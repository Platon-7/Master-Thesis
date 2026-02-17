"""
DSRL Environment Wrapper with RoboReward VLM Integration

This wrapper replaces the simulator's ground-truth reward with VLM-based rewards
from RoboReward-8B. It collects frames during the episode and scores at the end.

Key design decisions:
1. Frames are collected throughout the episode
2. VLM is called ONCE at episode end (as per RoboReward paper)
3. Sparse reward is given only at the final timestep
4. Compatible with DSRL's action chunking
"""

import numpy as np
import gym
from typing import List


class VLMRewardWrapperRobomimic(gym.Env):
    """
    Complete wrapper that combines observation processing + VLM rewards.
    This replaces ObservationWrapperRobomimic when using VLM rewards.
    """
    
    def __init__(
        self,
        env,
        task_name: str,
        roboreward_model=None,
        reward_offset: float = 1.0,  # Original DSRL uses offset
        frame_interval: int = 5,
        use_vlm_reward: bool = True,
    ):
        self.env = env
        self.task_name = task_name
        self.roboreward_model = roboreward_model
        self.reward_offset = reward_offset
        self.frame_interval = frame_interval
        self.use_vlm_reward = use_vlm_reward
        
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        
        # Episode state
        self.episode_frames: List[np.ndarray] = []
        self.step_count = 0
        self.episode_sim_reward = 0.0
        
        # Task instruction
        if roboreward_model is not None:
            self.task_instruction = roboreward_model.get_task_instruction(task_name)
        else:
            self.task_instruction = f"Complete the {task_name} task"
    
    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
    
    def reset(self, **kwargs):
        self.episode_frames = []
        self.step_count = 0
        self.episode_sim_reward = 0.0
        
        raw_obs = self.env.reset()
        obs = raw_obs['state'].flatten()
        
        # Capture first frame (from observation if available)
        self._capture_frame_from_env()
        
        return obs
    
    def step(self, action):
        raw_obs, sim_reward, done, info = self.env.step(action)
        obs = raw_obs['state'].flatten()
        
        self.step_count += 1
        self.episode_sim_reward += sim_reward
        
        # Collect frames only when using VLM rewards (skip rendering in eval mode)
        if self.use_vlm_reward and self.step_count % self.frame_interval == 0:
            self._capture_frame_from_env()
        
        # Compute reward based on mode
        if self.use_vlm_reward:
            # TRAINING MODE: Use VLM rewards
            if done:
                self._capture_frame_from_env()  # Final frame
                reward = self._compute_vlm_reward()
                info['vlm_reward'] = reward
                info['sim_reward'] = self.episode_sim_reward
                info['sim_success'] = 1 if sim_reward > 0 else 0
                info['num_frames_scored'] = len(self.episode_frames)
            else:
                reward = 0.0  # Sparse: 0 during episode
        else:
            # EVALUATION MODE: Use simulator rewards (ground-truth)
            # Skip VLM calls entirely to avoid expensive inference during eval
            reward = sim_reward - self.reward_offset

            if done:
                info['sim_reward'] = self.episode_sim_reward
                info['sim_success'] = 1 if self.episode_sim_reward > 0 else 0
        
        return obs, reward, done, info
    
    def _capture_frame_from_env(self):
        """Capture frame from environment render."""
        try:
            frame = self.env.render(mode='rgb_array')
            if frame is not None:
                self.episode_frames.append(frame)
        except Exception as e:
            print(f"Warning: failed to capture frame: {e}")
    
    def _compute_vlm_reward(self) -> float:
        """Compute VLM reward using raw RoboReward progress score {1,...,5}."""
        if self.roboreward_model is None or len(self.episode_frames) == 0:
            return 5.0 if self.episode_sim_reward > 0 else 1.0

        return self.roboreward_model.compute_reward(
            self.episode_frames,
            self.task_instruction,
            return_raw_score=True
        )
    
    def force_episode_end(self):
        """Force reward computation when episode ends due to TimeLimit.
        Call this from outer wrapper (ActionChunkWrapper) when TimeLimit is reached.
        """
        if not self.use_vlm_reward:
            # Even in eval mode, report sim metrics so evaluation logging works
            return {
                'sim_reward': self.episode_sim_reward,
                'sim_success': 1 if self.episode_sim_reward > 0 else 0,
            }
        
        self._capture_frame_from_env()  # Final frame
        reward = self._compute_vlm_reward()

        return {
            'vlm_reward': reward,
            'sim_reward': self.episode_sim_reward,
            'sim_success': 1 if self.episode_sim_reward > 0 else 0,
            'num_frames_scored': len(self.episode_frames),
        }
    
    def render(self, **kwargs):
        return self.env.render(**kwargs)


def make_vlm_robomimic_env(
    env_name: str,
    normalization_path: str,
    low_dim_keys: list,
    dppo_path: str,
    roboreward_model=None,
    use_vlm_reward: bool = True,
    frame_interval: int = 5,
):
    """
    Factory function to create a Robomimic environment with VLM rewards.
    
    This replaces the `make_robomimic_env` function when using VLM rewards.
    """
    from omegaconf import OmegaConf
    import json
    from dppo.env.gym_utils.wrapper import wrapper_dict
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    
    wrappers = OmegaConf.create({
        'robomimic_lowdim': {
            'normalization_path': normalization_path,
            'low_dim_keys': low_dim_keys,
        },
    })
    
    obs_modality_dict = {
        "low_dim": wrappers.robomimic_lowdim.low_dim_keys,
    }
    ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)
    
    robomimic_env_cfg_path = f'{dppo_path}/cfg/robomimic/env_meta/{env_name}.json'
    with open(robomimic_env_cfg_path, "r") as f:
        env_meta = json.load(f)
    env_meta["reward_shaping"] = False
    
    # Enable rendering for frame capture
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,  # Enable offscreen rendering
        use_image_obs=False,
    )
    env.env.hard_reset = False
    
    # Apply normalization wrapper
    for wrapper, args in wrappers.items():
        env = wrapper_dict[wrapper](env, **args)
    
    # Wrap with VLM reward wrapper
    env = VLMRewardWrapperRobomimic(
        env,
        task_name=env_name,
        roboreward_model=roboreward_model,
        use_vlm_reward=use_vlm_reward,
        frame_interval=frame_interval,
    )
    
    return env
