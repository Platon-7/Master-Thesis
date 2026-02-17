"""
DSRL Training with RoboReward VLM Rewards

This script trains DSRL on Robomimic tasks using RoboReward-8B as the reward function
instead of the ground-truth simulator rewards.

Usage:
    python train_dsrl_vlm.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml

Key modifications from original train_dsrl.py:
1. Loads RoboReward-8B model at startup
2. Uses VLMRewardWrapperRobomimic instead of ObservationWrapperRobomimic
3. Logs both VLM and simulator rewards for comparison
"""

import os
import warnings
warnings.filterwarnings("ignore")
import math
import torch
import random
import wandb
import numpy as np
import hydra
from omegaconf import OmegaConf
import gym
import sys
sys.path.append('./dppo')

from stable_baselines3 import SAC, DSRL
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from env_utils import DiffusionPolicyEnvWrapper, ActionChunkWrapper
from vlm_reward_wrapper import make_vlm_robomimic_env
from roboreward_wrapper import get_roboreward_model
from utils import load_base_policy, load_offline_data, collect_rollouts, LoggingCallback, VLMLoggingCallback

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

base_path = os.path.dirname(os.path.abspath(__file__))


@hydra.main(
    config_path=os.path.join(base_path, "cfg/robomimic"), config_name="dsrl_can.yaml", version_base=None
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # =========================================================
    # NEW: Load RoboReward model ONCE at startup
    # =========================================================
    use_vlm_reward = cfg.get('use_vlm_reward', True)
    
    # Determine VLM device (use GPU 1 if available, else same as policy)
    vlm_device = "cuda:1" if torch.cuda.device_count() > 1 else cfg.device
    
    if use_vlm_reward:
        print("\n" + "="*60)
        print("🤖 DSRL + RoboReward VLM Training (2 GPU Mode)")
        print("="*60)
        print(f"   GPU 0 ({cfg.device}): DSRL Policy (Diffusion + SAC)")
        print(f"   GPU 1 ({vlm_device}): RoboReward-8B VLM")
        print("="*60)
        roboreward_model = get_roboreward_model(device=vlm_device)
        print(f"Task: {cfg.env_name}")
        print(f"Task instruction: {roboreward_model.get_task_instruction(cfg.env_name)}")
        print("="*60 + "\n")
    else:
        roboreward_model = None
        print("\n⚠️  Running with SIMULATOR rewards (VLM disabled)\n")

    if cfg.use_wandb:
        # Add VLM info to wandb config
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        wandb_config['use_vlm_reward'] = use_vlm_reward
        wandb_config['reward_type'] = 'vlm_roboreward' if use_vlm_reward else 'simulator'
        
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.name + ("_vlm" if use_vlm_reward else "_sim"),
            group=cfg.wandb.group,
            monitor_gym=True,
            save_code=True,
            config=wandb_config,
        )

    MAX_STEPS = int(cfg.env.max_episode_steps / cfg.act_steps)

    num_env = cfg.env.n_envs
    
    # =========================================================
    # MODIFIED: Environment factory with VLM rewards (TRAINING ONLY)
    # =========================================================
    def make_train_env():
        """Training environment with VLM rewards."""
        if cfg.env_name in ['lift', 'can', 'square', 'transport']:
            env = make_vlm_robomimic_env(
                env_name=cfg.env_name,
                normalization_path=cfg.normalization_path,
                low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys,
                dppo_path=cfg.dppo_path,
                roboreward_model=roboreward_model,
                use_vlm_reward=use_vlm_reward,  # VLM rewards for training
                frame_interval=5,
            )
        else:
            raise ValueError(f"Unknown environment: {cfg.env_name}. "
                           f"Supported: lift, can, square, transport")
        
        env = ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)
        return env

    def make_eval_env():
        """
        Evaluation environment with SIMULATOR rewards (ground-truth).
        
        IMPORTANT: We use simulator rewards for evaluation to get trustworthy
        success metrics. The agent learns from VLM rewards, but we measure
        actual task success using the simulator's ground-truth rewards.
        """
        if cfg.env_name in ['lift', 'can', 'square', 'transport']:
            env = make_vlm_robomimic_env(
                env_name=cfg.env_name,
                normalization_path=cfg.normalization_path,
                low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys,
                dppo_path=cfg.dppo_path,
                roboreward_model=roboreward_model,  # Still need for logging VLM scores
                use_vlm_reward=False,  # ← SIMULATOR rewards for evaluation!
                frame_interval=5,
            )
        else:
            raise ValueError(f"Unknown environment: {cfg.env_name}.")
        
        env = ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)
        return env

    base_policy = load_base_policy(cfg)
    
    # Note: Using DummyVecEnv instead of SubprocVecEnv because the VLM model
    # needs to be shared across environments (can't pickle CUDA tensors)
    # For parallel training, you'd need to restructure the VLM calls
    env = make_vec_env(make_train_env, n_envs=num_env, vec_env_cls=DummyVecEnv)
    
    if cfg.algorithm == 'dsrl_sac':
        env = DiffusionPolicyEnvWrapper(env, cfg, base_policy)
    env.seed(cfg.seed + 1)
    
    post_linear_modules = None
    if cfg.train.use_layer_norm:
        post_linear_modules = [torch.nn.LayerNorm]

    net_arch = []
    for _ in range(cfg.train.num_layers):
        net_arch.append(cfg.train.layer_size)
    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, qf=net_arch),
        activation_fn=torch.nn.Tanh,
        log_std_init=0.0,
        post_linear_modules=post_linear_modules,
        n_critics=cfg.train.n_critics,
    )
    
    if cfg.algorithm == 'dsrl_sac':
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=cfg.train.actor_lr,
            buffer_size=20000000,
            learning_starts=1,
            batch_size=cfg.train.batch_size,
            tau=cfg.train.tau,
            gamma=cfg.train.discount,
            train_freq=cfg.train.train_freq,
            gradient_steps=cfg.train.utd,
            action_noise=None,
            optimize_memory_usage=False,
            ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,
            target_update_interval=1,
            target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,
            use_sde=False,
            sde_sample_freq=-1,
            tensorboard_log=cfg.logdir,
            verbose=1,
            policy_kwargs=policy_kwargs,
        )
    elif cfg.algorithm == 'dsrl_na':
        model = DSRL(
            "MlpPolicy",
            env,
            learning_rate=cfg.train.actor_lr,
            buffer_size=10000000,
            learning_starts=1,
            batch_size=cfg.train.batch_size,
            tau=cfg.train.tau,
            gamma=cfg.train.discount,
            train_freq=cfg.train.train_freq,
            gradient_steps=cfg.train.utd,
            action_noise=None,
            optimize_memory_usage=False,
            ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,
            target_update_interval=1,
            target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,
            use_sde=False,
            sde_sample_freq=-1,
            tensorboard_log=cfg.logdir,
            verbose=1,
            policy_kwargs=policy_kwargs,
            diffusion_policy=base_policy,
            diffusion_act_dim=(cfg.act_steps, cfg.action_dim),
            noise_critic_grad_steps=cfg.train.noise_critic_grad_steps,
            critic_backup_combine_type=cfg.train.critic_backup_combine_type,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.save_model_interval,
        save_path=cfg.logdir+'/checkpoint/',
        name_prefix='ft_policy_vlm' if use_vlm_reward else 'ft_policy',
        save_replay_buffer=cfg.save_replay_buffer,
        save_vecnormalize=True,
    )

    # =========================================================
    # EVALUATION ENV: Uses SIMULATOR rewards (ground-truth)
    # This ensures trustworthy success metrics!
    # =========================================================
    num_env_eval = cfg.env.n_eval_envs
    eval_env = make_vec_env(make_eval_env, n_envs=min(num_env_eval, 4), vec_env_cls=DummyVecEnv)
    if cfg.algorithm == 'dsrl_sac':
        eval_env = DiffusionPolicyEnvWrapper(eval_env, cfg, base_policy)
    eval_env.seed(cfg.seed + num_env + 1)

    # =========================================================
    # LOGGING: Use VLMLoggingCallback for VLM training
    # =========================================================
    csv_log_path = os.path.join(cfg.logdir, f"training_log_{cfg.env_name}.csv")
    
    if use_vlm_reward:
        logging_callback = VLMLoggingCallback(
            action_chunk=cfg.act_steps,
            eval_episodes=int(cfg.num_evals / num_env_eval),
            log_freq=MAX_STEPS,
            use_wandb=cfg.use_wandb,
            eval_env=eval_env,
            eval_freq=cfg.eval_interval,
            num_train_env=num_env,
            num_eval_env=min(num_env_eval, 4),
            algorithm=cfg.algorithm,
            max_steps=MAX_STEPS,
            deterministic_eval=cfg.deterministic_eval,
            csv_log_path=csv_log_path,
            vlm_success_threshold=4,  # VLM score >= 4 counts as success
        )
    else:
        # Original callback for simulator rewards
        logging_callback = LoggingCallback(
            action_chunk=cfg.act_steps,
            eval_episodes=int(cfg.num_evals / num_env_eval),
            log_freq=MAX_STEPS,
            use_wandb=cfg.use_wandb,
            eval_env=eval_env,
            eval_freq=cfg.eval_interval,
            num_train_env=num_env,
            num_eval_env=min(num_env_eval, 4),
            rew_offset=cfg.env.reward_offset,
            algorithm=cfg.algorithm,
            max_steps=MAX_STEPS,
            deterministic_eval=cfg.deterministic_eval,
        )

    logging_callback.evaluate(model, deterministic=False)
    if cfg.deterministic_eval:
        logging_callback.evaluate(model, deterministic=True)
    logging_callback.log_count += 1

    if cfg.load_offline_data:
        load_offline_data(model, cfg.offline_data_path, num_env)
    if cfg.train.init_rollout_steps > 0:
        collect_rollouts(model, env, cfg.train.init_rollout_steps, base_policy, cfg)
        logging_callback.set_timesteps(cfg.train.init_rollout_steps * num_env)

    callbacks = [checkpoint_callback, logging_callback]
    
    print("\n" + "="*60)
    print("🚀 Starting DSRL Training")
    print(f"   Algorithm: {cfg.algorithm}")
    print(f"   Reward: {'RoboReward VLM' if use_vlm_reward else 'Simulator'}")
    print(f"   Environment: {cfg.env_name}")
    print(f"   Parallel envs: {num_env}")
    print("="*60 + "\n")
    
    # Train the agent
    total_timesteps = cfg.get('total_timesteps', 250000)
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks
    )

    # Save the final model
    if len(cfg.name) > 0:
        model.save(cfg.logdir+"/checkpoint/final_vlm" if use_vlm_reward else cfg.logdir+"/checkpoint/final")

    # Close environment and wandb
    env.close()
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
