"""GPU-parallel ManiSkill environments for the existing rollout worker.

The CPU path (``maniskill_wrapper.ManiSkillSingleEnvWrapper`` + ``SyncVectorEnv``)
steps one physics simulation per environment on the CPU, which caps us at ~4
environments and ~50-80 steps/s. ManiSkill's own results use ``physx_cuda`` with
hundreds to thousands of environments simulated as batched tensors.

Two separate wins, and the second is the one that matters for this project:

  * ground-truth / ablation runs stop being hour-scale;
  * the reward model batches. ``RobometerReplayBuffer._compute_rewards_batch``
    scores a whole step's transitions in one VLM forward, so N parallel
    environments give an N-sample batch instead of N sequential 1-sample
    forwards. The VLM is the bottleneck in every reward-model arm (measured at
    5.65 steps/s with 4 environments), and batching is what moves it.

Why this is a thin adapter rather than a new rollout path: ``RolloutWorker``
already indexes per environment (``rewards[i]``, ``dones[i]``,
``extract_info_for_env(infos, i, num_envs)``), and that last helper already
understands gymnasium's dict-of-batched-arrays info format. So all that is
missing is presenting ManiSkill's batched GPU tensors as leading-dimension numpy
arrays. That is this file.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch


def _np(x: Any) -> Any:
    """Detach any torch tensor (possibly on GPU) to numpy, leave the rest alone."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


class ManiSkillGPUVectorWrapper(gym.vector.VectorEnv):
    """Present a GPU-batched ManiSkill env with the numpy vector-env conventions.

    Observations are ``{"state": (N, D) float32, "image": (N, H, W, 3) uint8}`` --
    the same keys the CPU wrapper emits, so downstream code (policy featurizers,
    the DINO wrapper, the reward-model buffer) needs no changes.
    """

    def __init__(
        self,
        env,
        num_envs: int,
        instruction: str,
        image_size: int = 224,
        use_full_state: bool = False,
        render_every_step: bool = True,
    ):
        self._env = env
        self.num_envs = num_envs
        self.language_instruction = instruction
        self.image_size = image_size
        self.use_full_state = use_full_state
        # Rendering N environments every step is the dominant sim-side cost. It is
        # required whenever a reward model consumes frames; GT-only runs can skip it.
        self.render_every_step = render_every_step

        obs, _ = self._env.reset()
        state = self._extract_state(obs)
        img = self._render() if render_every_step else np.zeros(
            (num_envs, image_size, image_size, 3), dtype=np.uint8
        )

        self.single_observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(-np.inf, np.inf, shape=(state.shape[1],), dtype=np.float32),
                "image": gym.spaces.Box(0, 255, shape=img.shape[1:], dtype=np.uint8),
            }
        )
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, num_envs)
        self.single_action_space = self._env.single_action_space
        self.action_space = self._env.action_space

    # ------------------------------------------------------------------
    def _extract_state(self, obs: Any) -> np.ndarray:
        """(N, D) float32 state: full privileged vector, or robot qpos only."""
        if self.use_full_state:
            arr = _np(obs)
            if isinstance(arr, np.ndarray):
                return arr.reshape(self.num_envs, -1).astype(np.float32)
        qpos = _np(self._env.unwrapped.agent.robot.get_qpos())
        return np.asarray(qpos).reshape(self.num_envs, -1).astype(np.float32)

    def _render(self) -> np.ndarray:
        """(N, image_size, image_size, 3) uint8 frames."""
        frames = self._env.unwrapped.render_rgb_array() if hasattr(
            self._env.unwrapped, "render_rgb_array"
        ) else self._env.render()
        if isinstance(frames, torch.Tensor):
            f = frames.detach()
            if f.dtype != torch.uint8:
                f = (f.clamp(0, 1) * 255).to(torch.uint8) if f.max() <= 1.0 else f.to(torch.uint8)
            # Resize on GPU (NHWC -> NCHW -> NHWC); far cheaper than a CPU round trip.
            if f.shape[1] != self.image_size or f.shape[2] != self.image_size:
                f = torch.nn.functional.interpolate(
                    f.permute(0, 3, 1, 2).float(),
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1).to(torch.uint8)
            return f.cpu().numpy()
        arr = np.asarray(_np(frames)).astype(np.uint8)
        return arr.reshape(self.num_envs, *arr.shape[-3:])

    def _obs(self, raw_obs: Any) -> Dict[str, np.ndarray]:
        img = self._render() if self.render_every_step else np.zeros(
            (self.num_envs, self.image_size, self.image_size, 3), dtype=np.uint8
        )
        return {"state": self._extract_state(raw_obs), "image": img}

    @staticmethod
    def _infos(info: Any) -> Dict[str, Any]:
        """Batched tensors -> batched numpy, keeping the dict-of-arrays layout
        that ``extract_info_for_env`` already slices per environment.

        Also folds ``final_info`` back into the top level, which is not cosmetic:
        ``ManiSkillVectorEnv`` auto-resets using gymnasium's OLD convention, so on
        the step an episode ends, ``info["success"]`` already describes the FRESH
        episode (False) and the real terminal flag sits in
        ``info["final_info"]["success"]``. Gymnasium 1.3's ``SyncVectorEnv`` -- our
        CPU path -- uses the new convention and returns terminal info directly,
        which is why CPU worked and GPU scored 0/64 with a policy known to succeed
        60% of the time. ManiSkill terminates on success, so without this merge the
        success flag is never observable at all.
        """
        if not isinstance(info, dict):
            return {}
        out: Dict[str, Any] = {}
        for k, v in info.items():
            if isinstance(v, dict):
                out[k] = ManiSkillGPUVectorWrapper._infos(v)
            else:
                out[k] = _np(v)

        final = out.pop("final_info", None)
        mask = out.pop("_final_info", None)
        out.pop("final_observation", None)
        out.pop("_final_observation", None)
        if isinstance(final, dict) and mask is not None:
            m = np.asarray(mask).astype(bool)
            for k, v in final.items():
                fv = np.asarray(v)
                if k in out and isinstance(out[k], np.ndarray) and out[k].shape[:1] == m.shape[:1]:
                    merged = np.array(out[k], copy=True)
                    merged[m] = fv[m]
                    out[k] = merged
                else:
                    out[k] = fv
        return out

    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        raw, info = self._env.reset(seed=seed, options=options)
        return self._obs(raw), self._infos(info)

    def step(self, actions: np.ndarray):
        raw, rew, term, trunc, info = self._env.step(actions)
        return (
            self._obs(raw),
            np.asarray(_np(rew)).reshape(-1).astype(np.float64),
            np.asarray(_np(term)).reshape(-1).astype(bool),
            np.asarray(_np(trunc)).reshape(-1).astype(bool),
            self._infos(info),
        )

    def get_language_instruction(self) -> str:
        return self.language_instruction

    def close(self):
        self._env.close()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make_maniskill_gpu_env(
    env_id: str,
    num_envs: int = 32,
    use_full_state: bool = False,
    max_episode_steps: Optional[int] = None,
    seed: Optional[int] = None,
    instruction: Optional[str] = None,
    image_size: int = 224,
    control_mode: Optional[str] = None,
    reward_mode: str = "normalized_dense",
    obs_mode: str = "state",
    render_every_step: bool = True,
    terminate_on_success: bool = True,
    reconfiguration_freq: Optional[int] = None,
    **_ignored,
) -> Tuple[ManiSkillGPUVectorWrapper, str]:
    """Build a GPU-parallel ManiSkill env behind the numpy vector-env interface."""
    import mani_skill.envs  # noqa: F401
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    from robometer_policy_learning.envs.maniskill_utils import assert_task_allowed, get_task_spec

    assert_task_allowed(env_id)
    spec = get_task_spec(env_id)
    instruction = instruction or spec.instruction

    # NOTE: reconfiguration_freq=1 (which ManiSkill's own demo env_kwargs use) is
    # NOT compatible with auto_reset here -- ManiSkill raises "With partial resets,
    # environment cannot be reconfigured automatically". Left at the default; a
    # per-episode geometry rebuild would need auto_reset disabled and manual resets.
    base = gym.make(
        env_id,
        num_envs=num_envs,
        obs_mode=obs_mode,
        control_mode=control_mode or spec.control_mode,
        render_mode="rgb_array",
        reward_mode=reward_mode,
        sim_backend="physx_cuda",
        max_episode_steps=max_episode_steps or spec.max_episode_steps,
    )
    # ignore_terminations=True is ManiSkill's own switch for "run to the horizon":
    # it stops success from ending the episode, which with a dense per-step reward
    # otherwise makes finishing the task cost the agent reward.
    vec = ManiSkillVectorEnv(base, num_envs=num_envs, auto_reset=True,
                             ignore_terminations=not terminate_on_success)
    if seed is not None:
        vec.reset(seed=seed)

    return (
        ManiSkillGPUVectorWrapper(
            vec,
            num_envs=num_envs,
            instruction=instruction,
            image_size=image_size,
            use_full_state=use_full_state,
            render_every_step=render_every_step,
        ),
        instruction,
    )
