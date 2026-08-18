"""ManiSkill3 -> robometer-policy-learning adapter.

Why this file exists
--------------------
ManiSkill3 is natively batched: even with ``num_envs=1`` it returns *torch*
tensors carrying a leading batch dimension, and its action space is shaped
accordingly.  ``gym.vector.SyncVectorEnv`` -- which the rest of this codebase
(rollout workers, replay buffers, DINO wrappers) is built around -- requires
each sub-environment to look like a plain unbatched numpy gym env.

``ManiSkillSingleEnvWrapper`` bridges the two: it strips the batch dimension,
converts torch -> numpy, and emits the ``{"state", "image"}`` observation dict
that ``ImageDictObsWrapper`` produces for MetaWorld, so every downstream
wrapper works unchanged.

Two behaviours differ deliberately from ``ImageDictObsWrapper``:

* **No vertical flip.** That wrapper flips because MuJoCo renders upside down.
  SAPIEN (ManiSkill's renderer) does not. Flipping here would feed upside-down
  frames to the VLM reward model and silently degrade every reward it returns.
* **Resize, not center-crop.** ManiSkill's render camera defaults to a much
  larger frame than MetaWorld's; center-cropping it to 224 would discard most
  of the scene, including the goal marker. We resize the full frame instead.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from robometer_policy_learning.envs.maniskill_utils import assert_task_allowed, get_task_spec


def _to_numpy(x: Any) -> Any:
    """Convert a torch tensor (possibly on GPU) to numpy, leaving others alone."""
    if hasattr(x, "detach") and hasattr(x, "cpu"):  # torch.Tensor duck-typing
        return x.detach().cpu().numpy()
    return x


def _debatch(x: np.ndarray, expected_ndim: int) -> np.ndarray:
    """Drop a leading singleton batch dim if present.

    Args:
        x: Array that may be shaped ``(1, ...)`` or already unbatched.
        expected_ndim: Rank of the unbatched array (1 for a state vector,
            3 for an HWC image).
    """
    x = np.asarray(x)
    if x.ndim == expected_ndim + 1 and x.shape[0] == 1:
        return x[0]
    return x


def _scalarize(x: Any) -> Any:
    """Reduce a possibly-batched tensor/array to a python scalar."""
    x = _to_numpy(x)
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return None
        return x.reshape(-1)[0].item()
    return x


def _resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """Resize an HWC uint8 image to ``size`` x ``size``."""
    if image.shape[0] == size and image.shape[1] == size:
        return image
    try:
        import cv2

        # cv2 wants (width, height); INTER_AREA is the right filter for downscaling.
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    except ImportError:
        from PIL import Image

        return np.asarray(Image.fromarray(image).resize((size, size), Image.BILINEAR))


class ManiSkillSingleEnvWrapper(gym.Wrapper):
    """Make one ManiSkill3 env behave like an unbatched numpy gym env.

    Emits observations as ``{"state": (D,) float32, "image": (S, S, 3) uint8}``.
    """

    def __init__(
        self,
        env: gym.Env,
        image_size: int = 224,
        use_full_state: bool = False,
        terminate_on_success: bool = True,
        proprio_dim: int = 9,
    ):
        super().__init__(env)
        self.image_size = image_size
        self.use_full_state = use_full_state
        # ManiSkill ends an episode the moment success fires. With a DENSE per-step
        # reward that inverts the incentive: succeeding early truncates the reward
        # stream, so loitering near the goal for the full horizon pays far more than
        # finishing. Observed directly on PullCube -- episode return rose 4.71 -> 21.43
        # while eval success fell 56% -> 10% and ent_coef collapsed to 0.002.
        # With this False the episode always runs to the horizon; success is still
        # reported in info, and eval measures success_once by OR-ing across steps.
        self.terminate_on_success = terminate_on_success
        self.proprio_dim = proprio_dim

        # --- action space: expose the unbatched view ------------------------
        act_space = env.action_space
        self._action_needs_batch_dim = False
        if isinstance(act_space, gym.spaces.Box) and len(act_space.shape) == 2 and act_space.shape[0] == 1:
            # ManiSkill handed us a (1, act_dim) space -> unwrap it, and remember
            # that step() has to put the batch dim back on.
            self._action_needs_batch_dim = True
            self.action_space = gym.spaces.Box(
                low=act_space.low[0], high=act_space.high[0], shape=act_space.shape[1:], dtype=act_space.dtype
            )
        else:
            self.action_space = act_space

        # --- observation space ---------------------------------------------
        # Probe the real env once; shapes vary by obs_mode/robot and guessing
        # them is how this class would silently break.
        raw_obs, _ = self.env.reset()
        state = self._extract_state(raw_obs)
        self._state_dim = int(state.shape[0])

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._state_dim,), dtype=np.float32),
                "image": gym.spaces.Box(low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8),
            }
        )

    # ------------------------------------------------------------------
    # observation plumbing
    # ------------------------------------------------------------------
    def _extract_state(self, raw_obs: Any) -> np.ndarray:
        """Build the proprioceptive (or full-state) vector as flat float32.

        ``use_full_state=True`` returns ManiSkill's whole privileged state
        vector (object poses, goal position, ...), which makes learning from
        scratch far more tractable but is not something a real robot has.
        ``False`` returns only the robot's own joint positions.
        """
        if self.use_full_state:
            flat = _to_numpy(raw_obs)
            if isinstance(flat, np.ndarray):
                arr = _debatch(flat, expected_ndim=1)
                return np.asarray(arr, dtype=np.float32).reshape(-1)
            # dict obs_mode: fall through to the concatenation branch below.

        # Proprioceptive only: read joint positions straight off the robot,
        # which is well-defined regardless of obs_mode (state/state_dict/rgb).
        # Skipped under use_full_state so a dict observation does not silently
        # collapse to proprioception when privileged state was requested.
        if not self.use_full_state:
            try:
                qpos = self.env.unwrapped.agent.robot.get_qpos()
                arr = _debatch(_to_numpy(qpos), expected_ndim=1)
                return np.asarray(arr, dtype=np.float32).reshape(-1)
            except Exception:
                pass

        # Fallbacks for non-standard agents or dict observations.
        if isinstance(raw_obs, dict):
            if not self.use_full_state:
                agent = raw_obs.get("agent", {})
                if isinstance(agent, dict) and "qpos" in agent:
                    arr = _debatch(_to_numpy(agent["qpos"]), expected_ndim=1)
                    return np.asarray(arr, dtype=np.float32).reshape(-1)

            # Recursively flatten every numeric leaf (state_dict obs_mode is
            # nested: {"agent": {"qpos": ...}, "extra": {...}}). Image leaves
            # are skipped -- they travel in the "image" key, not the state.
            flat: List[np.ndarray] = []

            def _collect(node: Any) -> None:
                if isinstance(node, dict):
                    for key in sorted(node):  # deterministic ordering
                        _collect(node[key])
                    return
                value = _to_numpy(node)
                if isinstance(value, np.ndarray) and value.dtype != np.uint8:
                    flat.append(np.asarray(value, dtype=np.float32).reshape(-1))

            _collect(raw_obs)
            if flat:
                concatenated = np.concatenate(flat)
                return concatenated if self.use_full_state else concatenated[: self.proprio_dim]
            raise RuntimeError("Could not derive a state vector from the ManiSkill observation dict.")

        arr = _debatch(_to_numpy(raw_obs), expected_ndim=1)
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        return arr if self.use_full_state else arr[: self.proprio_dim]

    def _render_image(self) -> np.ndarray:
        """Render the current frame as an HWC uint8 array, no flip."""
        image = self.env.render()
        image = _to_numpy(image)
        if image is None:
            raise RuntimeError("ManiSkill env.render() returned None; is render_mode='rgb_array' set?")
        image = _debatch(np.asarray(image), expected_ndim=3)
        if image.dtype != np.uint8:
            # Some render paths hand back float in [0, 1].
            if image.max() <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.shape[-1] == 4:  # drop alpha
            image = image[..., :3]
        image = _resize_image(image, self.image_size)
        return np.ascontiguousarray(image)

    def _build_obs(self, raw_obs: Any) -> Dict[str, np.ndarray]:
        return {"state": self._extract_state(raw_obs), "image": self._render_image()}

    @staticmethod
    def _clean_info(info: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten ManiSkill's batched tensor info into plain python values.

        ``success`` matters most: the rollout and evaluation workers do
        ``info.get("success", False)``, and a 1-element torch tensor there
        would be truthy-but-wrong rather than a real boolean.
        """
        if not isinstance(info, dict):
            return {}
        out: Dict[str, Any] = {}
        for key, value in info.items():
            converted = _to_numpy(value)
            if isinstance(converted, np.ndarray) and converted.size == 1:
                scalar = converted.reshape(-1)[0]
                out[key] = bool(scalar) if converted.dtype == np.bool_ else scalar.item()
            elif isinstance(converted, (bool, np.bool_)):
                out[key] = bool(converted)
            else:
                out[key] = converted
        # Keep both spellings the workers look for.
        if "success" in out:
            out["success"] = bool(out["success"])
            out.setdefault("is_success", out["success"])
        return out

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        kwargs: Dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = seed
        if options is not None:
            kwargs["options"] = options
        raw_obs, info = self.env.reset(**kwargs)
        return self._build_obs(raw_obs), self._clean_info(info)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if self._action_needs_batch_dim and action.ndim == 1:
            action = action[None, ...]
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        term = bool(_scalarize(terminated))
        if not self.terminate_on_success:
            # Suppress ONLY the termination signal. info["success"] is untouched, so
            # evaluation still measures success correctly (it ORs success across the
            # episode), and the agent no longer loses reward by finishing the task.
            term = False
        return (
            self._build_obs(raw_obs),
            float(_scalarize(reward)),
            term,
            bool(_scalarize(truncated)),
            self._clean_info(info),
        )


def make_maniskill_env(
    env_id: str,
    num_envs: int = 1,
    image_size: int = 224,
    use_full_state: bool = False,
    max_episode_steps: Optional[int] = None,
    control_mode: Optional[str] = None,
    reward_mode: str = "sparse",
    obs_mode: str = "state",
    sim_backend: str = "physx_cpu",
    seed: Optional[int] = None,
    instruction: Optional[str] = None,
    terminate_on_success: bool = True,
    sentence_model=None,
    chunk_size: Optional[int] = None,
    dinov2_model=None,
    dinov2_processor=None,
    dino_image_keys: Optional[List[str]] = None,
    device: Optional[str] = None,
) -> Tuple[gym.Env, str]:
    """Build a vectorized ManiSkill env matching this codebase's conventions.

    The wrapper stack mirrors the MetaWorld path in ``env_utils`` exactly:
    per-env dict observations -> ``SyncVectorEnv`` -> language -> action
    chunking -> DINO embeddings.

    Args:
        env_id: ManiSkill task id, e.g. ``"PullCube-v1"``. Rejected if it
            belongs to the FailSafe (PickCube/PushCube/StackCube) families.
        reward_mode: ``"sparse"`` for the reward-model experiments (the
            environment reward is then ignored/replaced), or
            ``"normalized_dense"`` / ``"dense"`` for the ground-truth control
            run that upper-bounds what the setup can reach.
        sim_backend: ``"physx_cpu"`` keeps one env per process, which is what
            ``SyncVectorEnv`` and the rest of this codebase expect.

    Returns:
        ``(env, instruction)``.
    """
    # Import here, not at module scope: ManiSkill is an optional dependency and
    # importing it also registers its envs with gymnasium.
    try:
        import mani_skill.envs  # noqa: F401  (registers ManiSkill env ids)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "ManiSkill is not installed. Install it with `pip install 'mani-skill>=3.0.0'` "
            "(or `pip install mani-skill`) and make sure a GPU/EGL-capable renderer "
            "is available for offscreen rendering."
        ) from exc

    assert_task_allowed(env_id)
    spec = get_task_spec(env_id)
    max_episode_steps = max_episode_steps or spec.max_episode_steps
    control_mode = control_mode or spec.control_mode
    instruction = instruction or spec.instruction

    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    def make_single_env(rank: int):
        def _thunk():
            env = gym.make(
                env_id,
                num_envs=1,
                # "state" gives a flat privileged vector. The wrapper decides
                # what to expose from it via use_full_state; the policy only
                # ever sees proprioception + DINO image features unless
                # use_full_state is on.
                obs_mode=obs_mode,
                control_mode=control_mode,
                render_mode="rgb_array",
                reward_mode=reward_mode,
                sim_backend=sim_backend,
                max_episode_steps=max_episode_steps,
            )
            env = ManiSkillSingleEnvWrapper(
                env,
                image_size=image_size,
            terminate_on_success=terminate_on_success,
                use_full_state=use_full_state,
            )
            if seed is not None:
                env.reset(seed=seed + rank)
            return env

        return _thunk

    env = gym.vector.SyncVectorEnv([make_single_env(i) for i in range(num_envs)])

    # Language: reuse the shared vector wrapper but pass the instruction
    # explicitly so it does not go looking in MetaWorld's TASK_TO_LANG.
    #
    # The instruction is attached either way. With a sentence model it is also
    # embedded into obs["language"] for language-conditioned policies; without
    # one, only the text is carried. It cannot be skipped in the no-model case:
    # RobometerRolloutWorker calls env.get_language_instruction() on every step
    # to condition the reward model, and the ManiSkill SAC arms run a plain MLP
    # policy with no sentence model at all.
    if sentence_model is not None:
        from robometer_policy_learning.envs.language_wrappers import LanguageInstructionVectorWrapper

        env = LanguageInstructionVectorWrapper(env, instruction=instruction, sentence_model=sentence_model)
    else:
        from robometer_policy_learning.envs.language_wrappers import VectorInstructionWrapper

        env = VectorInstructionWrapper(env, instruction=instruction)

    if chunk_size is not None:
        from robometer_policy_learning.envs.action_wrappers import VectorActionChunkingWrapper

        env = VectorActionChunkingWrapper(env, chunk_size=chunk_size, n_action_steps=1)

    if dinov2_model is not None:
        from robometer_policy_learning.envs.dino_wrapper import VectorDinoEmbeddingWrapper

        single_space = getattr(env, "single_observation_space", env.observation_space)
        if isinstance(single_space, gym.spaces.Dict) and "image" in single_space.spaces:
            env = VectorDinoEmbeddingWrapper(
                env,
                dinov2_model,
                dinov2_processor,
                device=device,
                image_keys=dino_image_keys or ["image"],
            )

    return env, instruction
