import gymnasium as gym
import gymnasium.vector as gym_vector
import numpy as np
from sentence_transformers import SentenceTransformer

from .metaworld_utils import TASK_TO_LANG
from robometer.utils.embedding_utils import compute_text_embeddings


class LanguageInstructionWrapper(gym.ObservationWrapper):
    """
    Observation wrapper that adds a fixed language encoding to observations
    and exposes it in the observation_space for single (non-vector) envs.
    """

    def __init__(self, env: gym.Env, task_name: str, sentence_model: SentenceTransformer):
        super().__init__(env)
        self.language_instruction = TASK_TO_LANG[task_name]
        # Convert to numpy float32 for compatibility with Box dtype
        enc = compute_text_embeddings(self.language_instruction, sentence_model)
        self.language_encoding = enc.cpu().numpy().astype(np.float32)

        # Update observation space to include 'language'
        orig_space = self.env.observation_space
        if isinstance(orig_space, gym.spaces.Dict):
            spaces_dict = dict(orig_space.spaces)
        else:
            spaces_dict = {"obs": orig_space}
        spaces_dict["language"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self.language_encoding.shape, dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(spaces_dict)

    def observation(self, obs):
        if isinstance(obs, dict):
            out = dict(obs)
            out["language"] = self.language_encoding
            return out
        return obs

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        # Add language_instruction to info
        if isinstance(info, dict):
            info = dict(info)
            info["language_instruction"] = self.language_instruction
        else:
            info = {"language_instruction": self.language_instruction}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Apply observation wrapper processing
        obs = self.observation(obs)
        # Add language_instruction to info
        if isinstance(info, dict):
            info = dict(info)
            info["language_instruction"] = self.language_instruction
        else:
            info = {"language_instruction": self.language_instruction}
        return obs, reward, terminated, truncated, info

    def get_language_instruction(self) -> str:
        return self.language_instruction

    def __getattr__(self, name):
        """Forward unknown attributes/methods to wrapped environment."""
        return getattr(self.env, name)


class VectorInstructionWrapper(gym_vector.VectorWrapper):
    """Expose a task's language instruction without touching observations.

    ``RobometerRolloutWorker`` calls ``env.get_language_instruction()`` on every
    step, because the reward model is conditioned on the task text. The only
    wrapper that provided it, ``LanguageInstructionVectorWrapper``, is attached
    solely when a ``SentenceTransformer`` is available -- it exists to embed the
    instruction into a 384-d ``obs["language"]`` vector. So a run with a reward
    model but no sentence model (every ManiSkill SAC arm: the policy is a plain
    MLP over image+state and has no language input) died with
    ``'SyncVectorEnv' object has no attribute 'get_language_instruction'``.

    Attaching the embedding wrapper regardless would be the wrong fix: it needs a
    sentence model to compute the encoding, and it widens the observation space
    with a key the policy is not built to consume. This wrapper carries the text
    only -- the accessor plus ``info["language_instruction"]`` -- and leaves the
    observation space untouched.
    """

    def __init__(self, env: gym_vector.VectorEnv, instruction: str):
        super().__init__(env)
        self.language_instruction = instruction

    def _add_instruction(self, info, n: int):
        if isinstance(info, dict):
            info = dict(info)
            info["language_instruction"] = [self.language_instruction] * n
            return info
        if isinstance(info, (list, tuple)):
            return [
                dict(x, language_instruction=self.language_instruction) if isinstance(x, dict) else x for x in info
            ]
        return info

    def _num_envs(self, obs) -> int:
        try:
            return self.env.num_envs
        except Exception:
            return len(obs) if hasattr(obs, "__len__") else 1

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, self._add_instruction(info, self._num_envs(obs))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._add_instruction(info, self._num_envs(obs))

    def get_language_instruction(self) -> str:
        return self.language_instruction

    def __getattr__(self, name):
        return getattr(self.env, name)


class LanguageInstructionVectorWrapper(gym_vector.VectorWrapper):
    """
    Vectorized version of LanguageInstructionWrapper for gymnasium.vector.VectorEnv.
    Attaches a per-env language instruction into the info dict and in the observation dict under 'language'.
    Also updates the observation space to include the 'language' key (shape=384).
    """

    def __init__(
        self,
        env: gym_vector.VectorEnv,
        task_name: str = None,
        sentence_model: SentenceTransformer = None,
        instruction: str = None,
    ):
        """
        Args:
            task_name: Meta-World task id, looked up in ``TASK_TO_LANG``.
            instruction: Explicit instruction string. Takes precedence over
                ``task_name`` so non-Meta-World suites (e.g. ManiSkill) can
                supply their own text without registering it in the
                Meta-World mapping.
        """
        super().__init__(env)
        if instruction is None:
            if task_name not in TASK_TO_LANG:
                raise KeyError(
                    f"No language instruction for task '{task_name}'. Either add it to "
                    f"metaworld_utils.TASK_TO_LANG or pass instruction=... explicitly."
                )
            instruction = TASK_TO_LANG[task_name]
        print("Using language instruction: ", instruction)
        self.language_instruction = instruction
        enc = compute_text_embeddings(self.language_instruction, sentence_model)
        # Convert to numpy array for consistency with image embeddings
        self.language_encoding = enc.cpu().numpy().astype(np.float32)

        # Update observation spaces to add 'language' for both single and batched views
        # Batched observation_space
        orig_space = self.env.observation_space
        if isinstance(orig_space, gym.spaces.Dict):
            obs_space_dict = dict(orig_space.spaces)
        else:
            obs_space_dict = {"obs": orig_space}
        obs_space_dict["language"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(384,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(obs_space_dict)

        # Single env observation space (critical for callers using single_observation_space)
        orig_single = getattr(self.env, "single_observation_space", None)
        if orig_single is None:
            orig_single = orig_space
        if isinstance(orig_single, gym.spaces.Dict):
            single_obs_space_dict = dict(orig_single.spaces)
        else:
            single_obs_space_dict = {"obs": orig_single}
        single_obs_space_dict["language"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(384,), dtype=np.float32)
        self.single_observation_space = gym.spaces.Dict(single_obs_space_dict)

    def _add_language_to_obs(self, obs, n):
        # Adds language instruction under key 'language' for each env
        # language_encoding is already a numpy array from __init__
        if isinstance(obs, dict):
            obs = dict(obs)
            # Create numpy array with shape (n, embedding_dim) instead of list
            obs["language"] = np.tile(self.language_encoding, (n, 1))  # (n, 384)
            return obs
        elif isinstance(obs, (list, tuple)):
            # for list/tuple of dicts
            for i in range(n):
                if isinstance(obs[i], dict):
                    obs[i] = dict(obs[i])
                    obs[i]["language"] = self.language_encoding  # (384,)
            return obs
        else:
            return obs  # in case of unknown obs format

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        try:
            n = self.env.num_envs
        except Exception:
            n = len(obs) if hasattr(obs, "__len__") else 1
        instructions = [self.language_instruction] * n
        # Add language to observation
        obs = self._add_language_to_obs(obs, n)
        # Add language to info
        if isinstance(info, dict):
            info = dict(info)
            info["language_instruction"] = instructions
        else:
            info = [{"language_instruction": self.language_instruction} for _ in range(n)]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        try:
            n = self.env.num_envs
        except Exception:
            n = len(obs) if hasattr(obs, "__len__") else 1
        instructions = [self.language_instruction] * n
        # Add language to observation
        obs = self._add_language_to_obs(obs, n)
        # Add language to info
        if isinstance(info, dict):
            info = dict(info)
            info["language_instruction"] = instructions
        else:
            info = [{"language_instruction": self.language_instruction} for _ in range(n)]
        return obs, reward, terminated, truncated, info

    def get_language_instruction(self) -> str:
        return self.language_instruction

    def __getattr__(self, name):
        return getattr(self.env, name)
