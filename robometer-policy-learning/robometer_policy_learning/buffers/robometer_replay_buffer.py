import numpy as np
from typing import List, Dict, Optional, Any, Tuple
import os
import hashlib
import pickle
import torch

from loguru import logger
from collections import deque
from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer
from robometer_policy_learning.buffers.h5_replay_buffer import H5ReplayBuffer

from robometer_policy_learning.utils.robometer_utils import (
    extract_rewards_from_output,
    extract_success_probs_from_output,
    extract_rewards_from_server_output,
)
from robometer_policy_learning.utils.gpu_utils import convert_to_numpy
from robometer.evals.eval_utils import raw_dict_to_sample, build_payload, post_batch_npy
from robometer.evals.eval_server import process_batch_helper
from robometer.utils.embedding_utils import compute_text_embeddings, compute_video_embeddings
from transformers import AutoModel, AutoImageProcessor
from sentence_transformers import SentenceTransformer
from robometer.utils.setup_utils import setup_batch_collator
from tqdm import tqdm


class RobometerReplayBuffer(ReplayBuffer):
    """
    Robometer replay buffer for storing and sampling experience transitions.
    Rewards are estimated before adding the transition to the buffer.
    """

    def __init__(
        self,
        reward_model=None,
        reward_model_config=None,
        use_relative_rewards: bool = False,
        use_eval_server: bool = False,
        eval_server_url: Optional[str] = None,
        eval_server_timeout: float = 120.0,
        reward_relabeling_keys: List[str] = ["image"],
        use_success_detection: bool = False,
        success_detection_duration: int = 2,
        success_detection_min_ep_steps: int = 0,
        success_detection_threshold: float = 0.65,
        add_estimated_reward: bool = False,
        icl_demo_path: Optional[str] = None,
        icl_demo_seed: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reward_model = reward_model
        self.use_eval_server = use_eval_server
        self.eval_server_url = eval_server_url
        self.eval_server_timeout = eval_server_timeout
        self.reward_relabeling_keys = reward_relabeling_keys
        self.use_success_detection = use_success_detection
        self.success_detection_duration = success_detection_duration
        # A fire before this many steps into the episode cannot be a real success --
        # the same guard as vlm_ibrl's ROBOMETER_MIN_EP_STEPS. Set it between the
        # latest "fake fire" and the earliest "real fire" reported by
        # scripts/causal_calib_maniskill.py. 0 disables the gate.
        self.success_detection_min_ep_steps = int(success_detection_min_ep_steps)
        self._ep_step = 0
        self.success_detection_threshold = success_detection_threshold
        self.add_estimated_reward = add_estimated_reward

        # Set max_frames once from config
        if reward_model_config is not None:
            self.max_frames = getattr(reward_model_config.data, "max_frames", 16)
        else:
            self.max_frames = 16

        if self.reward_model is not None:
            self.reward_model_config = reward_model_config
            self.processor = getattr(reward_model, "processor", None)
            self.tokenizer = getattr(reward_model, "tokenizer", None)
            if self.processor is None or self.tokenizer is None:
                raise ValueError(
                    "processor and tokenizer must be available on reward_model (reward_model.processor / reward_model.tokenizer)"
                )
            # Ensure use_multi_image is True for reward relabeling (process frames as images, not video)
            if not self.reward_model_config.data.use_multi_image:
                print("Warning: use_multi_image is False in config. Setting to True for reward relabeling.")
                self.reward_model_config.data.use_multi_image = True

            # Set up batch collator with inference=True for evaluation
            self.batch_collator = setup_batch_collator(
                self.processor, self.tokenizer, self.reward_model_config, is_eval=True
            )
        elif self.use_eval_server:
            if self.eval_server_url is None:
                raise ValueError("eval_server_url must be provided when use_eval_server=True")
            logger.info(f"Using eval_server at {self.eval_server_url} for reward computation")

        self.use_relative_rewards = use_relative_rewards
        if self.use_relative_rewards:
            self.prev_reward = {key: 0.0 for key in self.reward_relabeling_keys}
        self.success_tracker = {
            key: deque(maxlen=self.success_detection_duration) for key in self.reward_relabeling_keys
        }

        # --- in-context demonstrations (RoboRef-ICL / run1) -------------------
        # A bank of successful same-task demonstrations, produced by
        # scripts/generate_maniskill_icl_demos.py. When loaded, each query is
        # scored with a demonstration attached as sample.context_trajectory,
        # and the collator inserts <|demo_end|> between demo and query -- the
        # same input layout the model saw at training time. Mirrors the proven
        # path in vlm_ibrl_v3/env/robometer_utils.py.
        self.icl_demos = None
        self._icl_rng = np.random.default_rng(icl_demo_seed)
        if icl_demo_path:
            self._load_icl_demos(icl_demo_path)

    def _load_icl_demos(self, path: str) -> None:
        """Load a demo bank of shape (N, T, H, W, 3) uint8."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"icl_demo_path does not exist: {path}. Generate it with "
                f"scripts/generate_maniskill_icl_demos.py --task <TASK>."
            )
        # Fail loudly if the installed robometer predates ICL support. The
        # pinned submodule (upstream) has no context_trajectory field, so
        # assigning one would raise an opaque pydantic error deep in a rollout
        # instead of here, at startup, with a fix attached.
        try:
            from robometer.data.dataset_types import ProgressSample

            if "context_trajectory" not in getattr(ProgressSample, "model_fields", {}):
                raise ImportError
        except ImportError as exc:
            raise RuntimeError(
                "The installed `robometer` package has no ProgressSample.context_trajectory, "
                "so in-context demonstrations cannot be attached. This is the upstream "
                "submodule, which predates ICL. Install the RoboRef fork instead, e.g.\n"
                "    pip install -e /path/to/Master-Thesis/Robometer\n"
                "(see jobs/README_maniskill.md)."
            ) from exc

        data = np.load(path, allow_pickle=True)
        demos = np.asarray(data["frames"])
        if demos.ndim != 5:
            raise ValueError(f"ICL demo bank must be (N, T, H, W, 3); got {demos.shape} from {path}")
        self.icl_demos = demos
        instruction = str(data["instruction"]) if "instruction" in data else "<none>"
        logger.info(
            f"Loaded {demos.shape[0]} in-context demonstrations "
            f"({demos.shape[1]} frames each) from {path} | instruction: {instruction!r}"
        )
        if demos.shape[1] != self.max_frames:
            # Not fatal: raw_dict_to_sample resamples to max_frames. Still worth
            # surfacing, since a mismatch usually means the bank was built for a
            # different model.
            logger.warning(
                f"ICL demo bank has {demos.shape[1]} frames but the reward model expects "
                f"{self.max_frames}; frames will be resampled."
            )

    def _sample_icl_demo(self) -> Optional[np.ndarray]:
        """Draw one demonstration clip, or None if no bank is loaded."""
        if self.icl_demos is None or len(self.icl_demos) == 0:
            return None
        return self.icl_demos[self._icl_rng.integers(len(self.icl_demos))]

    def _attach_icl_context(self, sample, task: str, episode_id: int = 0):
        """Attach a demonstration to ``sample`` as its context trajectory.

        No-op when no demo bank is loaded, so the non-ICL arms are unaffected.
        """
        demo_frames = self._sample_icl_demo()
        if demo_frames is None:
            return sample
        icl_raw = dict(
            frames=np.asarray(demo_frames, dtype=np.uint8),
            task=task,
            # Offset the id so the demo cannot collide with the query's.
            id=int(episode_id) + 1_000_000,
            metadata=dict(subsequence_length=int(demo_frames.shape[0])),
            video_embeddings=None,
            text_embedding=None,
        )
        icl_sample = raw_dict_to_sample(
            raw_data=icl_raw, max_frames=self.max_frames, sample_type="progress"
        )
        sample.context_trajectory = icl_sample.trajectory
        return sample

    def _pad_to_max_frames(self, frames: np.ndarray) -> np.ndarray:
        """Front-pad a short clip up to ``max_frames`` by repeating the first frame.

        The reward model is scored on a GROWING prefix, so early in an episode it
        receives fewer frames than it was trained on. On LIBERO/Robomimic this is
        harmless -- episodes run 200-400 steps, so after ~16 steps the prefix is
        always long enough (their calibration simply starts at t=16). ManiSkill
        episodes are 50 steps and succeed around step 7, so a prefix NEVER reaches
        max_frames=16 for the fine-tuned checkpoints, and the heads collapse.

        Measured on PullCube at the GT success step, run2 (max_frames=16):
            7-11 frames  -> success_prob 0.053   (failures 0.024)
            padded to 16 -> success_prob 0.369   (failures 0.024, unchanged)
        Base (max_frames=8) is barely affected, since 7-11 frames already satisfies it
        -- which is exactly why only the fine-tuned models looked broken.

        Uses the SAME linspace-with-repeats rule as reward-model-study's ``sub16``
        (the sampler behind the validated MetaWorld/Robomimic numbers):

            idx = np.linspace(0, len(frames) - 1, max_frames).round()

        Robometer's own collator only ever REDUCES -- ``linspace_subsample_frames``
        returns the clip untouched when ``effective_total <= num_frames`` -- so the
        upsampling has to happen here, before the collator sees it. Spreading the
        repeats evenly matches how training clips were subsampled; front-padding with
        frame 0 instead makes the clip look stuck at the start and biases progress
        down (measured: run2 progress@success 0.203 -> 0.188).
        """
        if frames is None or len(frames) == 0:
            return frames
        n = int(len(frames))
        if n >= int(self.max_frames):
            return frames
        idx = np.linspace(0, n - 1, int(self.max_frames)).round().astype(int)
        return frames[idx]

    def _compute_reward_single(self, raw_data: Dict[str, Any]) -> Tuple[float, float]:
        """
        Compute reward for a single sample using either local reward model or eval_server.

        Args:
            raw_data: Dictionary containing frames, task, video_embeddings, text_embedding, etc.

        Returns:
            Reward value as float
        """
        if self.reward_model is not None:
            # Use local reward model
            sample = raw_dict_to_sample(
                raw_data=raw_data,
                max_frames=self.max_frames,
                sample_type="progress",
            )
            sample = self._attach_icl_context(
                sample, task=raw_data.get("task", ""), episode_id=raw_data.get("id", 0)
            )

            # Treat any C51 / discrete progress head as discrete-mode so its bin logits
            # get softmax-reduced to a [0,1] scalar. Fine-tuned models use loss-type
            # variants like "c51_asymmetric" (same 10-bin head, just an asymmetric loss);
            # matching only the literal "discrete" left those reads on the raw-logit path
            # -> suppressed/garbage reward. No-op for base Robometer-4B ("discrete").
            _plt = self.reward_model_config.loss.progress_loss_type.lower()
            is_discrete_mode = _plt == "discrete" or "c51" in _plt
            progress_discrete_bins = self.reward_model_config.loss.progress_discrete_bins
            outputs = process_batch_helper(
                model_type=self.reward_model_config.model.model_type,
                model=self.reward_model,
                tokenizer=self.tokenizer,
                batch_collator=self.batch_collator,
                device=self.reward_model.device,
                batch_data=[sample.model_dump()],
                job_id=0,
                is_discrete_mode=is_discrete_mode,
                num_bins=progress_discrete_bins,
            )
        elif self.use_eval_server:
            # Use eval_server
            sample = raw_dict_to_sample(
                raw_data=raw_data,
                max_frames=self.max_frames,
                sample_type="progress",
            )

            files, sample_data = build_payload([sample])
            outputs = post_batch_npy(self.eval_server_url, files, sample_data, timeout_s=self.eval_server_timeout)
        else:
            raise ValueError("Neither reward_model nor use_eval_server is set")

        rewards = extract_rewards_from_output(outputs)
        suceess_probs = extract_success_probs_from_output(outputs)
        return float(rewards[0]), float(suceess_probs[0])

    def _compute_rewards_batch(self, batch_raw: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
        """
        Compute rewards for a batch of samples using either local reward model or eval_server.

        Args:
            batch_raw: List of dictionaries, each containing frames, task, video_embeddings, text_embedding, etc.

        Returns:
            Tuple of (List of reward values as floats, List of success probabilities as floats)
        """
        if self.reward_model is not None:
            # Use local reward model
            samples = [
                self._attach_icl_context(
                    raw_dict_to_sample(
                        raw_data=raw_data_item,
                        max_frames=self.max_frames,
                        sample_type="progress",
                    ),
                    task=raw_data_item.get("task", ""),
                    episode_id=raw_data_item.get("id", 0),
                )
                for raw_data_item in batch_raw
            ]

            # Treat any C51 / discrete progress head as discrete-mode so its bin logits
            # get softmax-reduced to a [0,1] scalar. Fine-tuned models use loss-type
            # variants like "c51_asymmetric" (same 10-bin head, just an asymmetric loss);
            # matching only the literal "discrete" left those reads on the raw-logit path
            # -> suppressed/garbage reward. No-op for base Robometer-4B ("discrete").
            _plt = self.reward_model_config.loss.progress_loss_type.lower()
            is_discrete_mode = _plt == "discrete" or "c51" in _plt
            progress_discrete_bins = self.reward_model_config.loss.progress_discrete_bins
            outputs = process_batch_helper(
                model_type=self.reward_model_config.model.model_type,
                model=self.reward_model,
                tokenizer=self.tokenizer,
                batch_collator=self.batch_collator,
                device=self.reward_model.device,
                batch_data=[sample.model_dump() for sample in samples],
                job_id=0,
                is_discrete_mode=is_discrete_mode,
                num_bins=progress_discrete_bins,
            )
        elif self.use_eval_server:
            # Use eval_server
            samples = [
                raw_dict_to_sample(
                    raw_data=raw_data_item,
                    max_frames=self.max_frames,
                    sample_type="progress",
                )
                for raw_data_item in batch_raw
            ]

            files, sample_data = build_payload(samples)
            outputs = post_batch_npy(self.eval_server_url, files, sample_data, timeout_s=self.eval_server_timeout)
        else:
            raise ValueError("Neither reward_model nor use_eval_server is set")

        rewards_batch = extract_rewards_from_output(outputs)
        success_probs_batch = extract_success_probs_from_output(outputs)
        return rewards_batch.tolist(), success_probs_batch.tolist()

    def _add(
        self,
        language_instruction=None,
        video_frames=None,
        dino_embeddings=None,
        text_embedding=None,
        **kwargs,
    ):
        # Calculate reward using reward model or eval_server if available
        if self.reward_model is not None or self.use_eval_server:
            # Ensure text_embedding is a numpy array
            text_emb = convert_to_numpy(text_embedding)
            # Ensure dino_embeddings is a numpy array
            dino_embeddings = convert_to_numpy(dino_embeddings)
            avg_reward = 0.0
            for index, key in enumerate(self.reward_relabeling_keys):
                # Convert embeddings to proper format (common for both paths)
                if isinstance(dino_embeddings, list) and len(dino_embeddings) > 0:
                    # Convert list of embeddings to array [T, D]
                    dino_embeddings = np.array(dino_embeddings)
                # Take subset for this key: each key's embedding is a chunk of the list
                # since dino embeddings for each key are concatenated.
                video_embeddings_array = None
                if len(dino_embeddings) > 0:
                    embeddings_per_key = dino_embeddings.shape[1] // len(self.reward_relabeling_keys)
                    video_embeddings_array = dino_embeddings[
                        :, index * embeddings_per_key : (index + 1) * embeddings_per_key
                    ]

                _frames = np.array(video_frames[key]) if video_frames[key] is not None else np.array([])
                _frames = self._pad_to_max_frames(_frames)

                raw_data = dict(
                    frames=_frames,
                    task=language_instruction,
                    id=kwargs.get("episode_id"),
                    metadata=dict(
                        subsequence_length=int(len(_frames)),
                    ),
                    video_embeddings=video_embeddings_array,
                    text_embedding=text_emb,
                )

                reward, success_prob = self._compute_reward_single(raw_data)
                self.success_tracker[key].append(success_prob)

                # Apply relative rewards if enabled
                if self.use_relative_rewards:
                    current_reward = reward
                    reward = reward - self.prev_reward[key]
                    # Store original absolute reward
                    self.prev_reward[key] = current_reward
                    if kwargs.get("done") or kwargs.get("truncated"):
                        self.prev_reward[key] = 0.0
                avg_reward += reward

            avg_reward /= len(self.reward_relabeling_keys)

            # DISCRIMINATION LOGGING (RPL_LOG_DISCRIM=1): accumulate the RM reward
            # per episode and, when the episode ends, log it next to the GT success
            # label. Lets us measure whether the RM reward separates success from
            # failure on the live online distribution (no offline dataset needed).
            if os.environ.get("RPL_LOG_DISCRIM"):
                eid = kwargs.get("episode_id")
                if not hasattr(self, "_disc_acc"):
                    self._disc_acc, self._disc_succ, self._disc_len = {}, {}, {}
                self._disc_acc[eid] = self._disc_acc.get(eid, 0.0) + float(avg_reward)
                self._disc_len[eid] = self._disc_len.get(eid, 0) + 1
                if kwargs.get("is_success") or kwargs.get("success"):
                    self._disc_succ[eid] = True
                if kwargs.get("done") or kwargs.get("truncated"):
                    gt = bool(self._disc_succ.get(eid, False))
                    logger.info(
                        f"[DISCRIM] ep={eid} gt_success={int(gt)} "
                        f"rm_reward_sum={self._disc_acc[eid]:.4f} len={self._disc_len[eid]}"
                    )
                    self._disc_acc.pop(eid, None); self._disc_succ.pop(eid, None); self._disc_len.pop(eid, None)

            _gt_in = kwargs.get("reward")
            if self.add_estimated_reward:
                kwargs["reward"] += avg_reward
            else:
                kwargs["reward"] = avg_reward
            # REWARD-SOURCE PROOF (RPL_LOG_REWARD=1): first ~30 steps, show the env/GT
            # reward coming in, the VLM reward, and the final reward SAC trains on. If
            # final == vlm and != gt_in, the GT reward is overwritten (no leak).
            if os.environ.get("RPL_LOG_REWARD") and getattr(self, "_rwd_n", 0) < 30:
                self._rwd_n = getattr(self, "_rwd_n", 0) + 1
                try:
                    logger.info(
                        f"[REWARD-SRC] gt_in={float(_gt_in):.4f} vlm={float(avg_reward):.4f} "
                        f"final_train_reward={float(kwargs['reward']):.4f} add_est={self.add_estimated_reward}"
                    )
                except Exception:
                    pass
            if self.use_success_detection:
                self._ep_step += 1
                # Check if the episode is done based on majority vote of success probabilities
                vote = 0
                for key in self.reward_relabeling_keys:
                    for success_prob in self.success_tracker[key]:
                        if success_prob > float(self.success_detection_threshold):
                            vote += 1
                gate_open = self._ep_step >= self.success_detection_min_ep_steps
                if gate_open and vote > (
                    len(self.reward_relabeling_keys) * self.success_detection_duration / 2
                ):
                    kwargs["done"] = True
                    # THE reward-hacking signal. A fire with gt_success=0 is a FALSE
                    # termination: the policy got the episode ended (and the remaining
                    # cost avoided) without solving the task. One line per fire, so a
                    # run of "fired gt_success=0" at small step_in_ep means the
                    # threshold is too low; no lines at all means it is too high and
                    # TERMINATE=1 has silently degraded into TERMINATE=0.
                    self._n_fire = getattr(self, "_n_fire", 0) + 1
                    _gt_now = bool(kwargs.get("is_success") or kwargs.get("success"))
                    self._n_fire_false = getattr(self, "_n_fire_false", 0) + (0 if _gt_now else 1)
                    logger.info(
                        f"[DETECT] fired ep={kwargs.get('episode_id')} "
                        f"step_in_ep={self._ep_step} gt_success={int(_gt_now)} "
                        f"n_fire={self._n_fire} n_false={self._n_fire_false} "
                        f"false_rate={self._n_fire_false / max(1, self._n_fire):.2f}"
                    )
                if kwargs["done"] or kwargs["truncated"]:
                    for key in self.reward_relabeling_keys:
                        self.success_tracker[key].clear()
                    self._ep_step = 0

        super()._add(**kwargs)


class RobometerH5ReplayBuffer(H5ReplayBuffer):
    def __init__(
        self,
        reward_model=None,
        reward_model_config=None,
        use_relative_rewards: bool = False,
        use_eval_server: bool = False,
        eval_server_url: Optional[str] = None,
        eval_server_timeout: float = 120.0,
        sentence_model: SentenceTransformer = None,
        dinov2_model: AutoModel = None,
        dinov2_processor: AutoImageProcessor = None,
        reward_relabeling_keys: List[str] = ["image"],
        use_success_detection: bool = False,
        success_detection_duration: int = 2,
        success_detection_min_ep_steps: int = 0,
        success_detection_threshold: float = 0.65,
        add_estimated_reward: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_eval_server = use_eval_server
        self.eval_server_url = eval_server_url
        self.eval_server_timeout = eval_server_timeout
        self.reward_relabeling_keys = reward_relabeling_keys
        self.use_success_detection = use_success_detection
        self.success_detection_duration = success_detection_duration
        # A fire before this many steps into the episode cannot be a real success --
        # the same guard as vlm_ibrl's ROBOMETER_MIN_EP_STEPS. Set it between the
        # latest "fake fire" and the earliest "real fire" reported by
        # scripts/causal_calib_maniskill.py. 0 disables the gate.
        self.success_detection_min_ep_steps = int(success_detection_min_ep_steps)
        self._ep_step = 0
        self.success_detection_threshold = success_detection_threshold
        self.add_estimated_reward = add_estimated_reward

        # Set max_frames once from config
        if reward_model_config is not None:
            self.max_frames = getattr(reward_model_config.data, "max_frames", 16)
        else:
            self.max_frames = 16

        self.reward_model = reward_model
        if self.reward_model is not None:
            self.reward_model_config = reward_model_config
            self.processor = getattr(reward_model, "processor", None)
            self.tokenizer = getattr(reward_model, "tokenizer", None)
            if self.processor is None or self.tokenizer is None:
                raise ValueError(
                    "processor and tokenizer must be available on reward_model (reward_model.processor / reward_model.tokenizer)"
                )
            # Ensure use_multi_image is True for reward relabeling (process frames as images, not video)
            if not self.reward_model_config.data.use_multi_image:
                logger.warning("use_multi_image is False in config. Setting to True for reward relabeling.")
                self.reward_model_config.data.use_multi_image = True

            # Set up batch collator with inference=True for evaluation
            self.batch_collator = setup_batch_collator(
                self.processor, self.tokenizer, self.reward_model_config, is_eval=True
            )
        elif self.use_eval_server:
            if self.eval_server_url is None:
                raise ValueError("eval_server_url must be provided when use_eval_server=True")
            logger.info(f"Using eval_server at {self.eval_server_url} for reward computation")

        # Set up cache directory for embeddings
        h5_paths = kwargs.get("h5_paths")
        h5_paths_list = h5_paths if isinstance(h5_paths, list) else [h5_paths]
        self.embeddings_cache_dir = os.path.join(os.path.dirname(h5_paths_list[0]), ".embeddings_cache")
        os.makedirs(self.embeddings_cache_dir, exist_ok=True)

        # Since we need to add embeddings to observations for policy learning, we need to initialize the models.
        self.use_dino_embeddings = True
        self.dinov2_model = dinov2_model
        self.dinov2_processor = dinov2_processor
        self.sentence_model = sentence_model
        self.use_relative_rewards = use_relative_rewards

        # Always compute language and video embeddings
        self.precomputed_video_embeddings = self._load_or_compute_video_embeddings()
        self.text_embeddings_dict = self._load_or_compute_language_embeddings()

        # Relabel rewards if using Robometer rewards, otherwise only add embeddings to obs
        if self.reward_model is not None:
            self.relabel_rewards(verbose=True, batch_size=8)
        self.add_embeddings_to_obs()

    def relabel_rewards(self, verbose: bool = True, batch_size: Optional[int] = None):
        """
        Relabel rewards in the HDF5 cache using the reward model.

        This function iterates through all demos in the HDF5 cache, extracts video frames
        and language instructions, processes them through the reward model, and updates
        the cached rewards with the model's predictions.

        Args:
            verbose: Whether to print progress information
            batch_size: Batch size for reward model inference
        """
        logger.info("Starting to relabel rewards...")

        reward_keys: List[str] = (
            self.reward_relabeling_keys
            if hasattr(self, "reward_relabeling_keys") and self.reward_relabeling_keys is not None
            else []
        )
        if len(reward_keys) == 0:
            raise RuntimeError("reward_relabeling_keys must be non-empty to relabel rewards")

        # precomputed_video_embeddings are stored concatenated across keys: [T, D_total]
        any_demo_key = next(iter(self.hdf5_cache.keys()))
        total_emb_dim = int(self.precomputed_video_embeddings[any_demo_key].shape[-1])
        if total_emb_dim % len(reward_keys) != 0:
            raise RuntimeError(
                f"precomputed_video_embeddings last dim ({total_emb_dim}) is not divisible by "
                f"len(reward_relabeling_keys) ({len(reward_keys)})."
            )
        emb_dim_per_key = total_emb_dim // len(reward_keys)

        all_rewards_flat: Dict[str, List[float]] = {key: [] for key in reward_keys}
        all_success_probs_flat: Dict[str, List[float]] = {key: [] for key in reward_keys}
        demo_to_indices: Dict[str, tuple] = {}

        for key_idx, key in enumerate(reward_keys):
            # Load frames for this specific key
            all_traj_frames, _, _ = self._load_all_frames_for_demos(key=key)
            video_frames_by_demo = {demo_key: video_frames for demo_key, _, video_frames in all_traj_frames}

            all_raw_data_flat = []
            current_idx = 0

            for demo_key, cached_demo in self.hdf5_cache.items():
                h5_path, unique_episode_id = demo_key.split("::")
                episode_len = len(cached_demo["actions"])

                # Language instruction
                language_instruction = self._load_language_instruction_from_file(
                    h5_path,
                    cached_demo.get("original_demo_name", unique_episode_id.split("_", 1)[-1]),
                )
                text_emb = self.text_embeddings_dict[language_instruction]

                # Slice embeddings chunk corresponding to this key: [T, D_per_key]
                video_embeddings_all = self.precomputed_video_embeddings[demo_key]
                start_d = key_idx * emb_dim_per_key
                end_d = (key_idx + 1) * emb_dim_per_key
                video_embeddings = video_embeddings_all[:, start_d:end_d]

                video_frames = video_frames_by_demo[demo_key]

                if self.use_eval_server:
                    # If using eval server, append only full episode (as one item) to all_raw_data_flat
                    all_raw_data_flat.append(
                        dict(
                            frames=video_frames[:episode_len],
                            task=language_instruction,
                            id=unique_episode_id,
                            metadata=dict(subsequence_length=episode_len),
                            video_embeddings=video_embeddings[:episode_len],
                            text_embedding=text_emb,
                        )
                    )
                else:
                    # Record mapping indices once (must be identical across keys)
                    if key_idx == 0:
                        demo_to_indices[demo_key] = (current_idx, current_idx + episode_len)

                    for t in range(episode_len):
                        subseq_len = t + 1
                        all_raw_data_flat.append(
                            dict(
                                frames=video_frames[:subseq_len],
                                task=language_instruction,
                                id=unique_episode_id,
                                metadata=dict(subsequence_length=subseq_len),
                                video_embeddings=video_embeddings[:subseq_len],
                                text_embedding=text_emb,
                            )
                        )

                    current_idx += episode_len

            if verbose:
                logger.info(f"Computing rewards for all demos (key={key})...")

            effective_batch_size = batch_size if batch_size is not None and batch_size > 0 else 1024
            for batch_start in tqdm(
                range(0, len(all_raw_data_flat), effective_batch_size),
                desc=f"Computing rewards ({key})",
            ):
                batch_end = min(batch_start + effective_batch_size, len(all_raw_data_flat))
                batch_raw = all_raw_data_flat[batch_start:batch_end]
                rewards_batch, success_probs_batch = self._compute_rewards_batch(batch_raw)
                all_rewards_flat[key].extend(rewards_batch)
                all_success_probs_flat[key].extend(success_probs_batch)

        # Map rewards back to demos and average across keys per timestep
        demo_idx = 0
        for demo_key, cached_demo in self.hdf5_cache.items():
            per_key_rewards = []
            per_key_success_probs = []
            if self.use_eval_server:
                for key in reward_keys:
                    per_key_rewards.append(all_rewards_flat[key][demo_idx])
                    per_key_success_probs.append(all_success_probs_flat[key][demo_idx])
            else:
                start_idx, end_idx = demo_to_indices[demo_key]
                for key in reward_keys:
                    per_key_rewards.append(all_rewards_flat[key][start_idx:end_idx])
                    per_key_success_probs.append(all_success_probs_flat[key][start_idx:end_idx])
            demo_rewards = np.array(per_key_rewards, dtype=np.float32).mean(axis=0)
            if self.add_estimated_reward:
                cached_demo["rewards"] += demo_rewards
            else:
                cached_demo["rewards"] = demo_rewards
            if self.use_success_detection:
                # Majority voting for success probabilities
                demo_dones = np.zeros_like(per_key_success_probs[0], dtype=bool)
                window = self.success_detection_duration
                threshold = float(self.success_detection_threshold)
                for t in range(len(per_key_success_probs[0]) - window + 1):
                    # Gather success probabilities for all keys in the window [t:t+window]
                    votes = 0
                    total = 0
                    for key_probs in per_key_success_probs:
                        for i in range(window):
                            total += 1
                            if key_probs[t + i] > threshold:
                                votes += 1
                    # Majority voting: if more than half are successful, mark done
                    if votes > (total // 2):
                        demo_dones[t + window - 1] = True
                cached_demo["dones"] = demo_dones

            demo_idx += 1
            assert len(cached_demo["rewards"]) == len(cached_demo["actions"]) == len(cached_demo["dones"])

        logger.info(f"Reward relabeling complete!")

    # --------------------------
    # Loading and caching
    # --------------------------
    def _load_with_optimizations(self):
        if self.obs_keys is None:
            self.obs_keys = self._get_all_obs_keys(self.h5_paths[0])

        # Determine modalities by heuristic
        low_dim_keys: List[str] = []
        rgb_keys: List[str] = []
        # Combine reward_relabeling_keys with common image keywords
        reward_image_keys = set(
            self.reward_relabeling_keys
            if hasattr(self, "reward_relabeling_keys") and self.reward_relabeling_keys is not None
            else []
        )
        for key in self.obs_keys:
            is_image_key = any(img_kw in key.lower() for img_kw in ["image", "rgb", "camera", "cam"])
            is_reward_key = key in reward_image_keys
            if is_image_key or is_reward_key:
                if key not in rgb_keys:
                    rgb_keys.append(key)
            else:
                low_dim_keys.append(key)
        self.image_keys = rgb_keys
        self.low_dim_keys = low_dim_keys

        if self.hdf5_cache_mode in ["all", "low_dim"]:
            self._load_with_memory_cache()
        else:
            self._convert_to_transitions()

        self._print_dataset_statistics()
        self.image_loading_executor = None

    def _load_all_frames_for_demos(self, key: Optional[str] = None):
        """
        Helper method to load all frames for all demos.
        If key is provided, only load frames for that key.
        Returns tuple of (all_trajectory_frames, all_language_instructions, all_episode_lengths).
        """
        all_trajectory_frames = []
        all_language_instructions = {}
        all_episode_lengths = {}

        for demo_key, cached_demo in self.hdf5_cache.items():
            h5_path, unique_episode_id = demo_key.split("::")
            original_demo_name = cached_demo.get("original_demo_name", unique_episode_id.split("_", 1)[-1])

            # Episode length
            episode_len = len(cached_demo["actions"])
            all_episode_lengths[demo_key] = episode_len

            # Language instruction
            language_instruction = self._load_language_instruction_from_file(h5_path, original_demo_name)
            all_language_instructions[demo_key] = language_instruction

            img_key = key if key is not None else self.image_keys[0]

            # Load initial frame (t=0)
            initial_frame = self._load_obs_from_file(h5_path, original_demo_name, img_key, 0)

            # Load subsequent frames in a single read: shape [episode_len, H, W, C]
            with self._get_hdf5_file(h5_path) as file:
                demo_group = self._get_demo_group(file, original_demo_name)
                all_next_frames = np.array(demo_group["next_obs"][img_key][:episode_len])

            # Concatenate to [episode_len+1, H, W, C]
            video_frames = np.concatenate([initial_frame[np.newaxis, ...], all_next_frames], axis=0)

            all_trajectory_frames.append((demo_key, cached_demo, video_frames))

        return all_trajectory_frames, all_language_instructions, all_episode_lengths

    def _get_cache_key(self, cache_type: str) -> str:
        """
        Generate a cache key based on h5_paths, model names, and other relevant info.

        Args:
            cache_type: Type of cache ('video' or 'language')

        Returns:
            Cache key string
        """
        # Get h5_paths as a sorted list for consistent hashing
        h5_paths_list = sorted(self.h5_paths if isinstance(self.h5_paths, list) else [self.h5_paths])

        # Create a hash from h5_paths and cache type.
        # Include reward_relabeling_keys because video embeddings depend on which image keys we embed (and their order).
        reward_keys = (
            self.reward_relabeling_keys
            if hasattr(self, "reward_relabeling_keys") and self.reward_relabeling_keys is not None
            else []
        )
        hash_input = (
            f"{cache_type}_{h5_paths_list}_{reward_keys}_{self.sentence_model.get_sentence_embedding_dimension()}"
        )
        if self.use_dino_embeddings and self.dinov2_model is not None:
            hash_input += f"_dinov2_{self.dinov2_model.config.name_or_path}"

        # Use file modification times to detect dataset changes
        for h5_path in h5_paths_list:
            if os.path.exists(h5_path):
                mtime = os.path.getmtime(h5_path)
                hash_input += f"_{mtime}"

        cache_key = hashlib.md5(hash_input.encode()).hexdigest()
        return cache_key

    def _get_cache_path(self, cache_type: str) -> str:
        """Get the full path to the cache file."""
        cache_key = self._get_cache_key(cache_type)
        return os.path.join(self.embeddings_cache_dir, f"{cache_type}_embeddings_{cache_key}.pkl")

    def _load_video_embeddings_from_cache(self) -> Optional[Dict[str, np.ndarray]]:
        """Load video embeddings from cache if available."""
        cache_path = self._get_cache_path("video")
        if os.path.exists(cache_path):
            logger.info(f"Loading video embeddings from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                logger.info(f"Successfully loaded {len(cached_data)} video embeddings from cache")
                return cached_data
        return None

    def _save_video_embeddings_to_cache(self, embeddings: Dict[str, np.ndarray]):
        """Save video embeddings to cache."""
        cache_path = self._get_cache_path("video")
        logger.info(f"Saving video embeddings to cache: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(embeddings, f)
        logger.info(f"Successfully saved {len(embeddings)} video embeddings to cache")

    def _load_language_embeddings_from_cache(self) -> Optional[Dict[str, np.ndarray]]:
        """Load language embeddings from cache if available."""
        cache_path = self._get_cache_path("language")
        if os.path.exists(cache_path):
            logger.info(f"Loading language embeddings from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                logger.info(f"Successfully loaded {len(cached_data)} language embeddings from cache")
                return cached_data
        return None

    def _save_language_embeddings_to_cache(self, embeddings: Dict[str, np.ndarray]):
        """Save language embeddings to cache."""
        cache_path = self._get_cache_path("language")
        logger.info(f"Saving language embeddings to cache: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(embeddings, f)
        logger.info(f"Successfully saved {len(embeddings)} language embeddings to cache")

    def _load_or_compute_video_embeddings(self) -> Dict[str, np.ndarray]:
        """Load video embeddings from cache or compute them if not available."""
        cached_embeddings = self._load_video_embeddings_from_cache()
        if cached_embeddings is not None:
            return cached_embeddings

        # Compute embeddings if not in cache
        embeddings = self.compute_video_embeddings_for_trajectory()
        self._save_video_embeddings_to_cache(embeddings)
        return embeddings

    def _load_or_compute_language_embeddings(self) -> Dict[str, np.ndarray]:
        """Load language embeddings from cache or compute them if not available."""
        cached_embeddings = self._load_language_embeddings_from_cache()
        if cached_embeddings is not None:
            return cached_embeddings

        # Compute embeddings if not in cache
        embeddings = self.compute_language_embeddings_for_trajectory()
        self._save_language_embeddings_to_cache(embeddings)
        return embeddings

    def compute_video_embeddings_for_trajectory(self) -> Dict[str, np.ndarray]:
        """
        Compute DINO embeddings for all image keys in `self.reward_relabeling_keys`.

        Returns:
            Dictionary mapping demo_key to concatenated video embeddings array [T, D_total],
            where D_total = D_per_key * len(self.reward_relabeling_keys) and the concatenation order
            matches `self.reward_relabeling_keys`.
        """
        reward_keys: List[str] = (
            self.reward_relabeling_keys
            if hasattr(self, "reward_relabeling_keys") and self.reward_relabeling_keys is not None
            else []
        )
        if len(reward_keys) == 0:
            # Fallback to first image key if reward relabeling keys are not configured
            reward_keys = [self.image_keys[0]]

        logger.info(f"Computing video embeddings for all trajectories across keys: {reward_keys}")

        # Compute per-key embeddings, then concatenate per demo_key.
        per_key_embeddings: Dict[str, Dict[str, np.ndarray]] = {}

        for key in reward_keys:
            all_trajectory_frames, _, all_episode_lengths = self._load_all_frames_for_demos(key=key)

            # Collect all frames from all trajectories (for this key)
            all_frames_list = []
            trajectory_frame_indices = []
            current_frame_idx = 0

            for demo_idx, (demo_key, cached_demo, video_frames) in enumerate(all_trajectory_frames):
                num_frames = len(video_frames)
                trajectory_frame_indices.append((current_frame_idx, current_frame_idx + num_frames))
                all_frames_list.append(video_frames)
                current_frame_idx += num_frames

            if len(all_frames_list) == 0:
                logger.warning(f"No frames found for key={key}; skipping embeddings for this key")
                per_key_embeddings[key] = {}
                continue

            # Batch compute video embeddings for all frames for this key
            all_frames_array = np.concatenate(all_frames_list, axis=0)
            all_frame_embeddings = compute_video_embeddings(
                all_frames_array,
                self.dinov2_model,
                self.dinov2_processor,
                use_autocast=True,
                use_tqdm=True,
            )

            logger.info(
                f"[{key}] Computed {all_frame_embeddings.shape[0]} frame embeddings for {len(all_frames_list)} trajectories"
            )

            # Store per-timestep embeddings per trajectory
            key_embeddings: Dict[str, np.ndarray] = {}
            for demo_idx, (demo_key, cached_demo, video_frames) in enumerate(all_trajectory_frames):
                episode_len = all_episode_lengths[demo_key]
                start_idx, end_idx = trajectory_frame_indices[demo_idx]
                trajectory_embeddings = all_frame_embeddings[start_idx:end_idx]

                # One embedding per timestep (truncate or pad last as needed)
                embedding_per_timestep = []
                for t in range(episode_len):
                    if t < len(trajectory_embeddings):
                        embedding_per_timestep.append(trajectory_embeddings[t])
                    else:
                        embedding_per_timestep.append(trajectory_embeddings[-1])

                key_embeddings[demo_key] = np.stack(embedding_per_timestep[:episode_len], axis=0)

            per_key_embeddings[key] = key_embeddings

        # Concatenate per-key embeddings per demo_key in the configured order.
        precomputed_video_embeddings: Dict[str, np.ndarray] = {}
        demo_keys = list(self.hdf5_cache.keys()) if self.hdf5_cache is not None else []
        for demo_key in demo_keys:
            chunks = []
            for key in reward_keys:
                if demo_key not in per_key_embeddings.get(key, {}):
                    raise RuntimeError(
                        f"Missing video embeddings for demo_key={demo_key} key={key}. "
                        "Check that the image key exists in the dataset and frames were loaded correctly."
                    )
                chunks.append(per_key_embeddings[key][demo_key])
            precomputed_video_embeddings[demo_key] = np.concatenate(chunks, axis=-1)

        return precomputed_video_embeddings

    def compute_language_embeddings_for_trajectory(self) -> Dict[str, np.ndarray]:
        """
        Compute language embeddings for all unique instructions in the cache.

        Returns:
            Dictionary mapping text instruction to language embedding array [D]
        """
        logger.info("Computing language embeddings for all trajectories...")
        all_trajectory_frames, all_language_instructions, all_episode_lengths = self._load_all_frames_for_demos()

        # Compute language embeddings for all unique instructions
        unique_texts = list(set(all_language_instructions.values()))
        logger.info(f"Computing language embeddings for {len(unique_texts)} unique instructions...")

        text_embeddings_dict = {}
        for text in unique_texts:
            text_emb = compute_text_embeddings(text, self.sentence_model, use_autocast=True, show_progress_bar=False)
            text_embeddings_dict[text] = text_emb

        logger.info(f"Computed {len(text_embeddings_dict)} language embeddings")
        return text_embeddings_dict

    def _load_language_instruction_from_file(self, h5_path: str, demo: str):
        cache_key = f"{h5_path}::{demo}::language_instruction"
        # Lightweight cache on the instance
        if not hasattr(self, "_lang_instr_cache"):
            self._lang_instr_cache = {}
        if cache_key in self._lang_instr_cache:
            return self._lang_instr_cache[cache_key]
        with self._get_hdf5_file(h5_path) as f:
            grp = self._get_demo_group(f, demo)
            if "language_instruction" in grp:
                val = grp["language_instruction"][()]
                if isinstance(val, bytes):
                    val = val.decode("utf-8", errors="ignore")
                elif hasattr(val, "dtype") and getattr(val, "dtype", None).kind in {"S", "O"}:
                    val = val.astype(str)
                self._lang_instr_cache[cache_key] = val
                return val

    def add_embeddings_to_obs(self):
        """
        Add language encodings and DINO embeddings to the observation dicts of all demos in the HDF5 cache.
        This will add 'language' and 'dino_embedding' keys to each time step in 'obs' for each demo,
        reusing precomputed embeddings from self.text_embeddings_dict and self.precomputed_video_embeddings.
        """
        if self.hdf5_cache is None or len(self.hdf5_cache) == 0:
            logger.warning("HDF5 cache is empty or not available, cannot add embeddings")
            return

        # Ensure text embeddings dict exists
        if not hasattr(self, "text_embeddings_dict") or self.text_embeddings_dict is None:
            raise RuntimeError("text_embeddings_dict must be precomputed before calling add_embeddings_to_obs")

        for demo_key, cached_demo in self.hdf5_cache.items():
            # Get the language string for this demo (from the cache or directly if available)
            h5_path, unique_episode_id = demo_key.split("::")
            original_demo_name = cached_demo.get("original_demo_name", unique_episode_id.split("_", 1)[-1])
            if "language_instruction" in cached_demo:
                language_str = cached_demo["language_instruction"]
            else:
                language_str = self._load_language_instruction_from_file(h5_path, original_demo_name)
                cached_demo["language_instruction"] = language_str

            language_encoding = self.text_embeddings_dict[language_str]

            if "obs" in cached_demo and cached_demo["obs"] is not None:
                obs_dict = cached_demo["obs"]
                episode_len = len(cached_demo["actions"])

                # Add language encoding (same for all timesteps)
                lang_array = np.repeat(np.expand_dims(language_encoding, axis=0), episode_len, axis=0)
                obs_dict["language"] = lang_array

                # Add DINO embeddings (one per timestep) if available
                if self.use_dino_embeddings and demo_key in self.precomputed_video_embeddings:
                    video_embeddings = self.precomputed_video_embeddings[demo_key]  # [T, D]
                    obs_dict["dino_embedding"] = video_embeddings

                cached_demo["obs"] = obs_dict

        # Try to register 'language' and 'dino_embedding' as low_dim keys if needed
        if hasattr(self, "low_dim_keys"):
            if "language" not in self.low_dim_keys:
                self.low_dim_keys.append("language")
            if self.use_dino_embeddings and "dino_embedding" not in self.low_dim_keys:
                self.low_dim_keys.append("dino_embedding")
        if getattr(self, "obs_keys", None) is not None:
            if "language" not in self.obs_keys:
                self.obs_keys.append("language")
            if self.use_dino_embeddings and "dino_embedding" not in self.obs_keys:
                self.obs_keys.append("dino_embedding")

    def _compute_rewards_batch(self, batch_raw: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
        """
        Compute rewards for a batch of samples using either local reward model or eval_server.

        Args:
            batch_raw: List of dictionaries, each containing frames, task, video_embeddings, text_embedding, etc.

        Returns:
            Tuple of (List of reward values as floats, List of success probabilities as floats)
        """
        if self.reward_model is not None:
            # Use local reward model
            samples = [
                raw_dict_to_sample(
                    raw_data=raw_data_item,
                    max_frames=self.max_frames,
                    sample_type="progress",
                )
                for raw_data_item in batch_raw
            ]

            # Treat any C51 / discrete progress head as discrete-mode so its bin logits
            # get softmax-reduced to a [0,1] scalar. Fine-tuned models use loss-type
            # variants like "c51_asymmetric" (same 10-bin head, just an asymmetric loss);
            # matching only the literal "discrete" left those reads on the raw-logit path
            # -> suppressed/garbage reward. No-op for base Robometer-4B ("discrete").
            _plt = self.reward_model_config.loss.progress_loss_type.lower()
            is_discrete_mode = _plt == "discrete" or "c51" in _plt
            progress_discrete_bins = self.reward_model_config.loss.progress_discrete_bins
            outputs = process_batch_helper(
                model_type=self.reward_model_config.model.model_type,
                model=self.reward_model,
                tokenizer=self.tokenizer,
                batch_collator=self.batch_collator,
                device=self.reward_model.device,
                batch_data=[sample.model_dump() for sample in samples],
                job_id=0,
                is_discrete_mode=is_discrete_mode,
                num_bins=progress_discrete_bins,
            )
            rewards_batch = extract_rewards_from_output(outputs)
            success_probs_batch = extract_success_probs_from_output(outputs)
        elif self.use_eval_server:
            # Use eval_server
            samples = [
                raw_dict_to_sample(
                    raw_data=raw_data_item,
                    max_frames=self.max_frames,
                    sample_type="progress",
                )
                for raw_data_item in batch_raw
            ]

            files, sample_data = build_payload(samples)
            outputs = post_batch_npy(self.eval_server_url, files, sample_data, timeout_s=self.eval_server_timeout)
            rewards_batch, success_probs_batch = extract_rewards_from_server_output(outputs)
        else:
            raise ValueError("Neither reward_model nor use_eval_server is set")

        return rewards_batch.tolist(), success_probs_batch.tolist()
