#!/usr/bin/env python3
import io
import json
import tarfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Tuple, Union

import numpy as np
import random
import torch
from PIL import Image
from random import Random
from datasets import Dataset


class SampleSkipRequest(Exception):
    """Raised by samplers to signal that the current item should be skipped (bad data,
    label/frame mismatch, etc.). Upstream RBMDataset.__getitem__ catches this and
    retries with the next index. Use this instead of ValueError when the trajectory
    is unrecoverable but training should continue with other samples."""
    pass

from robometer.configs.experiment_configs import DataConfig
from robometer.data.datasets.helpers import (
    load_frames_from_npz,
    get_segment_indices_with_middle,
    compute_progress_from_segment,
    pad_trajectory_to_max_frames_torch,
    pad_trajectory_to_max_frames_np,
    compute_success_labels,
    create_trajectory_from_dict,
    load_embeddings_from_path,
    linspace_subsample_frames,
    convert_continuous_to_discrete_bins,
)
from robometer.data.dataset_types import Trajectory, ProgressSample, PreferenceSample
from robometer.utils.logger import get_logger
from robometer.data.dataset_category import is_preference_only_ds

logger = get_logger()

# Root under which relative pair-index frames_path entries resolve. Keyframe dataset on the HPC:
# /projects/prjs1958/robometer_frame_dataset/{metaworld,failsafe,droid,robometer,roboreward}/keyframes/.../shard-*.tar
_ROBOMETER_FRAME_DATASET_ROOT = Path("/projects/prjs1958/robometer_frame_dataset")

# Source-of-truth for episode → on-disk frames_path lookup. Per-split pair indices declare
# *which* partner pairs with which query but do not store the partner's tar-shard path; we
# resolve it at sampler init by streaming this file once. (Apr-30 audit found that without
# this resolution, the loader silently fell back to the query's path → demo == query for
# every ICL sample. The fix lives in _load_pair_index below.)
_PAIRS_UNIFIED_PATH = _ROBOMETER_FRAME_DATASET_ROOT / "pairs_unified.jsonl"

# --- Per-frame failure label rubric -------------------------------------------------------
# Maps curated ordinal labels {1,2,3,4,5} to [0, 1] progress decimals used by Robometer's
# training loop. Failures only emit 1..4; 5 is reserved for terminal success frames.
_FAILURE_LABEL_RUBRIC: Dict[int, float] = {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0}


def _frame_labels_to_decimals(labels: List[int]) -> List[float]:
    out: List[float] = []
    for v in labels:
        iv = int(v)
        if iv not in _FAILURE_LABEL_RUBRIC:
            raise ValueError(f"unexpected frame label {v!r}; expected int in {{1,2,3,4,5}}")
        out.append(_FAILURE_LABEL_RUBRIC[iv])
    return out


def _piecewise_linear_ramp(decimals: List[float]) -> List[float]:
    """Option A failure-label smoothing: linearly interpolate between label-transition frames
    so the stair-step rubric output takes on a ramp shape similar to success `t/T`."""
    seq = [float(v) for v in decimals]
    n = len(seq)
    if n <= 1:
        return list(seq)
    anchors: List[int] = [0]
    for i in range(1, n):
        if seq[i] != seq[i - 1]:
            anchors.append(i)
    if anchors[-1] != n - 1:
        anchors.append(n - 1)
    out: List[float] = [0.0] * n
    for a_prev, a_next in zip(anchors[:-1], anchors[1:]):
        v_prev, v_next = seq[a_prev], seq[a_next]
        span = a_next - a_prev
        out[a_prev] = v_prev
        out[a_next] = v_next
        if span <= 1:
            continue
        for j in range(1, span):
            t = j / span
            out[a_prev + j] = v_prev + t * (v_next - v_prev)
    return out


def _apply_failure_smoothing(decimals: List[float], mode: str) -> List[float]:
    if mode == "none":
        return list(decimals)
    if mode == "linear":
        return _piecewise_linear_ramp(decimals)
    raise ValueError(f"unknown failure_label_smoothing mode {mode!r}")


class RBMBaseSampler:
    """Base sampler class that provides trajectory retrieval functions for generating samples."""

    def __init__(
        self,
        config: DataConfig,
        dataset: Dataset,
        combined_indices: Dict[str, Any],
        dataset_success_cutoff_map: Optional[Dict[str, float]] = None,
        verbose: bool = True,
        random_seed: int = 42,
        pad_frames: bool = True,
        is_evaluation: bool = False,
    ):
        """Initialize sampler with dataset and indices.

        Args:
            config: Configuration object
            dataset: The loaded dataset
            combined_indices: Dictionary of combined indices from dataset loading
            dataset_success_cutoff_map: Dictionary mapping dataset names to success cutoff percentages
            verbose: Verbose flag
            random_seed: Random seed for deterministic sampling. Creates a local Random instance to avoid affecting global random state.
        """
        self.config = config
        self.dataset = dataset
        self.verbose = verbose
        self.dataset_success_cutoff_map = dataset_success_cutoff_map or {}
        self._local_random = Random(random_seed)
        self.pad_frames = pad_frames
        self._cached_ids = self.dataset["id"]
        self._cached_is_robot = self.dataset["is_robot"]

        # Build indices from combined_indices
        self._build_indices(combined_indices)

        # ICL (in-context demo) support: when enabled, each sample can carry a successful partner
        # trajectory prepended as context. Coin-flipped per sample at probability self._icl_prob.
        self._use_icl: bool = bool(getattr(config, "use_icl", False))
        self._icl_prob: float = float(getattr(config, "icl_prob", 0.0))
        self._icl_min_coverage: float = float(getattr(config, "icl_min_coverage", 0.40))
        # Ablation: when True, demo frames are replaced with all-zero pixels after
        # loading. Keeps the demo length / separator / position embeddings intact, but
        # removes the visual content. Used to test whether the VLM is genuinely using
        # the demo's visual signal vs. just reacting to longer context length.
        self._icl_demo_blank: bool = bool(getattr(config, "icl_demo_blank", False))
        self._is_evaluation: bool = bool(is_evaluation)
        self._pair_index: Dict[str, Dict[str, Any]] = {}
        if self._use_icl:
            pairs_index_path = getattr(config, "pairs_index_path", None)
            if not pairs_index_path:
                raise ValueError(
                    "data.use_icl=True but data.pairs_index_path is not set. "
                    "Build the pair index from pairs_unified.jsonl and point the config at it."
                )
            self._pair_index = self._load_pair_index(pairs_index_path)
            # Drift sniff is a *training-time* guard against HF/pairs_index mismatch on the
            # train pool. At eval/inference the framework still instantiates the train sampler
            # but it's never iterated, so its coverage is irrelevant — and pointing the config's
            # pairs_index_path at the test/eval index legitimately gives near-zero overlap with
            # the train HF dataset. Skip the assertion in eval to avoid false positives.
            if not self._is_evaluation:
                self._assert_icl_coverage(pairs_index_path)

    def _build_indices(self, combined_indices):
        """Build all index mappings from combined_indices.

        Args:
            combined_indices: Dictionary of combined indices from dataset loading
        """
        # Initialize index mappings from the loaded indices
        self.robot_trajectories = combined_indices["robot_trajectories"]
        self.human_trajectories = combined_indices["human_trajectories"]
        self.optimal_by_task = combined_indices["optimal_by_task"]
        self.suboptimal_by_task = combined_indices["suboptimal_by_task"]
        self.quality_indices = combined_indices["quality_indices"]
        self.task_indices = combined_indices["task_indices"]
        self.source_indices = combined_indices["source_indices"]
        self.partial_success_indices = combined_indices["partial_success_indices"]
        self.paired_human_robot_by_task = combined_indices["paired_human_robot_by_task"]
        self.tasks_with_multiple_quality_labels = combined_indices["tasks_with_multiple_quality_labels"]

        # Build mapping from data source -> available task instructions
        self._build_tasks_by_data_source()

    def _build_tasks_by_data_source(self):
        """Cache mapping from data_source to available task instructions."""
        self.tasks_by_data_source: Dict[str, List[str]] = {}

        all_tasks = self.dataset["task"]
        all_sources = self.dataset["data_source"]

        source_to_tasks: Dict[str, Set[str]] = {}
        for task, source in zip(all_tasks, all_sources):
            if task is None or source is None:
                continue
            if source not in source_to_tasks:
                source_to_tasks[source] = set()
            source_to_tasks[source].add(task)

        self.tasks_by_data_source = {source: list(tasks) for source, tasks in source_to_tasks.items()}

    def _generate_sample(self, item):
        """Generate a sample from an item.

        This method should be overridden by subclasses to implement their specific
        sample generation logic.

        Args:
            item: An item from the dataset (typically a trajectory dict)

        Returns:
            A sample object (e.g., PreferenceSample, ProgressSample)
        """
        raise NotImplementedError("Subclasses must implement _generate_sample")

    def _get_same_task_optimal(self, ref_traj: dict) -> dict | None:
        """Get optimal trajectory from same task (different from ref).

        Args:
            ref_traj: Reference trajectory

        Returns:
            Same task optimal trajectory dict or None if not available
        """
        task_name = ref_traj["task"]
        same_task_optimal_indices = self.optimal_by_task.get(task_name, [])
        if not same_task_optimal_indices:
            logger.trace(f"[BASE SAMPLER] _get_same_task_optimal: No optimal indices for task '{task_name}'")
            return None

        # Use cached IDs to check without loading full trajectories
        chosen_id = ref_traj["id"]
        random_idx = random.choice(same_task_optimal_indices)

        # Retry if the selected trajectory has the same ID as ref
        max_retries = min(10, len(same_task_optimal_indices))
        retries = 0
        while self._cached_ids[random_idx] == chosen_id and retries < max_retries:
            random_idx = random.choice(same_task_optimal_indices)
            retries += 1

        # If still matches after retries, fall back to filtering
        if self._cached_ids[random_idx] == chosen_id:
            filtered_indices = [idx for idx in same_task_optimal_indices if self._cached_ids[idx] != chosen_id]
            if filtered_indices:
                random_idx = random.choice(filtered_indices)
            else:
                # No other trajectories available
                logger.trace(
                    f"[BASE SAMPLER] _get_same_task_optimal: All trajectories have same ID '{chosen_id}' for task '{task_name}'"
                )
                return None

        result = self.dataset[random_idx]
        logger.trace(
            f"[BASE SAMPLER] _get_same_task_optimal: Found trajectory {result.get('id', 'unknown')} for task '{task_name}'"
        )
        return result

    def _get_same_task_suboptimal(self, ref_traj: dict) -> dict | None:
        """Get suboptimal trajectory from same task.

        For trajectories with partial_success, uses partial_success logic instead of quality_label logic.

        Args:
            ref_traj: Reference trajectory

        Returns:
            Suboptimal trajectory dict or None if not available
        """
        # Check if this trajectory uses partial_success
        use_partial_success = ref_traj.get("partial_success") is not None

        if use_partial_success:
            # For trajectories with partial_success, use partial_success logic
            return self._get_different_partial_success_traj(ref_traj)

        # For trajectories without partial_success, use the standard suboptimal logic
        task_name = ref_traj["task"]
        same_task_suboptimal_indices = self.suboptimal_by_task.get(task_name, [])
        if not same_task_suboptimal_indices:
            logger.trace(f"[BASE SAMPLER] _get_same_task_suboptimal: No suboptimal indices for task '{task_name}'")
            return None

        # Use cached IDs to check without loading full trajectories
        chosen_id = ref_traj["id"]
        random_idx = random.choice(same_task_suboptimal_indices)

        # Retry if the selected trajectory has the same ID as ref
        max_retries = min(10, len(same_task_suboptimal_indices))
        retries = 0
        while self._cached_ids[random_idx] == chosen_id and retries < max_retries:
            random_idx = random.choice(same_task_suboptimal_indices)
            retries += 1

        # If still matches after retries, fall back to filtering
        if self._cached_ids[random_idx] == chosen_id:
            filtered_indices = [idx for idx in same_task_suboptimal_indices if self._cached_ids[idx] != chosen_id]
            if filtered_indices:
                random_idx = random.choice(filtered_indices)
            else:
                # No other trajectories available
                logger.trace(
                    f"[BASE SAMPLER] _get_same_task_suboptimal: All trajectories have same ID '{chosen_id}' for task '{task_name}'"
                )
                return None

        result = self.dataset[random_idx]
        logger.trace(
            f"[BASE SAMPLER] _get_same_task_suboptimal: Found trajectory {result.get('id', 'unknown')} for task '{task_name}'"
        )
        return result

    def _get_different_video_traj(self, ref_traj: dict) -> dict | None:
        """Get trajectory from different task.

        Args:
            ref_traj: Reference trajectory

        Returns:
            Different task trajectory dict or None if not available
        """
        same_source_prob = self.config.traj_same_source_prob
        data_source = ref_traj.get("data_source")
        other_tasks = []

        if data_source and data_source in self.tasks_by_data_source and random.random() < same_source_prob:
            other_tasks = [task for task in self.tasks_by_data_source[data_source] if task != ref_traj["task"]]

        if not other_tasks:
            other_tasks = [task for task in self.optimal_by_task.keys() if task != ref_traj["task"]]

        if not other_tasks:
            logger.trace(
                f"[BASE SAMPLER] _get_different_video_traj: No other tasks available (ref task: '{ref_traj['task']}')"
            )
            return None

        # Try up to 2 times to find a valid task
        max_retries = 2
        other_task_indices = None
        other_task = None

        for attempt in range(max_retries):
            other_task = random.choice(other_tasks)
            if other_task not in self.optimal_by_task:
                logger.trace(
                    f"[BASE SAMPLER] _get_different_video_traj: Attempt {attempt + 1}/{max_retries}: Task '{other_task}' not found in optimal_by_task"
                )
                continue

            other_task_indices = self.optimal_by_task[other_task]
            if not other_task_indices:
                logger.trace(
                    f"[BASE SAMPLER] _get_different_video_traj: Attempt {attempt + 1}/{max_retries}: Task '{other_task}' has no optimal indices"
                )
                continue

            # Found a valid task with indices
            break

        if other_task_indices is None or not other_task_indices:
            logger.trace(
                f"[BASE SAMPLER] _get_different_video_traj: Failed to find valid task after {max_retries} attempts"
            )
            return None

        other_idx = random.choice(other_task_indices)
        result = self.dataset[other_idx]
        logger.trace(
            f"[BASE SAMPLER] _get_different_video_traj: Found trajectory {result.get('id', 'unknown')} from task '{other_task}'"
        )
        return result

    def _get_different_task_instruction(self, ref_traj: dict) -> dict | None:
        """Get the same trajectory but with a different task instruction.

        Args:
            ref_traj: Reference trajectory

        Returns:
            Trajectory dict with different task instruction or None if not available
        """
        same_source_prob = self.config.traj_same_source_prob
        data_source = ref_traj.get("data_source")
        candidate_tasks = []

        if data_source and data_source in self.tasks_by_data_source and random.random() < same_source_prob:
            candidate_tasks = [task for task in self.tasks_by_data_source[data_source] if task != ref_traj["task"]]

        if not candidate_tasks:
            candidate_tasks = [task for task in self.optimal_by_task.keys() if task != ref_traj["task"]]

        if not candidate_tasks:
            logger.trace(
                f"[BASE SAMPLER] _get_different_task_instruction: No candidate tasks available (ref task: '{ref_traj['task']}')"
            )
            return None

        other_task = random.choice(candidate_tasks)

        # Get embeddings_path and lang_vector from a random trajectory with the other_task
        other_task_indices = self.optimal_by_task.get(other_task, [])
        if not other_task_indices:
            logger.trace(f"[BASE SAMPLER] _get_different_task_instruction: Task '{other_task}' has no optimal indices")
            return None

        other_task_idx = random.choice(other_task_indices)
        other_task_traj = self.dataset[other_task_idx]

        # Create a copy of the trajectory with the task changed
        # Use embeddings_path and lang_vector from the other_task trajectory
        new_traj = ref_traj.copy()
        new_traj["task"] = other_task
        # Get embeddings_path and lang_vector from a random trajectory with the other_task
        if "embeddings_path" in other_task_traj:
            new_traj["embeddings_path"] = other_task_traj["embeddings_path"]
        if "lang_vector" in other_task_traj:
            new_traj["lang_vector"] = other_task_traj["lang_vector"]
        return new_traj

    def _get_paired_human_robot_traj(self, ref_traj: dict) -> dict | None:
        """Get paired human/robot trajectory for the same task.

        Given a reference trajectory, if it's a robot trajectory, returns a human trajectory
        from the same task. If it's a human trajectory, returns a robot trajectory from the
        same task.

        Args:
            ref_traj: Reference trajectory (can be robot or human)

        Returns:
            Paired trajectory dict (opposite type) or None if not available
        """
        task = ref_traj["task"]
        is_robot = ref_traj.get("is_robot", True)

        if task not in self.paired_human_robot_by_task:
            logger.trace(
                f"[BASE SAMPLER] _get_paired_human_robot_traj: Task '{task}' not in paired_human_robot_by_task"
            )
            return None

        task_pairs = self.paired_human_robot_by_task[task]

        # Get opposite type
        opposite_key = "human" if is_robot else "robot"
        opposite_indices = task_pairs.get(opposite_key, [])

        if not opposite_indices:
            logger.trace(f"[BASE SAMPLER] _get_paired_human_robot_traj: No {opposite_key} indices for task '{task}'")
            return None

        # Sample a paired trajectory and verify it's different from reference
        chosen_id = ref_traj["id"]
        available_indices = opposite_indices.copy()
        paired_traj = None

        # Add retry limit to prevent infinite loops
        max_retries = min(len(available_indices), 10)
        retries = 0

        logger.trace(
            f"[BASE SAMPLER] _get_paired_human_robot_traj: Looking for {opposite_key} trajectory (chosen_id: {chosen_id}, available: {len(available_indices)})"
        )

        while (paired_traj is None or paired_traj.get("id") == chosen_id) and retries < max_retries:
            retries += 1

            if not available_indices:
                logger.trace(
                    f"[BASE SAMPLER] _get_paired_human_robot_traj: No more available indices after {retries} retries"
                )
                return None

            paired_idx = random.choice(available_indices)
            paired_traj = self.dataset[paired_idx]

            # If it matches, remove this index and try again
            if paired_traj.get("id") == chosen_id:
                available_indices = [idx for idx in available_indices if idx != paired_idx]
                paired_traj = None
                continue

        # If we exhausted retries without finding a valid trajectory, return None
        if paired_traj is None or paired_traj.get("id") == chosen_id:
            logger.trace(
                f"[BASE SAMPLER] _get_paired_human_robot_traj: Failed to find valid paired trajectory after {max_retries} retries"
            )
            return None

        logger.trace(
            f"[BASE SAMPLER] _get_paired_human_robot_traj: Found paired trajectory {paired_traj.get('id', 'unknown')} on retry {retries}"
        )
        return paired_traj

    def _get_different_partial_success_traj(self, ref_traj: dict) -> dict | None:
        """Get trajectory from same task with different partial_success.

        Finds trajectories with either higher or lower partial_success than the reference,
        using absolute difference for threshold checking.

        Args:
            ref_traj: Reference trajectory

        Returns:
            Trajectory dict with different partial_success from same task or None if not available
        """
        task_name = ref_traj["task"]
        ref_partial_success = ref_traj.get("partial_success")

        # Check if partial_success is available
        if ref_partial_success is None:
            logger.trace(
                f"[BASE SAMPLER] _get_different_partial_success_traj: No partial_success for trajectory {ref_traj.get('id', 'unknown')}"
            )
            return None

        # Get minimum threshold from config
        min_threshold = getattr(self.config, "partial_success_threshold", 0.2)

        # Get all trajectories from the same task
        same_task_indices = self.task_indices.get(task_name, [])
        if not same_task_indices:
            logger.trace(
                f"[BASE SAMPLER] _get_different_partial_success_traj: No trajectories found for task '{task_name}'"
            )
            return None

        # Filter to trajectories with different partial_success that meet the threshold requirement
        # Uses absolute difference to allow both higher and lower partial_success
        chosen_id = ref_traj["id"]
        candidate_indices = []

        for idx in same_task_indices:
            # Skip if same trajectory
            if self._cached_ids[idx] == chosen_id:
                continue

            # Get partial_success for this trajectory
            traj_dict = self.dataset[idx]
            traj_partial_success = traj_dict.get("partial_success", None)

            if traj_partial_success is None:
                logger.trace(
                    f"[BASE SAMPLER] _get_different_partial_success_traj: No partial_success for trajectory {traj_dict.get('id', 'unknown')}, task '{task_name}'"
                )
                continue

            # Include if partial_success differs from reference by at least the threshold (using abs)
            partial_success_diff = abs(ref_partial_success - traj_partial_success)
            if partial_success_diff >= min_threshold:
                candidate_indices.append(idx)

        if not candidate_indices:
            logger.trace(
                f"[BASE SAMPLER] _get_different_partial_success_traj: No trajectories with different partial_success (threshold: {min_threshold}) for task '{task_name}' (ref: {ref_partial_success})"
            )
            return None

        # Randomly select from candidates
        selected_idx = random.choice(candidate_indices)
        result = self.dataset[selected_idx]
        result_partial_success = result.get("partial_success")
        # If ref_partial_success is 1.0, direction is always "lower" since 1.0 is the maximum
        if ref_partial_success == 1.0:
            direction = "lower"
        else:
            direction = "higher" if result_partial_success > ref_partial_success else "lower"
        logger.trace(
            f"[BASE SAMPLER] _get_different_partial_success_traj: Found trajectory {result.get('id', 'unknown')} with partial_success {result_partial_success} ({direction} than {ref_partial_success}, abs diff: {abs(ref_partial_success - result_partial_success):.3f}, threshold: {min_threshold})"
        )
        return result

    def _get_subsample_indices(
        self, data, direction: str = "bidirectional", max_frames: int = None
    ) -> Optional[Tuple[int, int, int]]:
        """Get start, middle, and end indices for subsample strategy.

        Samples three random frames from the trajectory. The relationship between indices
        follows three main scenarios:
        1. start < middle < end: forward progress - normal forward progression through trajectory
        2. start < end < middle: rewind progress - forward from start to end, then continues to middle (simulating rewind/backtrack)
        3. end < middle < start: reverse progress - backward from start through middle to end (full backward traversal)

        Args:
            data: Trajectory data (frames or embeddings) to sample from
            direction: Sampling direction - "forward" (start < middle < end),
                      "reverse" (end < middle < start),
                      "rewind" (start < end < middle),
                      or "bidirectional" (any of the 3 orderings)
            max_frames: Maximum number of frames to subsample. If 1, returns only start. If 2, returns start and end.

        Returns:
            Tuple of (start_idx, middle_idx, end_idx), or None if insufficient frames
            For max_frames == 1: returns (start_idx, None, None)
            For max_frames == 2: returns (start_idx, None, end_idx)
        """
        num_frames_total = len(data) if hasattr(data, "__len__") else data.shape[0]

        # Handle edge cases for max_frames == 1 or 2
        if max_frames == 1:
            # Randomly sample 1 frame
            random_idx = random.randint(0, num_frames_total - 1)
            logger.trace(f"[BASE SAMPLER] _get_subsample_indices: max_frames=1, randomly sampled idx={random_idx}")
            return (random_idx, None, None)

        if max_frames == 2:
            # Sample 2 frames: either forward (start < end) or reverse (end < start)
            # No rewind possible with only 2 frames
            if direction == "reverse":
                # Reverse: sample end first, then start (end < start)
                end_idx = random.randint(0, num_frames_total - 2)
                start_idx = random.randint(end_idx + 1, num_frames_total - 1)
            else:
                # Forward: sample start first, then end (start < end)
                start_idx = random.randint(0, num_frames_total - 2)
                end_idx = random.randint(start_idx + 1, num_frames_total - 1)
            logger.trace(
                f"[BASE SAMPLER] _get_subsample_indices: max_frames=2, start_idx={start_idx}, end_idx={end_idx}, direction={direction}"
            )
            return (start_idx, None, end_idx)

        if num_frames_total < 3:
            logger.trace(f"[BASE SAMPLER] _get_subsample_indices: Not enough frames ({num_frames_total})")
            return None

        # Sample three random distinct frames
        frame_indices = sorted(random.sample(range(num_frames_total), 3))
        frame1_idx, frame2_idx, frame3_idx = frame_indices

        # Determine start, middle, and end based on direction
        # We only care about 3 cases:
        # 1. start < middle < end: forward progress
        # 2. start < end < middle: rewind progress
        # 3. end < middle < start: reverse progress

        if direction == "forward":
            # Case 1: start < middle < end
            start_idx = frame1_idx
            middle_idx = frame2_idx
            end_idx = frame3_idx
        elif direction == "reverse":
            # Case 3: end < middle < start
            end_idx = frame1_idx
            middle_idx = frame2_idx
            start_idx = frame3_idx
        elif direction == "rewind":
            # Case 2: start < end < middle
            start_idx = frame1_idx
            end_idx = frame2_idx
            middle_idx = frame3_idx
        else:  # bidirectional (default)
            # Randomly choose from the 3 cases
            pattern = random.choice([1, 2, 3])
            if pattern == 1:  # start < middle < end: forward progress
                start_idx = frame1_idx
                middle_idx = frame2_idx
                end_idx = frame3_idx
            elif pattern == 2:  # start < end < middle: rewind progress
                start_idx = frame1_idx
                end_idx = frame2_idx
                middle_idx = frame3_idx
            else:  # pattern == 3: end < middle < start: reverse progress
                end_idx = frame1_idx
                middle_idx = frame2_idx
                start_idx = frame3_idx

        logger.trace(
            f"[BASE SAMPLER] _get_subsample_indices: Selected indices start={start_idx}, middle={middle_idx}, end={end_idx} "
            f"from {num_frames_total} total frames (direction: {direction})"
        )
        return start_idx, middle_idx, end_idx

    # ---------------------------------------------------------------------
    # ICL (in-context demo) helpers
    # ---------------------------------------------------------------------

    def _load_pair_index(self, pairs_index_path: str) -> Dict[str, Dict[str, Any]]:
        """Stream a JSONL pair index into a dict keyed by episode_id, then resolve every
        partner_episode_id back to a tar-shard path by cross-referencing pairs_unified.jsonl.

        Each pair-index row declares "use partner X as demo for query Y" via:
          - episode_id            (the query)
          - partner_episode_id    (the partner, the demo)
          - partner_task          (semantic check)
          - frames_path           ("<shard>::<episode_dir>" for the QUERY)
        but does NOT store the partner's frames_path — that lives in pairs_unified.jsonl
        keyed by episode_id. We resolve here, once at sampler init, and inject
        `partner_frames_path` into each pair_index row. _load_partner_trajectory then
        requires this field to exist (no silent fallback to the query's path).

        Rows with a null partner_episode_id are silently skipped. Rows whose episode_id is
        not in the current dataset's id column are skipped too, so the in-memory index stays
        proportional to the training set, not to the full 628k-row pairs_unified.jsonl.
        """
        path = Path(pairs_index_path)
        if not path.is_file():
            raise FileNotFoundError(f"pairs_index_path does not exist: {pairs_index_path}")

        episode_ids_in_dataset: Set[str] = set(self._cached_ids)
        pair_index: Dict[str, Dict[str, Any]] = {}

        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not row.get("partner_episode_id"):
                    continue
                episode_id = row.get("episode_id")
                if episode_id not in episode_ids_in_dataset:
                    continue
                pair_index[episode_id] = row

        # --- Resolve partner_frames_path from pairs_unified.jsonl -----------------------
        # This is the part that prevents the Apr-30 self-demo bug: pair indices don't carry
        # the partner's tar-shard path, so without this lookup the loader's
        # `pair_row.get("partner_frames_path") or pair_row.get("frames_path")` falls back to
        # the query's path → demo == query.
        needed_partner_ids: Set[str] = {
            row["partner_episode_id"] for row in pair_index.values()
        }
        if needed_partner_ids:
            if not _PAIRS_UNIFIED_PATH.is_file():
                raise FileNotFoundError(
                    f"pairs_unified.jsonl not found at {_PAIRS_UNIFIED_PATH}; cannot resolve "
                    f"partner_frames_path for ICL. The pair-index files alone do not carry the "
                    f"partner's on-disk tar path."
                )
            resolved: Dict[str, str] = {}
            with _PAIRS_UNIFIED_PATH.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    eid = r.get("episode_id")
                    if eid in needed_partner_ids:
                        fp = r.get("frames_path")
                        if fp:
                            resolved[eid] = fp
                    if len(resolved) == len(needed_partner_ids):
                        break  # early exit — got all of them
            missing = needed_partner_ids - set(resolved)
            if missing:
                raise RuntimeError(
                    f"[BASE SAMPLER] Could not resolve partner_frames_path for {len(missing)} "
                    f"partner_episode_ids in {pairs_index_path} via {_PAIRS_UNIFIED_PATH}. "
                    f"First 3 missing: {list(missing)[:3]}. The pair index references partners "
                    f"that are not present in the source-of-truth pairs_unified.jsonl."
                )
            for row in pair_index.values():
                row["partner_frames_path"] = resolved[row["partner_episode_id"]]

        n_pool = len(episode_ids_in_dataset)
        n_match = len(pair_index)
        coverage = (n_match / n_pool) if n_pool > 0 else 0.0
        logger.info(
            f"[BASE SAMPLER] ICL pair index loaded: {n_match} / {n_pool} episodes "
            f"({coverage:.1%} partner coverage in this pool); "
            f"resolved {len(needed_partner_ids)} partner_frames_paths from pairs_unified.jsonl"
        )
        return pair_index

    def _assert_icl_coverage(self, pairs_index_path: str) -> None:
        """Hard-fail at startup if partner coverage is below `data.icl_min_coverage`.

        Catches drift between the HF Arrow dataset and the pair index — e.g. the Apr-25
        bake-off where train.jsonl was rebuilt after the HF cache was generated, leaving
        only ~50% of HF episode IDs findable in the new pair index. ICL silently fired at
        half the configured rate. Default threshold (0.40) is a drift sniff, not a target.
        """
        n_pool = len(self._cached_ids)
        n_match = len(self._pair_index)
        if n_pool == 0:
            return
        coverage = n_match / n_pool
        if coverage < self._icl_min_coverage:
            raise RuntimeError(
                f"[ICL DRIFT] Partner coverage in this training pool is {coverage:.1%} "
                f"({n_match} / {n_pool}), below the configured floor "
                f"data.icl_min_coverage={self._icl_min_coverage:.2f}. "
                f"This usually means the HF Arrow dataset and pairs_index_path were built "
                f"from different snapshots of the splits. Rebuild the HF dataset with "
                f"`SPLIT=<name> sbatch jobs/preprocess_split.job`, or align the pair index "
                f"to the existing HF IDs, or — if you legitimately expect low coverage — "
                f"lower data.icl_min_coverage. pairs_index_path={pairs_index_path}"
            )

    def _resolve_shard_path(self, shard_rel_or_abs: str) -> Path:
        """Resolve a shard path from the pair index. Absolute paths pass through;
        relative paths are resolved against the keyframe dataset root."""
        p = Path(shard_rel_or_abs)
        if p.is_absolute():
            return p
        return _ROBOMETER_FRAME_DATASET_ROOT / p

    def _load_partner_frames(self, shard_path: Path, episode_dir: str) -> np.ndarray:
        """Extract all .jpg frames for a given episode_dir from a tar shard, in sorted order.

        Returns an (T, H, W, 3) uint8 ndarray. Matches the layout produced by
        Robometer-LoRA/data_adapters/robometer_frames_loader.py::TarKeyframeLoader.
        """
        prefix = episode_dir.rstrip("/") + "/"
        frames: List[np.ndarray] = []
        with tarfile.open(shard_path, "r") as tf:
            members = [
                m for m in tf.getmembers()
                if m.name.startswith(prefix) and m.name.endswith(".jpg")
            ]
            members.sort(key=lambda m: m.name)
            for m in members:
                f = tf.extractfile(m)
                if f is None:
                    continue
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
                frames.append(np.asarray(img, dtype=np.uint8))
        if not frames:
            raise RuntimeError(
                f"No JPG frames found for episode_dir={episode_dir!r} inside {shard_path}"
            )
        return np.stack(frames, axis=0)

    def _load_partner_trajectory(self, pair_row: Dict[str, Any]) -> Trajectory:
        """Build a bare Trajectory for a partner episode. The partner is used only as
        in-context demo, so no progress labels / success labels / embeddings are computed.

        REQUIRES `partner_frames_path` to be set on the row. _load_pair_index resolves this
        at sampler init by joining against pairs_unified.jsonl. We deliberately do NOT fall
        back to `frames_path` (the query's path) — that fallback was the cause of the Apr-30
        self-demo bug where every ICL sample silently loaded the query as its own demo.
        """
        frames_path = pair_row.get("partner_frames_path")
        if not frames_path or "::" not in frames_path:
            raise RuntimeError(
                f"pair_row missing partner_frames_path for "
                f"partner_episode_id={pair_row.get('partner_episode_id')!r}. "
                f"This must be populated at _load_pair_index time via the pairs_unified.jsonl "
                f"cross-reference. Got: {frames_path!r}. (Falling back to the query's frames_path "
                f"is forbidden — see the Apr-30 self-demo audit.)"
            )
        shard_str, episode_dir = frames_path.split("::", 1)
        shard_path = self._resolve_shard_path(shard_str)
        frames = self._load_partner_frames(shard_path, episode_dir)

        # ABLATION: zero out the demo's visual content while keeping its shape +
        # position. Lets us test whether the VLM is using the demo's visual signal
        # or just the extra context length / separator token.
        if self._icl_demo_blank:
            frames = np.zeros_like(frames)

        return Trajectory(
            id=pair_row.get("partner_episode_id"),
            task=pair_row.get("partner_task") or pair_row.get("task"),
            frames=frames,
            frames_shape=frames.shape,
            is_robot=True,
            quality_label="successful",
            data_source=pair_row.get("partner_source") or pair_row.get("data_source"),
        )

    def _maybe_attach_icl_context(self, sample):
        """Per-sample Bernoulli(icl_prob) decides whether to populate sample.context_trajectory.
        No-op unless self._use_icl is True."""
        if not self._use_icl or sample is None:
            return sample
        if self._local_random.random() >= self._icl_prob:
            return sample

        # Pick the trajectory whose partner we look up. Progress samples have one; preference
        # samples use the chosen (labeled) side as the anchor.
        if isinstance(sample, ProgressSample):
            query_id = sample.trajectory.id
        elif isinstance(sample, PreferenceSample):
            query_id = sample.chosen_trajectory.id
        else:
            return sample

        pair = self._pair_index.get(query_id)
        # Soft fallback covers two cases that would otherwise drop or crash the sample:
        #   (1) `pair is None`: no entry for this query in the pair index.
        #   (2) `pair` exists but `partner_frames_path` was not resolved at index-load time
        #       (the partner_episode_id wasn't found in pairs_unified.jsonl). _load_partner_
        #       trajectory raises RuntimeError on this — which propagates as a silent dataloader
        #       drop and surfaces as `compile_results.py: "No valid policy ranking data found"`
        #       when most samples in a dataset hit this case (e.g. eval failsafe at 14% partner
        #       resolution; eval metaworld at 26%).
        # In both cases the right behavior is: train/eval this sample ICL-off rather than crash.
        partner_path = pair.get("partner_frames_path") if pair is not None else None
        if pair is None or not partner_path or "::" not in partner_path:
            logger.trace(
                f"[BASE SAMPLER] ICL coin-on but partner unavailable for episode {query_id!r} "
                f"(pair_missing={pair is None}, partner_path={partner_path!r}) — using ICL-off."
            )
            return sample
        sample.context_trajectory = self._load_partner_trajectory(pair)
        return sample

    def _get_traj_from_data(
        self,
        traj: dict | Trajectory,
        subsample_strategy: str | None = None,
        frame_indices: List[int] | None = None,
        metadata: Dict[str, Any] | None = None,
        pad_frames: bool = True,
    ) -> Trajectory:
        """Load, subsample, and optionally pad trajectory data and create a Trajectory object.

        Args:
            traj: Trajectory dict or Trajectory object
            subsample_strategy: Optional strategy for subsampling ("subsample_forward", "subsample_reverse", "subsample_rewind", or None for default/bidirectional). Ignored if frame_indices is provided.
            frame_indices: Optional list of specific frame indices to use. If provided, subsample_strategy is ignored.
            metadata: Optional metadata dict to merge into trajectory metadata.
            pad_frames: Whether to pad the trajectory data to max_frames.

        Returns:
            Trajectory object with loaded and subsampled data (padded)
        """
        # Initialize variables
        frames = None
        video_embeddings = None
        text_embedding = None
        data = None

        if isinstance(traj, Trajectory):
            # If already a Trajectory, just return it
            return traj

        # Load from dict
        # Check if text_embedding is already provided in the dict (for samplers that need to override it)
        if "text_embedding" in traj and traj["text_embedding"] is not None:
            text_embedding = traj["text_embedding"]

        if self.config.load_embeddings and traj.get("embeddings_path"):
            embeddings = load_embeddings_from_path(traj["embeddings_path"])
            video_embeddings = embeddings["video_embeddings"]
            # Only use loaded text_embedding if not already provided in dict
            if text_embedding is None:
                text_embedding = embeddings["text_embedding"]
            data = video_embeddings
        else:
            if isinstance(traj["frames"], str):
                frames = load_frames_from_npz(traj["frames"])
            else:
                frames = traj["frames"]
            data = frames

        # Get total frames for progress computation
        if hasattr(data, "shape"):
            num_frames_total = data.shape[0]
        else:
            num_frames_total = len(data)

        ds_key = traj["data_source"]
        success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)

        # Determine which indices to use (construct indices first, then subsample uniformly)
        if frame_indices is not None:
            # Use provided frame indices directly
            indices = frame_indices
        elif subsample_strategy is not None:
            # Use subsampling strategy
            # Get subsample indices (handles edge cases for max_frames == 1 or 2)
            if subsample_strategy == "subsample_forward":
                strategy_indices = self._get_subsample_indices(
                    data, direction="forward", max_frames=self.config.max_frames
                )
            elif subsample_strategy == "subsample_reverse":
                strategy_indices = self._get_subsample_indices(
                    data, direction="reverse", max_frames=self.config.max_frames
                )
            elif subsample_strategy == "subsample_rewind":
                strategy_indices = self._get_subsample_indices(
                    data, direction="rewind", max_frames=self.config.max_frames
                )
            else:
                strategy_indices = self._get_subsample_indices(
                    data, direction="bidirectional", max_frames=self.config.max_frames
                )

            if strategy_indices is None:
                logger.trace("[BASE SAMPLER] _get_traj_from_data: Failed to get uniform sample indices")
                return None

            start_idx, middle_idx, end_idx = strategy_indices

            logger.trace(
                f"[BASE SAMPLER] _get_traj_from_data: Subsampling trajectory with strategy: {subsample_strategy}, start_idx: {start_idx}, middle_idx: {middle_idx}, end_idx: {end_idx}"
            )

            # Use middle_idx only for rewind strategy (requires at least 3 frames)
            use_middle = subsample_strategy == "subsample_rewind" and middle_idx is not None and num_frames_total >= 3

            # Use get_segment_indices_with_middle to construct indices
            indices = get_segment_indices_with_middle(
                num_frames_total=num_frames_total,
                start_idx=start_idx,
                end_idx=end_idx,
                middle_idx=middle_idx if use_middle else None,
                max_frames=self.config.max_frames,
            )
        else:
            # No subsampling strategy or indices provided - use all frames
            indices = list(range(num_frames_total))

        # Extract data using indices
        subsampled = data[indices]

        # Get partial_success early to pass to compute_progress_from_segment
        partial_success = traj.get("partial_success")

        # Compute progress
        target_progress = compute_progress_from_segment(
            num_frames_total=num_frames_total,
            frame_indices=indices,
            progress_pred_type=self.config.progress_pred_type,
            success_cutoff=success_cutoff,
            partial_success=partial_success,
        )

        # --- Failure path: override target_progress with curated per-frame labels when present
        # Loader emits `frame_labels` as a List[int in {1..4}] on failure trajectories; we map
        # each to a rubric decimal ({1→0, 2→0.25, 3→0.5, 4→0.75}) and optionally smooth via
        # piecewise-linear interpolation so the shape matches the smooth `t/T` success signal.
        raw_frame_labels = traj.get("frame_labels")
        if raw_frame_labels is not None:
            if not isinstance(raw_frame_labels, (list, tuple)):
                raise TypeError(
                    f"frame_labels must be a list, got {type(raw_frame_labels).__name__} "
                    f"(trajectory id={traj.get('id')!r})"
                )
            if len(raw_frame_labels) != num_frames_total:
                # Bad data: the curated frame_labels were authored against a different
                # number of frames than the actual loaded trajectory has (e.g., labels
                # built from a 16-frame view but the default keyframes/ view has fewer).
                # Skip-and-retry instead of crashing the whole training run — the
                # DataLoader worker will receive the SampleSkipRequest, RBMDataset's
                # __getitem__ wraps a retry loop that advances to the next index.
                raise SampleSkipRequest(
                    f"frame_labels length mismatch: got {len(raw_frame_labels)}, expected "
                    f"{num_frames_total} (trajectory id={traj.get('id')!r})"
                )
            # Subsample labels along the same indices the frames were subsampled along, then
            # rubric-map + smooth. indices is guaranteed to be a list at this point.
            sub_labels = [raw_frame_labels[i] for i in indices]
            decimals = _frame_labels_to_decimals(sub_labels)
            smoothing_mode = getattr(self.config, "failure_label_smoothing", "none")
            target_progress = _apply_failure_smoothing(decimals, smoothing_mode)

        # --- Success path ------------------------------------------------------------
        # Two optional transforms for success-trajectory labels, both governed by config:
        #
        # 1) CORN bucketization (losses.md §Loss 1): when progress_loss_type == "corn_asymmetric",
        #    snap the smooth t/T decimals to the nearest 25-quantile {0, 0.25, 0.5, 0.75, 1.0}.
        #    This matches the resolution of the failure rubric so both sides live in the same
        #    5-level ordinal space (ordinal 1..5 at loss time).
        #
        # 2) Gaussian regulariser (shared with Loss 2 via `data.success_label_noise_std`):
        #    per-frame N(0, σ²) added to the (possibly bucketized) decimals, clipped to [0, 1].
        #    Skips frames 0 and T-1 — those are label anchors: frame 0 is the starting state
        #    (ordinal 1), frame T-1 is the terminal success frame (ordinal 5). Anchoring keeps
        #    every CORN threshold supervised by at least one confident positive and one
        #    confident negative per success trajectory.
        elif traj.get("quality_label") == "successful":
            # (1) Loss-1 bucketization.
            loss_type_str = str(getattr(self.config, "progress_loss_type", "") or "").lower()
            if loss_type_str == "corn_asymmetric" and target_progress:
                target_progress = [round(float(v) * 4.0) / 4.0 for v in target_progress]

            # (2) Gaussian noise on interior frames only.
            noise_std = float(getattr(self.config, "success_label_noise_std", 0.0) or 0.0)
            if noise_std > 0.0 and target_progress and len(target_progress) >= 3:
                noisy: List[float] = [float(target_progress[0])]
                for v in target_progress[1:-1]:
                    perturbed = float(v) + self._local_random.gauss(0.0, noise_std)
                    noisy.append(max(0.0, min(1.0, perturbed)))
                noisy.append(float(target_progress[-1]))
                target_progress = noisy

        # Subsample uniformly if needed (if we have more frames than max_frames)
        current_frame_count = len(subsampled) if hasattr(subsampled, "__len__") else subsampled.shape[0]
        if current_frame_count > self.config.max_frames:
            subsampled, frame_indices_subsample = linspace_subsample_frames(subsampled, self.config.max_frames)
            # Update indices and target_progress
            if target_progress and len(target_progress) == current_frame_count:
                target_progress = [target_progress[idx] for idx in frame_indices_subsample]
            indices = [indices[idx] for idx in frame_indices_subsample] if isinstance(indices, list) else indices

        # Pad if needed
        if target_progress and pad_frames:
            if self.config.load_embeddings:
                subsampled, target_progress = pad_trajectory_to_max_frames_torch(
                    subsampled, target_progress, self.config.max_frames
                )
            else:
                subsampled, target_progress = pad_trajectory_to_max_frames_np(
                    subsampled, target_progress, self.config.max_frames
                )

        # Create predict_last_frame_mask: mark the last frame if partial_success < 1.0
        # If predict_last_frame_partial_progress is True and partial_success < 1.0 and the last original frame is in the subsampled indices,
        # mark all positions where it appears with 1.0, all others 0.0. Otherwise, all 1.0s.
        final_frame_count = len(subsampled)
        predict_last_frame_mask = [1.0] * final_frame_count  # Default: all 1.0s (no masking)

        if self.config.predict_last_frame_partial_progress and partial_success is not None:
            if partial_success == 1.0 and not is_preference_only_ds(traj["data_source"]):
                pass
            else:
                last_original_frame_idx = num_frames_total - 1
                if isinstance(indices, list) and last_original_frame_idx in indices:
                    # Find all positions where the last frame index appears
                    last_frame_positions = [
                        i for i, idx in enumerate(indices) if idx == last_original_frame_idx and i < final_frame_count
                    ]
                    if last_frame_positions:
                        # Mark all positions where the last frame appears with 1.0, all others 0.0
                        predict_last_frame_mask = [0.0] * final_frame_count
                        for pos in last_frame_positions:
                            predict_last_frame_mask[pos] = 1.0
                else:
                    predict_last_frame_mask = [0.0] * final_frame_count

        # Update frames_shape
        frames_shape = subsampled.shape if hasattr(subsampled, "shape") else tuple()

        # Set frames or video_embeddings
        if self.config.load_embeddings:
            video_embeddings = subsampled
        else:
            frames = subsampled

        # Compute success labels
        success_label = compute_success_labels(
            target_progress=target_progress,
            data_source=traj["data_source"],
            dataset_success_percent=self.dataset_success_cutoff_map,
            max_success=self.config.max_success,
            quality_label=traj.get("quality_label"),
        )

        # Convert partial_success and target_progress to discrete bins if in discrete mode
        if self.config.progress_loss_type.lower() in ("discrete", "c51_asymmetric"):
            if partial_success is not None:
                partial_success = convert_continuous_to_discrete_bins(
                    [partial_success], self.config.progress_discrete_bins
                )[0]
            target_progress = convert_continuous_to_discrete_bins(target_progress, self.config.progress_discrete_bins)

        trajectory = create_trajectory_from_dict(
            traj,
            overrides={
                "frames": frames,
                "frames_shape": frames_shape,
                "video_embeddings": video_embeddings,
                "text_embedding": text_embedding,
                "target_progress": target_progress,
                "success_label": success_label,
                "partial_success": partial_success,
                "predict_last_frame_mask": predict_last_frame_mask,
                "metadata": metadata,
            },
        )
        return trajectory
