#!/usr/bin/env python3
"""CPU-only smoke test for the Phase-2 label-harmonization changes.

Covers three effects in the sampler / collator:
  1) Failure trajectories with per-frame `frame_labels` produce rubric decimals as
     `target_progress` (and the piecewise-linear smoother does the right thing).
  2) Success trajectories with `data.success_label_noise_std > 0` get Gaussian-perturbed
     labels (deterministic given the sampler's seed, clipped to [0, 1]).
  3) `should_compute_progress` returns 1.0 for failures whose data_source is one of our
     keyframe datasets (robometer_frames_*).

Does NOT call into torch / the real VLM — we bypass RBMBaseSampler's dataset plumbing by
instantiating a minimal subclass and handing it a fake trajectory dict directly.

Run:
    python Robometer-LoRA/scripts/smoke_test_labels.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

# Make the upstream Robometer package importable when the script is run from the repo root.
_HERE = Path(__file__).resolve()
_ROBOMETER_DIR = _HERE.parent.parent.parent / "Robometer"
if str(_ROBOMETER_DIR) not in sys.path:
    sys.path.insert(0, str(_ROBOMETER_DIR))

import numpy as np

# Load via collators first: Robometer's samplers package has an internal circular-import chain
# that only surfaces when samplers.base is the very first entry point. Going through rbm_heads
# triggers the full dependency graph in a safe order so subsequent imports from samplers.base
# resolve without cycling.
from robometer.data.collators.rbm_heads import should_compute_progress  # noqa: F401

from robometer.data.samplers.base import (  # noqa: E402
    RBMBaseSampler,
    _apply_failure_smoothing,
    _frame_labels_to_decimals,
    _piecewise_linear_ramp,
)


def _make_sampler(failure_label_smoothing: str = "none", success_label_noise_std: float = 0.0) -> RBMBaseSampler:
    """Build a minimal RBMBaseSampler without a real HF dataset. We only exercise
    `_get_traj_from_data`, which does not touch the index maps or cached ids."""

    class _MinimalSampler(RBMBaseSampler):
        def __init__(self, cfg):
            self.config = cfg
            self.verbose = False
            self.dataset_success_cutoff_map = {}
            import random as _random
            self._local_random = _random.Random(42)
            self.pad_frames = True
            # Leave the rest (self.dataset, self._cached_ids, combined indices) unset — the
            # methods we call don't touch them.

    cfg = SimpleNamespace(
        load_embeddings=False,
        max_frames=16,
        max_success=1.0,
        progress_pred_type="absolute_wrt_total_frames",
        progress_loss_type="l2",  # keep as continuous for easy inspection
        progress_discrete_bins=5,
        predict_last_frame_partial_progress=False,
        use_icl=False,
        icl_prob=0.0,
        pairs_index_path=None,
        failure_label_smoothing=failure_label_smoothing,
        success_label_noise_std=success_label_noise_std,
    )
    return _MinimalSampler(cfg)


def _fake_failure_traj(frame_labels, num_frames=None):
    # `num_frames` independently controls the frames-tensor length so we can construct
    # mismatched cases for negative tests. Defaults to len(frame_labels) for the happy path.
    if num_frames is None:
        num_frames = len(frame_labels)
    frames = np.random.randint(0, 255, size=(num_frames, 32, 32, 3), dtype=np.uint8)
    return {
        "id": "fake_failure_0",
        "task": "pick up the red block",
        "frames": frames,
        "data_source": "robometer_frames_metaworld",
        "quality_label": "failure",
        "frame_labels": frame_labels,
        "partial_success": None,
    }


def _fake_success_traj(num_frames: int = 16):
    frames = np.random.randint(0, 255, size=(num_frames, 32, 32, 3), dtype=np.uint8)
    return {
        "id": "fake_success_0",
        "task": "pick up the red block",
        "frames": frames,
        "data_source": "robometer_frames_metaworld",
        "quality_label": "successful",
        "frame_labels": None,
        "partial_success": None,
    }


def _approx_equal(a, b, tol=1e-6):
    if len(a) != len(b):
        return False
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Pure-function checks (no sampler involved)
# ---------------------------------------------------------------------------

def test_rubric_mapping():
    assert _frame_labels_to_decimals([1, 2, 3, 4]) == [0.0, 0.25, 0.5, 0.75]
    assert _frame_labels_to_decimals([5]) == [1.0]
    try:
        _frame_labels_to_decimals([0])  # invalid
    except ValueError:
        pass
    else:
        raise AssertionError("invalid label 0 should have raised ValueError")


def test_linear_ramp_and_dispatch():
    fail = [0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
    smoothed = _piecewise_linear_ramp(fail)
    assert smoothed[0] == 0.0 and smoothed[3] == 0.25 and smoothed[7] == 0.5 and smoothed[9] == 0.75
    assert all(smoothed[i] <= smoothed[i + 1] for i in range(len(smoothed) - 1)), "expected non-decreasing"
    # dispatch
    assert _apply_failure_smoothing(fail, "none") == fail
    assert _apply_failure_smoothing(fail, "linear") == smoothed


# ---------------------------------------------------------------------------
# Sampler-path checks
# ---------------------------------------------------------------------------

def test_failure_labels_without_smoothing():
    sampler = _make_sampler(failure_label_smoothing="none")
    raw = [1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4]
    traj = _fake_failure_traj(raw)
    t = sampler._get_traj_from_data(traj)
    expected = [0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
    assert _approx_equal(t.target_progress, expected), f"got {t.target_progress}"


def test_failure_labels_with_linear_smoothing():
    sampler = _make_sampler(failure_label_smoothing="linear")
    raw = [1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4]
    traj = _fake_failure_traj(raw)
    t = sampler._get_traj_from_data(traj)
    # Anchors (label-change indices + endpoints): 0, 3, 7, 9, 15.
    assert abs(t.target_progress[0] - 0.0) < 1e-6
    assert abs(t.target_progress[3] - 0.25) < 1e-6
    assert abs(t.target_progress[7] - 0.5) < 1e-6
    assert abs(t.target_progress[9] - 0.75) < 1e-6
    # Between anchor 0 and anchor 3, linear ramp → frame 1 = 0.0833..., frame 2 = 0.1666...
    assert abs(t.target_progress[1] - 0.25 / 3) < 1e-6
    assert abs(t.target_progress[2] - 0.5 / 3) < 1e-6


def test_failure_length_mismatch_raises():
    sampler = _make_sampler()
    # 16 frames but only 3 labels → length mismatch path.
    traj = _fake_failure_traj([1, 2, 3], num_frames=16)
    try:
        sampler._get_traj_from_data(traj)
    except ValueError as e:
        assert "frame_labels length mismatch" in str(e)
    else:
        raise AssertionError("length mismatch should have raised ValueError")


def test_success_noise_is_deterministic_and_bounded():
    sampler_a = _make_sampler(success_label_noise_std=0.05)
    sampler_b = _make_sampler(success_label_noise_std=0.05)
    t_a = sampler_a._get_traj_from_data(_fake_success_traj())
    t_b = sampler_b._get_traj_from_data(_fake_success_traj())
    # Same seed → same perturbed labels
    assert _approx_equal(t_a.target_progress, t_b.target_progress)
    # All values clipped to [0, 1]
    assert all(0.0 <= v <= 1.0 for v in t_a.target_progress)
    # Some noise actually landed — not identical to the clean t/T
    sampler_clean = _make_sampler(success_label_noise_std=0.0)
    t_clean = sampler_clean._get_traj_from_data(_fake_success_traj())
    assert not _approx_equal(t_a.target_progress, t_clean.target_progress, tol=1e-6)


# ---------------------------------------------------------------------------
# Collator `should_compute_progress` check
# ---------------------------------------------------------------------------

def test_should_compute_progress_unmasks_our_failures():
    # Our keyframe dataset: failures should contribute to loss
    assert should_compute_progress(
        quality_label="failure",
        data_gen_strategy="forward_progress",
        data_source="robometer_frames_metaworld",
    ) == 1.0
    # A vanilla "failure" from a generic dataset is still masked out
    assert should_compute_progress(
        quality_label="failure",
        data_gen_strategy="forward_progress",
        data_source="some_other_dataset",
    ) == 0.0


def main() -> int:
    np.random.seed(0)
    test_rubric_mapping()
    test_linear_ramp_and_dispatch()
    test_failure_labels_without_smoothing()
    test_failure_labels_with_linear_smoothing()
    test_failure_length_mismatch_raises()
    test_success_noise_is_deterministic_and_bounded()
    test_should_compute_progress_unmasks_our_failures()
    print("=" * 60)
    print("P2 label-harmonization smoke test: all checks passed.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
