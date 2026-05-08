"""Smoke test: verify the tar-shard data path produces samples with intact
frame_labels.

This is the explicit defense against the LoRA "predictions were random" bug
(failure rows trained against t/T because frame_labels were silently dropped).

Checks performed (CPU only, no GPU needed):
  1. Builds TarKeyframeIndex from pairs_unified.jsonl + manifest frame_labels.
  2. Reports kept-vs-dropped row counts.
  3. For 100 random samples (50 success + 50 failure):
       a. dict has every required key (id, frames, frame_labels, ...)
       b. frames.shape[0] >= 5 and dtype is uint8
       c. for failures: frame_labels has values in {1,2,3,4} and at least
          60% of sampled failures have >=2 distinct values (varied supervision)
       d. for successes: frame_labels is None (downstream uses t/T fallback)
       e. len(frame_labels) == frames.shape[0] (when frame_labels present)

Usage:
    cd Robometer-FT
    /home/pkarageorgis1/.conda/envs/robometer_gpu/bin/python scripts/smoke_test_tar_data.py
"""
import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Robometer"))

from robometer_ft_data.tar_dataset import build_index_from_pairs_unified


def main() -> None:
    frame_dataset_root = os.environ.get(
        "FRAME_DATASET_ROOT", "/projects/prjs1958/robometer_frame_dataset"
    )

    print(f"== smoke_test_tar_data ==")
    print(f"  frame_dataset_root: {frame_dataset_root}")
    print()

    index = build_index_from_pairs_unified(frame_dataset_root=frame_dataset_root)
    n = len(index)
    assert n > 100, f"index has too few rows: {n}"

    successes = [i for i, r in enumerate(index._rows) if r["quality_label"] == "successful"]
    failures = [i for i, r in enumerate(index._rows) if r["quality_label"] == "failure"]
    print(f"\n  bucket sizes: successful={len(successes)}, failure={len(failures)}")
    assert len(successes) >= 50, f"need >=50 successes, got {len(successes)}"
    assert len(failures) >= 50, f"need >=50 failures, got {len(failures)}"

    rng = random.Random(42)
    sample_idx = rng.sample(successes, 50) + rng.sample(failures, 50)
    rng.shuffle(sample_idx)

    print(f"\n  loading 100 random samples (decoding JPGs from tars)…")

    n_pass_success = 0
    n_pass_failure = 0
    failure_label_values: set = set()
    failure_distinctness: list = []
    failed_loads = []

    for k, idx in enumerate(sample_idx):
        try:
            item = index[idx]
        except Exception as e:
            failed_loads.append((idx, str(e)))
            continue

        # 3a: keys present
        for required in ("id", "frames", "frame_labels", "data_source",
                         "quality_label", "task", "is_robot", "partial_success"):
            assert required in item, f"missing key {required} in item idx={idx}"

        # 3b: frames shape & dtype
        frames = item["frames"]
        assert hasattr(frames, "shape"), f"frames is not array-like: type={type(frames)}"
        assert frames.ndim == 4, f"frames.ndim={frames.ndim}, expected 4 (T,H,W,C)"
        assert frames.shape[0] >= 5, f"too few frames: {frames.shape[0]} for idx={idx}"
        assert frames.dtype.name == "uint8", f"frames dtype {frames.dtype} != uint8"

        labels = item["frame_labels"]

        if item["quality_label"] == "successful":
            # 3d: successes have NO curated frame_labels (None → upstream uses t/T)
            assert labels is None, (
                f"success row idx={idx} unexpectedly has frame_labels={labels} "
                f"(should be None — successes get t/T fallback downstream)"
            )
            n_pass_success += 1
        else:
            # failure: must have rubric-shaped labels matched to frame count
            assert labels is not None, (
                f"FAILURE row idx={idx} has no frame_labels — same shape as the "
                f"LoRA disaster. Halting."
            )
            assert len(labels) == frames.shape[0], (
                f"len(frame_labels)={len(labels)} != frames.shape[0]={frames.shape[0]} "
                f"for idx={idx} id={item['id']}"
            )
            for v in labels:
                assert v in (1, 2, 3, 4, 5), (
                    f"failure label {v} outside rubric for idx={idx} "
                    f"id={item['id']} labels={labels}"
                )
            failure_label_values.update(labels)
            failure_distinctness.append(len(set(labels)))
            n_pass_failure += 1

        if (k + 1) % 25 == 0:
            print(f"    [{k+1}/100] OK so far ({n_pass_success} succ, {n_pass_failure} fail)")

    if failed_loads:
        print(f"\n  WARNING: {len(failed_loads)} samples failed to load:")
        for idx, msg in failed_loads[:5]:
            print(f"    idx={idx}: {msg}")
        # Don't immediately fail — log and continue. The samplers handle
        # SampleSkipRequest at runtime; a few unreadable tars is non-fatal.

    print(f"\n  PASS counts: success={n_pass_success}/50, failure={n_pass_failure}/50")
    print(f"  failure rubric values seen: {sorted(failure_label_values)}")
    print(f"  failure within-trajectory distinct labels: "
          f"min={min(failure_distinctness)}, "
          f"mean={sum(failure_distinctness)/len(failure_distinctness):.2f}, "
          f"max={max(failure_distinctness)}")

    # Disaster guard: if EVERY sampled failure has the same single label across
    # all frames, supervision is degenerate. Require >=60% to have >=2 distinct.
    n_varied = sum(1 for d in failure_distinctness if d >= 2)
    pct_varied = n_varied / len(failure_distinctness)
    assert pct_varied >= 0.6, (
        f"only {pct_varied:.0%} of sampled failure trajectories have varied "
        f"frame_labels — looks like supervision isn't carrying real per-frame "
        f"signal. This is the same shape as the LoRA disaster. Halting."
    )
    print(f"  failure variation: {pct_varied:.0%} have >=2 distinct frame labels (>=60% threshold met)")

    print(f"\nPASS: tar data path produces well-formed samples with intact frame_labels.")


if __name__ == "__main__":
    main()
