"""Offline confusion-matrix evaluation for Robometer-family critics on MetaWorld demos.

Loads a small set of demonstration trajectories with known per-frame
ground-truth labels (success / no-success), runs a Robometer-architecture
checkpoint on prefix-clips of each trajectory, and reports a confusion
matrix at a chosen threshold on ``success_prob``.

Pairs perfectly with the IBRL PoC: it isolates reward-model quality from
policy-learning noise, so you can directly compare two checkpoints on the
same prefix-clip eval. Same `--vlm`-style ``--model-path`` API as
``robometer_smoke.py``.

Example::

    python tools/robometer_offline_cm.py \\
        --model-path /scratch-shared/$USER/Robometer_FT_consolidated/run1_icl_ours_step3000 \\
        --task CoffeePush \\
        --past-len 4 --prefix-stride 5

Run on a GPU node with the demo2reward env active and Robometer/ on
PYTHONPATH (set_env.sh handles both).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import h5py
import numpy as np

from env.robometer_utils import get_robometer_4b


# Task descriptions mirror env/vlm_prompts.py:METAWORLD_TASK_DESCRIPTIONS.
TASK_DESCRIPTIONS = {
    "Assembly":   "pick up a nut and place it onto a peg",
    "BoxClose":   "grasp the cover and close the box with it",
    "CoffeePush": "push a mug under a coffee machine",
    "StickPull":  "grasp a stick and pull a box with the stick",
}


def past_frames_single_video(t: int, num_frames: int) -> List[int]:
    """Mirror env/vlm_envs.py:past_frames_single_video.

    Returns ``num_frames`` indices into ``[0, t)`` evenly spaced (start at 0).
    The IBRL env then appends the current frame itself; we do the same in
    ``build_clip`` below.
    """
    if num_frames == 0:
        return []
    n_avail = min(t + 1, num_frames)
    step = t / num_frames if num_frames > 0 else 1
    return [round(k * step) for k in range(n_avail)]


def build_clip(frames_hwc: np.ndarray, end_t: int, past_len: int) -> List[np.ndarray]:
    """Build the prefix clip the way the MetaWorld IBRL env would: ``past_len``
    indices from the past + the current frame at ``end_t-1``.

    Args:
        frames_hwc: (T, H, W, 3) uint8 array.
        end_t: prefix length (1-indexed); use frames [0..end_t-1].
        past_len: number of past frames to include before the current one.

    Returns:
        List of (H, W, 3) uint8 frames. Length is up to past_len + 1.
    """
    cur_idx = end_t - 1
    idx = past_frames_single_video(cur_idx, past_len)
    clip = [frames_hwc[i] for i in idx]
    clip.append(frames_hwc[cur_idx])
    return clip


def load_demo_rgb_and_rewards(hdf5_path: str, demo_idx: int, camera: str):
    """Read one demo from the MetaWorld release hdf5.

    Returns:
        frames_hwc: (T, H, W, 3) uint8 — last 3 channels of the stacked
            ``{camera}_image`` field (the current-frame slice).
        rewards: (T,) float — ground-truth binary 0/1 per frame.
    """
    with h5py.File(hdf5_path, "r") as f:
        demo = f[f"data/demo_{demo_idx}"]
        img = demo[f"obs/{camera}_image"][:]  # (T, C, H, W), C is typically 6 (frame-stack 2)
        rewards = demo["rewards"][:].astype(np.float32)
    if img.ndim != 4:
        raise ValueError(f"Expected 4D image array, got shape {img.shape}")
    # Take the current-frame slice: last 3 channels. (Channels are
    # frame-stacked: earlier frame in [0:3], current in [3:6].)
    if img.shape[1] >= 3:
        cur = img[:, -3:, :, :]
    else:
        raise ValueError(f"Need >=3 channels, got {img.shape[1]}")
    frames_hwc = np.transpose(cur, (0, 2, 3, 1)).astype(np.uint8)
    return frames_hwc, rewards


def find_success_t(rewards: np.ndarray) -> int | None:
    """First index where reward hits 1.0; None if no success in this demo."""
    hits = np.where(rewards >= 1.0)[0]
    return int(hits[0]) if len(hits) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True,
                        help="HF id or local path to a Robometer-architecture checkpoint")
    parser.add_argument("--task", default="CoffeePush",
                        choices=list(TASK_DESCRIPTIONS.keys()))
    parser.add_argument("--data-dir", default="release/data/metaworld",
                        help="Root containing <Task>_frame_stack_1_224x224_end_on_success/")
    parser.add_argument("--camera", default="corner2",
                        help="Camera name (matches the IBRL env default for MetaWorld)")
    parser.add_argument("--num-demos", type=int, default=3,
                        help="How many demos to consume (release ships 3 per MW task)")
    parser.add_argument("--past-len", type=int, default=4,
                        help="Past frames included before current. Ignored if --full-clip.")
    parser.add_argument("--full-clip", action="store_true",
                        help="Pass the entire prefix [0:end_t] to the scorer instead of "
                             "subsampling to past_len+1 frames. The scorer's internal "
                             "linspace_subsample then reduces to its configured max_frames "
                             "(16 for Robometer). This matches the training-time input "
                             "format. Strongly recommended for Robometer-family critics.")
    parser.add_argument("--prefix-stride", type=int, default=5,
                        help="Generate a prefix clip every N frames within each demo")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold on success_prob for CM tabulation")
    parser.add_argument("--output", default=None,
                        help="Optional JSON dump of per-clip results")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset_path = os.path.join(
        args.data_dir, f"{args.task}_frame_stack_1_224x224_end_on_success", "dataset.hdf5"
    )
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Demo dataset not found at {dataset_path}. The release bundle ships "
            f"three demos per MetaWorld task; check that you ran the unzip step "
            f"and that --task / --data-dir resolve correctly."
        )

    task_desc = TASK_DESCRIPTIONS[args.task]

    print(f"=" * 64)
    print(f"Model:     {args.model_path}")
    print(f"Task:      {args.task}  —  {task_desc!r}")
    print(f"Camera:    {args.camera}")
    print(f"Demos:     {args.num_demos} from {dataset_path}")
    print(f"past_len:  {args.past_len} (+ current frame)")
    print(f"stride:    every {args.prefix_stride} frames within each demo")
    print(f"threshold: success_prob > {args.threshold}")
    print(f"=" * 64)

    print(f"\nLoading scorer on {args.device} ...")
    scorer = get_robometer_4b(model_path=args.model_path, device=args.device)
    print(f"OK. model_type={scorer._model_type} discrete={scorer._is_discrete} "
          f"num_bins={scorer._num_bins} max_frames={scorer.max_frames}")
    print()

    results = []
    tp = fp = tn = fn = 0

    for demo_idx in range(args.num_demos):
        frames_hwc, rewards = load_demo_rgb_and_rewards(dataset_path, demo_idx, args.camera)
        T = len(frames_hwc)
        success_t = find_success_t(rewards)
        print(f"demo {demo_idx}: T={T:>4}  success_t={success_t}  "
              f"({'SUCCESS' if success_t is not None else 'NO-SUCCESS'})")

        prefix_ends = list(range(args.prefix_stride, T + 1, args.prefix_stride))
        if T not in prefix_ends:
            prefix_ends.append(T)

        for end_t in prefix_ends:
            # Ground truth = the env's per-frame reward at end_t-1. This
            # correctly handles MetaWorld tasks where the success criterion
            # is *non-sticky* (e.g., CoffeePush, StickPull): reward[t]
            # drops back to 0 if the trajectory leaves the success region.
            # Earlier "once-success-always-success" labeling penalized the
            # model for correctly tracking the trajectory leaving success.
            gt = int(rewards[end_t - 1] >= 1.0)
            if args.full_clip:
                clip = [frames_hwc[i] for i in range(end_t)]
            else:
                clip = build_clip(frames_hwc, end_t, args.past_len)
            out = scorer(clip, task=task_desc, episode_id=demo_idx)
            pred = int(out["success_prob"] > args.threshold)

            results.append(dict(
                demo=demo_idx,
                end_t=end_t,
                success_t=success_t,
                gt=gt,
                pred=pred,
                progress_reward=out["progress_reward"],
                success_prob=out["success_prob"],
            ))

            if gt == 1:
                tp += pred == 1
                fn += pred == 0
            else:
                fp += pred == 1
                tn += pred == 0

    n_pos = tp + fn
    n_neg = fp + tn
    tpr = tp / n_pos if n_pos else float("nan")
    fnr = fn / n_pos if n_pos else float("nan")
    tnr = tn / n_neg if n_neg else float("nan")
    fpr = fp / n_neg if n_neg else float("nan")

    print(f"\n{'=' * 64}")
    print(f"Confusion matrix  (threshold = success_prob > {args.threshold})")
    print(f"  Clips:     {len(results)}   positives={n_pos}   negatives={n_neg}")
    print(f"                Pred=1     Pred=0")
    print(f"      GT=1    TP={tp:>4}   FN={fn:>4}     TPR={tpr:.3f}  FNR={fnr:.3f}")
    print(f"      GT=0    FP={fp:>4}   TN={tn:>4}     FPR={fpr:.3f}  TNR={tnr:.3f}")

    # Continuous distribution per class — more informative than the binary CM
    # for models that emit calibrated probabilities.
    pos = [r["success_prob"] for r in results if r["gt"] == 1]
    neg = [r["success_prob"] for r in results if r["gt"] == 0]
    print()
    if pos:
        print(f"success_prob | GT=1  (n={len(pos)}):  "
              f"mean={np.mean(pos):.4f}  min={min(pos):.4f}  max={max(pos):.4f}  "
              f"median={float(np.median(pos)):.4f}")
    if neg:
        print(f"success_prob | GT=0  (n={len(neg)}):  "
              f"mean={np.mean(neg):.4f}  min={min(neg):.4f}  max={max(neg):.4f}  "
              f"median={float(np.median(neg)):.4f}")

    pos_pg = [r["progress_reward"] for r in results if r["gt"] == 1]
    neg_pg = [r["progress_reward"] for r in results if r["gt"] == 0]
    if pos_pg:
        print(f"progress      | GT=1  (n={len(pos_pg)}): "
              f"mean={np.mean(pos_pg):.4f}  min={min(pos_pg):.4f}  max={max(pos_pg):.4f}")
    if neg_pg:
        print(f"progress      | GT=0  (n={len(neg_pg)}): "
              f"mean={np.mean(neg_pg):.4f}  min={min(neg_pg):.4f}  max={max(neg_pg):.4f}")

    print(f"{'=' * 64}")

    if args.output:
        with open(args.output, "w") as fp_out:
            json.dump(dict(args=vars(args),
                           confusion_matrix=dict(tp=tp, fp=fp, tn=tn, fn=fn,
                                                 tpr=tpr, fpr=fpr, tnr=tnr, fnr=fnr),
                           results=results),
                      fp_out, indent=2)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
