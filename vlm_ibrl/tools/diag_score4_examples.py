"""Inference on hand-picked score=4 (near-success failure) examples.

Goal: isolate whether our curated score-4 data is actually learned by the
fine-tuned models, by running three models on the same 16-frame clips and
comparing success_prob predictions to ground truth (terminal_reward = 4 on
a 1-4 scale → expect high success_prob ≈ 0.7-1.0 if the model learned it).

Three examples:
  1. MetaWorld Assembly score=4 (HELD-OUT from FT training)
  2. MetaWorld BinPicking score=4 (IN-DISTRIBUTION for FT training)
  3. Failsafe Pick score=4 (IN-DISTRIBUTION for FT training)

Three models:
  a. Robometer-FT run1 step 3000 (asymmetric loss + ICL training)
  b. Qwen3.5-FT run4 step 6500 (asymmetric loss + ICL training, latest)
  c. Robometer-4B baseline (no FT)

Each model is run with ICL on AND off. ICL frames come from the
``paired_success_id`` of each scored episode.

Outputs a markdown table to stdout.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl")
sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer")

from env.robometer_utils import get_robometer_4b


DATA_ROOT = Path("/projects/prjs1958/robometer_frame_dataset")

# (label, source-dir, archive, scored-episode-id, paired-success-id, task-desc)
EXAMPLES = [
    (
        "MW-Assembly (held-out)",
        DATA_ROOT / "metaworld",
        "metaworld_assembly_v3",
        "metaworld_assembly_v3_score4_inst0000_corner2_freeze_score_4",
        "metaworld_assembly_v3_success_inst0005_corner2",
        "pick up a nut and place it onto a peg",
    ),
    (
        "MW-BinPicking (in-train)",
        DATA_ROOT / "metaworld",
        "metaworld_bin_picking_v3",
        "metaworld_bin_picking_v3_score4_inst0000_corner2_freeze_score_4",
        "metaworld_bin_picking_v3_success_inst0006_corner2",
        "grasp the puck from one bin and place it into another bin",
    ),
    (
        "FS-Pick (in-train)",
        DATA_ROOT / "failsafe",
        "failsafe_pick",
        "failsafe_pick_pick_s4_grasp_freeze_inst0000_front_score_4",
        "failsafe_pick_pick_success_inst0012_wrist",
        "pick up the red cube and lift it to the goal position",
    ),
]

MODELS = [
    ("Robometer-4B baseline", "robometer/Robometer-4B"),
    ("Robometer-FT run1 s3000", "/scratch-shared/pkarageorgis1/Robometer_FT_consolidated/run1_icl_ours_step3000"),
    ("Qwen3.5-FT run4 s6500",  "/scratch-shared/pkarageorgis1/Qwen35_FT_phase1_consolidated/run4_step6500"),
]


def load_episode_frames(family_dir: Path, archive: str, episode_id: str, success: bool = False) -> List[np.ndarray]:
    """Load the 16 JPEG keyframes for a specific episode from its tar shard."""
    keyframes_dir = family_dir / ("keyframes_success" if success else "keyframes") / archive
    idx_path = keyframes_dir / "shard_index.json"
    with open(idx_path) as f:
        idx = json.load(f)
    if episode_id not in idx:
        raise KeyError(f"episode_id {episode_id!r} not in {idx_path}")
    shard_path = keyframes_dir / idx[episode_id]

    frames_by_idx: Dict[int, np.ndarray] = {}
    prefix = episode_id + "/"
    with tarfile.open(shard_path, "r") as tf:
        for m in tf.getmembers():
            if not m.isfile() or not m.name.startswith(prefix) or not m.name.endswith(".jpg"):
                continue
            fname = m.name[len(prefix):]
            # frame_NN_*.jpg
            try:
                frame_idx = int(fname.split("_")[1])
            except (IndexError, ValueError):
                continue
            f = tf.extractfile(m)
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
            frames_by_idx[frame_idx] = np.asarray(img)
    if not frames_by_idx:
        raise RuntimeError(f"no frames extracted from {shard_path} for prefix {prefix}")
    return [frames_by_idx[k] for k in sorted(frames_by_idx)]


def score_episode(scorer, frames: List[np.ndarray], task_desc: str, icl_frames: List[np.ndarray] | None) -> Dict[str, float]:
    """Run the scorer on a 16-frame clip; return success_prob and progress_reward.

    The IBRL scorer returns aggregated trajectory-level scalars (not per-frame
    arrays): ``success_prob`` is the model's P(trajectory succeeded), and
    ``progress_reward`` is the decoded scalar progress signal.
    """
    out = scorer(
        frames=frames,
        task=task_desc,
        icl_frames=icl_frames,
    )
    return {
        "success_prob":    float(out.get("success_prob", float("nan"))),
        "progress_reward": float(out.get("progress_reward", float("nan"))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="filter to a subset of model labels (case-insensitive substring)")
    args = ap.parse_args()

    chosen_models = MODELS
    if args.models:
        keys = [m.lower() for m in args.models]
        chosen_models = [m for m in MODELS if any(k in m[0].lower() for k in keys)]
    print(f"Models to test: {[m[0] for m in chosen_models]}")
    print()

    # Pre-load all frames (CPU only, fast)
    print("=== loading frames ===")
    examples_data = []
    for label, family_dir, archive, eid, paired, task_desc in EXAMPLES:
        clip = load_episode_frames(family_dir, archive, eid, success=False)
        icl = load_episode_frames(family_dir, archive, paired, success=True)
        print(f"  {label}: clip={len(clip)} frames  icl={len(icl)} frames")
        examples_data.append((label, eid, task_desc, clip, icl))
    print()

    # Run each model on each example × {no-ICL, +ICL}
    results = []  # list of dicts
    for model_label, model_path in chosen_models:
        print(f"=== model: {model_label} ({model_path}) ===")
        try:
            scorer = get_robometer_4b(model_path=model_path)
        except Exception as e:
            print(f"  FAILED to load model: {e}")
            continue

        for ex_label, eid, task_desc, clip, icl in examples_data:
            for icl_mode, icl_frames in [("no-ICL", None), ("+ICL", icl)]:
                try:
                    pred = score_episode(scorer, clip, task_desc, icl_frames)
                    results.append({
                        "model": model_label,
                        "example": ex_label,
                        "icl": icl_mode,
                        **pred,
                    })
                    print(f"  {ex_label:<26}  {icl_mode:<6}  success_prob={pred['success_prob']:.4f}  progress={pred['progress_reward']:.4f}")
                except Exception as e:
                    print(f"  {ex_label:<26}  {icl_mode:<6}  FAILED: {e}")
                    results.append({
                        "model": model_label,
                        "example": ex_label,
                        "icl": icl_mode,
                        "error": str(e),
                    })
        del scorer
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass
        print()

    # ---- summary table ----
    print()
    print("=" * 100)
    print("SUMMARY — success_prob @ last frame (GT terminal_reward = 4; well-learned model should output high)")
    print("=" * 100)
    header = f"{'Model':<26}  {'Example':<26}  {'ICL':<6}  {'success_prob':>12}  {'progress':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['model']:<26}  {r['example']:<26}  {r['icl']:<6}  ERROR: {r['error'][:60]}")
        else:
            print(f"{r['model']:<26}  {r['example']:<26}  {r['icl']:<6}  {r['success_prob']:>12.4f}  {r['progress_reward']:>8.4f}")
    print()


if __name__ == "__main__":
    main()
