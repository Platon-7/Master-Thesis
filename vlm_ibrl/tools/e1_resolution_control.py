"""E1 — resolution control.

Tests whether the on-policy AUC collapse is caused by image RESOLUTION shift
(IBRL feeds 224x224; training used native keyframe resolution) vs genuine
trajectory-content shift.

Method: score the SAME curated keyframe trajectories two ways with the same
checkpoint — (a) native resolution, (b) resized to 224x224 — frame count held
at 16 for both. Compute success-vs-failure AUC per condition.

  - If AUC drops sharply from native -> 224  => resolution sensitivity (fixable
    by rendering IBRL frames at training res, or resizing in the scorer).
  - If AUC ~ unchanged  => resolution innocent; the IBRL collapse is genuine
    trajectory-content / live-rollout-domain shift.

Tasks: CoffeePush (held-out, the IBRL task) + an in-training control task
(button_press) where native AUC should be high — so we can see if resolution
sensitivity is general or task-specific.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl")
sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer")

from env.robometer_utils import get_robometer_4b

DATA_ROOT = Path("/projects/prjs1958/robometer_frame_dataset/metaworld")

# (task_archive, task_description, in_training?)
TASKS = [
    ("metaworld_coffee_push_v3", "push a mug under a coffee machine", False),
    ("metaworld_button_press_v3", "press a button", True),
]


def load_trajectories(archive: str, success: bool, max_n: int) -> List[List[np.ndarray]]:
    """Load up to max_n trajectories (each a list of frames) from a keyframe tar."""
    kf_dir = DATA_ROOT / ("keyframes_success" if success else "keyframes") / archive
    idx = json.loads((kf_dir / "shard_index.json").read_text())
    # group episode_ids by shard for efficient reads
    by_shard: Dict[str, List[str]] = {}
    for eid, shard in idx.items():
        by_shard.setdefault(shard, []).append(eid)

    trajs = []
    for shard, eids in by_shard.items():
        if len(trajs) >= max_n:
            break
        with tarfile.open(kf_dir / shard, "r") as tf:
            members_by_ep: Dict[str, list] = {}
            for m in tf.getmembers():
                if not m.isfile() or not m.name.endswith(".jpg"):
                    continue
                ep = m.name.split("/")[0]
                members_by_ep.setdefault(ep, []).append(m)
            for eid in eids:
                if len(trajs) >= max_n:
                    break
                if eid not in members_by_ep:
                    continue
                members = sorted(members_by_ep[eid], key=lambda m: m.name)
                frames = []
                for m in members:
                    f = tf.extractfile(m)
                    frames.append(np.asarray(Image.open(io.BytesIO(f.read())).convert("RGB")))
                if frames:
                    trajs.append(frames)
    return trajs


def resize_224(frames: List[np.ndarray]) -> List[np.ndarray]:
    return [np.array(Image.fromarray(f).resize((224, 224), Image.BILINEAR), dtype=np.uint8)
            for f in frames]


def auc(pos: List[float], neg: List[float]) -> float:
    if not pos or not neg:
        return float("nan")
    a = np.concatenate([np.asarray(pos), np.asarray(neg)])
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    ranks = np.empty(len(a))
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        ranks[i:j + 1] = (i + j + 2) / 2
        i = j + 1
    inv = np.empty_like(order, dtype=float)
    inv[order] = ranks
    return float((inv[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--n-per-class", type=int, default=30)
    ap.add_argument("--icl", action="store_true", help="use a held-out success demo as ICL context")
    args = ap.parse_args()

    print(f"loading scorer: {args.model_path}")
    scorer = get_robometer_4b(model_path=args.model_path)
    print(f"max_frames={scorer.max_frames}\n")

    for archive, task_desc, in_train in TASKS:
        tag = "in-train" if in_train else "HELD-OUT"
        print(f"================ {archive}  ({tag}) ================")
        succ = load_trajectories(archive, success=True, max_n=args.n_per_class)
        fail = load_trajectories(archive, success=False, max_n=args.n_per_class)
        print(f"  loaded {len(succ)} success + {len(fail)} failure trajectories")

        icl_frames = None
        if args.icl:
            # use the FIRST success demo as ICL; exclude it from the scored set
            if len(succ) > 1:
                icl_frames = succ[0]
                succ = succ[1:]
                print(f"  ICL: using 1 held-out success demo ({len(icl_frames)} frames) as context")

        for cond, transform in [("native", lambda f: f), ("224x224", resize_224)]:
            sp_succ, sp_fail = [], []
            for fr in succ:
                out = scorer(transform(fr), task=task_desc,
                             icl_frames=(transform(icl_frames) if icl_frames is not None else None))
                sp_succ.append(float(out["success_prob"]))
            for fr in fail:
                out = scorer(transform(fr), task=task_desc,
                             icl_frames=(transform(icl_frames) if icl_frames is not None else None))
                sp_fail.append(float(out["success_prob"]))
            a = auc(sp_succ, sp_fail)
            msp_s = float(np.nanmean(sp_succ)) if sp_succ else float("nan")
            msp_f = float(np.nanmean(sp_fail)) if sp_fail else float("nan")
            print(f"    {cond:<8}  AUC={a:.3f}   sp|succ={msp_s:.4f}  sp|fail={msp_f:.4f}  Δ={msp_s-msp_f:+.4f}")
        print()


if __name__ == "__main__":
    main()
