"""Unified eval: score one checkpoint on one dataset (OOD or in-dist), ICL on/off,
dump per-trajectory predictions + ground truth for BOTH heads.

Output (one JSONL row per trajectory):
  episode_id, source, gt_success (0/1), gt_progress[T],
  succ_last (success head, last frame), succ_per_frame[T] (success head),
  prog_last (progress head, last frame), prog_per_frame[T] (progress head)

Datasets:
  OOD  = /projects/prjs1958/robometer_frame_eval_dataset  (packed keyframe tars;
         GT label from subdir: keyframes=failure, keyframes_success=successful;
         GT progress from meta.json frame_labels, or monotonic ramp for successes)
  IND  = /projects/prjs1958/robometer_frames_splits_full eval split joined with
         /projects/prjs1958/robometer_frame_dataset/<source> keyframes + scores

Metrics are computed downstream by compute_metrics.py from these dumps so every
model/cell uses identical definitions.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl")
sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer")
from env.robometer_utils import get_robometer_4b

OOD_ROOT = Path("/projects/prjs1958/robometer_frame_eval_dataset")
IND_SPLIT = Path("/projects/prjs1958/robometer_frames_splits_full")
IND_DATA = Path("/projects/prjs1958/robometer_frame_dataset")
PACKED_SUBDIRS = {"keyframes": "failure", "keyframes_success": "successful",
                  "keyframes_orphan_success": "successful"}


def _resolve_shard(shard_value, episode_id: str) -> Tuple[str, str]:
    """shard_index values are either a tar-filename string (metaworld/failsafe/
    robometer) or a dict {'shard': ..., 'inner_dir': ...} (droid). Return
    (tar_filename, in-tar-prefix)."""
    if isinstance(shard_value, dict):
        inner = shard_value.get("inner_dir") or episode_id
        return shard_value["shard"], inner.rstrip("/") + "/"
    return shard_value, episode_id + "/"


def _load_tar_episode(tar_path: Path, prefix: str) -> Tuple[List[np.ndarray], dict]:
    frames_by_idx: Dict[int, np.ndarray] = {}
    meta = {}
    with tarfile.open(tar_path, "r") as tf:
        for m in tf.getmembers():
            if not m.isfile() or not m.name.startswith(prefix):
                continue
            if m.name.endswith(".jpg"):
                try:
                    fi = int(m.name[len(prefix):].split("_")[1])
                except (IndexError, ValueError):
                    continue
                frames_by_idx[fi] = np.asarray(Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB"))
            elif m.name.endswith("meta.json"):
                meta = json.loads(tf.extractfile(m).read())
    frames = [frames_by_idx[k] for k in sorted(frames_by_idx)]
    return frames, meta


def _shard_index(kf_dir: Path) -> Dict[str, str]:
    idx_path = kf_dir / "shard_index.json"
    return json.loads(idx_path.read_text()) if idx_path.exists() else {}


def gt_progress_from(meta_frame_labels, n_frames: int, gt_success: int) -> List[float]:
    """Per-frame GT progress. Use frame_labels if present; else monotonic ramp
    for successes (0->1), zeros for failures lacking labels."""
    if meta_frame_labels:
        fl = [float(x) for x in meta_frame_labels]
        if len(fl) == n_frames:
            return fl
        # resample to n_frames
        idx = np.linspace(0, len(fl) - 1, n_frames).round().astype(int)
        return [fl[i] for i in idx]
    if gt_success == 1:
        return list(np.linspace(0.0, 1.0, n_frames))
    return [0.0] * n_frames


def iter_ood(max_per_source: int | None = None) -> Iterator[dict]:
    """Yield {episode_id, source, frames, gt_success, gt_progress} for the OOD set.

    OOD keyframes have NO shard_index.json — each shard-*.tar holds episodes as
    top-level dirs (episode_id/frame_*.jpg + episode_id/meta.json). GT label from
    the subdir (keyframes=failure, keyframes_success*=successful)."""
    for subdir, lbl in PACKED_SUBDIRS.items():
        root = OOD_ROOT / subdir
        if not root.exists():
            continue
        for fam_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            count = 0
            for shard in sorted(fam_dir.glob("shard-*.tar")):
                if max_per_source and count >= max_per_source:
                    break
                with tarfile.open(shard, "r") as tf:
                    by_ep: Dict[str, list] = {}
                    metas: Dict[str, dict] = {}
                    for m in tf.getmembers():
                        if not m.isfile():
                            continue
                        ep = m.name.split("/")[0]
                        if m.name.endswith(".jpg"):
                            by_ep.setdefault(ep, []).append(m)
                        elif m.name.endswith("meta.json"):
                            metas[ep] = json.loads(tf.extractfile(m).read())
                    for ep in sorted(by_ep):
                        if max_per_source and count >= max_per_source:
                            break
                        members = sorted(by_ep[ep], key=lambda m: m.name)
                        frames = [np.asarray(Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB"))
                                  for m in members]
                        if not frames:
                            continue
                        meta = metas.get(ep, {})
                        gt = 1 if (meta.get("label", lbl) == "successful") else 0
                        yield {"episode_id": ep, "source": fam_dir.name, "frames": frames,
                               "gt_success": gt, "task": meta.get("task", "complete the task"),
                               "gt_progress": gt_progress_from(meta.get("frame_labels"), len(frames), gt)}
                        count += 1


def _ind_label_maps() -> Dict[str, Dict[str, int]]:
    """For each source, map episode_id -> gt_success using successes/failures manifests."""
    out: Dict[str, Dict[str, int]] = {}
    for src in ["droid", "metaworld", "failsafe", "robometer"]:
        man = IND_DATA / src / "manifests"
        if not man.exists():
            continue
        m: Dict[str, int] = {}
        for f in man.glob("*_successes.jsonl"):
            for line in open(f):
                m[json.loads(line)["episode_id"]] = 1
        for f in man.glob("*_failures.jsonl"):
            for line in open(f):
                m[json.loads(line)["episode_id"]] = 0
        out[src] = m
    return out


def _ind_frame_labels() -> Dict[str, List[float]]:
    """episode_id -> frame_labels from the scored jsonls (all sources)."""
    fl: Dict[str, List[float]] = {}
    for src in ["droid", "metaworld", "failsafe", "robometer"]:
        sc = IND_DATA / src / "scores"
        if not sc.exists():
            continue
        for f in sc.glob("*.jsonl"):
            for line in open(f):
                r = json.loads(line)
                if "frame_labels" in r:
                    fl[r["episode_id"]] = r["frame_labels"]
    return fl


def iter_indist(max_per_source: int | None) -> Iterator[dict]:
    """GT label derived from the keyframe subdir the episode lives in
    (keyframes=failure, keyframes_success=successful) — robust across sources
    whose manifest ids differ from the eval-split / keyframe ids (e.g. failsafe)."""
    frame_labels = _ind_frame_labels()
    # failsafe dropped: its eval-split episode ids don't match the keyframe
    # tars in robometer_frame_dataset (dataset-version mismatch).
    for src in ["droid", "metaworld", "robometer"]:
        split = IND_SPLIT / f"eval_{src}.jsonl"
        if not split.exists():
            continue
        eids = [json.loads(l)["episode_id"] for l in open(split)]
        if max_per_source:
            eids = eids[:max_per_source]
        # arch_idx: eid -> (archive_dir, shard_value, gt_success)
        arch_idx: Dict[str, Tuple[Path, object, int]] = {}
        for sub, gt in [("keyframes", 0), ("keyframes_success", 1)]:
            base = IND_DATA / src / sub
            if not base.exists():
                continue
            for arch in base.iterdir():
                if not arch.is_dir():
                    continue
                for eid, shard_value in _shard_index(arch).items():
                    arch_idx.setdefault(eid, (arch, shard_value, gt))
        for eid in eids:
            if eid not in arch_idx:
                continue
            arch, shard_value, gt = arch_idx[eid]
            shard, prefix = _resolve_shard(shard_value, eid)
            frames, meta = _load_tar_episode(arch / shard, prefix)
            if not frames:
                continue
            yield {"episode_id": eid, "source": src, "frames": frames, "gt_success": gt,
                   "task": meta.get("task", "complete the task"),
                   "gt_progress": gt_progress_from(frame_labels.get(eid), len(frames), gt)}


def load_icl_demo() -> List[np.ndarray]:
    """A curated CoffeePush success demo as ICL context (16 frames)."""
    kf = IND_DATA / "metaworld" / "keyframes_success" / "metaworld_coffee_push_v3"
    idx = _shard_index(kf)
    eid, shard = next(iter(idx.items()))
    frames, _ = _load_tar_episode(kf / shard, eid)
    i = np.linspace(0, len(frames) - 1, 16).round().astype(int)
    return [frames[k] for k in i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset", choices=["ood", "indist"], required=True)
    ap.add_argument("--icl", action="store_true")
    ap.add_argument("--max-per-source", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"loading scorer: {args.model_path}")
    scorer = get_robometer_4b(model_path=args.model_path)
    icl_frames = load_icl_demo() if args.icl else None
    print(f"ICL: {'on (16-frame curated CoffeePush success demo)' if args.icl else 'off'}")

    it = iter_ood(args.max_per_source) if args.dataset == "ood" else iter_indist(args.max_per_source)
    n = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fout:
        for rec in it:
            out = scorer(rec["frames"], task=rec.get("task", "complete the task"),
                         icl_frames=icl_frames, detailed=True)
            row = {
                "episode_id": rec["episode_id"], "source": rec["source"],
                "gt_success": rec["gt_success"], "gt_progress": rec["gt_progress"],
                "succ_last": out["success_prob"],
                "succ_per_frame": out.get("success_probs_per_frame", [out["success_prob"]]),
                "prog_last": out["progress_reward"],
                "prog_per_frame": out.get("progress_per_frame", [out["progress_reward"]]),
            }
            fout.write(json.dumps(row) + "\n")
            n += 1
            if n % 50 == 0:
                print(f"  scored {n}  (last: {rec['source']} gt={rec['gt_success']} succ={out['success_prob']:.3f})", flush=True)
    print(f"done — {n} trajectories -> {args.out}")


if __name__ == "__main__":
    main()
