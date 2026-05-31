"""Trajectory-level inference over the canonical ``robometer/rbm-1m-ood`` test set.

Runs a Robometer-architecture scorer over every episode of the canonical
OOD eval set (782 episodes) and dumps one row per trajectory to parquet.
Output is consumed by ``tools/build_results_table.py`` for the comparison
table.

Two source modes:

* ``--source raw_hf`` (default) — loads from the canonical LeRobot v3 mp4
  at ``raw_hf/videos/.../file-000.mp4`` (32 frames/episode @ 10fps). The
  scorer's internal ``linspace_subsample`` reduces to its config's
  ``max_frames`` (8 for Robometer-4B, 16 for FT).

* ``--source packed`` — loads from the 16-frame JPEG keyframes already
  staged at ``$eval_root/keyframes{,_success,_orphan_success}/<family>/shard-*.tar``
  (produced by your ``Real-World-Failures/Robometer/rbm-1m-ood-pipeline/``
  scripts; each episode = 16 JPEGs + meta.json, where the 16 frames were
  picked by ``linspace(0, 31, 16)`` on the raw mp4). The scorer then does
  ``linspace(16, max_frames)`` internally.

  For the FT column (max_frames=16) ``packed`` and ``raw_hf`` are
  bit-equivalent. For the baseline column (max_frames=8) the picked raw
  indices differ slightly: raw_hf → [0,4,9,13,18,22,27,31];
  packed → [0,4,8,12,19,23,27,31]. ``packed`` matches the two-stage
  ``linspace(32,16)→linspace(16,8)`` pipeline that mirrors the
  preprocessing convention shared between the FT training pipeline and
  Robometer's own released pipeline.

Label policy (per user direction, 2026-05-19): ``suboptimal`` is treated
as ``failure``. Binary label is ``1`` iff ``quality_label == 'successful'``.

Example::

    python tools/inference_eval_set.py \\
        --model-path /scratch-shared/$USER/Robometer_FT_consolidated/run1_step3000 \\
        --out /scratch-shared/$USER/vlm_ibrl_results_table/run1_step3000.parquet \\
        --source packed
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
from typing import Dict, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from env.robometer_utils import get_robometer_4b


DEFAULT_RAW_HF_ROOT = "/projects/prjs1958/robometer_frame_eval_dataset/raw_hf"
DEFAULT_PACKED_ROOT = "/projects/prjs1958/robometer_frame_eval_dataset"
PACKED_SUBDIRS = {
    "keyframes": "failure",
    "keyframes_success": "successful",
    "keyframes_orphan_success": "successful",
}


def load_episode_metadata(eval_root: str) -> List[Dict]:
    """Return one record per episode (sorted by episode_index).

    Joins ``meta/episodes/.../file-000.parquet`` (episode-level: tasks,
    length, mp4 frame range) with the frame-level data parquet (carries
    id / quality_label / data_source as episode-constant fields).
    """
    ep_tbl = pq.read_table(
        os.path.join(eval_root, "meta/episodes/chunk-000/file-000.parquet"),
        columns=["episode_index", "tasks", "length",
                 "dataset_from_index", "dataset_to_index"],
    ).to_pylist()

    fr_tbl = pq.read_table(
        os.path.join(eval_root, "data/chunk-000/file-000.parquet"),
        columns=["id", "quality_label", "data_source",
                 "episode_index", "frame_index"],
    )
    by_ep: Dict[int, Dict] = {}
    for i in range(fr_tbl.num_rows):
        if fr_tbl["frame_index"][i].as_py() != 0:
            continue
        ei = fr_tbl["episode_index"][i].as_py()
        by_ep[ei] = dict(
            id=fr_tbl["id"][i].as_py(),
            quality_label=fr_tbl["quality_label"][i].as_py(),
            data_source=fr_tbl["data_source"][i].as_py(),
        )

    out: List[Dict] = []
    for ep in sorted(ep_tbl, key=lambda r: r["episode_index"]):
        ei = ep["episode_index"]
        if ei not in by_ep:
            raise KeyError(
                f"episode_index {ei} present in meta/episodes/ but missing from data/ "
                f"frame parquet — eval set is corrupt."
            )
        fr = by_ep[ei]
        ql = fr["quality_label"]
        # Binary label: suboptimal collapses to failure (per user 2026-05-19).
        label = 1 if ql == "successful" else 0
        out.append(dict(
            id=fr["id"],
            episode_index=ei,
            data_source=fr["data_source"],
            task=ep["tasks"][0] if ep["tasks"] else "",
            quality_label_raw=ql,
            label=label,
            mp4_frame_start=int(ep["dataset_from_index"]),
            mp4_frame_end=int(ep["dataset_to_index"]),
            n_frames=int(ep["length"]),
        ))
    return out


def load_packed_episodes(packed_root: str) -> List[Dict]:
    """Scan the packed shard layout and return one record per episode.

    Reads JPEGs + meta.json out of each ``shard-*.tar`` and returns dicts
    with keys: id, data_source, task, quality_label_raw, label, frames
    (np.ndarray of shape (16, H, W, 3) uint8).
    """
    out: List[Dict] = []
    for subdir, expected_label in PACKED_SUBDIRS.items():
        root = os.path.join(packed_root, subdir)
        if not os.path.isdir(root):
            continue
        families = sorted(f for f in os.listdir(root)
                          if os.path.isdir(os.path.join(root, f)))
        for fam in families:
            fam_dir = os.path.join(root, fam)
            shards = sorted(s for s in os.listdir(fam_dir) if s.endswith(".tar"))
            for shard_name in shards:
                shard_path = os.path.join(fam_dir, shard_name)
                with tarfile.open(shard_path, "r") as tf:
                    # Group entries by episode_id (first path component)
                    entries: Dict[str, List[tarfile.TarInfo]] = {}
                    for member in tf.getmembers():
                        if not member.isfile():
                            continue
                        ep_id = member.name.split("/", 1)[0]
                        entries.setdefault(ep_id, []).append(member)
                    for ep_id, members in entries.items():
                        # Pull meta.json
                        meta = None
                        jpgs: Dict[str, bytes] = {}
                        for m in members:
                            name = m.name.split("/", 1)[1]
                            f = tf.extractfile(m)
                            data = f.read() if f is not None else b""
                            if name == "meta.json":
                                meta = json.loads(data.decode("utf-8"))
                            elif name.endswith(".jpg"):
                                jpgs[name] = data
                        if meta is None or not jpgs:
                            continue

                        # Decode JPEGs in filename order — names sort
                        # lexically by ``frame_NN_...`` so this matches the
                        # writer's emission order.
                        frame_names = sorted(jpgs.keys())
                        frames = np.stack([
                            np.asarray(Image.open(io.BytesIO(jpgs[n])).convert("RGB"))
                            for n in frame_names
                        ], axis=0)

                        ql = expected_label
                        # Trust shard subdir for label, but fall back to
                        # meta in case the layout shifts.
                        if "label" in meta:
                            ql = meta["label"]
                        label = 1 if ql == "successful" else 0
                        out.append(dict(
                            id=meta.get("episode_id", ep_id),
                            data_source=meta.get("family", fam),
                            task=meta.get("task", ""),
                            quality_label_raw=ql,
                            label=label,
                            frames=frames,
                        ))
    # Stable order by data_source then id for deterministic logging
    out.sort(key=lambda r: (r["data_source"], r["id"]))
    return out


def decode_full_mp4(mp4_path: str, expected_frames: int) -> np.ndarray:
    """Decode the entire mp4 into ``(N, H, W, 3)`` uint8 in display order.

    For a 50 MB / 25024-frame OOD mp4 this is ~5.4 GB of CPU RAM and
    decodes in ~30 s on a typical compute node. Doing this once at startup
    is far simpler — and safer — than seeking per-episode through h264
    with potential B-frame reordering.
    """
    import av
    container = av.open(mp4_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    frames: List[np.ndarray] = []
    pts_keys: List[int] = []
    for frame in container.decode(stream):
        # frame.pts is monotonic w.r.t. display order; collect to be safe
        # then re-sort if PTS-out-of-order packets appear (B-frames).
        pts_keys.append(frame.pts if frame.pts is not None else len(frames))
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()

    if len(frames) != expected_frames:
        raise RuntimeError(
            f"mp4 decode produced {len(frames)} frames, expected {expected_frames}. "
            f"Cannot guarantee episode → frame alignment.")

    # If PTS were strictly monotonic, decode order == display order. If any
    # packet was out of order (B-frames), sort by PTS to restore display order.
    if pts_keys != sorted(pts_keys):
        order = sorted(range(len(frames)), key=lambda k: pts_keys[k])
        frames = [frames[i] for i in order]

    arr = np.stack(frames, axis=0)
    return arr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True,
                        help="HF id or local path to a Robometer-architecture checkpoint")
    parser.add_argument("--out", required=True,
                        help="Output parquet path (per-episode trajectory scores)")
    parser.add_argument("--source", choices=["raw_hf", "packed"], default="raw_hf",
                        help="Where to load frames from. 'raw_hf' = canonical "
                             "32-frame mp4. 'packed' = your 16-frame keyframe "
                             "shards (matches FT training pipeline; gives "
                             "linspace(32,16)->linspace(16,8) for baseline).")
    parser.add_argument("--eval-root", default=None,
                        help="Override eval root. Defaults: raw_hf -> "
                             f"{DEFAULT_RAW_HF_ROOT}; packed -> {DEFAULT_PACKED_ROOT}.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after N episodes (smoke-test). 0 = all 782.")
    args = parser.parse_args()

    eval_root = args.eval_root or (DEFAULT_RAW_HF_ROOT if args.source == "raw_hf"
                                   else DEFAULT_PACKED_ROOT)

    print("=" * 72)
    print("Inference on canonical rbm-1m-ood (one trajectory-level score per episode)")
    print(f"  model_path : {args.model_path}")
    print(f"  out        : {args.out}")
    print(f"  source     : {args.source}")
    print(f"  eval_root  : {eval_root}")
    print(f"  device     : {args.device}")
    print(f"  limit      : {args.limit or 'all'}")
    print("=" * 72)

    # --- 1) Load episode frames + metadata ---------------------------------
    if args.source == "raw_hf":
        eps = load_episode_metadata(eval_root)
        n_pos = sum(1 for e in eps if e["label"] == 1)
        n_neg = sum(1 for e in eps if e["label"] == 0)
        print(f"Loaded {len(eps)} episodes: {n_pos} pos / {n_neg} neg")

        expected = max(e["mp4_frame_end"] for e in eps)
        mp4_path = os.path.join(eval_root, "videos/video/chunk-000/file-000.mp4")
        print(f"Decoding {mp4_path}  (expecting {expected} frames) ...")
        t0 = time.time()
        full_video = decode_full_mp4(mp4_path, expected_frames=expected)
        print(f"  decoded {full_video.shape} dtype={full_video.dtype}  "
              f"in {time.time()-t0:.1f}s  ({full_video.nbytes/1e9:.2f} GB)")
    else:
        print(f"Loading packed episodes from {eval_root}/{{keyframes,keyframes_success,keyframes_orphan_success}}/* ...")
        t0 = time.time()
        eps = load_packed_episodes(eval_root)
        n_pos = sum(1 for e in eps if e["label"] == 1)
        n_neg = sum(1 for e in eps if e["label"] == 0)
        # Frames are already embedded per-episode; no mp4 in this path.
        full_video = None
        print(f"  loaded {len(eps)} episodes ({n_pos} pos / {n_neg} neg) "
              f"in {time.time()-t0:.1f}s  "
              f"(frames per ep: {eps[0]['frames'].shape if eps else 'n/a'})")

    # --- 3) Load scorer -----------------------------------------------------
    print(f"\nLoading scorer ...")
    t0 = time.time()
    scorer = get_robometer_4b(model_path=args.model_path, device=args.device)
    print(f"  scorer ready ({time.time()-t0:.1f}s)  "
          f"model_type={scorer._model_type}  discrete={scorer._is_discrete}  "
          f"num_bins={scorer._num_bins}  max_frames={scorer.max_frames}")

    # --- 4) Per-episode inference ------------------------------------------
    n_run = len(eps) if not args.limit else min(args.limit, len(eps))
    results: List[Dict] = []
    print(f"\nRunning inference over {n_run} episodes ...")
    t_start = time.time()
    last_log = t_start
    for i, ep in enumerate(eps[:n_run]):
        if args.source == "raw_hf":
            clip_arr = full_video[ep["mp4_frame_start"]:ep["mp4_frame_end"]]
            ep_id_int = ep["episode_index"]
        else:
            clip_arr = ep["frames"]
            ep_id_int = i  # opaque episode_index for the scorer's bookkeeping
        clip_list = [clip_arr[k] for k in range(clip_arr.shape[0])]
        out = scorer(clip_list, task=ep["task"], episode_id=ep_id_int)
        results.append(dict(
            id=ep["id"],
            episode_index=ep.get("episode_index", i),
            data_source=ep["data_source"],
            task=ep["task"],
            quality_label_raw=ep["quality_label_raw"],
            label=ep["label"],
            success_prob_last=float(out["success_prob"]),
            progress_last=float(out["progress_reward"]),
        ))
        if (i + 1) % 25 == 0 or i + 1 == n_run or (time.time() - last_log) > 60:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n_run - i - 1) / rate if rate > 0 else float("inf")
            print(f"  [{i+1:4d}/{n_run}]  elapsed={elapsed/60:5.1f}m  "
                  f"rate={rate:.2f}eps/s  ETA={eta/60:5.1f}m  "
                  f"last(success_prob={out['success_prob']:.3f}, "
                  f"progress={out['progress_reward']:.3f})")
            last_log = time.time()

    # --- 5) Persist ---------------------------------------------------------
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tbl = pa.Table.from_pylist(results)
    pq.write_table(tbl, args.out)
    print(f"\nWrote {args.out}  ({len(results)} rows, "
          f"{os.path.getsize(args.out)/1e3:.1f} kB)")

    # --- 6) Quick sanity summary -------------------------------------------
    pos_sp = [r["success_prob_last"] for r in results if r["label"] == 1]
    neg_sp = [r["success_prob_last"] for r in results if r["label"] == 0]
    if pos_sp and neg_sp:
        print(f"\nsuccess_prob_last | label=1 (n={len(pos_sp)}):  "
              f"mean={np.mean(pos_sp):.4f}  median={float(np.median(pos_sp)):.4f}")
        print(f"success_prob_last | label=0 (n={len(neg_sp)}):  "
              f"mean={np.mean(neg_sp):.4f}  median={float(np.median(neg_sp)):.4f}")
        gap = float(np.mean(pos_sp) - np.mean(neg_sp))
        print(f"mean-gap (pos - neg): {gap:+.4f}  "
              f"({'separates' if gap > 0 else 'INVERTED — model broken / mislabeled'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
