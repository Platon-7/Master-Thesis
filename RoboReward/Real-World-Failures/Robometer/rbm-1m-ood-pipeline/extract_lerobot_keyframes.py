#!/usr/bin/env python3
"""
extract_lerobot_keyframes.py — extract 16 uniformly-spaced keyframes from each
episode of a LeRobot v3.0 dataset, in the loose-JPEG layout that
score_robometer_failures.py + pack_scored_output.py expect.

Specifically targets `robometer/rbm-1m-ood` which is in LeRobot v3.0 format:
  - data/chunk-XXX/file-XXX.parquet     ← per-frame rows (id, quality_label, data_source, ...)
  - meta/episodes/chunk-XXX/file-XXX.parquet  ← per-episode rows (length, video file, timestamps)
  - meta/tasks.parquet                  ← task strings indexed by task_index
  - videos/<data_source>/chunk-XXX/file-XXX.mp4  ← videos packed with multiple episodes per file

Output layout (matches FOR_LEONARDO.md $DATA_DIR):
  $OUT_DIR/
    keyframes/<archive>/<episode_id>/frame_NN_X.Xs.jpg     # 16 JPEGs per episode
    manifests/<archive>_failures.jsonl                     # one line per failure episode
    manifests/<archive>_successes.jsonl                    # one line per success episode
    manifests/<archive>_suboptimal.jsonl                   # one line per suboptimal episode

Each manifest line:
    {
      "episode_id": "<id from parquet>",
      "archive": "<data_source>",
      "family": "<data_source>",          # we use data_source as both — they're already family-level
      "task": "<natural language task>",
      "label": "successful" | "failure" | "suboptimal",
      "n_source_frames": <int>,
      "n_keyframes": 16,
      "keyframes_dir": "<archive>/<episode_id>"
    }

Usage:
    python extract_lerobot_keyframes.py \
        --raw-dir /projects/prjs1958/robometer_frame_eval_dataset/raw_hf \
        --out-dir /projects/prjs1958/robometer_frame_eval_dataset/staging
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd


N_KEYFRAMES = 16
JPEG_QUALITY = 95


def load_meta(raw_dir: Path):
    """Load LeRobot meta+data parquets. Returns (episodes_df, frames_df, tasks_list)."""
    info_path = raw_dir / "meta" / "info.json"
    with info_path.open() as f:
        info = json.load(f)

    # Episodes (per-episode metadata: length, video file pointer, timestamps)
    ep_paths = sorted((raw_dir / "meta" / "episodes").rglob("*.parquet"))
    if not ep_paths:
        sys.exit(f"No episode parquets under {raw_dir / 'meta' / 'episodes'}")
    episodes_df = pd.concat([pd.read_parquet(p) for p in ep_paths], ignore_index=True)

    # Per-frame rows (we only need first row per episode for data_source + quality_label)
    frame_paths = sorted((raw_dir / "data").rglob("*.parquet"))
    if not frame_paths:
        sys.exit(f"No data parquets under {raw_dir / 'data'}")
    frames_df = pd.concat([pd.read_parquet(p) for p in frame_paths], ignore_index=True)

    # Tasks
    tasks_path = raw_dir / "meta" / "tasks.parquet"
    tasks_df = pd.read_parquet(tasks_path)
    # tasks.parquet has columns ['task_index', 'task'] or similar — adapt
    if "task" in tasks_df.columns:
        tasks = tasks_df.set_index("task_index")["task"].to_dict() if "task_index" in tasks_df.columns else {i: t for i, t in enumerate(tasks_df["task"])}
    else:
        # Best-effort
        tasks = {i: str(row.iloc[0]) for i, row in tasks_df.iterrows()}

    return info, episodes_df, frames_df, tasks


def episode_first_row(frames_df: pd.DataFrame, ep_idx: int):
    """First per-frame row for this episode (gives us data_source, quality_label, id, task_index)."""
    rows = frames_df[frames_df["episode_index"] == ep_idx]
    if len(rows) == 0:
        return None
    return rows.iloc[0]


def extract_keyframes_from_video(
    video_path: Path,
    from_ts: float,
    to_ts: float,
    n_source_frames: int,
    n_keyframes: int = N_KEYFRAMES,
):
    """Decode the [from_ts, to_ts) window of `video_path`, return n_keyframes uniformly-spaced
    BGR frames + their timestamps."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    # Seek to from_ts
    cap.set(cv2.CAP_PROP_POS_MSEC, from_ts * 1000.0)

    # Read n_source_frames frames sequentially from from_ts
    frames = []
    for _ in range(n_source_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < n_keyframes:
        # Fallback: re-read entire video and clamp
        raise ValueError(
            f"video {video_path} window [{from_ts:.3f}-{to_ts:.3f}] yielded only "
            f"{len(frames)} frames, need >= {n_keyframes}"
        )

    # Uniformly sample n_keyframes indices from 0..len(frames)-1
    if len(frames) == n_keyframes:
        sampled_idx = list(range(n_keyframes))
    else:
        sampled_idx = np.linspace(0, len(frames) - 1, n_keyframes).round().astype(int).tolist()

    sampled_frames = [frames[i] for i in sampled_idx]
    # Timestamp per sampled frame relative to from_ts
    sampled_ts = [from_ts + i / fps for i in sampled_idx]
    return sampled_frames, sampled_ts


def write_keyframes(keyframes, timestamps, ep_dir: Path):
    """Save BGR frames as frame_NN_T.TTs.jpg in ep_dir."""
    ep_dir.mkdir(parents=True, exist_ok=True)
    for i, (frame, ts) in enumerate(zip(keyframes, timestamps)):
        rel_ts = ts - timestamps[0]
        fname = f"frame_{i:02d}_{rel_ts:.2f}s.jpg"
        cv2.imwrite(str(ep_dir / fname), frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--max-episodes", type=int, default=-1, help="stop after N episodes (debug)")
    ap.add_argument("--skip-existing", action="store_true", default=True, help="skip episode dirs that already have 16 keyframes")
    args = ap.parse_args()

    info, episodes_df, frames_df, tasks = load_meta(args.raw_dir)
    print(f"Total episodes: {len(episodes_df)}, total frames: {len(frames_df)}, tasks: {len(tasks)}")
    print(f"Video path template: {info['video_path']}")

    keyframes_root = args.out_dir / "keyframes"
    manifests_root = args.out_dir / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)
    keyframes_root.mkdir(parents=True, exist_ok=True)

    # Group manifests by archive+label (we'll write all at end)
    manifest_lines: dict[tuple[str, str], list[dict]] = defaultdict(list)

    n_done = 0
    n_skipped = 0
    n_failed = 0

    for ep_idx in range(len(episodes_df)):
        ep_row = episodes_df.iloc[ep_idx]
        first_frame_row = episode_first_row(frames_df, ep_row["episode_index"])
        if first_frame_row is None:
            print(f"  [skip] ep {ep_idx}: no frame rows", file=sys.stderr)
            n_failed += 1
            continue

        ep_id = str(first_frame_row["id"])
        archive = str(first_frame_row["data_source"])
        quality = str(first_frame_row["quality_label"]).strip().lower()
        if quality in ("failed", "fail"):
            quality = "failure"
        elif quality in ("success", "successful"):
            quality = "successful"
        elif quality == "suboptimal":
            quality = "suboptimal"
        else:
            quality = "failure" if quality not in ("successful", "failure", "suboptimal") else quality

        # Task lookup
        task_idx = int(first_frame_row["task_index"])
        task_str = tasks.get(task_idx, "")
        if isinstance(task_str, (list, np.ndarray)) and len(task_str) > 0:
            task_str = str(task_str[0])
        task_str = str(task_str).strip()
        # Episode tasks list (more reliable) — overrides
        ep_tasks = ep_row.get("tasks")
        if ep_tasks is not None:
            try:
                if hasattr(ep_tasks, "__len__") and len(ep_tasks) > 0:
                    task_str = str(ep_tasks[0]).strip()
            except (TypeError, KeyError):
                pass

        # Locate video. LeRobot v3 uses video_key='video' for single-camera datasets;
        # all 782 episodes share the single packed mp4 at videos/video/chunk-XXX/file-XXX.mp4.
        video_key = "video"
        v_chunk = int(ep_row["videos/video/chunk_index"])
        v_file = int(ep_row["videos/video/file_index"])
        from_ts = float(ep_row["videos/video/from_timestamp"])
        to_ts = float(ep_row["videos/video/to_timestamp"])
        n_source = int(ep_row["length"])

        video_path = args.raw_dir / info["video_path"].format(
            video_key=video_key, chunk_index=v_chunk, file_index=v_file
        )
        if not video_path.exists():
            print(f"  [skip] ep {ep_idx} ({ep_id}): video missing {video_path}", file=sys.stderr)
            n_failed += 1
            continue

        # Output dirs
        ep_keyframes_dir = keyframes_root / archive / ep_id
        if args.skip_existing and ep_keyframes_dir.is_dir():
            existing_jpegs = sorted(ep_keyframes_dir.glob("frame_*.jpg"))
            if len(existing_jpegs) == N_KEYFRAMES:
                # Already extracted; still record in manifest
                manifest_lines[(archive, quality)].append({
                    "episode_id": ep_id,
                    "archive": archive,
                    "family": archive,
                    "task": task_str,
                    "label": quality,
                    "n_source_frames": n_source,
                    "n_keyframes": N_KEYFRAMES,
                    # NOTE: must be prefixed with 'keyframes/' — score_robometer_failures.py
                    # resolves the keyframe directory as `data_dir / episode["keyframes_dir"]`
                    "keyframes_dir": f"keyframes/{archive}/{ep_id}",
                })
                n_skipped += 1
                if (n_skipped + n_done) % 50 == 0:
                    print(f"  [progress] {n_done} extracted, {n_skipped} skipped, {n_failed} failed")
                continue

        # Extract
        try:
            keyframes, timestamps = extract_keyframes_from_video(
                video_path, from_ts, to_ts, n_source, N_KEYFRAMES
            )
        except Exception as e:
            print(f"  [fail] ep {ep_idx} ({ep_id}): {type(e).__name__}: {e}", file=sys.stderr)
            n_failed += 1
            continue

        write_keyframes(keyframes, timestamps, ep_keyframes_dir)

        manifest_lines[(archive, quality)].append({
            "episode_id": ep_id,
            "archive": archive,
            "family": archive,
            "task": task_str,
            "label": quality,
            "n_source_frames": n_source,
            "n_keyframes": N_KEYFRAMES,
            "keyframes_dir": f"keyframes/{archive}/{ep_id}",
        })
        n_done += 1
        if n_done % 50 == 0:
            print(f"  [progress] {n_done} extracted, {n_skipped} skipped, {n_failed} failed")

        if args.max_episodes > 0 and (n_done + n_skipped) >= args.max_episodes:
            print(f"  [debug-cap] reached max-episodes={args.max_episodes}")
            break

    # Write manifests (one per archive+label; existing pipeline expects <archive>_failures.jsonl etc.)
    label_to_suffix = {"failure": "failures", "successful": "successes", "suboptimal": "suboptimal"}
    counts = defaultdict(int)
    for (archive, label), lines in sorted(manifest_lines.items()):
        suffix = label_to_suffix.get(label, label)
        out_path = manifests_root / f"{archive}_{suffix}.jsonl"
        with out_path.open("w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        counts[label] += len(lines)
        print(f"  manifest: {out_path.name} ({len(lines)} episodes)")

    print(f"\nDone.")
    print(f"  extracted: {n_done}")
    print(f"  skipped (already done): {n_skipped}")
    print(f"  failed: {n_failed}")
    for label, n in counts.items():
        print(f"  {label}: {n} episodes")


if __name__ == "__main__":
    main()
