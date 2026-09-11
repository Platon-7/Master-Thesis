#!/usr/bin/env python3
"""
extract_failures.py — Stage 1 of the RoboMeter failure scoring pipeline.

Downloads a tar archive from robometer/processed_datasets on HuggingFace,
identifies failure trajectories via index_mappings.json, extracts 8 evenly-spaced
keyframes from each, and writes a JSONL manifest for the scoring stage.

Usage:
    python extract_failures.py \\
        --archive jesbu1_oxe_rfm_oxe_bridge_v2 \\
        --output-dir /data/robometer_failures \\
        --hf-token hf_xxx

Output layout:
    /data/robometer_failures/
      keyframes/
        jesbu1_oxe_rfm_oxe_bridge_v2/
          ep_000123_<uuid>/
            frame_0_0.00s.jpg
            frame_1_...jpg
            ...
            frame_7_...jpg
      manifests/
        jesbu1_oxe_rfm_oxe_bridge_v2.jsonl   <- one line per failure episode

Each JSONL line:
    {
      "episode_id": "ep_000123_<uuid>",
      "archive": "jesbu1_oxe_rfm_oxe_bridge_v2",
      "category": "real_robot",
      "task": "Pick up the banana and place it in the bowl.",
      "robometer_label": "failure",
      "n_source_frames": 32,
      "keyframes_dir": "keyframes/jesbu1_oxe_rfm_oxe_bridge_v2/ep_000123_<uuid>"
    }
"""

import argparse
import io
import json
import os
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def find_snapshot(hf_cache_dir: str, repo_id: str, filename: str) -> Path | None:
    """Return cached file path if it already exists in the HF cache."""
    safe_repo = repo_id.replace("/", "--")
    base = Path(hf_cache_dir) / f"datasets--{safe_repo}" / "snapshots"
    if not base.exists():
        return None
    for snapshot in base.iterdir():
        candidate = snapshot / filename
        if candidate.exists():
            return candidate
    return None


def download_archive(archive_name: str, hf_token: str, hf_cache_dir: str) -> Path:
    """Download the .tar from HuggingFace (or return cached path)."""
    from huggingface_hub import hf_hub_download

    filename = f"{archive_name}.tar"
    cached = find_snapshot(hf_cache_dir, "robometer/processed_datasets", filename)
    if cached:
        print(f"  Using cached: {cached}")
        return cached

    print(f"  Downloading {filename} …")
    local = hf_hub_download(
        repo_id="robometer/processed_datasets",
        filename=filename,
        repo_type="dataset",
        cache_dir=hf_cache_dir,
        token=hf_token,
    )
    return Path(local)


def build_index_to_uuid(tf: tarfile.TarFile) -> dict[int, str]:
    """Build {trajectory_index: uuid} from embedding filenames inside the tar.

    Embedding files are named:  trajectory_{index:06d}_{uuid}_embeddings.pt
    """
    pattern = re.compile(r"trajectory_(\d+)_([0-9a-f\-]+)_embeddings\.pt$")
    mapping = {}
    for m in tf.getmembers():
        match = pattern.search(m.name)
        if match:
            idx = int(match.group(1))
            uuid = match.group(2)
            mapping[idx] = uuid
    return mapping


def build_index_to_task(task_indices: dict) -> dict[int, str]:
    """Invert task_indices: {task_name: [idx, ...]} → {idx: task_name}."""
    result = {}
    for task, indices in task_indices.items():
        for idx in indices:
            result[idx] = task
    return result


def extract_keyframes(frames_array: np.ndarray, n_keyframes: int = 8) -> list[np.ndarray]:
    """Sample n_keyframes evenly spaced from a (T, H, W, 3) uint8 array."""
    T = frames_array.shape[0]
    if T <= n_keyframes:
        indices = list(range(T))
        # pad by repeating last frame
        while len(indices) < n_keyframes:
            indices.append(T - 1)
    else:
        indices = [int(round(i * (T - 1) / (n_keyframes - 1))) for i in range(n_keyframes)]
    return [frames_array[i] for i in indices], indices


def save_keyframes(frames: list[np.ndarray], indices: list[int], fps: float,
                   out_dir: Path) -> list[str]:
    """Save keyframes as JPEG and return relative paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for kf_idx, (frame, src_idx) in enumerate(zip(frames, indices)):
        timestamp = src_idx / fps
        fname = f"frame_{kf_idx}_{timestamp:.2f}s.jpg"
        Image.fromarray(frame).save(out_dir / fname, quality=90)
        paths.append(str(out_dir / fname))
    return paths


# ────────────────────────────────────────────────────────────────────────────
# Core extraction
# ────────────────────────────────────────────────────────────────────────────

def extract_dataset_failures(
    archive_name: str,
    category: str,
    tar_path: Path,
    output_dir: Path,
    n_keyframes: int = 8,
    fps: float = 8.0,
    include_partial_success: bool = False,
    resume: bool = True,
) -> int:
    """Extract failure keyframes from one archive. Returns number of episodes processed."""

    manifest_path = output_dir / "manifests" / f"{archive_name}.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Load already-done episodes for resumption
    done_episodes = set()
    if resume and manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                entry = json.loads(line)
                done_episodes.add(entry["episode_id"])
        print(f"  Resuming: {len(done_episodes)} episodes already done")

    print(f"  Opening tar: {tar_path}")
    with tarfile.open(tar_path) as tf:
        members = tf.getmembers()

        # Read index_mappings.json
        mapping_member = next(
            (m for m in members if m.name.endswith("index_mappings.json")), None
        )
        if mapping_member is None:
            print("  WARNING: index_mappings.json not found — skipping")
            return 0

        index_mappings = json.loads(tf.extractfile(mapping_member).read())
        quality_indices = index_mappings.get("quality_indices", {})
        task_indices = index_mappings.get("task_indices", {})

        # Collect failure indices
        failure_indices = set(quality_indices.get("failure", []))
        if include_partial_success:
            failure_indices |= set(quality_indices.get("partial_success", []))

        if not failure_indices:
            print("  No failures found in this archive")
            return 0

        print(f"  Found {len(failure_indices)} failure trajectories")

        # Build index → uuid (from embeddings directory filenames)
        idx_to_uuid = build_index_to_uuid(tf)
        if not idx_to_uuid:
            print("  WARNING: Could not build index→UUID map (no embeddings dir?)")
            # Fallback: try to match by ordering of frames files
            frame_members = sorted(
                [m for m in members if m.name.endswith(".npz")],
                key=lambda m: m.name,
            )
            idx_to_uuid = {i: Path(m.name).stem.replace("trajectory_", "")
                           for i, m in enumerate(frame_members)}

        # Build index → task
        idx_to_task = build_index_to_task(task_indices)

        # Build uuid → npz member for fast lookup
        uuid_to_member = {}
        for m in members:
            if m.name.endswith(".npz") and "/frames/" in m.name:
                # trajectory_<uuid>.npz
                stem = Path(m.name).stem  # e.g. "trajectory_7cd5a40c-..."
                uuid = stem.replace("trajectory_", "")
                uuid_to_member[uuid] = m

        processed = 0
        with open(manifest_path, "a") as mf:
            for traj_idx in sorted(failure_indices):
                uuid = idx_to_uuid.get(traj_idx)
                if uuid is None:
                    continue

                episode_id = f"ep_{traj_idx:06d}_{uuid}"

                if episode_id in done_episodes:
                    continue

                task = idx_to_task.get(traj_idx, "unknown task")

                # Locate NPZ member
                npz_member = uuid_to_member.get(uuid)
                if npz_member is None:
                    print(f"  WARNING: NPZ not found for index {traj_idx} / uuid {uuid}")
                    continue

                # Load frames
                raw = tf.extractfile(npz_member).read()
                npz = np.load(io.BytesIO(raw), allow_pickle=True)
                frames_array = npz["frames"]  # (T, H, W, 3) uint8

                # Extract 8 keyframes
                keyframes, src_indices = extract_keyframes(frames_array, n_keyframes)

                # Save
                ep_dir = output_dir / "keyframes" / archive_name / episode_id
                save_keyframes(keyframes, src_indices, fps, ep_dir)

                # Write manifest entry
                entry = {
                    "episode_id": episode_id,
                    "archive": archive_name,
                    "category": category,
                    "task": task,
                    "robometer_label": "failure",
                    "n_source_frames": int(frames_array.shape[0]),
                    "keyframes_dir": str(ep_dir.relative_to(output_dir)),
                }
                mf.write(json.dumps(entry) + "\n")
                mf.flush()

                processed += 1
                if processed % 100 == 0:
                    print(f"  {processed}/{len(failure_indices)} episodes processed")

    print(f"  Done: {processed} new episodes extracted")
    return processed


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract RoboMeter failure keyframes")
    parser.add_argument(
        "--archive", required=True,
        help="Archive name (e.g. jesbu1_oxe_rfm_oxe_bridge_v2). "
             "Use 'all' to process every non-skipped single-file archive."
    )
    parser.add_argument(
        "--output-dir", default="/data/robometer_failures",
        help="Root output directory (default: /data/robometer_failures)"
    )
    parser.add_argument(
        "--hf-token", default=os.environ.get("HF_TOKEN", ""),
        help="HuggingFace token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--hf-cache-dir", default=os.environ.get("HF_HOME", "/data/hf_cache"),
        help="HuggingFace cache directory"
    )
    parser.add_argument(
        "--n-keyframes", type=int, default=8,
        help="Number of keyframes to extract per episode (default: 8)"
    )
    parser.add_argument(
        "--fps", type=float, default=8.0,
        help="Assumed frame rate of source video (for timestamp labels, default: 8.0)"
    )
    parser.add_argument(
        "--include-partial-success", action="store_true",
        help="Also extract partial_success episodes (not just failures)"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Do not resume — reprocess from scratch"
    )
    parser.add_argument(
        "--category", default=None,
        help="Filter by category when --archive all (real_robot, simulated, human_egocentric, eval_benchmark)"
    )
    parser.add_argument(
        "--list-datasets", action="store_true",
        help="Print available datasets and exit"
    )
    args = parser.parse_args()

    # Import catalog
    sys.path.insert(0, str(Path(__file__).parent))
    from datasets_catalog import DATASETS, list_datasets

    if args.list_datasets:
        for d in list_datasets(skip_done=True, no_splits=True):
            print(f"  [{d['category']}] {d['archive']}")
            print(f"    {d['description']}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.archive == "all":
        datasets_to_run = list_datasets(
            category=args.category,
            skip_done=True,
            no_splits=True,
        )
    else:
        # Find in catalog
        match = next((d for d in DATASETS if d["archive"] == args.archive), None)
        if match is None:
            # Allow running unlisted archives with unknown category
            datasets_to_run = [{"archive": args.archive, "category": "unknown", "split_parts": False}]
        else:
            datasets_to_run = [match]

    total_episodes = 0
    for ds in datasets_to_run:
        archive = ds["archive"]
        category = ds["category"]

        if ds.get("split_parts"):
            print(f"\nSKIPPING {archive} (split archive — requires manual assembly)")
            continue

        print(f"\n{'='*60}")
        print(f"Dataset: {archive}")
        print(f"Category: {category}")
        print(f"{'='*60}")

        try:
            tar_path = download_archive(archive, args.hf_token, args.hf_cache_dir)
        except Exception as e:
            print(f"  ERROR downloading {archive}: {e}")
            continue

        try:
            n = extract_dataset_failures(
                archive_name=archive,
                category=category,
                tar_path=tar_path,
                output_dir=output_dir,
                n_keyframes=args.n_keyframes,
                fps=args.fps,
                include_partial_success=args.include_partial_success,
                resume=not args.no_resume,
            )
            total_episodes += n
        except Exception as e:
            import traceback
            print(f"  ERROR processing {archive}: {e}")
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"Extraction complete. Total new episodes: {total_episodes}")
    print(f"Manifests written to: {output_dir}/manifests/")
    print(f"Keyframes written to: {output_dir}/keyframes/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
