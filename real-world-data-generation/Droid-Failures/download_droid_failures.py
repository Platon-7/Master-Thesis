#!/usr/bin/env python3
"""
Download DROID Failure Episodes from GCS (Fast Bulk Version)
=============================================================

Uses `gsutil -m cp` with wildcards for parallel bulk downloads instead of
per-episode sequential calls.

Strategy:
  1. For each lab, bulk-download all metadata JSONs in one gsutil -m command
  2. For each lab, bulk-download all mono MP4s (exclude stereo) in one gsutil -m command
  3. Locally reorganize into flat episode directories and build manifest

Usage:
    # Download all failures (bulk)
    python download_droid_failures.py --output-dir /var/scratch/pkarageo/droid_failures_raw

    # Download specific labs only
    python download_droid_failures.py --output-dir /var/scratch/pkarageo/droid_failures_raw --labs GuptaLab CLVR

Author: Platon Karageorgis
"""

import os
import sys
import json
import argparse
import subprocess
import re
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


GCS_BUCKET = "gs://gresearch/robotics/droid_raw/1.0.1"
MANIFEST_FILENAME = "download_manifest.json"

DROID_LABS = [
    "AUTOLab", "CLVR", "GuptaLab", "ILIAD", "IPRL", "IRIS",
    "PennPAL", "RAD", "RAIL", "REAL", "RPL", "TRI", "WEIRD",
]


def run_gsutil(args: List[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    """Run a gsutil command and return the result."""
    cmd = ["gsutil"] + args
    print(f"  Running: {' '.join(cmd[:6])}...")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def bulk_download_lab(lab: str, output_dir: str) -> int:
    """Bulk download all failure episodes for a single lab using gsutil -m rsync.

    Downloads the entire failure/ tree preserving directory structure, then
    we process it locally. This is MUCH faster than per-episode gsutil calls.

    Returns number of files downloaded.
    """
    lab_gcs = f"{GCS_BUCKET}/{lab}/failure/"
    lab_local = os.path.join(output_dir, "_raw", lab, "failure")
    os.makedirs(lab_local, exist_ok=True)

    # Use gsutil -m rsync to mirror the GCS tree locally
    # -r = recursive, -x = exclude pattern (stereo mp4s, svo files, h5 files)
    # rsync skips already-downloaded files automatically (resume-safe)
    exclude_pattern = r".*(-stereo\.mp4|\.svo|\.h5|svo_names\.json)$"

    result = run_gsutil([
        "-m", "rsync", "-r",
        "-x", exclude_pattern,
        lab_gcs,
        lab_local,
    ], timeout=7200)  # 2hr timeout per lab

    if result.returncode != 0:
        print(f"  WARNING: gsutil rsync for {lab} had errors: {result.stderr[:300]}")
        # rsync may partially succeed, so continue

    # Count downloaded files
    file_count = sum(1 for _ in Path(lab_local).rglob("*") if _.is_file())
    return file_count


def reorganize_lab(lab: str, output_dir: str) -> List[Dict]:
    """Reorganize a downloaded lab's failure tree into flat episode directories.

    Raw structure:  _raw/{LAB}/failure/{date}/{episode_name}/metadata_*.json + recordings/MP4/*.mp4
    Target:         {LAB}_{date}_{episode_name}/metadata.json + {cam}.mp4

    Returns list of episode info dicts.
    """
    raw_lab_dir = os.path.join(output_dir, "_raw", lab, "failure")
    if not os.path.exists(raw_lab_dir):
        return []

    episodes = []

    # Walk date directories
    for date_dir in sorted(Path(raw_lab_dir).iterdir()):
        if not date_dir.is_dir():
            continue
        date_str = date_dir.name

        for episode_dir in sorted(date_dir.iterdir()):
            if not episode_dir.is_dir():
                continue

            episode_name = episode_dir.name
            # Filesystem-safe ID
            safe_name = episode_name.replace(":", "-")
            episode_id = f"{lab}_{date_str}_{safe_name}"

            # Find metadata JSON
            metadata_files = list(episode_dir.glob("metadata_*.json"))
            if not metadata_files:
                metadata_files = list(episode_dir.glob("*.json"))
            if not metadata_files:
                continue

            # Parse metadata
            try:
                with open(metadata_files[0]) as f:
                    metadata = json.load(f)
            except Exception:
                continue

            # Map camera serials to names
            camera_map = {}
            for cam_type in ["wrist", "ext1", "ext2"]:
                serial_key = f"{cam_type}_cam_serial"
                if serial_key in metadata:
                    camera_map[metadata[serial_key]] = cam_type

            # Create target episode directory
            target_dir = os.path.join(output_dir, episode_id)
            os.makedirs(target_dir, exist_ok=True)

            # Copy/link metadata
            target_metadata = os.path.join(target_dir, "metadata.json")
            if not os.path.exists(target_metadata):
                try:
                    os.link(str(metadata_files[0]), target_metadata)
                except OSError:
                    import shutil
                    shutil.copy2(str(metadata_files[0]), target_metadata)

            # Find and copy/link mono MP4s
            mp4_dir = episode_dir / "recordings" / "MP4"
            cameras_found = []
            if mp4_dir.exists():
                for mp4_file in mp4_dir.glob("*.mp4"):
                    if "stereo" in mp4_file.name.lower():
                        continue
                    serial = mp4_file.stem
                    cam_name = camera_map.get(serial, f"unknown_{serial}")
                    target_mp4 = os.path.join(target_dir, f"{cam_name}.mp4")
                    if not os.path.exists(target_mp4):
                        try:
                            os.link(str(mp4_file), target_mp4)
                        except OSError:
                            import shutil
                            shutil.copy2(str(mp4_file), target_mp4)
                    cameras_found.append(cam_name)

            episodes.append({
                "episode_id": episode_id,
                "task": metadata.get("current_task", ""),
                "success": metadata.get("success", False),
                "lab": metadata.get("lab", lab),
                "uuid": metadata.get("uuid", ""),
                "trajectory_length": metadata.get("trajectory_length", 0),
                "cameras_downloaded": sorted(cameras_found),
                "num_cameras": len(cameras_found),
            })

    return episodes


def load_manifest(output_dir: str) -> Dict:
    manifest_path = os.path.join(output_dir, MANIFEST_FILENAME)
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)
    return {"completed_episodes": {}, "completed_labs": [], "started_at": datetime.now().isoformat()}


def save_manifest(output_dir: str, manifest: Dict):
    manifest_path = os.path.join(output_dir, MANIFEST_FILENAME)
    manifest["last_updated"] = datetime.now().isoformat()
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Bulk download DROID failure episodes from GCS")
    parser.add_argument("--output-dir", type=str, default="/var/scratch/pkarageo/droid_failures_raw",
                        help="Output directory")
    parser.add_argument("--labs", nargs="+", default=None,
                        help="Specific labs to download (default: all)")
    args = parser.parse_args()

    labs = args.labs or DROID_LABS
    os.makedirs(args.output_dir, exist_ok=True)
    manifest = load_manifest(args.output_dir)

    print(f"{'='*60}")
    print(f"DROID Failure Download (Bulk rsync)")
    print(f"{'='*60}")
    print(f"Labs: {labs}")
    print(f"Output: {args.output_dir}")
    print(f"Already completed labs: {manifest.get('completed_labs', [])}")
    print(f"{'='*60}\n")

    total_episodes = 0
    total_files = 0

    for lab in labs:
        if lab in manifest.get("completed_labs", []):
            print(f"\n[{lab}] Already downloaded, skipping.")
            continue

        print(f"\n{'='*40}")
        print(f"[{lab}] Bulk downloading failures...")
        print(f"{'='*40}")

        # Step 1: Bulk download with gsutil -m rsync
        file_count = bulk_download_lab(lab, args.output_dir)
        print(f"  Files downloaded: {file_count}")
        total_files += file_count

        # Step 2: Reorganize into flat episode dirs
        print(f"  Reorganizing into episode directories...")
        episodes = reorganize_lab(lab, args.output_dir)
        print(f"  Episodes organized: {len(episodes)}")
        total_episodes += len(episodes)

        # Update manifest
        for ep in episodes:
            manifest["completed_episodes"][ep["episode_id"]] = ep
        if "completed_labs" not in manifest:
            manifest["completed_labs"] = []
        manifest["completed_labs"].append(lab)
        save_manifest(args.output_dir, manifest)

        # Print sample tasks
        tasks = set(ep["task"] for ep in episodes[:20] if ep["task"])
        print(f"  Sample tasks: {list(tasks)[:5]}")

    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"Total episodes: {total_episodes}")
    print(f"Total files: {total_files}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
