#!/usr/bin/env python3
"""
Preprocess DROID Failure Videos
================================

Converts raw DROID videos (1280x720 @ 60fps) to RoboReward format (320x192 @ 10fps).
Processes all 3 cameras per episode.

Transform: scale to 320x180, then pad to 320x192 with 6px black letterbox top+bottom.

Usage:
    python preprocess_droid_videos.py \
        --input-dir /var/scratch/pkarageo/droid_failures_raw \
        --output-dir /var/scratch/pkarageo/droid_failures_processed \
        --workers 8

Author: Platon Karageorgis
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional


def preprocess_video(input_path: str, output_path: str) -> Optional[str]:
    """Convert a single video to RoboReward format using ffmpeg.

    - Scale 1280x720 -> 320x180 (maintaining 16:9 aspect ratio)
    - Pad 320x180 -> 320x192 (6px black letterbox top+bottom)
    - Resample 60fps -> 10fps
    - Re-encode H.264

    Returns error message on failure, None on success.
    """
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return None  # Already processed

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", "scale=320:180,pad=320:192:0:6:black",
        "-r", "10",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-an",  # Remove audio
        "-loglevel", "error",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return f"ffmpeg error: {result.stderr[:200]}"
        return None
    except subprocess.TimeoutExpired:
        return "ffmpeg timeout (120s)"
    except Exception as e:
        return str(e)


def process_single_episode(args_tuple):
    """Process all cameras for a single episode. Used by ProcessPoolExecutor."""
    episode_dir, output_dir, episode_id = args_tuple
    results = []

    for cam in ["ext1", "ext2", "wrist"]:
        input_path = os.path.join(episode_dir, f"{cam}.mp4")
        if not os.path.exists(input_path):
            continue

        output_filename = f"{episode_id}_{cam}.mp4"
        output_path = os.path.join(output_dir, output_filename)

        error = preprocess_video(input_path, output_path)
        results.append({
            "episode_id": episode_id,
            "camera": cam,
            "input": input_path,
            "output": output_path,
            "output_filename": output_filename,
            "success": error is None,
            "error": error,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Preprocess DROID failure videos")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory with raw DROID episodes (from download_droid_failures.py)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for processed videos")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel ffmpeg workers")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect episodes to process
    manifest_path = os.path.join(args.input_dir, "download_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        episode_ids = list(manifest.get("completed_episodes", {}).keys())
    else:
        # Fall back to directory listing
        episode_ids = [
            d for d in os.listdir(args.input_dir)
            if os.path.isdir(os.path.join(args.input_dir, d)) and d != "__pycache__"
        ]

    print(f"{'='*60}")
    print(f"Preprocessing DROID Failure Videos")
    print(f"{'='*60}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Episodes: {len(episode_ids)}")
    print(f"Workers: {args.workers}")
    print(f"{'='*60}\n")

    # Build work items
    work_items = []
    for episode_id in episode_ids:
        episode_dir = os.path.join(args.input_dir, episode_id)
        if os.path.isdir(episode_dir):
            work_items.append((episode_dir, args.output_dir, episode_id))

    # Process in parallel
    all_results = []
    success_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_episode, item): item for item in work_items}

        for i, future in enumerate(as_completed(futures)):
            episode_results = future.result()
            all_results.extend(episode_results)

            for r in episode_results:
                if r["success"]:
                    success_count += 1
                else:
                    fail_count += 1
                    print(f"  FAILED: {r['episode_id']}_{r['camera']}: {r['error']}")

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(work_items)} episodes | {success_count} videos OK, {fail_count} failed")

    # Save processing report
    report = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "total_episodes": len(work_items),
        "total_videos_success": success_count,
        "total_videos_failed": fail_count,
        "processed_videos": [r for r in all_results if r["success"]],
        "failed_videos": [r for r in all_results if not r["success"]],
    }

    report_path = os.path.join(args.output_dir, "preprocess_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Videos processed: {success_count}")
    print(f"Videos failed: {fail_count}")
    print(f"Report: {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
