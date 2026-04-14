#!/usr/bin/env python3
"""
Package DROID Failures as RoboReward JSONL
============================================

Takes scored+verified results from label_droid_failures.py and packages them
into RoboReward-compatible JSONL format. Optionally merges with existing metadata.

Output format per line:
{
    "file_name": "droid_failures/{episode_id}_{camera}.mp4",
    "task": "Rearrange pillows on sofa.",
    "reward": 2,
    "gpt5_mini_check": "Qwen3-VL verification: ..."
}

Usage:
    # Package only
    python package_droid_failures.py \
        --scored-results /var/scratch/pkarageo/droid_failures_labeled/scored_results.jsonl \
        --video-dir /var/scratch/pkarageo/droid_failures_processed \
        --output /var/scratch/pkarageo/droid_failures_packaged/metadata.jsonl

    # Package and merge with existing RoboReward metadata
    python package_droid_failures.py \
        --scored-results /var/scratch/pkarageo/droid_failures_labeled/scored_results.jsonl \
        --video-dir /var/scratch/pkarageo/droid_failures_processed \
        --output /var/scratch/pkarageo/droid_failures_packaged/metadata.jsonl \
        --merge-with /var/scratch/pkarageo/roboreward_dataset/metadata.jsonl \
        --copy-videos-to /var/scratch/pkarageo/roboreward_dataset/droid_failures

Author: Platon Karageorgis
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Set
from collections import Counter


def load_scored_results(results_path: str) -> List[Dict]:
    """Load scored results JSONL, keeping only verified entries."""
    results = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("verified", False):
                results.append(entry)
    return results


def package_as_roboreward(results: List[Dict], video_rel_prefix: str = "droid_failures") -> List[Dict]:
    """Convert scored results to RoboReward metadata format."""
    packaged = []
    seen_filenames: Set[str] = set()

    for entry in results:
        video_key = entry["video_key"]
        filename = f"{video_rel_prefix}/{video_key}.mp4"

        # Skip duplicates
        if filename in seen_filenames:
            continue
        seen_filenames.add(filename)

        score = entry["score"]
        if score < 1 or score > 5:
            continue

        packaged.append({
            "file_name": filename,
            "task": entry["task"],
            "reward": score,
            "gpt5_mini_check": f"Qwen3-VL verification: {entry.get('verify_output', '')[:500]}",
        })

    return packaged


def validate_packaged(packaged: List[Dict]) -> Dict:
    """Validate packaged entries and return statistics."""
    score_dist = Counter(e["reward"] for e in packaged)
    filenames = [e["file_name"] for e in packaged]
    unique_filenames = set(filenames)

    return {
        "total_entries": len(packaged),
        "unique_filenames": len(unique_filenames),
        "duplicates": len(filenames) - len(unique_filenames),
        "score_distribution": dict(sorted(score_dist.items())),
        "scores_valid": all(1 <= e["reward"] <= 5 for e in packaged),
        "all_have_task": all(bool(e["task"]) for e in packaged),
    }


def main():
    parser = argparse.ArgumentParser(description="Package DROID failures as RoboReward JSONL")
    parser.add_argument("--scored-results", type=str, required=True,
                        help="Path to scored_results.jsonl from label_droid_failures.py")
    parser.add_argument("--video-dir", type=str, required=True,
                        help="Directory containing processed videos")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL path")
    parser.add_argument("--merge-with", type=str, default=None,
                        help="Existing metadata.jsonl to merge with")
    parser.add_argument("--copy-videos-to", type=str, default=None,
                        help="Copy processed videos to this directory")
    parser.add_argument("--video-rel-prefix", type=str, default="droid_failures",
                        help="Relative path prefix for file_name field")
    args = parser.parse_args()

    print(f"{'='*60}")
    print("Packaging DROID Failures as RoboReward JSONL")
    print(f"{'='*60}\n")

    # Load scored results
    results = load_scored_results(args.scored_results)
    print(f"Verified results loaded: {len(results)}")

    # Package
    packaged = package_as_roboreward(results, args.video_rel_prefix)
    print(f"Packaged entries: {len(packaged)}")

    # Validate
    stats = validate_packaged(packaged)
    print(f"\nValidation:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if stats["duplicates"] > 0:
        print(f"\nWARNING: {stats['duplicates']} duplicate file_name entries detected!")

    # Write output JSONL
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for entry in packaged:
            f.write(json.dumps(entry) + "\n")
    print(f"\nOutput written: {args.output}")

    # Optionally copy videos
    if args.copy_videos_to:
        os.makedirs(args.copy_videos_to, exist_ok=True)
        copied = 0
        missing = 0
        for entry in packaged:
            video_key = os.path.basename(entry["file_name"]).replace(".mp4", "")
            src = os.path.join(args.video_dir, f"{video_key}.mp4")
            dst = os.path.join(args.copy_videos_to, f"{video_key}.mp4")
            if os.path.exists(src):
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                copied += 1
            else:
                missing += 1
        print(f"\nVideos copied to {args.copy_videos_to}: {copied} (missing: {missing})")

    # Optionally merge with existing metadata
    if args.merge_with and os.path.exists(args.merge_with):
        print(f"\nMerging with existing metadata: {args.merge_with}")

        existing = []
        with open(args.merge_with) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(json.loads(line))

        existing_count = len(existing)
        existing_filenames = {e["file_name"] for e in existing}

        # Add only new entries
        new_entries = [e for e in packaged if e["file_name"] not in existing_filenames]
        merged = existing + new_entries

        merged_path = args.merge_with + ".merged"
        with open(merged_path, "w") as f:
            for entry in merged:
                f.write(json.dumps(entry) + "\n")

        print(f"  Existing entries: {existing_count}")
        print(f"  New entries added: {len(new_entries)}")
        print(f"  Skipped (already in existing): {len(packaged) - len(new_entries)}")
        print(f"  Total merged: {len(merged)}")
        print(f"  Merged file: {merged_path}")
        print(f"\n  To apply: mv {merged_path} {args.merge_with}")

    print(f"\n{'='*60}")
    print("Packaging complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
