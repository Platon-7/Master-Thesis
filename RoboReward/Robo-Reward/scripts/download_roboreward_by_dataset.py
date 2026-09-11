#!/usr/bin/env python3
"""
RoboReward Video Downloader (By Dataset Only)
==============================================

Downloads videos from HuggingFace RoboReward dataset filtered only by
dataset source and reward (not by specific task).

This is adapted to work with RoboReward's extreme task diversity.

Author: Platon Karageorgis
"""

import os
import sys
import json
import re
import argparse
import cv2
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datasets import load_dataset, Value
from tqdm import tqdm

# Import from original script
sys.path.insert(0, os.path.dirname(__file__))

DEFAULT_OUTPUT_DIR = "/var/scratch/pkarageo/roboreward_tasks"
DEFAULT_NUM_SUCCESS = 20
DEFAULT_NUM_FAILURE = 20
DEFAULT_REWARD_THRESHOLD = 4

@dataclass
class VideoMetadata:
    """Metadata for a downloaded video."""
    video_filename: str
    hf_path: str
    task: str
    reward: int
    dataset_source: str
    episode_id: str
    label: str

    def to_dict(self):
        return asdict(self)


def parse_hf_video_path(hf_path: str) -> Tuple[str, str]:
    """Parse HF path to extract dataset and episode ID."""
    dataset_source = 'unknown'
    episode_id = 'unknown'

    if 'train/' in hf_path:
        match = re.search(r'/train/([^/]+)/([^/]+\.mp4)', hf_path)
        if match:
            dataset_source = match.group(1)
            episode_filename = match.group(2)
            episode_id = episode_filename.replace('.mp4', '')

    return dataset_source, episode_id


def convert_hf_to_https(hf_path: str) -> str:
    """Convert HF hf:// URL to HTTPS."""
    if not hf_path.startswith('hf://'):
        return hf_path

    path_part = hf_path.replace('hf://datasets/', '')
    if '@' in path_part:
        repo_part, hash_and_path = path_part.split('@', 1)
        hash_part, file_path = hash_and_path.split('/', 1)
        return f"https://huggingface.co/datasets/{repo_part}/resolve/{hash_part}/{file_path}"
    return hf_path.replace('hf://', 'https://')


def download_video(hf_path: str, output_path: str) -> bool:
    """Download video from HF."""
    try:
        https_url = convert_hf_to_https(hf_path)
        response = requests.get(https_url, stream=True, timeout=30)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        return False


def validate_video(video_path: str) -> Optional[Dict]:
    """Validate video with OpenCV."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if fps <= 0 or frame_count <= 0:
            return None

        return {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': frame_count / fps if fps > 0 else 0
        }
    except:
        return None


def main():
    parser = argparse.ArgumentParser(description="Download RoboReward videos by dataset")
    parser.add_argument('--dataset', type=str, required=True, choices=['droid', 'bridge'],
                        help='Dataset source')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--num-success', type=int, default=DEFAULT_NUM_SUCCESS)
    parser.add_argument('--num-failure', type=int, default=DEFAULT_NUM_FAILURE)
    parser.add_argument('--reward-threshold', type=int, default=DEFAULT_REWARD_THRESHOLD)
    parser.add_argument('--task-keyword', type=str, default=None,
                        help='Optional keyword to filter tasks (e.g., "pot", "move")')
    args = parser.parse_args()

    # Create directories
    success_dir = os.path.join(args.output_dir, "success")
    failure_dir = os.path.join(args.output_dir, "failure")
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(failure_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"📥 Downloading {args.dataset.upper()} Videos")
    print(f"{'='*80}")
    print(f"Target: {args.num_success} success + {args.num_failure} failure")
    if args.task_keyword:
        print(f"Keyword filter: '{args.task_keyword}'")
    print(f"Output: {args.output_dir}")
    print(f"{'='*80}\n")

    # Load dataset
    print("⬇️  Loading RoboReward dataset...")
    dataset = load_dataset("teetone/RoboReward", split="train", streaming=True)
    dataset = dataset.cast_column("video", Value("string"))
    print("✅ Dataset loaded\n")

    # Counters
    success_count = 0
    failure_count = 0
    samples_checked = 0
    downloaded_success = []
    downloaded_failure = []

    print("🔍 Searching for matching samples...\n")
    dataset_iter = iter(dataset)

    while success_count < args.num_success or failure_count < args.num_failure:
        try:
            sample = next(dataset_iter)
            samples_checked += 1

            if samples_checked % 1000 == 0:
                print(f"  Checked {samples_checked:,} | Found: {success_count}S / {failure_count}F")

            # Parse metadata
            dataset_source, episode_id = parse_hf_video_path(sample['video'])

            # Filter by dataset
            if dataset_source != args.dataset:
                continue

            # Filter by keyword if specified
            if args.task_keyword and args.task_keyword.lower() not in sample['task'].lower():
                continue

            # Determine label
            reward = int(sample['reward'])
            if reward >= args.reward_threshold:
                if success_count >= args.num_success:
                    continue
                label = "success"
            else:
                if failure_count >= args.num_failure:
                    continue
                label = "failure"

            # Download
            filename = f"{dataset_source}_{episode_id}.mp4"
            output_dir = success_dir if label == "success" else failure_dir
            output_path = os.path.join(output_dir, filename)

            if samples_checked <= 100 or (success_count + failure_count) % 5 == 0:
                print(f"\n  📥 [{label}] {filename} (reward: {reward})")
                print(f"     Task: {sample['task'][:70]}")

            if download_video(sample['video'], output_path):
                video_info = validate_video(output_path)
                if video_info:
                    metadata = VideoMetadata(
                        video_filename=filename,
                        hf_path=sample['video'],
                        task=sample['task'],
                        reward=reward,
                        dataset_source=dataset_source,
                        episode_id=episode_id,
                        label=label
                    )

                    if label == "success":
                        downloaded_success.append(metadata)
                        success_count += 1
                    else:
                        downloaded_failure.append(metadata)
                        failure_count += 1
                else:
                    os.remove(output_path)

        except StopIteration:
            print(f"\n⚠️  Reached end of dataset")
            break
        except Exception as e:
            if samples_checked % 1000 == 0:
                print(f"⚠️  Error: {e}")
            continue

    # Save metadata
    print(f"\n{'='*80}")
    print(f"✅ DOWNLOAD COMPLETE")
    print(f"{'='*80}")
    print(f"Samples checked: {samples_checked:,}")
    print(f"Success: {success_count}/{args.num_success}")
    print(f"Failure: {failure_count}/{args.num_failure}")
    print(f"{'='*80}\n")

    # Save metadata files
    metadata_json = {
        "dataset_source": args.dataset,
        "task_keyword": args.task_keyword,
        "download_config": {
            "num_success": args.num_success,
            "num_failure": args.num_failure,
            "reward_threshold": args.reward_threshold
        },
        "statistics": {
            "samples_checked": samples_checked,
            "success_count": success_count,
            "failure_count": failure_count
        }
    }

    with open(os.path.join(args.output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata_json, f, indent=2)

    with open(os.path.join(success_dir, "samples.json"), 'w') as f:
        json.dump([m.to_dict() for m in downloaded_success], f, indent=2)

    with open(os.path.join(failure_dir, "samples.json"), 'w') as f:
        json.dump([m.to_dict() for m in downloaded_failure], f, indent=2)

    print("💾 Metadata saved\n")


if __name__ == "__main__":
    main()
