#!/usr/bin/env python3
"""
RoboReward Video Downloader
============================

Downloads videos from HuggingFace RoboReward dataset for specific tasks
and organizes them into success/failure folders for FPS confusion experiments.

Key functionalities:
- Stream RoboReward dataset to find task-specific samples
- Filter by task instruction and dataset source (DROID, Bridge)
- Download videos from HuggingFace URLs
- Organize into structured directories (success/failure)
- Validate downloaded videos with OpenCV

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
from urllib.parse import urlparse

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_OUTPUT_DIR = "/var/scratch/pkarageo/roboreward_tasks"
DEFAULT_NUM_SUCCESS = 20
DEFAULT_NUM_FAILURE = 20
DEFAULT_REWARD_THRESHOLD = 4
DEFAULT_DATASETS = ['droid', 'bridge']

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DownloadConfig:
    """Configuration for video download."""
    task_instruction: str
    output_dir: str
    num_success: int = DEFAULT_NUM_SUCCESS
    num_failure: int = DEFAULT_NUM_FAILURE
    reward_threshold: int = DEFAULT_REWARD_THRESHOLD
    dataset_filter: List[str] = None

    def __post_init__(self):
        if self.dataset_filter is None:
            self.dataset_filter = DEFAULT_DATASETS

    def to_dict(self):
        return asdict(self)


@dataclass
class VideoMetadata:
    """Metadata for a downloaded video."""
    video_filename: str
    hf_path: str
    task: str
    reward: int
    dataset_source: str
    episode_id: str
    label: str  # "success" or "failure"

    def to_dict(self):
        return asdict(self)


# ============================================================================
# VIDEO DOWNLOADER CLASS
# ============================================================================

class RoboRewardDownloader:
    """Main class for downloading RoboReward videos."""

    def __init__(self, config: DownloadConfig):
        self.config = config
        self.task_dir = config.output_dir
        self.success_dir = os.path.join(self.task_dir, "success")
        self.failure_dir = os.path.join(self.task_dir, "failure")

        # Create directories
        os.makedirs(self.success_dir, exist_ok=True)
        os.makedirs(self.failure_dir, exist_ok=True)

        # Statistics
        self.downloaded_success = []
        self.downloaded_failure = []
        self.failed_downloads = []

    @staticmethod
    def slugify_task(task: str) -> str:
        """
        Convert task instruction to a filesystem-safe slug.

        Args:
            task: Task instruction text

        Returns:
            Slugified string
        """
        # Take first 50 chars, lowercase, replace spaces and special chars
        slug = task.lower()[:50]
        slug = re.sub(r'[^a-z0-9]+', '_', slug)
        slug = slug.strip('_')
        return slug

    @staticmethod
    def parse_hf_video_path(hf_path: str) -> Tuple[str, str]:
        """
        Parse HuggingFace video path to extract dataset source and episode ID.

        Format: hf://datasets/teetone/RoboReward@{hash}/train/{dataset}/{episode}.mp4

        Args:
            hf_path: HuggingFace path

        Returns:
            Tuple of (dataset_source, episode_id)
        """
        dataset_source = 'unknown'
        episode_id = 'unknown'

        if 'train/' in hf_path:
            match = re.search(r'/train/([^/]+)/([^/]+\.mp4)', hf_path)
            if match:
                dataset_source = match.group(1)
                episode_filename = match.group(2)
                episode_id = episode_filename.replace('.mp4', '')

        return dataset_source, episode_id

    @staticmethod
    def convert_hf_to_https(hf_path: str) -> str:
        """
        Convert HuggingFace hf:// URL to HTTPS URL.

        Example:
            hf://datasets/teetone/RoboReward@abc123/train/droid/ep0.mp4
            -> https://huggingface.co/datasets/teetone/RoboReward/resolve/abc123/train/droid/ep0.mp4

        Args:
            hf_path: HuggingFace path with hf:// protocol

        Returns:
            HTTPS URL
        """
        if not hf_path.startswith('hf://'):
            return hf_path

        # Remove hf://datasets/ prefix
        path_part = hf_path.replace('hf://datasets/', '')

        # Split on @ to get repo and hash
        if '@' in path_part:
            repo_part, hash_and_path = path_part.split('@', 1)
            # hash_and_path = "abc123/train/droid/ep0.mp4"
            hash_part, file_path = hash_and_path.split('/', 1)

            # Construct HTTPS URL
            https_url = f"https://huggingface.co/datasets/{repo_part}/resolve/{hash_part}/{file_path}"
            return https_url
        else:
            # Fallback: just replace hf:// with https://
            return hf_path.replace('hf://', 'https://')

    def download_video_from_hf(self, hf_path: str, output_path: str) -> bool:
        """
        Download video from HuggingFace to local file.

        Args:
            hf_path: HuggingFace path (hf:// format)
            output_path: Local output path

        Returns:
            True if successful, False otherwise
        """
        try:
            https_url = self.convert_hf_to_https(hf_path)

            # Stream download with progress bar
            response = requests.get(https_url, stream=True, timeout=30)
            response.raise_for_status()

            # Get total size for progress bar
            total_size = int(response.headers.get('content-length', 0))

            # Write to file
            with open(output_path, 'wb') as f:
                if total_size > 0:
                    # Progress bar for large files
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(output_path)) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                else:
                    # No progress bar if size unknown
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            return True

        except Exception as e:
            print(f"  ❌ Download failed: {e}")
            return False

    def validate_video(self, video_path: str) -> Optional[Dict]:
        """
        Validate that video can be read with OpenCV.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video info if valid, None otherwise
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            cap.release()

            # Basic validation
            if fps <= 0 or frame_count <= 0:
                return None

            return {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': frame_count / fps if fps > 0 else 0
            }

        except Exception as e:
            print(f"  ⚠️  Validation error: {e}")
            return None

    def download_task_subset(self) -> Dict:
        """
        Main method to download videos for a specific task.

        Returns:
            Dictionary with download statistics
        """
        print(f"\n{'='*80}")
        print(f"📥 Downloading Videos for Task")
        print(f"{'='*80}")
        print(f"Task: {self.config.task_instruction}")
        print(f"Target: {self.config.num_success} success + {self.config.num_failure} failure")
        print(f"Datasets: {', '.join(self.config.dataset_filter)}")
        print(f"Output: {self.task_dir}")
        print(f"{'='*80}\n")

        # Load dataset
        print("⬇️  Loading RoboReward dataset (streaming mode)...")
        dataset = load_dataset("teetone/RoboReward", split="train", streaming=True)
        dataset = dataset.cast_column("video", Value("string"))
        print("✅ Dataset loaded\n")

        # Counters
        success_count = 0
        failure_count = 0
        samples_checked = 0

        # Stream through dataset
        print("🔍 Searching for matching samples...\n")
        dataset_iter = iter(dataset)

        while (success_count < self.config.num_success or failure_count < self.config.num_failure):
            try:
                sample = next(dataset_iter)
                samples_checked += 1

                # Progress indicator
                if samples_checked % 500 == 0:
                    print(f"  Checked {samples_checked:,} samples | "
                          f"Found: {success_count} success, {failure_count} failure")

                # Filter by task
                if sample['task'] != self.config.task_instruction:
                    continue

                # Parse metadata
                dataset_source, episode_id = self.parse_hf_video_path(sample['video'])

                # Filter by dataset
                if dataset_source not in self.config.dataset_filter:
                    continue

                # Determine label
                reward = int(sample['reward'])
                if reward >= self.config.reward_threshold:
                    label = "success"
                    if success_count >= self.config.num_success:
                        continue
                else:
                    label = "failure"
                    if failure_count >= self.config.num_failure:
                        continue

                # Download video
                filename = f"{dataset_source}_{episode_id}.mp4"
                output_dir = self.success_dir if label == "success" else self.failure_dir
                output_path = os.path.join(output_dir, filename)

                print(f"\n  📥 Downloading [{label}] {filename}...")
                print(f"     Reward: {reward} | Dataset: {dataset_source}")

                success = self.download_video_from_hf(sample['video'], output_path)

                if success:
                    # Validate video
                    video_info = self.validate_video(output_path)
                    if video_info:
                        print(f"  ✅ Validated: {video_info['frame_count']} frames, "
                              f"{video_info['fps']:.1f} fps, {video_info['duration']:.1f}s")

                        # Save metadata
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
                            self.downloaded_success.append(metadata)
                            success_count += 1
                        else:
                            self.downloaded_failure.append(metadata)
                            failure_count += 1
                    else:
                        print(f"  ⚠️  Validation failed, deleting file")
                        os.remove(output_path)
                        self.failed_downloads.append(filename)
                else:
                    self.failed_downloads.append(filename)

            except StopIteration:
                print(f"\n⚠️  Reached end of dataset after {samples_checked:,} samples")
                break
            except Exception as e:
                print(f"⚠️  Error processing sample: {e}")
                continue

        # Summary
        print(f"\n{'='*80}")
        print(f"✅ DOWNLOAD COMPLETE")
        print(f"{'='*80}")
        print(f"Samples checked: {samples_checked:,}")
        print(f"Success videos: {success_count}/{self.config.num_success}")
        print(f"Failure videos: {failure_count}/{self.config.num_failure}")
        print(f"Failed downloads: {len(self.failed_downloads)}")
        print(f"{'='*80}\n")

        # Save metadata
        self.save_metadata()

        return {
            'samples_checked': samples_checked,
            'success_downloaded': success_count,
            'failure_downloaded': failure_count,
            'failed_downloads': len(self.failed_downloads)
        }

    def save_metadata(self):
        """Save metadata files to task directory."""
        print("💾 Saving metadata...")

        # Task metadata
        task_slug = self.slugify_task(self.config.task_instruction)
        dataset_dist = {}
        reward_dist = {}

        # Count dataset and reward distributions
        for metadata_list in [self.downloaded_success, self.downloaded_failure]:
            for m in metadata_list:
                dataset_dist[m.dataset_source] = dataset_dist.get(m.dataset_source, 0) + 1
                reward_dist[m.reward] = reward_dist.get(m.reward, 0) + 1

        metadata_json = {
            "task_instruction": self.config.task_instruction,
            "task_slug": task_slug,
            "download_config": self.config.to_dict(),
            "statistics": {
                "total_downloaded": len(self.downloaded_success) + len(self.downloaded_failure),
                "success_count": len(self.downloaded_success),
                "failure_count": len(self.downloaded_failure),
                "dataset_distribution": dataset_dist,
                "reward_distribution": reward_dist
            }
        }

        metadata_path = os.path.join(self.task_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata_json, f, indent=2)
        print(f"  ✅ Saved: {metadata_path}")

        # Success samples metadata
        success_metadata_path = os.path.join(self.success_dir, "samples.json")
        with open(success_metadata_path, 'w') as f:
            json.dump([m.to_dict() for m in self.downloaded_success], f, indent=2)
        print(f"  ✅ Saved: {success_metadata_path}")

        # Failure samples metadata
        failure_metadata_path = os.path.join(self.failure_dir, "samples.json")
        with open(failure_metadata_path, 'w') as f:
            json.dump([m.to_dict() for m in self.downloaded_failure], f, indent=2)
        print(f"  ✅ Saved: {failure_metadata_path}\n")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download RoboReward videos for specific tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 20 success + 20 failure videos for a task
  python download_roboreward_videos.py \\
      --task "Take the lid off the pot and place it on the plate." \\
      --task-slug pot_on_plate_task \\
      --num-success 20 \\
      --num-failure 20

  # Download from specific datasets only
  python download_roboreward_videos.py \\
      --task "rearrange pillows on sofa" \\
      --task-slug pillows_task \\
      --datasets droid bridge
        """
    )

    parser.add_argument(
        '--task',
        type=str,
        required=True,
        help='Task instruction (exact match from RoboReward dataset)'
    )

    parser.add_argument(
        '--task-slug',
        type=str,
        required=True,
        help='Filesystem-safe slug for task (e.g., "pot_on_plate_task")'
    )

    parser.add_argument(
        '--output-base',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Base output directory (default: {DEFAULT_OUTPUT_DIR})'
    )

    parser.add_argument(
        '--num-success',
        type=int,
        default=DEFAULT_NUM_SUCCESS,
        help=f'Number of success videos to download (default: {DEFAULT_NUM_SUCCESS})'
    )

    parser.add_argument(
        '--num-failure',
        type=int,
        default=DEFAULT_NUM_FAILURE,
        help=f'Number of failure videos to download (default: {DEFAULT_NUM_FAILURE})'
    )

    parser.add_argument(
        '--reward-threshold',
        type=int,
        default=DEFAULT_REWARD_THRESHOLD,
        help=f'Reward threshold for success (>= threshold = success) (default: {DEFAULT_REWARD_THRESHOLD})'
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        default=DEFAULT_DATASETS,
        help=f'Dataset sources to include (default: {" ".join(DEFAULT_DATASETS)})'
    )

    return parser.parse_args()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point."""
    args = parse_args()

    # Create download configuration
    output_dir = os.path.join(args.output_base, args.task_slug)

    config = DownloadConfig(
        task_instruction=args.task,
        output_dir=output_dir,
        num_success=args.num_success,
        num_failure=args.num_failure,
        reward_threshold=args.reward_threshold,
        dataset_filter=args.datasets
    )

    # Create downloader and run
    downloader = RoboRewardDownloader(config)
    stats = downloader.download_task_subset()

    print("\n✅ Download process complete!")
    print(f"\nNext steps:")
    print(f"1. Verify videos: ls {output_dir}/success/ && ls {output_dir}/failure/")
    print(f"2. Run augmentation matching: python match_augmentations.py --task-dir {output_dir}")
    print(f"3. Run FPS experiment with task slug: {args.task_slug}\n")


if __name__ == "__main__":
    main()
