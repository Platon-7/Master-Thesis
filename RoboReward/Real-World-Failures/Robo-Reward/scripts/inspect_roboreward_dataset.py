#!/usr/bin/env python3
"""
RoboReward Dataset Inspector
=============================

Analyzes the RoboReward dataset to identify tasks from specific OXE sub-datasets
(DROID, Bridge) with good success/failure balance for FPS confusion experiments.

Key functionalities:
- Extract metadata from video paths (dataset source, episode ID)
- Build task index grouped by (task, dataset_source)
- Filter tasks from DROID/Bridge datasets
- Find tasks with balanced success/failure samples (reward >= 4 vs reward < 4)

Author: Platon Karageorgis
"""

import os
import sys
import json
import pickle
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datasets import load_dataset, Value

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "/var/scratch/pkarageo/roboreward_dataset"
NUM_SAMPLES_TO_INSPECT = 45072  # FULL train split
TARGET_DATASETS = None  # None = analyze ALL datasets (29 total)
MIN_SUCCESS_SAMPLES = 5  # Very low threshold given dataset diversity
MIN_FAILURE_SAMPLES = 5  # Very low threshold given dataset diversity
REWARD_SUCCESS_THRESHOLD = 4  # reward >= 4 = success, < 4 = failure

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SampleMetadata:
    """Metadata extracted from a RoboReward sample."""
    video_path: str
    task: str
    reward: int
    dataset_source: str
    episode_id: str
    gpt5_mini_check: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class TaskStatistics:
    """Statistics for a specific task."""
    task: str
    dataset_source: str
    total_samples: int
    success_samples: int  # reward >= 4
    failure_samples: int  # reward < 4
    reward_distribution: Dict[int, int]
    sample_episodes: List[str]  # Sample episode IDs for reference

    def is_balanced(self, min_success: int = 20, min_failure: int = 20) -> bool:
        """Check if task has sufficient success and failure samples."""
        return self.success_samples >= min_success and self.failure_samples >= min_failure

    def to_dict(self):
        return {
            'task': self.task,
            'dataset_source': self.dataset_source,
            'total_samples': self.total_samples,
            'success_samples': self.success_samples,
            'failure_samples': self.failure_samples,
            'reward_distribution': self.reward_distribution,
            'sample_episodes': self.sample_episodes[:5]  # Only save first 5 for reference
        }


# ============================================================================
# DATASET INSPECTOR CLASS
# ============================================================================

class RoboRewardInspector:
    """Main class for inspecting RoboReward dataset."""

    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Statistics trackers
        self.dataset_summary = defaultdict(int)
        self.reward_distribution = defaultdict(int)
        self.task_index = defaultdict(lambda: defaultdict(list))  # {task: {dataset: [samples]}}

    @staticmethod
    def extract_metadata(sample: dict) -> SampleMetadata:
        """
        Extract metadata from a RoboReward sample.

        Video path format: hf://datasets/teetone/RoboReward@{hash}/train/{dataset_name}/{episode_id}.mp4

        Args:
            sample: Raw sample from dataset

        Returns:
            SampleMetadata with parsed information
        """
        video_path = sample.get('video', '')
        task = sample.get('task', 'unknown')
        reward = int(sample.get('reward', 0))
        gpt5_check = sample.get('gpt5_mini_check', None)

        # Parse video path to extract dataset source and episode ID
        # Example: hf://datasets/teetone/RoboReward@abc123/train/droid/episode_42.mp4
        dataset_source = 'unknown'
        episode_id = 'unknown'

        if isinstance(video_path, str) and 'train/' in video_path:
            # Extract dataset name (between train/ and next /)
            match = re.search(r'/train/([^/]+)/([^/]+\.mp4)', video_path)
            if match:
                dataset_source = match.group(1)
                episode_filename = match.group(2)
                episode_id = episode_filename.replace('.mp4', '')

        return SampleMetadata(
            video_path=video_path,
            task=task,
            reward=reward,
            dataset_source=dataset_source,
            episode_id=episode_id,
            gpt5_mini_check=gpt5_check
        )

    def build_task_index(self, num_samples: int = NUM_SAMPLES_TO_INSPECT) -> Dict:
        """
        Stream dataset and build index of tasks grouped by dataset source.

        Args:
            num_samples: Number of samples to inspect from the dataset

        Returns:
            Task index dictionary
        """
        print(f"\n{'='*80}")
        print(f"📊 Building Task Index from {num_samples:,} samples...")
        print(f"{'='*80}\n")

        # Load dataset with streaming
        print("⬇️  Loading RoboReward dataset (streaming mode)...")
        dataset = load_dataset("teetone/RoboReward", split="train", streaming=True)

        # Cast video column to string to avoid torchcodec issues
        dataset = dataset.cast_column("video", Value("string"))
        print("✅ Dataset loaded (video column cast to string)\n")

        # Stream and process samples
        dataset_iter = iter(dataset)
        for i in range(num_samples):
            try:
                sample = next(dataset_iter)
                metadata = self.extract_metadata(sample)

                # Update statistics
                self.dataset_summary[metadata.dataset_source] += 1
                self.reward_distribution[metadata.reward] += 1

                # Add to task index
                self.task_index[metadata.task][metadata.dataset_source].append(metadata)

                # Progress indicator
                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i+1:,} samples...")

            except StopIteration:
                print(f"\n⚠️  Reached end of dataset at {i:,} samples")
                break
            except Exception as e:
                print(f"⚠️  Error processing sample {i}: {e}")
                continue

        print(f"\n✅ Task index built from {i+1:,} samples")
        return dict(self.task_index)

    def filter_by_datasets(self, datasets: List[str] = TARGET_DATASETS) -> Dict:
        """
        Filter task index to only include specific datasets.

        Args:
            datasets: List of dataset names to keep (e.g., ['droid', 'bridge']).
                     If None, keep all datasets.

        Returns:
            Filtered task index
        """
        if datasets is None:
            print(f"\n{'='*80}")
            print(f"🔍 Analyzing ALL datasets (no filtering)")
            print(f"{'='*80}\n")
            return dict(self.task_index)

        print(f"\n{'='*80}")
        print(f"🔍 Filtering for datasets: {', '.join(datasets)}")
        print(f"{'='*80}\n")

        filtered_index = defaultdict(lambda: defaultdict(list))

        for task, dataset_dict in self.task_index.items():
            for dataset_name in datasets:
                if dataset_name in dataset_dict:
                    filtered_index[task][dataset_name] = dataset_dict[dataset_name]

        # Convert to regular dict
        filtered_index = {task: dict(ds_dict) for task, ds_dict in filtered_index.items()}

        print(f"✅ Filtered to {len(filtered_index)} tasks from target datasets\n")
        return filtered_index

    def compute_task_statistics(
        self,
        task_index: Dict,
        reward_threshold: int = REWARD_SUCCESS_THRESHOLD
    ) -> List[TaskStatistics]:
        """
        Compute statistics for each task in the index.

        Args:
            task_index: Task index to analyze
            reward_threshold: Threshold for success (>= threshold = success)

        Returns:
            List of TaskStatistics objects
        """
        print(f"\n{'='*80}")
        print(f"📈 Computing Task Statistics")
        print(f"   Success threshold: reward >= {reward_threshold}")
        print(f"{'='*80}\n")

        all_stats = []

        for task, dataset_dict in task_index.items():
            for dataset_source, samples in dataset_dict.items():
                # Count successes and failures
                success_count = sum(1 for s in samples if s.reward >= reward_threshold)
                failure_count = sum(1 for s in samples if s.reward < reward_threshold)

                # Reward distribution
                reward_dist = defaultdict(int)
                for s in samples:
                    reward_dist[s.reward] += 1

                # Episode IDs
                episode_ids = [s.episode_id for s in samples]

                stats = TaskStatistics(
                    task=task,
                    dataset_source=dataset_source,
                    total_samples=len(samples),
                    success_samples=success_count,
                    failure_samples=failure_count,
                    reward_distribution=dict(reward_dist),
                    sample_episodes=episode_ids
                )

                all_stats.append(stats)

        # Sort by total samples (descending)
        all_stats.sort(key=lambda x: x.total_samples, reverse=True)

        print(f"✅ Computed statistics for {len(all_stats)} task-dataset pairs\n")
        return all_stats

    def find_balanced_tasks(
        self,
        task_stats: List[TaskStatistics],
        min_success: int = MIN_SUCCESS_SAMPLES,
        min_failure: int = MIN_FAILURE_SAMPLES
    ) -> List[TaskStatistics]:
        """
        Find tasks with sufficient success and failure samples.

        Args:
            task_stats: List of task statistics
            min_success: Minimum number of success samples required
            min_failure: Minimum number of failure samples required

        Returns:
            List of balanced tasks
        """
        print(f"\n{'='*80}")
        print(f"🎯 Finding Balanced Tasks")
        print(f"   Min success samples: {min_success}")
        print(f"   Min failure samples: {min_failure}")
        print(f"{'='*80}\n")

        balanced = [stat for stat in task_stats if stat.is_balanced(min_success, min_failure)]

        print(f"✅ Found {len(balanced)} balanced tasks\n")
        return balanced

    def print_summary(self):
        """Print summary statistics of the dataset inspection."""
        print(f"\n{'='*80}")
        print("📋 DATASET SUMMARY")
        print(f"{'='*80}\n")

        # Dataset distribution
        print("Dataset Distribution:")
        total = sum(self.dataset_summary.values())
        for dataset, count in sorted(self.dataset_summary.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {dataset:30s}: {count:6,} ({percentage:5.1f}%)")
        print(f"  {'TOTAL':30s}: {total:6,}\n")

        # Reward distribution
        print("Reward Distribution:")
        for reward in sorted(self.reward_distribution.keys()):
            count = self.reward_distribution[reward]
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  Reward {reward}: {count:6,} ({percentage:5.1f}%)")
        print()

    def save_results(
        self,
        task_index: Dict,
        task_stats: List[TaskStatistics],
        balanced_tasks: List[TaskStatistics]
    ):
        """Save all results to disk."""
        print(f"\n{'='*80}")
        print(f"💾 Saving Results to {self.output_dir}")
        print(f"{'='*80}\n")

        # 1. Dataset summary
        summary_path = os.path.join(self.output_dir, "dataset_summary.json")
        summary_data = {
            "total_samples_inspected": sum(self.dataset_summary.values()),
            "dataset_distribution": dict(self.dataset_summary),
            "reward_distribution": dict(self.reward_distribution),
            "total_unique_tasks": len(task_index),
            "balanced_tasks_count": len(balanced_tasks)
        }
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"✅ Saved dataset summary: {summary_path}")

        # 2. Task index (pickle for Python objects)
        task_index_path = os.path.join(self.output_dir, "task_index.pkl")
        with open(task_index_path, 'wb') as f:
            pickle.dump(task_index, f)
        print(f"✅ Saved task index: {task_index_path}")

        # 3. Task statistics (all)
        stats_path = os.path.join(self.output_dir, "all_task_statistics.json")
        stats_data = [stat.to_dict() for stat in task_stats]
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
        print(f"✅ Saved all task statistics: {stats_path}")

        # 4. Balanced tasks CSV (human-readable)
        csv_filename = "all_datasets_tasks.csv" if TARGET_DATASETS is None else "droid_bridge_tasks.csv"
        csv_path = os.path.join(self.output_dir, csv_filename)
        with open(csv_path, 'w') as f:
            # Header
            f.write("task,dataset_source,total_samples,success_samples,failure_samples,reward_1,reward_2,reward_3,reward_4,reward_5\n")

            # Rows
            for stat in balanced_tasks:
                rd = stat.reward_distribution
                f.write(f'"{stat.task}",{stat.dataset_source},{stat.total_samples},'
                       f'{stat.success_samples},{stat.failure_samples},'
                       f'{rd.get(1,0)},{rd.get(2,0)},{rd.get(3,0)},{rd.get(4,0)},{rd.get(5,0)}\n')
        print(f"✅ Saved balanced tasks CSV: {csv_path}")

        # 5. Balanced tasks JSON (for programmatic use)
        balanced_json_path = os.path.join(self.output_dir, "balanced_tasks.json")
        balanced_data = [stat.to_dict() for stat in balanced_tasks]
        with open(balanced_json_path, 'w') as f:
            json.dump(balanced_data, f, indent=2)
        print(f"✅ Saved balanced tasks JSON: {balanced_json_path}\n")

    def print_balanced_tasks_preview(self, balanced_tasks: List[TaskStatistics], n: int = 10):
        """Print a preview of the top N balanced tasks."""
        print(f"\n{'='*80}")
        print(f"🏆 TOP {n} BALANCED TASKS (by total samples)")
        print(f"{'='*80}\n")

        for i, stat in enumerate(balanced_tasks[:n], 1):
            print(f"{i}. [{stat.dataset_source}] {stat.task[:60]}")
            print(f"   Total: {stat.total_samples} | Success: {stat.success_samples} | Failure: {stat.failure_samples}")
            print(f"   Reward dist: {stat.reward_distribution}")
            print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point."""

    print("\n" + "="*80)
    print("🤖 RoboReward Dataset Inspector")
    print("="*80)

    # Initialize inspector
    inspector = RoboRewardInspector(output_dir=OUTPUT_DIR)

    # Step 1: Build task index
    task_index = inspector.build_task_index(num_samples=NUM_SAMPLES_TO_INSPECT)

    # Step 2: Filter for target datasets
    filtered_index = inspector.filter_by_datasets(datasets=TARGET_DATASETS)

    # Step 3: Compute statistics
    task_stats = inspector.compute_task_statistics(filtered_index)

    # Step 4: Find balanced tasks
    balanced_tasks = inspector.find_balanced_tasks(
        task_stats,
        min_success=MIN_SUCCESS_SAMPLES,
        min_failure=MIN_FAILURE_SAMPLES
    )

    # Step 5: Print summary
    inspector.print_summary()
    inspector.print_balanced_tasks_preview(balanced_tasks, n=10)

    # Step 6: Save results
    inspector.save_results(filtered_index, task_stats, balanced_tasks)

    print("="*80)
    print("✅ INSPECTION COMPLETE")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Review balanced tasks: {os.path.join(OUTPUT_DIR, 'droid_bridge_tasks.csv')}")
    print(f"2. Select 2-3 tasks for video download")
    print(f"3. Run download_roboreward_videos.py with selected tasks\n")


if __name__ == "__main__":
    main()
