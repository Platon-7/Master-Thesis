#!/usr/bin/env python3
"""
RoboReward Augmentation Matcher
================================

Creates heuristic links between negative samples and potential source demonstrations
in RoboReward dataset downloads.

Since RoboReward lacks explicit augmentation metadata, this script uses heuristics:
1. Task + dataset source matching
2. Episode ID proximity analysis
3. Temporal pattern detection (if available)

Each match is assigned a confidence score based on the evidence strength.

Author: Platon Karageorgis
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class VideoSample:
    """Represents a single video sample."""
    video_filename: str
    hf_path: str
    task: str
    reward: int
    dataset_source: str
    episode_id: str
    label: str  # "success" or "failure"

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)


@dataclass
class Match:
    """Represents a match between success and failure samples."""
    match_id: str
    confidence: str  # "low", "medium", "high"
    success_samples: List[Dict]
    failure_samples: List[Dict]
    reasoning: str
    metadata: Optional[Dict] = None

    def to_dict(self):
        return {
            'match_id': self.match_id,
            'confidence': self.confidence,
            'success_samples': self.success_samples,
            'failure_samples': self.failure_samples,
            'reasoning': self.reasoning,
            'metadata': self.metadata or {}
        }


# ============================================================================
# AUGMENTATION MATCHER CLASS
# ============================================================================

class AugmentationMatcher:
    """Main class for heuristic augmentation matching."""

    def __init__(self, task_dir: str):
        self.task_dir = task_dir
        self.success_dir = os.path.join(task_dir, "success")
        self.failure_dir = os.path.join(task_dir, "failure")

        # Load metadata
        self.metadata = self._load_metadata()
        self.success_samples = self._load_samples("success")
        self.failure_samples = self._load_samples("failure")

        # Matches
        self.matches = []

    def _load_metadata(self) -> dict:
        """Load task metadata."""
        metadata_path = os.path.join(self.task_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            return json.load(f)

    def _load_samples(self, label: str) -> List[VideoSample]:
        """Load samples from success/failure directories."""
        samples_dir = self.success_dir if label == "success" else self.failure_dir
        samples_path = os.path.join(samples_dir, "samples.json")

        if not os.path.exists(samples_path):
            print(f"⚠️  Warning: No samples.json found in {samples_dir}")
            return []

        with open(samples_path, 'r') as f:
            data = json.load(f)

        return [VideoSample.from_dict(d) for d in data]

    @staticmethod
    def extract_episode_number(episode_id: str) -> Optional[int]:
        """
        Extract numeric episode number from episode ID.

        Examples:
            "episode_42" -> 42
            "ep_123" -> 123
            "042" -> 42
            "trajectory_5" -> 5

        Returns:
            Episode number or None if not found
        """
        # Try to find numbers in the episode ID
        numbers = re.findall(r'\d+', episode_id)
        if numbers:
            return int(numbers[0])
        return None

    def match_by_task_and_dataset(self) -> List[Match]:
        """
        Level 1 matching: Group samples by dataset source.

        All samples already share the same task (from download filter).
        This groups by dataset source and creates basic matches.

        Confidence: LOW (only task + dataset match)
        """
        print(f"\n{'='*80}")
        print("🔍 Level 1: Task + Dataset Matching")
        print(f"{'='*80}\n")

        # Group samples by dataset
        success_by_dataset = defaultdict(list)
        failure_by_dataset = defaultdict(list)

        for sample in self.success_samples:
            success_by_dataset[sample.dataset_source].append(sample)

        for sample in self.failure_samples:
            failure_by_dataset[sample.dataset_source].append(sample)

        # Create matches for each dataset
        matches = []
        for dataset in set(list(success_by_dataset.keys()) + list(failure_by_dataset.keys())):
            success_list = success_by_dataset.get(dataset, [])
            failure_list = failure_by_dataset.get(dataset, [])

            if success_list and failure_list:
                match = Match(
                    match_id=f"{dataset}_group",
                    confidence="low",
                    success_samples=[{
                        'video_path': f"success/{s.video_filename}",
                        'reward': s.reward,
                        'episode_id': s.episode_id
                    } for s in success_list],
                    failure_samples=[{
                        'video_path': f"failure/{f.video_filename}",
                        'reward': f.reward,
                        'episode_id': f.episode_id
                    } for f in failure_list],
                    reasoning=f"Same dataset ({dataset}), same task, different episodes",
                    metadata={
                        'dataset': dataset,
                        'num_success': len(success_list),
                        'num_failure': len(failure_list)
                    }
                )
                matches.append(match)
                print(f"  ✅ {dataset}: {len(success_list)} success ↔ {len(failure_list)} failure")

        print(f"\n📊 Created {len(matches)} dataset-level matches\n")
        return matches

    def match_by_episode_proximity(self, matches: List[Match]) -> List[Match]:
        """
        Level 2 matching: Analyze episode ID proximity.

        Looks for patterns where episode numbers are close together,
        which might indicate related samples.

        Upgrades confidence to MEDIUM if proximity patterns detected.
        """
        print(f"\n{'='*80}")
        print("🔍 Level 2: Episode Proximity Analysis")
        print(f"{'='*80}\n")

        refined_matches = []

        for match in matches:
            # Extract episode numbers
            success_episodes = []
            failure_episodes = []

            for s in match.success_samples:
                ep_num = self.extract_episode_number(s['episode_id'])
                if ep_num is not None:
                    success_episodes.append(ep_num)

            for f in match.failure_samples:
                ep_num = self.extract_episode_number(f['episode_id'])
                if ep_num is not None:
                    failure_episodes.append(ep_num)

            # Analyze proximity
            if success_episodes and failure_episodes:
                success_range = (min(success_episodes), max(success_episodes))
                failure_range = (min(failure_episodes), max(failure_episodes))

                # Check if ranges overlap or are close
                gap = min(
                    abs(success_range[0] - failure_range[1]),
                    abs(failure_range[0] - success_range[1])
                )

                # If ranges overlap or are very close, upgrade confidence
                if gap <= 50:  # Arbitrary threshold
                    match.confidence = "medium"
                    match.reasoning += f" | Episode proximity: success [{success_range[0]}-{success_range[1]}], failure [{failure_range[0]}-{failure_range[1]}], gap={gap}"
                    match.metadata['episode_proximity'] = {
                        'success_range': success_range,
                        'failure_range': failure_range,
                        'gap': gap
                    }
                    print(f"  ⬆️  Upgraded {match.match_id} to MEDIUM (episode gap: {gap})")
                else:
                    print(f"  ➡️  {match.match_id} remains LOW (episode gap: {gap})")
            else:
                print(f"  ⚠️  {match.match_id}: Could not extract episode numbers")

            refined_matches.append(match)

        print()
        return refined_matches

    def detect_temporal_patterns(self, matches: List[Match]) -> List[Match]:
        """
        Level 3 matching: Detect temporal clipping patterns.

        Looks for filename patterns that suggest temporal augmentation:
        - episode_X vs episode_X_clip
        - episode_X vs episode_X_temporal
        - Similar patterns

        Upgrades confidence to HIGH if clear patterns found.
        """
        print(f"\n{'='*80}")
        print("🔍 Level 3: Temporal Pattern Detection")
        print(f"{'='*80}\n")

        refined_matches = []

        for match in matches:
            found_patterns = []

            # Check each success-failure pair for naming patterns
            for s in match.success_samples:
                s_base = s['episode_id']
                for f in match.failure_samples:
                    f_base = f['episode_id']

                    # Check various patterns
                    if s_base in f_base or f_base in s_base:
                        # One is substring of the other (e.g., "ep1" and "ep1_clip")
                        found_patterns.append(f"{s_base} ↔ {f_base}")
                    elif s_base.replace('_clip', '') == f_base or f_base.replace('_clip', '') == s_base:
                        # Clip suffix pattern
                        found_patterns.append(f"{s_base} ↔ {f_base} (clip pattern)")

            if found_patterns:
                match.confidence = "high"
                match.reasoning += f" | Temporal patterns detected: {len(found_patterns)} pairs"
                match.metadata['temporal_patterns'] = found_patterns[:5]  # Save first 5 examples
                print(f"  🎯 {match.match_id} upgraded to HIGH ({len(found_patterns)} patterns)")
            else:
                print(f"  ➡️  {match.match_id}: No temporal patterns detected")

            refined_matches.append(match)

        print()
        return refined_matches

    def run_matching(self) -> List[Match]:
        """
        Run all matching levels sequentially.

        Returns:
            List of matches with confidence scores
        """
        print(f"\n{'='*80}")
        print("🤖 RoboReward Augmentation Matcher")
        print(f"{'='*80}")
        print(f"Task: {self.metadata['task_instruction']}")
        print(f"Success samples: {len(self.success_samples)}")
        print(f"Failure samples: {len(self.failure_samples)}")
        print(f"{'='*80}")

        # Level 1: Basic matching by dataset
        matches = self.match_by_task_and_dataset()

        # Level 2: Episode proximity analysis
        matches = self.match_by_episode_proximity(matches)

        # Level 3: Temporal pattern detection
        matches = self.detect_temporal_patterns(matches)

        self.matches = matches
        return matches

    def save_results(self):
        """Save matching results to JSON file."""
        print(f"\n{'='*80}")
        print("💾 Saving Matching Results")
        print(f"{'='*80}\n")

        # Count by confidence
        confidence_counts = defaultdict(int)
        for match in self.matches:
            confidence_counts[match.confidence] += 1

        # Build output data
        output_data = {
            "task": self.metadata['task_instruction'],
            "task_slug": self.metadata.get('task_slug', 'unknown'),
            "matching_method": "task_dataset_proximity",
            "summary": {
                "total_matches": len(self.matches),
                "confidence_distribution": dict(confidence_counts),
                "success_samples": len(self.success_samples),
                "failure_samples": len(self.failure_samples)
            },
            "matches": [match.to_dict() for match in self.matches],
            "limitations": [
                "No explicit linking metadata in RoboReward dataset",
                "Matching based on heuristics with confidence scores",
                "Cannot guarantee true augmentation relationships",
                "Episode proximity threshold (50) is arbitrary and may need tuning",
                "Temporal patterns may be rare in this dataset"
            ],
            "methodology": {
                "level_1": "Task + dataset source matching (confidence: LOW)",
                "level_2": "Episode ID proximity analysis (upgrade to MEDIUM if gap <= 50)",
                "level_3": "Temporal filename pattern detection (upgrade to HIGH if found)"
            }
        }

        # Save to file
        output_path = os.path.join(self.task_dir, "matching_info.json")
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"✅ Saved: {output_path}")
        print(f"\n📊 Confidence Distribution:")
        for conf, count in sorted(confidence_counts.items()):
            print(f"  {conf.upper():10s}: {count} matches")
        print()

    def print_summary(self):
        """Print a summary of matching results."""
        print(f"\n{'='*80}")
        print("📋 MATCHING SUMMARY")
        print(f"{'='*80}\n")

        for i, match in enumerate(self.matches, 1):
            print(f"{i}. {match.match_id} (Confidence: {match.confidence.upper()})")
            print(f"   Success: {len(match.success_samples)} samples")
            print(f"   Failure: {len(match.failure_samples)} samples")
            print(f"   Reasoning: {match.reasoning[:100]}...")
            print()


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Match augmentations in downloaded RoboReward videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python match_augmentations.py \\
      --task-dir /var/scratch/pkarageo/roboreward_tasks/pot_on_plate_task
        """
    )

    parser.add_argument(
        '--task-dir',
        type=str,
        required=True,
        help='Task directory containing success/failure folders'
    )

    return parser.parse_args()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point."""
    args = parse_args()

    # Validate task directory
    if not os.path.exists(args.task_dir):
        print(f"❌ Error: Task directory not found: {args.task_dir}")
        sys.exit(1)

    # Create matcher and run
    matcher = AugmentationMatcher(args.task_dir)
    matches = matcher.run_matching()

    # Print summary
    matcher.print_summary()

    # Save results
    matcher.save_results()

    print("="*80)
    print("✅ MATCHING COMPLETE")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Review matches: cat {args.task_dir}/matching_info.json")
    print(f"2. Run FPS experiment with this task")
    print(f"3. Analyze confidence scores to assess match quality\n")


if __name__ == "__main__":
    main()
