#!/usr/bin/env python3
"""
Analyze which tasks in RoboReward have both success and failure examples
WITHOUT downloading the full dataset.
"""

import json
from typing import Dict, List
import pandas as pd
import os


def load_task_statistics(path: str) -> List[Dict]:
    """Load pre-computed task statistics."""
    with open(path, 'r') as f:
        return json.load(f)


def find_pairable_tasks(stats: List[Dict], min_success=1, min_failure=1) -> List[Dict]:
    """
    Find tasks with at least min_success AND min_failure examples.

    Returns list of dicts with:
      - task_key: "dataset__task_description"
      - dataset: source dataset name
      - task: task description
      - success_count: number of successful trajectories
      - failure_count: number of failed trajectories
      - max_pairs: min(success_count, failure_count)
    """
    pairable = []

    for task_data in stats:
        success = task_data.get('success_samples', 0)
        failure = task_data.get('failure_samples', 0)

        if success >= min_success and failure >= min_failure:
            dataset = task_data['dataset_source']
            task = task_data['task']
            task_key = f"{dataset}__{task}"

            pairable.append({
                'task_key': task_key,
                'dataset': dataset,
                'task': task,
                'success_count': success,
                'failure_count': failure,
                'max_pairs': min(success, failure),
                'total_samples': task_data['total_samples'],
                'sample_episodes': task_data.get('sample_episodes', [])
            })

    # Sort by max_pairs descending (tasks with most pairing potential)
    pairable.sort(key=lambda x: x['max_pairs'], reverse=True)
    return pairable


def main():
    stats_path = '/var/scratch/pkarageo/roboreward_dataset/all_task_statistics.json'

    print("="*80)
    print("ROBOREWARD PAIRABLE TASKS ANALYZER")
    print("="*80)

    print("\n⬇️  Loading task statistics...")
    stats = load_task_statistics(stats_path)
    print(f"✅ Loaded {len(stats)} unique task-dataset pairs")

    print("\n" + "="*80)
    print("🔍 FINDING PAIRABLE TASKS")
    print("="*80)

    # Find all tasks with at least 1 success and 1 failure
    pairable = find_pairable_tasks(stats, min_success=1, min_failure=1)

    print(f"\n✅ Found {len(pairable)} tasks with both success AND failure examples")

    # Filter for droid and bridge datasets
    droid_bridge = [t for t in pairable if t['dataset'] in ['droid', 'bridge']]
    print(f"✅ {len(droid_bridge)} from droid/bridge datasets")

    # Save all pairable tasks
    output_csv = '/var/scratch/pkarageo/roboreward_dataset/pairable_tasks_all.csv'
    df_all = pd.DataFrame(pairable)
    df_all.to_csv(output_csv, index=False)
    print(f"\n💾 Saved all pairable tasks to: {output_csv}")

    # Save droid/bridge only
    output_csv_db = '/var/scratch/pkarageo/roboreward_dataset/pairable_tasks_droid_bridge.csv'
    df_db = pd.DataFrame(droid_bridge)
    df_db.to_csv(output_csv_db, index=False)
    print(f"💾 Saved droid/bridge tasks to: {output_csv_db}")

    # Show top 30 overall
    print("\n" + "="*80)
    print("TOP 30 TASKS BY PAIRING POTENTIAL (All Datasets)")
    print("="*80)

    print(f"\n{'Dataset':<20} {'Pairs':<6} {'Success':<8} {'Failure':<8} Task")
    print("-" * 120)
    for task in pairable[:30]:
        task_truncated = task['task'][:70] + "..." if len(task['task']) > 70 else task['task']
        print(f"{task['dataset']:<20} {task['max_pairs']:<6} "
              f"{task['success_count']:<8} {task['failure_count']:<8} "
              f"{task_truncated}")

    # Show top 20 droid/bridge
    if droid_bridge:
        print("\n" + "="*80)
        print("TOP 20 TASKS (DROID/BRIDGE Only)")
        print("="*80)

        print(f"\n{'Dataset':<20} {'Pairs':<6} {'Success':<8} {'Failure':<8} Task")
        print("-" * 120)
        for task in droid_bridge[:20]:
            task_truncated = task['task'][:70] + "..." if len(task['task']) > 70 else task['task']
            print(f"{task['dataset']:<20} {task['max_pairs']:<6} "
                  f"{task['success_count']:<8} {task['failure_count']:<8} "
                  f"{task_truncated}")

    # Statistics
    print("\n" + "="*80)
    print("📊 STATISTICS")
    print("="*80)

    print("\n🌍 ALL DATASETS:")
    print(f"  Total pairable tasks: {len(pairable)}")
    print(f"  Total possible pairs: {sum(t['max_pairs'] for t in pairable)}")
    if pairable:
        print(f"  Average pairs per task: {sum(t['max_pairs'] for t in pairable) / len(pairable):.2f}")
        print(f"  Median pairs per task: {sorted([t['max_pairs'] for t in pairable])[len(pairable)//2]}")
        print(f"  Max pairs for single task: {max(t['max_pairs'] for t in pairable)}")

    print("\n🤖 DROID/BRIDGE ONLY:")
    if droid_bridge:
        print(f"  Pairable tasks: {len(droid_bridge)}")
        print(f"  Total possible pairs: {sum(t['max_pairs'] for t in droid_bridge)}")
        print(f"  Average pairs per task: {sum(t['max_pairs'] for t in droid_bridge) / len(droid_bridge):.2f}")
        print(f"  Median pairs per task: {sorted([t['max_pairs'] for t in droid_bridge])[len(droid_bridge)//2]}")
        print(f"  Max pairs for single task: {max(t['max_pairs'] for t in droid_bridge)}")
    else:
        print(f"  ⚠️  No pairable tasks found in droid/bridge datasets!")

    # Recommendations
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)

    if len(pairable) > 100:
        print("\n✅ Good news! Plenty of pairable tasks available.")
        print(f"   Recommendation: Select top 20-30 tasks and download videos for exact matching.")
        if len(droid_bridge) > 20:
            print(f"   Focus on DROID/BRIDGE datasets for consistency with current experiments.")
    elif len(pairable) > 20:
        print("\n⚠️  Moderate number of pairable tasks.")
        print(f"   Recommendation: Download all pairable task videos, supplement with semantic matching.")
    else:
        print("\n❌ Limited exact matches available.")
        print(f"   Recommendation: Primarily use semantic matching based on task similarity.")
        print(f"   Consider downloading popular tasks and using cosine similarity for the rest.")

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nReview results in:")
    print(f"  - {output_csv}")
    print(f"  - {output_csv_db}")


if __name__ == "__main__":
    main()
