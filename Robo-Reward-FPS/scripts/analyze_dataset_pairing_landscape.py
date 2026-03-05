#!/usr/bin/env python3
"""
Comprehensive analysis of pairable tasks across ALL RoboReward subdatasets.
Shows which datasets have pairing potential and which don't.
"""

import json
from typing import Dict, List
from collections import defaultdict
import pandas as pd


def load_task_statistics(path: str) -> List[Dict]:
    """Load pre-computed task statistics."""
    with open(path, 'r') as f:
        return json.load(f)


def analyze_by_dataset(stats: List[Dict], min_success=1, min_failure=1) -> Dict:
    """
    Analyze pairing potential per dataset.

    Returns dict with dataset-level statistics.
    """
    dataset_stats = defaultdict(lambda: {
        'total_tasks': 0,
        'pairable_tasks': 0,
        'total_samples': 0,
        'success_samples': 0,
        'failure_samples': 0,
        'total_pairs': 0,
        'tasks': [],
        'success_only_tasks': 0,
        'failure_only_tasks': 0,
        'mixed_tasks': 0  # Tasks with both but not enough for pairing
    })

    for task_data in stats:
        dataset = task_data['dataset_source']
        success = task_data.get('success_samples', 0)
        failure = task_data.get('failure_samples', 0)
        total = task_data['total_samples']

        # Update dataset stats
        dataset_stats[dataset]['total_tasks'] += 1
        dataset_stats[dataset]['total_samples'] += total
        dataset_stats[dataset]['success_samples'] += success
        dataset_stats[dataset]['failure_samples'] += failure

        # Categorize task
        if success >= min_success and failure >= min_failure:
            # Pairable: has both success and failure
            max_pairs = min(success, failure)
            dataset_stats[dataset]['pairable_tasks'] += 1
            dataset_stats[dataset]['total_pairs'] += max_pairs
            dataset_stats[dataset]['tasks'].append({
                'task': task_data['task'],
                'success': success,
                'failure': failure,
                'pairs': max_pairs
            })
        elif success > 0 and failure == 0:
            # Success-only task
            dataset_stats[dataset]['success_only_tasks'] += 1
        elif success == 0 and failure > 0:
            # Failure-only task
            dataset_stats[dataset]['failure_only_tasks'] += 1
        elif success > 0 and failure > 0:
            # Has both but not enough for pairing threshold
            dataset_stats[dataset]['mixed_tasks'] += 1

    return dict(dataset_stats)


def main():
    stats_path = '/var/scratch/pkarageo/roboreward_dataset/all_task_statistics.json'

    print("="*100)
    print("ROBOREWARD PAIRING LANDSCAPE ANALYSIS")
    print("="*100)

    print("\n⬇️  Loading task statistics...")
    stats = load_task_statistics(stats_path)
    print(f"✅ Loaded {len(stats)} unique task-dataset pairs")

    print("\n🔍 Analyzing pairing potential per dataset...")
    dataset_analysis = analyze_by_dataset(stats, min_success=1, min_failure=1)

    # Sort datasets by pairable tasks (descending)
    sorted_datasets = sorted(
        dataset_analysis.items(),
        key=lambda x: x[1]['pairable_tasks'],
        reverse=True
    )

    print("\n" + "="*100)
    print("📊 PAIRING POTENTIAL BY DATASET")
    print("="*100)

    print(f"\n{'Dataset':<45} {'Tasks':<7} {'Pairable':<9} {'Pairs':<7} {'Status':<20}")
    print("-" * 100)

    for dataset_name, data in sorted_datasets:
        # Determine dataset status
        has_successes = data['success_samples'] > 0
        has_failures = data['failure_samples'] > 0
        has_pairable = data['pairable_tasks'] > 0

        if has_pairable:
            status = f"✅ {data['pairable_tasks']} pairable"
        elif has_successes and has_failures:
            status = "⚠️  Mixed, no pairs"
        elif has_successes and not has_failures:
            status = "❌ Success-only"
        elif has_failures and not has_successes:
            status = "❌ Failure-only"
        else:
            status = "❓ No data"

        print(f"{dataset_name:<45} {data['total_tasks']:<7} "
              f"{data['pairable_tasks']:<9} {data['total_pairs']:<7} {status:<20}")

    # Summary statistics
    print("\n" + "="*100)
    print("📈 SUMMARY STATISTICS")
    print("="*100)

    total_datasets = len(dataset_analysis)
    datasets_with_pairs = sum(1 for d in dataset_analysis.values() if d['pairable_tasks'] > 0)
    datasets_without_pairs = total_datasets - datasets_with_pairs

    total_pairable_tasks = sum(d['pairable_tasks'] for d in dataset_analysis.values())
    total_pairs = sum(d['total_pairs'] for d in dataset_analysis.values())

    print(f"\n🌍 Overall:")
    print(f"  Total datasets: {total_datasets}")
    print(f"  Datasets with pairable tasks: {datasets_with_pairs} ({datasets_with_pairs/total_datasets*100:.1f}%)")
    print(f"  Datasets without pairable tasks: {datasets_without_pairs} ({datasets_without_pairs/total_datasets*100:.1f}%)")
    print(f"  Total pairable tasks: {total_pairable_tasks}")
    print(f"  Total possible pairs: {total_pairs}")

    # Top datasets with pairing potential
    print("\n" + "="*100)
    print("🏆 TOP 10 DATASETS BY PAIRING POTENTIAL")
    print("="*100)

    top_10 = sorted_datasets[:10]
    print(f"\n{'Rank':<6} {'Dataset':<45} {'Pairable Tasks':<15} {'Total Pairs':<12} {'% Pairable':<12}")
    print("-" * 100)

    for rank, (dataset_name, data) in enumerate(top_10, 1):
        pairable_pct = (data['pairable_tasks'] / data['total_tasks'] * 100) if data['total_tasks'] > 0 else 0
        print(f"{rank:<6} {dataset_name:<45} {data['pairable_tasks']:<15} "
              f"{data['total_pairs']:<12} {pairable_pct:>10.1f}%")

    # Datasets WITHOUT any pairable tasks - categorize WHY
    print("\n" + "="*100)
    print("❌ DATASETS WITH ZERO PAIRABLE TASKS (Categorized by Problem)")
    print("="*100)

    zero_pair_datasets = [(name, data) for name, data in sorted_datasets if data['pairable_tasks'] == 0]

    if zero_pair_datasets:
        # Categorize the problems
        success_only = []
        failure_only = []
        mixed_no_pairs = []

        for dataset_name, data in zero_pair_datasets:
            has_successes = data['success_samples'] > 0
            has_failures = data['failure_samples'] > 0

            if has_successes and not has_failures:
                success_only.append((dataset_name, data))
            elif has_failures and not has_successes:
                failure_only.append((dataset_name, data))
            elif has_successes and has_failures:
                mixed_no_pairs.append((dataset_name, data))

        print(f"\n🔴 Problem Type 1: SUCCESS-ONLY datasets ({len(success_only)} datasets)")
        print("    → No failures at all, everything is successful")
        if success_only:
            print(f"\n    {'Dataset':<50} {'Tasks':<8} {'Samples':<9}")
            print("    " + "-" * 70)
            for dataset_name, data in success_only[:10]:
                print(f"    {dataset_name:<50} {data['total_tasks']:<8} {data['success_samples']:<9}")
            if len(success_only) > 10:
                print(f"    ... and {len(success_only) - 10} more")

        print(f"\n🔴 Problem Type 2: FAILURE-ONLY datasets ({len(failure_only)} datasets)")
        print("    → No successes at all, everything fails")
        if failure_only:
            print(f"\n    {'Dataset':<50} {'Tasks':<8} {'Samples':<9}")
            print("    " + "-" * 70)
            for dataset_name, data in failure_only[:10]:
                print(f"    {dataset_name:<50} {data['total_tasks']:<8} {data['failure_samples']:<9}")
            if len(failure_only) > 10:
                print(f"    ... and {len(failure_only) - 10} more")

        print(f"\n🟡 Problem Type 3: MIXED but NO TASK-LEVEL PAIRS ({len(mixed_no_pairs)} datasets)")
        print("    → Has both successes and failures overall, but no single task has both")
        print("    → Solution: Semantic matching required")
        if mixed_no_pairs:
            print(f"\n    {'Dataset':<50} {'Tasks':<8} {'Success':<9} {'Failure':<9}")
            print("    " + "-" * 80)
            for dataset_name, data in mixed_no_pairs:
                print(f"    {dataset_name:<50} {data['total_tasks']:<8} "
                      f"{data['success_samples']:<9} {data['failure_samples']:<9}")

        print(f"\n📊 Summary:")
        print(f"    Total datasets with no pairs: {len(zero_pair_datasets)}")
        print(f"      - Success-only: {len(success_only)}")
        print(f"      - Failure-only: {len(failure_only)}")
        print(f"      - Mixed but no task pairs: {len(mixed_no_pairs)}")
    else:
        print("\n✅ All datasets have at least some pairable tasks!")

    # Example tasks from top datasets
    print("\n" + "="*100)
    print("📋 EXAMPLE PAIRABLE TASKS (Top 3 Datasets)")
    print("="*100)

    for rank, (dataset_name, data) in enumerate(top_10[:3], 1):
        print(f"\n{rank}. {dataset_name} ({data['pairable_tasks']} pairable tasks)")
        print("-" * 100)

        # Show top 5 tasks by pair count
        top_tasks = sorted(data['tasks'], key=lambda x: x['pairs'], reverse=True)[:5]

        for i, task in enumerate(top_tasks, 1):
            task_truncated = task['task'][:75] + "..." if len(task['task']) > 75 else task['task']
            print(f"  {i}. {task_truncated}")
            print(f"     Success: {task['success']}, Failure: {task['failure']}, Max Pairs: {task['pairs']}")

    # Save comprehensive results
    print("\n" + "="*100)
    print("💾 SAVING RESULTS")
    print("="*100)

    # Save dataset-level summary
    dataset_summary = []
    for dataset_name, data in sorted_datasets:
        pairable_pct = (data['pairable_tasks'] / data['total_tasks'] * 100) if data['total_tasks'] > 0 else 0

        # Determine problem type
        has_successes = data['success_samples'] > 0
        has_failures = data['failure_samples'] > 0
        has_pairable = data['pairable_tasks'] > 0

        if has_pairable:
            problem_type = "pairable"
        elif has_successes and has_failures:
            problem_type = "mixed_no_pairs"
        elif has_successes and not has_failures:
            problem_type = "success_only"
        elif has_failures and not has_successes:
            problem_type = "failure_only"
        else:
            problem_type = "no_data"

        dataset_summary.append({
            'dataset': dataset_name,
            'total_tasks': data['total_tasks'],
            'pairable_tasks': data['pairable_tasks'],
            'pairable_pct': pairable_pct,
            'total_pairs': data['total_pairs'],
            'total_samples': data['total_samples'],
            'success_samples': data['success_samples'],
            'failure_samples': data['failure_samples'],
            'success_only_tasks': data['success_only_tasks'],
            'failure_only_tasks': data['failure_only_tasks'],
            'problem_type': problem_type
        })

    df_summary = pd.DataFrame(dataset_summary)
    summary_path = '/var/scratch/pkarageo/roboreward_dataset/dataset_pairing_summary.csv'
    df_summary.to_csv(summary_path, index=False)
    print(f"✅ Dataset summary: {summary_path}")

    # Save detailed pairable tasks per dataset
    detailed_path = '/var/scratch/pkarageo/roboreward_dataset/pairable_tasks_by_dataset.json'
    detailed_data = {
        dataset_name: {
            'stats': {
                'total_tasks': data['total_tasks'],
                'pairable_tasks': data['pairable_tasks'],
                'total_pairs': data['total_pairs']
            },
            'tasks': data['tasks']
        }
        for dataset_name, data in dataset_analysis.items()
        if data['pairable_tasks'] > 0  # Only save datasets with pairable tasks
    }

    with open(detailed_path, 'w') as f:
        json.dump(detailed_data, f, indent=2)
    print(f"✅ Detailed tasks: {detailed_path}")

    print("\n" + "="*100)
    print("✅ ANALYSIS COMPLETE")
    print("="*100)

    print("\n💡 Key Insights:")
    if datasets_with_pairs > 0:
        top_dataset = sorted_datasets[0]
        print(f"  • Best dataset for pairing: {top_dataset[0]} ({top_dataset[1]['pairable_tasks']} tasks, {top_dataset[1]['total_pairs']} pairs)")

    if datasets_without_pairs > 0:
        print(f"  • {datasets_without_pairs} datasets have NO pairable tasks (only successes OR failures, not both)")

    print(f"  • Total pairing potential across all datasets: {total_pairs} pairs")

    print("\n📁 Review results in:")
    print(f"  - {summary_path}")
    print(f"  - {detailed_path}")


if __name__ == "__main__":
    main()
