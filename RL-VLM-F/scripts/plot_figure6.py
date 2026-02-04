#!/usr/bin/env python3
"""Plot VLM accuracy vs. query difficulty (Figure 6) from vlm_query_log.csv."""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


QueryEntry = Tuple[float, float, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot stacked accuracy histogram from VLM query logs"
    )
    parser.add_argument(
        "log_csv",
        type=Path,
        help="Path to vlm_query_log.csv logged during training",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figure6_query_accuracy.png"),
        help="Output PNG file (default: figure6_query_accuracy.png)",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=10,
        help="Number of difficulty bins (default: 10)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Sweep Into – VLM Accuracy vs. Difficulty",
        help="Plot title",
    )
    return parser.parse_args()


def load_entries(csv_path: Path) -> List[QueryEntry]:
    entries: List[QueryEntry] = []
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"ground_truth_score_1", "ground_truth_score_2", "vlm_predicted_label"}
        if not required.issubset(reader.fieldnames or {}):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")
        for row in reader:
            try:
                score1 = float(row["ground_truth_score_1"])
                score2 = float(row["ground_truth_score_2"])
                label = int(float(row["vlm_predicted_label"]))
            except (TypeError, ValueError):
                continue
            entries.append((score1, score2, label))
    if not entries:
        raise ValueError(f"No valid rows found in {csv_path}")
    return entries


def determine_category(score1: float, score2: float, label: int) -> str:
    if label == -1:
        return "no_preference"
    if score1 == score2:
        return "correct" if label == -1 else "incorrect"
    gt_label = 0 if score1 > score2 else 1
    return "correct" if label == gt_label else "incorrect"


def compute_histogram(entries: List[QueryEntry], num_bins: int):
    diffs = [abs(s1 - s2) for s1, s2, _ in entries]
    max_diff = max(diffs)
    if max_diff == 0:
        max_diff = 1.0
    bin_edges = np.linspace(0.0, max_diff, num_bins + 1)
    counts = np.zeros((num_bins, 3), dtype=float)
    totals = np.zeros(num_bins, dtype=float)
    category_idx = {"correct": 0, "incorrect": 1, "no_preference": 2}

    for (score1, score2, label), diff in zip(entries, diffs):
        idx = np.searchsorted(bin_edges, diff, side="right") - 1
        idx = min(max(idx, 0), num_bins - 1)
        cat = determine_category(score1, score2, label)
        counts[idx, category_idx[cat]] += 1
        totals[idx] += 1

    for i, total in enumerate(totals):
        if total > 0:
            counts[i] = (counts[i] / total) * 100.0
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = (bin_edges[1] - bin_edges[0]) * 0.9 if num_bins > 0 else 0.1
    return bin_centers, bin_width, counts


def plot_histogram(bin_centers, bin_width, counts, output: Path, title: str) -> None:
    # FIX: Added "No Preference" to labels so zip() iterates 3 times
    labels = ["Correct", "Incorrect", "No Preference"]
    colors = ["#4daf4a", "#e41a1c", "#377eb8"]  # Green, Red, Blue
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(bin_centers))
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.bar(
            bin_centers,
            counts[:, i],
            width=bin_width,
            bottom=bottom,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.3,
        )
        bottom += counts[:, i]
        
    ax.set_xlabel("Query difficulty |Δscore| (Low=Hard, High=Easy)")
    ax.set_ylabel("Percentage of queries (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    # Move legend outside if it blocks bars, or keep mostly upper left/right
    ax.legend(loc="best")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    print(f"Saved Figure 6 to {output}")


def main() -> None:
    args = parse_args()
    entries = load_entries(args.log_csv)
    centers, width, counts = compute_histogram(entries, args.num_bins)
    plot_histogram(centers, width, counts, args.output, args.title)


if __name__ == "__main__":
    main()
