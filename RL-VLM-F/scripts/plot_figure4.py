#!/usr/bin/env python3
"""Generate Sweep-Into learning curves (Figure 4) from eval.csv."""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot learning curves (success rate vs environment steps) from eval.csv"
    )
    parser.add_argument(
        "eval_csv",
        type=Path,
        help="Path to eval.csv produced by run_qwen_container.job",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figure4_learning_curve.png"),
        help="Path to save the resulting plot (default: figure4_learning_curve.png)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Moving-average window (in evaluations). Set to 1 to disable smoothing.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Sweep Into – Success Rate",
        help="Plot title.",
    )
    return parser.parse_args()


def load_learning_curve(csv_path: Path) -> Tuple[List[float], List[float]]:
    steps: List[float] = []
    success_rates: List[float] = []
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if "step" not in reader.fieldnames or "success_rate" not in reader.fieldnames:
            raise ValueError("CSV must contain 'step' and 'success_rate' columns")
        for row in reader:
            try:
                step = float(row["step"])
                raw_sr = float(row["success_rate"])
            except (TypeError, ValueError):
                continue
            if raw_sr > 1.0:
                raw_sr /= 100.0
            steps.append(step)
            success_rates.append(max(0.0, min(1.0, raw_sr)))
    if not steps:
        raise ValueError(f"No valid rows found in {csv_path}")
    return steps, success_rates


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1 or window > len(values):
        return values
    averaged: List[float] = []
    cumulative = [0.0]
    for v in values:
        cumulative.append(cumulative[-1] + v)
    for idx in range(window, len(cumulative)):
        averaged.append((cumulative[idx] - cumulative[idx - window]) / window)
    pad = [values[i] for i in range(window - 1)]
    return pad + averaged


def plot_curve(
    steps: List[float],
    success_rates: List[float],
    output_path: Path,
    title: str,
    smooth_window: int,
) -> None:
    smoothed = moving_average(success_rates, smooth_window)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, smoothed, label=f"Success Rate (window={smooth_window})", linewidth=2)
    if smooth_window > 1:
        ax.scatter(steps, success_rates, s=10, alpha=0.4, label="Raw evals")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title(title)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved plot to {output_path}")


def main() -> None:
    args = parse_args()
    steps, success = load_learning_curve(args.eval_csv)
    plot_curve(steps, success, args.output, args.title, args.smooth_window)


if __name__ == "__main__":
    main()
