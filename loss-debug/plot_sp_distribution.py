"""Plot per-episode VLM success_prob distribution on IBRL-policy rollouts,
split by env-GT success/failure label, for 3 reward models.

Compares:
  - Robometer-4B baseline (no FT)
  - Robometer-FT run1 step-3000 (asymmetric + ICL)
  - Qwen3.5-FT run4 step-6500 (asymmetric + ICL)

For each model, render one PNG with overlaid histograms:
  - success-conditional sp distribution
  - failure-conditional sp distribution
  - vertical line at the IBRL-training τ for context
"""
from __future__ import annotations
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


RUNS = [
    {
        "label": "Robometer-4B baseline",
        "csv": "/scratch-shared/pkarageorgis1/sp_dist_eval/robometer_4b_baseline.csv",
        "tau_train": 0.6,
        "beta_train": 0.5,
        "out": "/gpfs/home3/pkarageorgis1/Master-Thesis/loss-debug/sp_dist/robometer_4b.png",
    },
    {
        "label": "Robometer-FT s3000 (asymmetric + ICL)",
        "csv": "/scratch-shared/pkarageorgis1/sp_dist_eval/robometer_ft_run1_s3000.csv",
        "tau_train": 0.0192,
        "beta_train": 0.0,
        "out": "/gpfs/home3/pkarageorgis1/Master-Thesis/loss-debug/sp_dist/robometer_ft_s3000.png",
    },
    {
        "label": "Qwen3.5-FT s6500 (asymmetric + ICL)",
        "csv": "/scratch-shared/pkarageorgis1/sp_dist_eval/qwen35_ft_run4_s6500.csv",
        "tau_train": 0.25,
        "beta_train": 0.0,
        "out": "/gpfs/home3/pkarageorgis1/Master-Thesis/loss-debug/sp_dist/qwen35_ft_s6500.png",
    },
]


def load(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    sp_succ, sp_fail = [], []
    for r in rows:
        sp = float(r["sp"])
        if math.isnan(sp):
            continue
        if r["env_success"] == "1":
            sp_succ.append(sp)
        else:
            sp_fail.append(sp)
    return np.asarray(sp_succ), np.asarray(sp_fail)


def render(run):
    sp_succ, sp_fail = load(run["csv"])
    Path(run["out"]).parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    # Use log scale on the x-axis would be nice for the compressed FT models but
    # keep linear for direct comparability. Range: 0 to max(sp_succ + sp_fail) + 0.05.
    upper = float(np.concatenate([sp_succ, sp_fail]).max()) + 0.05 if (len(sp_succ) + len(sp_fail)) else 1.0
    upper = min(max(upper, 0.3), 1.05)
    bins = np.linspace(0, upper, 41)

    ax.hist(sp_fail, bins=bins, color="#e0473b", alpha=0.55,
            label=f"failure (n={len(sp_fail)}, μ={sp_fail.mean():.3f})" if len(sp_fail) else "failure (n=0)",
            density=False, edgecolor="#a01710", linewidth=0.4)
    ax.hist(sp_succ, bins=bins, color="#3aaa55", alpha=0.7,
            label=f"success (n={len(sp_succ)}, μ={sp_succ.mean():.3f})" if len(sp_succ) else "success (n=0)",
            density=False, edgecolor="#1a662d", linewidth=0.4)

    # τ line
    ax.axvline(run["tau_train"], color="#222222", linestyle="--", linewidth=1.2,
               label=f"τ used in IBRL training = {run['tau_train']}")

    # Mean markers
    if len(sp_succ):
        ax.axvline(sp_succ.mean(), color="#1a662d", linestyle=":", linewidth=1.0, alpha=0.8)
    if len(sp_fail):
        ax.axvline(sp_fail.mean(), color="#a01710", linestyle=":", linewidth=1.0, alpha=0.8)

    ax.set_xlabel("VLM success_prob at episode truncation")
    ax.set_ylabel("episode count")
    ax.set_xlim(0, upper)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.set_title(
        f"{run['label']} — sp distribution on IBRL-policy rollouts (200 episodes)",
        fontsize=12, pad=10,
    )
    # Inverted-separation annotation
    if len(sp_succ) and len(sp_fail):
        sep = sp_succ.mean() - sp_fail.mean()
        if sep < 0:
            ax.text(0.5, 0.85,
                    f"sp(success) − sp(failure) = {sep:+.4f}\nINVERTED — model rates failures higher than successes on IBRL trajectories",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=10, color="#a01710", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff0ee", edgecolor="#a01710"))
        else:
            ax.text(0.5, 0.85,
                    f"sp(success) − sp(failure) = {sep:+.4f}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=10, color="#1a662d", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#eefff0", edgecolor="#1a662d"))

    fig.tight_layout()
    fig.savefig(run["out"], dpi=160, bbox_inches="tight")
    fig.savefig(run["out"].replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {run['out']}")


def main():
    for run in RUNS:
        render(run)
    print("done")


if __name__ == "__main__":
    main()
