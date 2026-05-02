"""fig_8: progress loss + 3 per-source eval Kendall trajectories.

Uses run.history() for full 10-eval-round fidelity. Clean 4-panel layout, no overlapping
titles, no suptitle (each panel is self-explanatory).
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import wandb


OUT = "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer-LoRA/results/presentation"
RUNS = {"loss1": "rl5u04gb", "loss2": "p4z06ajv"}
LABELS = {"loss1": "L1 — CORN asym", "loss2": "L2 — C51 + BCE asym"}
COLORS = {"loss1": "#2E86AB", "loss2": "#C0392B"}

EVAL_KEYS = {
    "droid":     "eval_p_rank/kendall_avg_robometer_frames_eval_droid",
    "robometer": "eval_p_rank/kendall_avg_robometer_frames_eval_robometer",
    "metaworld": "eval_p_rank/kendall_avg_robometer_frames_eval_metaworld",
    "failsafe":  "eval_p_rank/kendall_avg_robometer_frames_eval_failsafe",
}
TRAIN_KEY = "train/prog_loss"


def fetch(run_id):
    api = wandb.Api(timeout=60)
    run = api.run(f"nlp-squad/Robometer_LoRA/{run_id}")
    eval_df = run.history(keys=list(EVAL_KEYS.values()), samples=10000, pandas=True)
    eval_df = eval_df.dropna(how="all", subset=list(EVAL_KEYS.values())).sort_values("_step").reset_index(drop=True)
    train_df = run.history(keys=[TRAIN_KEY], samples=10000, pandas=True)
    train_df = train_df.dropna(subset=[TRAIN_KEY]).sort_values("_step").reset_index(drop=True)
    return eval_df, train_df


def smooth(arr, window=20):
    if len(arr) < window:
        return arr
    kern = np.ones(window) / window
    return np.convolve(arr, kern, mode="same")


def main():
    print("Fetching wandb runs...")
    histories = {m: fetch(rid) for m, rid in RUNS.items()}

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "normal",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    fig.subplots_adjust(hspace=0.40, wspace=0.25, top=0.95, bottom=0.08, left=0.08, right=0.97)

    # Common helper for an eval-Kendall panel
    def kendall_panel(ax, source_key, title):
        for m in ["loss1", "loss2"]:
            eval_df, _ = histories[m]
            steps = eval_df["_step"].astype(int).values
            vals = eval_df[EVAL_KEYS[source_key]].values
            ax.plot(steps, vals, marker="o", color=COLORS[m], linewidth=1.8,
                    markersize=5, label=LABELS[m])
        ax.axhline(0, color="#95a5a6", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Kendall ρ")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.set_axisbelow(True)

    # Panel 1 — Train progress loss
    ax = axes[0, 0]
    for m in ["loss1", "loss2"]:
        _, train = histories[m]
        steps = train["_step"].astype(int).values
        vals = train[TRAIN_KEY].values
        order = np.argsort(steps)
        steps, vals = steps[order], vals[order]
        smoothed = smooth(vals, window=max(20, len(vals) // 100))
        ax.plot(steps, smoothed, color=COLORS[m], linewidth=2, label=LABELS[m])
        ax.plot(steps, vals, color=COLORS[m], alpha=0.12, linewidth=0.4)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Train progress loss")
    ax.set_title("Training loss (smoothed)")
    ax.legend(loc="upper right")
    ax.set_axisbelow(True)

    # Panel 2 — Failsafe Kendall (deciding)
    kendall_panel(axes[0, 1], "failsafe",
                  "Eval ranking on Failsafe  (deciding split, n=30)")

    # Panel 3 — Robometer Kendall (replaces previous "mean")
    kendall_panel(axes[1, 0], "robometer",
                  "Eval ranking on Robometer  (n=2,705)")

    # Panel 4 — Droid Kendall (largest)
    kendall_panel(axes[1, 1], "droid",
                  "Eval ranking on DROID  (n=976)")

    fig.savefig(f"{OUT}/fig_8_training_dynamics.png")
    plt.close(fig)
    print(f"Wrote {OUT}/fig_8_training_dynamics.png")


if __name__ == "__main__":
    main()
