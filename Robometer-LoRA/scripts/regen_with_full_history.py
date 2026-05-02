"""Re-extract training & eval curves using run.history() (which doesn't drop pages
the way scan_history() does), and regenerate the affected tables and figures.

Updates: table_12, fig_8 (Failsafe panel), fig_9 (per-source eval grid).
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import wandb


OUT = "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer-LoRA/results/presentation"
RUNS = {"loss1": "rl5u04gb", "loss2": "p4z06ajv"}
LABELS = {"loss1": "Loss 1 (CORN, ours)", "loss2": "Loss 2 (C51 + BCE asym)"}
COLORS = {"loss1": "#2E86AB", "loss2": "#E63946"}

EVAL_KEYS = {
    "droid":     "eval_p_rank/kendall_avg_robometer_frames_eval_droid",
    "robometer": "eval_p_rank/kendall_avg_robometer_frames_eval_robometer",
    "metaworld": "eval_p_rank/kendall_avg_robometer_frames_eval_metaworld",
    "failsafe":  "eval_p_rank/kendall_avg_robometer_frames_eval_failsafe",
}


def fetch_eval_history(run_id):
    api = wandb.Api(timeout=60)
    run = api.run(f"nlp-squad/Robometer_LoRA/{run_id}")
    df = run.history(keys=list(EVAL_KEYS.values()), samples=10000, pandas=True)
    df = df.dropna(how="all", subset=list(EVAL_KEYS.values())).sort_values("_step").reset_index(drop=True)
    return df


def main():
    print("Fetching eval history (all 10 rounds per run) ...")
    histories = {m: fetch_eval_history(rid) for m, rid in RUNS.items()}
    for m, df in histories.items():
        print(f"  {m}: {len(df)} eval rounds, steps = {df['_step'].astype(int).tolist()}")

    # ---- table_12: replace stale ----
    csv_path = f"{OUT}/table_12_eval_kendall_per_split.csv"
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["model", "step", "kendall_droid", "kendall_robometer",
                    "kendall_metaworld", "kendall_failsafe"])
        for m, df in histories.items():
            for _, row in df.iterrows():
                w.writerow([m, int(row["_step"]),
                            row[EVAL_KEYS["droid"]], row[EVAL_KEYS["robometer"]],
                            row[EVAL_KEYS["metaworld"]], row[EVAL_KEYS["failsafe"]]])
    print(f"Rewrote {csv_path} (10 rounds per model)")

    # ---- fig_9: per-source eval grid ----
    plt.style.use("seaborn-v0_8-whitegrid")
    SPLITS = ["droid", "robometer", "metaworld", "failsafe"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for idx, sp in enumerate(SPLITS):
        ax = axes[idx // 2, idx % 2]
        for m, df in histories.items():
            steps = df["_step"].astype(int).values
            vals = df[EVAL_KEYS[sp]].values
            ax.plot(steps, vals, marker="o", color=COLORS[m], label=LABELS[m], linewidth=1.8)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Kendall ranking corr")
        title = sp + (" (deciding)" if sp == "failsafe" else "")
        ax.set_title(title)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Eval Kendall ranking correlation across training — all 10 eval rounds, by source", fontsize=12)
    fig.tight_layout()
    out_path = f"{OUT}/fig_9_eval_per_source.png"
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Rewrote {out_path}")

    # ---- fig_8 panel 2 (Failsafe trajectory): regenerate just that panel inline ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, df in histories.items():
        steps = df["_step"].astype(int).values
        vals = df[EVAL_KEYS["failsafe"]].values
        ax.plot(steps, vals, marker="o", color=COLORS[m], label=LABELS[m], linewidth=2, markersize=7)
        # Annotate peak and final
        peak_idx = np.argmax(vals)
        ax.annotate(f"peak={vals[peak_idx]:.2f}",
                    xy=(steps[peak_idx], vals[peak_idx]),
                    xytext=(8, 8), textcoords="offset points", fontsize=8, color=COLORS[m])
        ax.annotate(f"end={vals[-1]:.2f}",
                    xy=(steps[-1], vals[-1]),
                    xytext=(8, -12), textcoords="offset points", fontsize=8, color=COLORS[m])
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="random")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Failsafe Kendall ρ")
    ax.set_title("Failsafe (deciding split per losses.md) — Kendall ρ across training\nAll 10 eval rounds. Peak ≠ final for both losses.")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out_path = f"{OUT}/fig_1_failsafe_auc_trajectory.png"
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Rewrote {out_path}  (replaces stale 4-point version)")

    print("\nDone — table_12, fig_1, fig_9 now reflect all 10 eval rounds.")


if __name__ == "__main__":
    main()
