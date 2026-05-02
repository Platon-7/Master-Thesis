"""Pull training-time diagnostics from wandb for both bake-off runs and plot.

Wandb run IDs:
  loss1: nlp-squad/Robometer_LoRA/rl5u04gb
  loss2: nlp-squad/Robometer_LoRA/p4z06ajv

Writes:
  results/presentation/fig_8_training_dynamics.png       (4-panel grid)
  results/presentation/fig_9_eval_per_source.png         (eval kendall ranking per source)
  results/presentation/fig_10_per_layer_grad_norms.png   (top-5 per-layer grad norms)
  results/presentation/table_10_training_curves.csv      (sampled scalar values)
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import wandb


OUT = "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer-LoRA/results/presentation"
os.makedirs(OUT, exist_ok=True)

RUNS = {
    "loss1": "nlp-squad/Robometer_LoRA/rl5u04gb",
    "loss2": "nlp-squad/Robometer_LoRA/p4z06ajv",
}
LABELS = {"loss1": "Loss 1 (CORN, ours)", "loss2": "Loss 2 (C51 + BCE asym)"}
COLORS = {"loss1": "#2E86AB", "loss2": "#E63946"}


def fetch(run_path):
    api = wandb.Api(timeout=60)
    run = api.run(run_path)
    print(f"  {run_path}: {run.state}, n_steps={run.summary.get('_step', '?')}")
    return list(run.scan_history(page_size=2000))


def stack_metric(rows, key):
    steps, values = [], []
    for r in rows:
        v = r.get(key)
        if v is None or isinstance(v, dict):
            continue
        s = r.get("_step")
        if s is None:
            continue
        steps.append(s); values.append(v)
    return np.array(steps), np.array(values)


def main():
    print("Fetching wandb runs ...")
    histories = {m: fetch(p) for m, p in RUNS.items()}

    plt.style.use("seaborn-v0_8-whitegrid")

    # ============== fig 8: 4-panel core training dynamics ==============
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # P1 — train/prog_loss (main training loss)
    ax = axes[0, 0]
    for m, rows in histories.items():
        steps, vals = stack_metric(rows, "train/prog_loss")
        if len(steps) > 100:
            # Smooth with rolling mean for clarity
            order = np.argsort(steps)
            steps, vals = steps[order], vals[order]
            window = max(20, len(steps) // 100)
            smooth = np.convolve(vals, np.ones(window) / window, mode="same")
            ax.plot(steps, smooth, color=COLORS[m], label=LABELS[m], linewidth=2)
            ax.plot(steps, vals, color=COLORS[m], alpha=0.15, linewidth=0.5)
    ax.set_xlabel("Training step"); ax.set_ylabel("train/prog_loss (smoothed)")
    ax.set_title("Training progress loss")
    ax.legend(loc="upper right", fontsize=9)

    # P2 — Failsafe progress kendall (eval, deciding split per losses.md)
    ax = axes[0, 1]
    for m, rows in histories.items():
        steps, vals = stack_metric(rows, "eval_p_rank/kendall_avg_robometer_frames_eval_failsafe")
        if len(steps):
            order = np.argsort(steps)
            ax.plot(steps[order], vals[order], marker="o", color=COLORS[m], label=LABELS[m], linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Training step"); ax.set_ylabel("Kendall correlation")
    ax.set_title("Failsafe eval — Kendall ranking corr\n(deciding split per losses.md)")
    ax.legend(loc="lower right", fontsize=9)

    # P3 — gradient norm (pre-clip)
    ax = axes[1, 0]
    for m, rows in histories.items():
        steps, vals = stack_metric(rows, "optim/preclip_grad_norm")
        if len(steps):
            order = np.argsort(steps)
            steps, vals = steps[order], vals[order]
            if len(steps) > 500:
                window = max(20, len(steps) // 100)
                smooth = np.convolve(vals, np.ones(window) / window, mode="same")
                ax.plot(steps, smooth, color=COLORS[m], label=LABELS[m], linewidth=2)
                ax.plot(steps, vals, color=COLORS[m], alpha=0.15, linewidth=0.5)
            else:
                ax.plot(steps, vals, color=COLORS[m], label=LABELS[m], linewidth=1.5)
    ax.set_xlabel("Training step"); ax.set_ylabel("‖grad‖ (pre-clip)")
    ax.set_title("Gradient norm (pre-clip)")
    ax.legend(loc="upper right", fontsize=9)
    # Cap y at 99th percentile to suppress spikes
    all_vals = np.concatenate([stack_metric(h, "optim/preclip_grad_norm")[1]
                                for h in histories.values()
                                if len(stack_metric(h, "optim/preclip_grad_norm")[1])])
    if len(all_vals):
        ax.set_ylim(0, np.percentile(all_vals, 99))

    # P4 — Spearman correlation (training)
    ax = axes[1, 1]
    for m, rows in histories.items():
        steps, vals = stack_metric(rows, "train/spearman_corr")
        if len(steps):
            order = np.argsort(steps)
            steps, vals = steps[order], vals[order]
            window = max(20, len(steps) // 100)
            smooth = np.convolve(vals, np.ones(window) / window, mode="same")
            ax.plot(steps, smooth, color=COLORS[m], label=LABELS[m], linewidth=2)
            ax.plot(steps, vals, color=COLORS[m], alpha=0.15, linewidth=0.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Training step"); ax.set_ylabel("Spearman ρ (train, smoothed)")
    ax.set_title("Train Spearman ρ — predictions vs targets")
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("Training dynamics — bake-off runs (7500 steps each)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_8_training_dynamics.png", dpi=150)
    plt.close()
    print(f"Wrote {OUT}/fig_8_training_dynamics.png")

    # ============== fig 9: per-source eval Kendall ranking ==============
    SPLITS = ["eval_droid", "eval_robometer", "eval_metaworld", "eval_failsafe"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for idx, split in enumerate(SPLITS):
        ax = axes[idx // 2, idx % 2]
        key = f"eval_p_rank/kendall_avg_robometer_frames_{split}"
        for m, rows in histories.items():
            steps, vals = stack_metric(rows, key)
            if len(steps):
                order = np.argsort(steps)
                ax.plot(steps[order], vals[order], marker="o", color=COLORS[m],
                        label=LABELS[m], linewidth=1.8)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Training step"); ax.set_ylabel("Kendall ranking corr")
        ax.set_title(split.replace("eval_", "") + (" (deciding)" if split == "eval_failsafe" else ""))
        if idx == 0:
            ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Eval Kendall ranking correlation across training — by source", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_9_eval_per_source.png", dpi=150)
    plt.close()
    print(f"Wrote {OUT}/fig_9_eval_per_source.png")

    # ============== fig 10: per-layer top-5 gradient norms ==============
    # Find which params show up most often in top-5 per step, then plot rolling means
    def top_layer_grads(rows):
        """Return dict[param_name -> (steps[], values[])] for top-K parameters seen."""
        per_param = defaultdict(lambda: ([], []))
        for r in rows:
            s = r.get("_step")
            if s is None:
                continue
            for k, v in r.items():
                if k.startswith("optim/top_preclip_grad_norm_") and v is not None and not isinstance(v, dict):
                    # key fmt: optim/top_preclip_grad_norm_<rank>_<param_name>
                    parts = k.split("_", 5)
                    if len(parts) >= 6:
                        param = parts[5]
                        per_param[param][0].append(s); per_param[param][1].append(v)
        return per_param

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, m in enumerate(["loss1", "loss2"]):
        per_param = top_layer_grads(histories[m])
        # Pick the top-5 params by total appearance count
        ranked = sorted(per_param.items(), key=lambda kv: -len(kv[1][0]))[:5]
        ax = axes[idx]
        for i, (pname, (ss, vs)) in enumerate(ranked):
            ss_arr, vs_arr = np.array(ss), np.array(vs)
            order = np.argsort(ss_arr)
            ss_arr, vs_arr = ss_arr[order], vs_arr[order]
            # Smooth
            if len(ss_arr) > 30:
                window = max(10, len(ss_arr) // 50)
                vs_arr = np.convolve(vs_arr, np.ones(window) / window, mode="same")
            short = pname.replace("language_base_layers.", "L").replace("self_attn.", "")[:50]
            ax.plot(ss_arr, vs_arr, label=short, linewidth=1.4, alpha=0.85)
        ax.set_xlabel("Training step"); ax.set_ylabel("‖grad‖ (top-5 params)")
        ax.set_title(LABELS[m])
        ax.legend(loc="upper right", fontsize=7)
        ax.set_yscale("log")
    fig.suptitle("Per-layer gradient norms — top-5 most-active parameters per run", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_10_per_layer_grad_norms.png", dpi=150)
    plt.close()
    print(f"Wrote {OUT}/fig_10_per_layer_grad_norms.png")

    # ============== Table 10: sampled scalar curves ==============
    csv_path = f"{OUT}/table_10_training_curves.csv"
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["model", "step", "train_prog_loss", "train_success_loss",
                    "preclip_grad_norm", "train_spearman_corr",
                    "eval_failsafe_kendall", "eval_droid_kendall",
                    "eval_robometer_kendall", "eval_metaworld_kendall"])
        for m, rows in histories.items():
            for r in rows:
                step = r.get("_step")
                if step is None:
                    continue
                ev_fs = r.get("eval_p_rank/kendall_avg_robometer_frames_eval_failsafe")
                ev_dr = r.get("eval_p_rank/kendall_avg_robometer_frames_eval_droid")
                ev_rb = r.get("eval_p_rank/kendall_avg_robometer_frames_eval_robometer")
                ev_mw = r.get("eval_p_rank/kendall_avg_robometer_frames_eval_metaworld")
                tl = r.get("train/prog_loss")
                sl = r.get("train/success_loss")
                gn = r.get("optim/preclip_grad_norm")
                sp = r.get("train/spearman_corr")
                # Subsample: keep every 50th training step OR any row with eval data
                if (tl is not None and step % 50 != 0) and ev_fs is None:
                    continue
                w.writerow([m, step, tl, sl, gn, sp, ev_fs, ev_dr, ev_rb, ev_mw])
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
