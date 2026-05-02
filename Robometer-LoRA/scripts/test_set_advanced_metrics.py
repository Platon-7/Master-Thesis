"""Advanced metrics on the held-out test set:

  1. Threshold-matched recall — for each model, compute the τ that matches baseline's FPR
     to find each model's TPR at the same operating point. Definitive answer to "could
     baseline match L1 just by raising threshold?"
  2. Precision-Recall AUC (PR-AUC) per model — emphasizes precision, what matters for RL.
  3. Expected Calibration Error (ECE) per model — per-frame success_probs vs success_labels,
     10-bin binning. Plus reliability diagram (fig_7).
  4. FPR sweep at τ ∈ {0.3, 0.5, 0.7, 0.9} — stability check across operating points.

Reads the existing test JSONs under /projects/prjs1958/LoRA_weights/test_eval/<model>/...
Writes:
  results/presentation/table_6_threshold_matched_recall.csv
  results/presentation/table_7_pr_auc.csv
  results/presentation/table_8_ece.csv
  results/presentation/table_9_fpr_sweep.csv
  results/presentation/fig_7_reliability_diagram.png
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, average_precision_score, precision_recall_curve, roc_auc_score,
)


TEST_PATHS = {
    "baseline": "/projects/prjs1958/LoRA_weights/test_eval/baseline/robometer_lora_loss2_c51_asymmetric/eval_results/policy_ranking_robometer_frames_test.json",
    "loss1":    "/projects/prjs1958/LoRA_weights/test_eval/loss1/robometer_lora_loss1_corn_asymmetric/eval_results/policy_ranking_robometer_frames_test.json",
    "loss2":    "/projects/prjs1958/LoRA_weights/test_eval/loss2/robometer_lora_loss2_c51_asymmetric/eval_results/policy_ranking_robometer_frames_test.json",
}
LABELS = {
    "baseline": "Baseline (Robometer-4B)",
    "loss1":    "Loss 1 (CORN, ours)",
    "loss2":    "Loss 2 (C51 + BCE asym)",
}
COLORS = {"baseline": "#666666", "loss1": "#2E86AB", "loss2": "#E63946"}
OUT = "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer-LoRA/results/presentation"


def load_trajectory_predictions(path):
    """Return (traj_scores [N], traj_labels [N]) using max-frame success_probs."""
    with open(path) as f:
        recs = json.load(f)
    seen = set()
    scores, labels = [], []
    for r in recs:
        tid = r["id"]
        if tid in seen:
            continue
        seen.add(tid)
        scores.append(float(np.array(r["success_probs"]).max()))
        labels.append(1 if r["quality_label"] == "successful" else 0)
    return np.array(scores), np.array(labels)


def load_per_frame_predictions(path):
    """Return (per_frame_probs [M], per_frame_labels [M]) flattened across all records."""
    with open(path) as f:
        recs = json.load(f)
    seen = set()
    probs, labels = [], []
    for r in recs:
        tid = r["id"]
        if tid in seen:
            continue
        seen.add(tid)
        sp = np.array(r["success_probs"]).flatten()
        sl = np.array(r["success_labels"]).flatten()
        # Trim to common length in case of any shape mismatch
        n = min(len(sp), len(sl))
        probs.extend(sp[:n].tolist())
        labels.extend(sl[:n].astype(int).tolist())
    return np.array(probs), np.array(labels)


def main():
    # Load trajectory-level + per-frame data
    traj_data = {m: load_trajectory_predictions(p) for m, p in TEST_PATHS.items()}
    frame_data = {m: load_per_frame_predictions(p) for m, p in TEST_PATHS.items()}

    for m, (s, l) in traj_data.items():
        n = len(l); n_succ = int(l.sum())
        print(f"[{m}] traj-level: n={n}, n_succ={n_succ}, n_fail={n - n_succ}")

    # ------- 1. Threshold-matched recall -------
    # For each model, find τ where FPR = baseline's FPR @ 0.5 (or 0.0 = lowest possible).
    # Then report TPR at that τ. Also report TPR at FPR=0 strict (the most stringent).
    print("\n=== 1. Threshold-matched recall ===")
    baseline_scores, baseline_labels = traj_data["baseline"]

    targets = []  # [(label, target_fpr), ...]
    # Target 1: FPR = 0 (strictest — what L1 is at τ=0.5)
    targets.append(("FPR=0.000 (strictest)", 0.000))
    # Target 2: FPR = baseline's @ 0.5 — i.e. how much TPR we lose to match baseline's mistake rate
    base_fpr_at_05 = float(((baseline_scores > 0.5) & (baseline_labels == 0)).sum() /
                           max((baseline_labels == 0).sum(), 1))
    targets.append((f"FPR={base_fpr_at_05:.3f} (baseline @ τ=0.5)", base_fpr_at_05))

    rows = []
    for label, target_fpr in targets:
        row = [label]
        for m in ["baseline", "loss1", "loss2"]:
            scores, labs = traj_data[m]
            fail_scores = scores[labs == 0]
            succ_scores = scores[labs == 1]
            if len(fail_scores) == 0 or len(succ_scores) == 0:
                row.append("undef")
                continue
            # τ* = smallest threshold giving FPR ≤ target_fpr
            if target_fpr == 0.0:
                # τ* must be > max failure score
                tau = float(fail_scores.max()) + 1e-9
            else:
                # τ* = the (1-target_fpr)-quantile of failure scores
                tau = float(np.quantile(fail_scores, 1.0 - target_fpr))
            tpr = float((succ_scores > tau).sum() / len(succ_scores))
            row.append(f"τ*={tau:.4f}, TPR={tpr:.3f}")
        rows.append(row)

    with open(f"{OUT}/table_6_threshold_matched_recall.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["target", "Baseline", "Loss 1 (CORN)", "Loss 2 (C51)"])
        w.writerows(rows)
    for r in rows:
        print(f"  {r[0]}")
        for m, v in zip(["baseline", "loss1", "loss2"], r[1:]):
            print(f"    {LABELS[m]:<26} {v}")

    # ------- 2. PR-AUC -------
    print("\n=== 2. Precision-Recall AUC ===")
    pr_rows = []
    for m in ["baseline", "loss1", "loss2"]:
        scores, labs = traj_data[m]
        if len(set(labs)) > 1:
            roc = roc_auc_score(labs, scores)
            pr = average_precision_score(labs, scores)
        else:
            roc, pr = float("nan"), float("nan")
        pr_rows.append([LABELS[m], f"{roc:.3f}", f"{pr:.3f}"])
        print(f"  {LABELS[m]:<26}  ROC-AUC = {roc:.3f}   PR-AUC = {pr:.3f}")

    with open(f"{OUT}/table_7_pr_auc.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["model", "ROC-AUC", "PR-AUC"])
        w.writerows(pr_rows)

    # ------- 3. ECE + reliability diagram -------
    # Computed on per-frame (probability, label) pairs across all test trajectories.
    print("\n=== 3. Expected Calibration Error (per-frame) ===")
    ece_rows = []
    bin_edges = np.linspace(0, 1, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    for idx, m in enumerate(["baseline", "loss1", "loss2"]):
        probs, labs = frame_data[m]
        ece = 0.0
        n_total = len(probs)
        bin_acc = np.zeros(10); bin_conf = np.zeros(10); bin_count = np.zeros(10)
        for i in range(10):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if i == 9:  # include 1.0
                mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
            count = mask.sum()
            if count == 0:
                continue
            acc = labs[mask].mean()
            conf = probs[mask].mean()
            bin_acc[i] = acc; bin_conf[i] = conf; bin_count[i] = count
            ece += (count / n_total) * abs(acc - conf)
        print(f"  {LABELS[m]:<26}  ECE = {ece:.4f}  (n_frames = {n_total})")
        ece_rows.append([LABELS[m], f"{ece:.4f}", n_total])

        # Reliability diagram
        ax = axes[idx]
        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="perfect calibration")
        widths = 0.09
        nonempty = bin_count > 0
        ax.bar(bin_centers[nonempty], bin_acc[nonempty], width=widths,
               alpha=0.7, color=COLORS[m], edgecolor="black", linewidth=0.5,
               label="actual")
        ax.set_xlabel("Predicted P(success)")
        if idx == 0:
            ax.set_ylabel("Empirical P(success)")
        ax.set_title(f"{LABELS[m]}\nECE = {ece:.4f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Reliability diagrams — per-frame P(success) vs empirical frequency", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_7_reliability_diagram.png", dpi=150)
    plt.close()

    with open(f"{OUT}/table_8_ece.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["model", "ECE", "n_frames"])
        w.writerows(ece_rows)

    # ------- 4. FPR sweep at τ ∈ {0.3, 0.5, 0.7, 0.9} -------
    print("\n=== 4. FPR sweep across thresholds ===")
    taus = [0.3, 0.5, 0.7, 0.9]
    fpr_rows = []
    header = ["model"] + [f"FPR @ τ={t}" for t in taus]
    print(f"  {'Model':<26}  " + "  ".join([f"τ={t}" for t in taus]))
    for m in ["baseline", "loss1", "loss2"]:
        scores, labs = traj_data[m]
        fail_scores = scores[labs == 0]
        n_fail = len(fail_scores)
        row = [LABELS[m]]
        nums = []
        for tau in taus:
            fpr = float((fail_scores > tau).sum() / n_fail) if n_fail else None
            nums.append(fpr)
            row.append(f"{fpr:.3f}" if fpr is not None else "undef")
        fpr_rows.append(row)
        print(f"  {LABELS[m]:<26}  " + "  ".join([f"{n:.3f}" for n in nums]))

    with open(f"{OUT}/table_9_fpr_sweep.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(fpr_rows)

    print(f"\nWritten:")
    for f in ["table_6_threshold_matched_recall.csv", "table_7_pr_auc.csv",
              "table_8_ece.csv", "table_9_fpr_sweep.csv", "fig_7_reliability_diagram.png"]:
        print(f"  {OUT}/{f}")


if __name__ == "__main__":
    main()
