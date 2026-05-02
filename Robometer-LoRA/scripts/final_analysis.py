"""Definitive 3-way comparison with correct head-per-model.

P(success) source per model:
  Baseline   → success_probs    (pretrained binary success head, sigmoid)
  Loss 2     → success_probs    (asym-BCE-trained binary success head, sigmoid)
  Loss 1     → progress_signal  (CORN-decoded scalar in [0,1] — losses.md §Loss 1 uses
                                 σ(z_{t,5}); the decoded scalar is the closest spec-faithful
                                 signal we have given the eval JSONs store the source-decoded
                                 scalar, not the 4 raw threshold logits)

Outputs (all to results/presentation/):
  table_4_test_set_3way.csv         — per-source AUC/FPR (overwrites)
  table_5_test_set_headline.csv     — Group A pooled summary (overwrites)
  table_6_threshold_matched_recall.csv
  table_7_pr_auc.csv
  table_8_ece.csv
  table_9_fpr_sweep.csv
  fig_5_test_set_fpr.png            — per-source FPR (overwrites)
  fig_6_test_set_auc.png            — per-source AUC (overwrites)
  fig_7_reliability_diagram.png     — reliability diagrams (overwrites; uses correct head)
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
)


OUT = "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer-LoRA/results/presentation"
os.makedirs(OUT, exist_ok=True)

import json
TEST_PATHS = {
    "baseline": "/projects/prjs1958/LoRA_weights/test_eval/baseline/robometer_lora_loss2_c51_asymmetric/eval_results/policy_ranking_robometer_frames_test.json",
    "loss1":    "/projects/prjs1958/LoRA_weights/test_eval/loss1/robometer_lora_loss1_corn_asymmetric/eval_results/policy_ranking_robometer_frames_test.json",
    "loss2":    "/projects/prjs1958/LoRA_weights/test_eval/loss2/robometer_lora_loss2_c51_asymmetric/eval_results/policy_ranking_robometer_frames_test.json",
}
LABELS = {
    "baseline": "Robometer-4B (baseline)",
    "loss1":    "L1 — CORN asym",
    "loss2":    "L2 — C51 + BCE asym",
}
COLORS = {"baseline": "#7F8C8D", "loss1": "#2E86AB", "loss2": "#C0392B"}
ORDER = ["baseline", "loss1", "loss2"]


# ============================================================================
# Per-model P(success) extraction
# ============================================================================

def _c51_decode(logits, num_bins=10):
    """C51 [T, num_bins] logits → scalar [T] in [0,1] via softmax × bin_centers."""
    pp = np.asarray(logits)
    if pp.ndim == 2:
        ex = np.exp(pp - pp.max(axis=-1, keepdims=True))
        probs = ex / ex.sum(axis=-1, keepdims=True)
        return (probs * np.linspace(0, 1, num_bins)).sum(axis=-1)
    return pp


def per_frame_p_success(rec, model):
    """Per-frame P(success) using the head-per-model mapping (losses.md compliant).

    Loss 1: σ(z_{t,5}) — top CORN cumulative-threshold sigmoid.
        We extract this from `progress_pred_raw` (shape [T, 4], the raw logits) when
        available; falls back to the source-decoded scalar (1/4) Σ σ(z_k) if not.
    Loss 2 / Baseline: σ(success_logit) from the binary success head.
    """
    if model == "loss1":
        raw = rec.get("progress_pred_raw")
        if raw is not None:
            # raw is [T, 4] cumulative-threshold logits for k ∈ {2, 3, 4, 5}.
            # σ(z_{t,5}) = sigmoid of the LAST threshold logit.
            arr = np.asarray(raw)
            if arr.ndim == 2 and arr.shape[-1] == 4:
                z5 = arr[..., -1]
                return 1.0 / (1.0 + np.exp(-z5))
        # Fallback: source-decoded scalar
        return np.asarray(rec["progress_pred"]).flatten()
    # Loss 2 / Baseline: trained / pretrained binary success head
    return np.asarray(rec["success_probs"]).flatten()


def trajectory_p_success(rec, model):
    """Trajectory-level P(success) = max over per-frame predictions."""
    return float(per_frame_p_success(rec, model).max())


# ============================================================================
# Data loading
# ============================================================================

def load_records(path):
    with open(path) as f:
        recs = json.load(f)
    seen = set()
    out = []
    for r in recs:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        out.append(r)
    return out


# ============================================================================
# Metric helpers
# ============================================================================

def fpr_at_tau(scores, labels, tau=0.5):
    s, l = np.asarray(scores), np.asarray(labels)
    fail = l == 0
    return float((s[fail] > tau).sum() / fail.sum()) if fail.any() else None


def collect_per_source(recs, model):
    """Return {data_source: {"score": [...], "label": [...]}}."""
    by_src = defaultdict(lambda: {"score": [], "label": []})
    for r in recs:
        score = trajectory_p_success(r, model)
        label = 1 if r["quality_label"] == "successful" else 0
        by_src[r["data_source"]]["score"].append(score)
        by_src[r["data_source"]]["label"].append(label)
    return by_src


def collect_pooled(recs, model, source_filter=None):
    scores, labels = [], []
    for r in recs:
        if source_filter and r["data_source"] not in source_filter:
            continue
        scores.append(trajectory_p_success(r, model))
        labels.append(1 if r["quality_label"] == "successful" else 0)
    return np.array(scores), np.array(labels)


def collect_per_frame(recs, model):
    probs, labels = [], []
    for r in recs:
        sig = per_frame_p_success(r, model)
        sl = np.asarray(r["success_labels"]).flatten()
        n = min(len(sig), len(sl))
        probs.extend(sig[:n].tolist())
        labels.extend(sl[:n].astype(int).tolist())
    return np.array(probs), np.array(labels)


# ============================================================================
# Main analysis
# ============================================================================

GROUP_A_SOURCES = {
    "robometer_frames_auto_eval", "robometer_frames_racer", "robometer_frames_libero",
    "robometer_frames_mit_franka", "robometer_frames_usc_franka",
    "robometer_frames_usc_xarm", "robometer_frames_utd_so101",
    "robometer_frames_usc_koch", "robometer_frames_usc_trossen",
}


def main():
    print("Loading records ...")
    records = {m: load_records(p) for m, p in TEST_PATHS.items()}
    for m, recs in records.items():
        print(f"  {m}: {len(recs)} unique trajectories")

    # ---------- Table 4 — per-source 3-way ----------
    print("\n[Table 4] per-source AUC/FPR (using correct head per model)")
    header = ["data_source"]
    for m in ORDER:
        header += [f"{m}_n", f"{m}_n_fail", f"{m}_n_succ", f"{m}_AUC", f"{m}_FPR_at_0.5"]

    rows = []
    sources = sorted({r["data_source"] for recs in records.values() for r in recs})
    for src in sources:
        row = [src]
        for m in ORDER:
            recs_in_src = [r for r in records[m] if r["data_source"] == src]
            scores = [trajectory_p_success(r, m) for r in recs_in_src]
            labels = [1 if r["quality_label"] == "successful" else 0 for r in recs_in_src]
            n = len(labels)
            n_succ = sum(labels)
            n_fail = n - n_succ
            auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else None
            fpr = fpr_at_tau(scores, labels)
            row += [n, n_fail, n_succ, auc, fpr]
        rows.append(row)

    with open(f"{OUT}/table_4_test_set_3way.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    # ---------- Table 5 — Group A pooled headline ----------
    print("\n[Table 5] Group A pooled headline")
    headline_rows = []
    for metric_name in ["n", "AUC", "PR-AUC", "FPR @ τ=0.5"]:
        row = [metric_name]
        for m in ORDER:
            scores, labels = collect_pooled(records[m], m, source_filter=GROUP_A_SOURCES)
            if metric_name == "n":
                row.append(int(len(labels)))
            elif metric_name == "AUC":
                row.append(f"{roc_auc_score(labels, scores):.3f}" if len(set(labels)) > 1 else "—")
            elif metric_name == "PR-AUC":
                row.append(f"{average_precision_score(labels, scores):.3f}" if len(set(labels)) > 1 else "—")
            elif metric_name == "FPR @ τ=0.5":
                v = fpr_at_tau(scores, labels)
                row.append(f"{v:.3f}" if v is not None else "—")
        headline_rows.append(row)
    with open(f"{OUT}/table_5_test_set_headline.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + [LABELS[m] for m in ORDER])
        w.writerows(headline_rows)
    for r in headline_rows:
        print("  " + "  ".join(str(x).rjust(12) for x in r))

    # ---------- Table 6 — Threshold-matched recall (Group A pooled) ----------
    print("\n[Table 6] Threshold-matched recall (Group A pooled)")
    base_scores, base_labels = collect_pooled(records["baseline"], "baseline", GROUP_A_SOURCES)
    base_fpr_at_05 = fpr_at_tau(base_scores, base_labels, 0.5)
    targets = [("Strict — zero false positives (FPR = 0.000)", 0.0),
               ("Deploy threshold (FPR = 0.050)", 0.05),
               (f"Baseline default τ=0.5 (FPR = {base_fpr_at_05:.3f})", base_fpr_at_05)]

    tmr_rows = []
    for label, target_fpr in targets:
        row = [label]
        for m in ORDER:
            scores, labels = collect_pooled(records[m], m, GROUP_A_SOURCES)
            fail_scores = scores[labels == 0]
            succ_scores = scores[labels == 1]
            if target_fpr == 0.0:
                tau = float(fail_scores.max()) + 1e-9
            else:
                tau = float(np.quantile(fail_scores, 1.0 - target_fpr))
            tpr = float((succ_scores > tau).sum() / len(succ_scores)) if len(succ_scores) else None
            row.append(f"τ*={tau:.4f}, TPR={tpr:.3f}" if tpr is not None else "—")
        tmr_rows.append(row)
    with open(f"{OUT}/table_6_threshold_matched_recall.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["target"] + [LABELS[m] for m in ORDER])
        w.writerows(tmr_rows)
    for r in tmr_rows:
        print("  " + r[0])
        for m, v in zip(ORDER, r[1:]):
            print(f"    {LABELS[m]:<28} {v}")

    # ---------- Table 7 — PR-AUC ----------
    print("\n[Table 7] PR-AUC + ROC-AUC (Group A pooled)")
    pr_rows = []
    for m in ORDER:
        scores, labels = collect_pooled(records[m], m, GROUP_A_SOURCES)
        roc = roc_auc_score(labels, scores)
        pr = average_precision_score(labels, scores)
        pr_rows.append([LABELS[m], f"{roc:.3f}", f"{pr:.3f}"])
        print(f"  {LABELS[m]:<28}  ROC-AUC = {roc:.3f}   PR-AUC = {pr:.3f}")
    with open(f"{OUT}/table_7_pr_auc.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["model", "ROC-AUC", "PR-AUC"])
        w.writerows(pr_rows)

    # ---------- Table 8 — ECE (per-frame, correct head per model) ----------
    print("\n[Table 8] ECE (per-frame, correct head per model)")
    ece_rows = []
    bin_data = {}
    for m in ORDER:
        probs, labels = collect_per_frame(records[m], m)
        edges = np.linspace(0, 1, 11)
        centers = (edges[:-1] + edges[1:]) / 2
        bin_acc = np.zeros(10); bin_conf = np.zeros(10); bin_count = np.zeros(10)
        ece = 0.0
        n_total = len(probs)
        for i in range(10):
            mask = (probs >= edges[i]) & (probs < edges[i + 1]) if i < 9 else (probs >= edges[i]) & (probs <= edges[i + 1])
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            bin_acc[i] = labels[mask].mean()
            bin_conf[i] = probs[mask].mean()
            bin_count[i] = cnt
            ece += (cnt / n_total) * abs(bin_acc[i] - bin_conf[i])
        ece_rows.append([LABELS[m], f"{ece:.4f}", n_total, f"{probs.min():.3f}", f"{probs.max():.3f}", f"{np.median(probs):.3f}"])
        bin_data[m] = (bin_acc, bin_conf, bin_count, centers, ece, n_total, probs.min(), probs.max())
        print(f"  {LABELS[m]:<28}  ECE = {ece:.4f}  range=[{probs.min():.3f}, {probs.max():.3f}]  n={n_total}")
    with open(f"{OUT}/table_8_ece.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["model", "ECE", "n_frames", "pred_min", "pred_max", "pred_median"])
        w.writerows(ece_rows)

    # ---------- Table 9 — FPR sweep ----------
    print("\n[Table 9] FPR sweep (Group A pooled)")
    taus = [0.3, 0.5, 0.7, 0.9]
    sweep_rows = []
    for m in ORDER:
        scores, labels = collect_pooled(records[m], m, GROUP_A_SOURCES)
        row = [LABELS[m]] + [f"{fpr_at_tau(scores, labels, t):.3f}" for t in taus]
        sweep_rows.append(row)
        print(f"  {LABELS[m]:<28} " + " ".join(f"τ={t}: {fpr_at_tau(scores, labels, t):.3f}" for t in taus))
    with open(f"{OUT}/table_9_fpr_sweep.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["model"] + [f"FPR @ τ={t}" for t in taus])
        w.writerows(sweep_rows)

    # ============================================================================
    # FIGURES — clean, professional, single-line titles
    # ============================================================================
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

    short_src = lambda s: s.replace("robometer_frames_", "")
    bar_width = 0.27

    # ---------- Fig 5 — per-source FPR ----------
    # Filter to sources with statistically meaningful failure samples (n_fail ≥ 20).
    # Below that, a single false positive drives FPR to 33-100% — noise, not signal.
    MIN_N_FAIL = 20
    src_n_fail = {}
    for src in sources:
        # Use baseline as the canonical record set (all 3 models cover same trajectories)
        n_fail = sum(1 for r in records["baseline"]
                     if r["data_source"] == src and r["quality_label"] == "failure")
        src_n_fail[src] = n_fail
    sources_fpr = sorted([s for s in sources if src_n_fail[s] >= MIN_N_FAIL])

    x = np.arange(len(sources_fpr))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, m in enumerate(ORDER):
        vals = []
        for src in sources_fpr:
            sub = [r for r in records[m] if r["data_source"] == src]
            scores = [trajectory_p_success(r, m) for r in sub]
            labels = [1 if r["quality_label"] == "successful" else 0 for r in sub]
            v = fpr_at_tau(scores, labels)
            vals.append(v if v is not None else 0)
        ax.bar(x + (i - 1) * bar_width, vals, bar_width, label=LABELS[m],
               color=COLORS[m], edgecolor="white", linewidth=0.5)
    ax.axhline(0.05, color="#27ae60", linestyle=":", linewidth=1, alpha=0.8)
    # Annotate sample sizes on x-axis
    xtick_labels = [f"{short_src(s)}\n(n={src_n_fail[s]})" for s in sources_fpr]
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=0, ha="center", fontsize=9)
    ax.set_ylabel("FPR @ τ = 0.5")
    ax.set_title(f"False positive rate per source  (sources with n_fail ≥ {MIN_N_FAIL}; lower is better)")
    ax.legend(loc="upper right")
    ax.set_axisbelow(True)
    # 5% deploy threshold annotation in upper-right corner of plot
    ax.text(len(sources_fpr) - 0.5, 0.058, "5% deploy threshold",
            color="#27ae60", fontsize=8, ha="right", va="bottom")
    fig.savefig(f"{OUT}/fig_5_test_set_fpr.png")
    plt.close(fig)

    # ---------- Fig 6 — per-source AUC ----------
    # AUC requires both classes; also filter to n ≥ 30 (15+15 minimum for any signal)
    MIN_N_BOTH = 30
    src_n = {}
    for src in sources:
        n_total = sum(1 for r in records["baseline"] if r["data_source"] == src)
        n_succ = sum(1 for r in records["baseline"]
                     if r["data_source"] == src and r["quality_label"] == "successful")
        src_n[src] = (n_total, n_succ)
    sources_auc = sorted([s for s in sources
                          if src_n[s][1] > 0 and src_n[s][0] - src_n[s][1] > 0
                          and src_n[s][0] >= MIN_N_BOTH and s in GROUP_A_SOURCES])

    x = np.arange(len(sources_auc))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, m in enumerate(ORDER):
        vals = []
        for src in sources_auc:
            sub = [r for r in records[m] if r["data_source"] == src]
            scores = [trajectory_p_success(r, m) for r in sub]
            labels = [1 if r["quality_label"] == "successful" else 0 for r in sub]
            v = roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.5
            vals.append(v)
        ax.bar(x + (i - 1) * bar_width, vals, bar_width, label=LABELS[m],
               color=COLORS[m], edgecolor="white", linewidth=0.5)
    ax.axhline(0.5, color="#7f8c8d", linestyle=":", linewidth=0.8, alpha=0.6)
    xtick_labels = [f"{short_src(s)}\n(n={src_n[s][0]})" for s in sources_auc]
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=0, ha="center", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"ROC-AUC per source  (Group A, n ≥ {MIN_N_BOTH}; higher is better)")
    ax.legend(loc="lower right")
    ax.set_axisbelow(True)
    fig.savefig(f"{OUT}/fig_6_test_set_auc.png")
    plt.close(fig)

    # ---------- Fig 7 — Reliability diagrams ----------
    # Each bar: a bin of predicted P(success). Bar height = empirical positive rate among
    # frames in that bin. Diagonal = perfect calibration. We annotate the active prediction
    # range per panel because some models (L1) only predict over a narrow [0, 0.26] band
    # — empty bins on the right convey "model never predicts there", not "no data".
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for idx, m in enumerate(ORDER):
        bin_acc, bin_conf, bin_count, centers, ece, n_total, lo, hi = bin_data[m]
        ax = axes[idx]
        nonempty = bin_count > 0
        # Diagonal reference (perfect calibration)
        ax.plot([0, 1], [0, 1], "--", color="#95a5a6", linewidth=1, alpha=0.7)
        # Bars
        bars = ax.bar(centers[nonempty], bin_acc[nonempty], width=0.085,
                      color=COLORS[m], alpha=0.85, edgecolor="white", linewidth=0.5)
        # Operating-range annotation in upper-left
        ax.text(0.03, 0.95, f"prediction range:\n[{lo:.3f}, {hi:.3f}]",
                fontsize=8, color="#444", va="top",
                bbox=dict(facecolor="white", edgecolor="#ccc", linewidth=0.5, pad=3))
        ax.set_xlabel("Predicted P(success)")
        if idx == 0:
            ax.set_ylabel("Empirical P(success)")
        ax.set_title(f"{LABELS[m]}    ECE = {ece:.3f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axisbelow(True)
    fig.suptitle("Reliability diagrams: per-frame predicted vs empirical success probability  (n_frames = 9,632 each)",
                 y=1.02, fontsize=11)
    fig.savefig(f"{OUT}/fig_7_reliability_diagram.png")
    plt.close(fig)

    # Also remove the now-redundant fig_7b
    redundant = f"{OUT}/fig_7b_reliability_progress_signal.png"
    if os.path.exists(redundant):
        os.remove(redundant)
        print(f"\nRemoved redundant: {redundant}")
    redundant_csv = f"{OUT}/table_8b_ece_progress_signal.csv"
    if os.path.exists(redundant_csv):
        os.remove(redundant_csv)
        print(f"Removed redundant: {redundant_csv}")

    print(f"\nAll outputs in: {OUT}")


if __name__ == "__main__":
    main()
