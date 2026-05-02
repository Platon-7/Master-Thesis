"""3-way comparison on the held-out test set: baseline vs Loss 1 (CORN) vs Loss 2 (C51).

Reads policy_ranking_*.json from /projects/prjs1958/LoRA_weights/test_eval/<model>/
Computes per-source AUC + FPR for the test split, sliced by data_source.
Writes:
  results/presentation/table_4_test_set_3way.csv  — full per-source table
  results/presentation/table_5_test_set_headline.csv  — single-row summary
  results/presentation/fig_5_test_set_fpr.png  — 3-bar FPR per source
  results/presentation/fig_6_test_set_auc.png  — 3-bar AUC per source (Group A only)
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


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
os.makedirs(OUT, exist_ok=True)


def score_corn(rec):
    return float(np.array(rec["progress_pred"]).max())


def score_c51(rec, nb=10):
    pp = np.array(rec["progress_pred"])
    if pp.ndim == 2:
        ex = np.exp(pp - pp.max(axis=-1, keepdims=True))
        probs = ex / ex.sum(axis=-1, keepdims=True)
        return float((probs * np.linspace(0, 1, nb)).sum(axis=-1).max())
    return float(pp.max())


def fpr_at(scores, labels, tau=0.5):
    a, l = np.array(scores), np.array(labels)
    fm = l == 0
    return float((a[fm] > tau).sum() / fm.sum()) if fm.sum() else None


def load_per_source(path, score_fn):
    with open(path) as f:
        recs = json.load(f)
    by_src = defaultdict(lambda: {"sp": [], "sm": [], "labs": []})
    seen_traj = set()
    for r in recs:
        tid = r["id"]
        if tid in seen_traj:
            continue
        seen_traj.add(tid)
        src = r["data_source"]
        by_src[src]["sp"].append(score_fn(r))
        by_src[src]["sm"].append(float(np.array(r["success_probs"]).max()))
        by_src[src]["labs"].append(1 if r["quality_label"] == "successful" else 0)
    return by_src


def main():
    print("Loading test JSONs ...")
    per_src = {}
    for model, path in TEST_PATHS.items():
        fn = score_corn if model == "loss1" else score_c51
        per_src[model] = load_per_source(path, fn)

    sources = sorted({s for d in per_src.values() for s in d.keys()})

    rows = []
    for src in sources:
        row = [src]
        for model in ["baseline", "loss1", "loss2"]:
            d = per_src[model].get(src, {"sp": [], "sm": [], "labs": []})
            n = len(d["labs"])
            n_succ = sum(d["labs"])
            n_fail = n - n_succ
            if len(set(d["labs"])) > 1:
                auc_p = roc_auc_score(d["labs"], d["sp"])
                auc_s = roc_auc_score(d["labs"], d["sm"])
            else:
                auc_p = None
                auc_s = None
            fpr_p = fpr_at(d["sp"], d["labs"])
            fpr_s = fpr_at(d["sm"], d["labs"])
            row.extend([n, n_fail, n_succ, auc_p, auc_s, fpr_p, fpr_s])
        rows.append(row)

    # Table 4: per-source 3-way
    with open(f"{OUT}/table_4_test_set_3way.csv", "w") as f:
        w = csv.writer(f)
        cols = ["data_source"]
        for m in ["baseline", "loss1", "loss2"]:
            cols.extend([f"{m}_n", f"{m}_n_fail", f"{m}_n_succ",
                         f"{m}_AUC_progress", f"{m}_AUC_success",
                         f"{m}_FPR_progress", f"{m}_FPR_success"])
        w.writerow(cols)
        w.writerows(rows)

    # Print pretty
    print(f"\n{'data_source':<28} {'n':>4}  | base AUCs FPRs | L1 AUCs FPRs | L2 AUCs FPRs")
    print("-" * 110)
    for row in rows:
        src = row[0]
        n_base = row[1]
        out = f"{src:<28} {n_base:>4}  |"
        for i, m in enumerate(["baseline", "loss1", "loss2"]):
            base = 1 + i * 7
            auc_s = row[base + 4]
            fpr_s = row[base + 6]
            auc_str = f"{auc_s:.3f}" if auc_s is not None else " — "
            fpr_str = f"{fpr_s:.3f}" if fpr_s is not None else " — "
            out += f" {auc_str} {fpr_str} |"
        print(out)

    # Table 5: aggregate (Group A only — sim families have only failures)
    GROUP_A = ["robometer_frames_auto_eval", "robometer_frames_racer", "robometer_frames_libero",
               "robometer_frames_mit_franka", "robometer_frames_usc_franka",
               "robometer_frames_usc_xarm", "robometer_frames_utd_so101",
               "robometer_frames_usc_koch", "robometer_frames_usc_trossen"]
    SIM = ["robometer_frames_metaworld", "robometer_frames_failsafe"]

    with open(f"{OUT}/table_5_test_set_headline.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["metric", "Baseline", "Loss 1 (CORN)", "Loss 2 (C51)"])

        # Aggregate Group A: pool all per-source predictions
        for metric, score_key in [("AUC_success (Group A pooled)", "sm"),
                                   ("AUC_progress (Group A pooled)", "sp")]:
            row = [metric]
            for model in ["baseline", "loss1", "loss2"]:
                all_scores, all_labs = [], []
                for src in GROUP_A:
                    d = per_src[model].get(src, {"sp": [], "sm": [], "labs": []})
                    all_scores.extend(d[score_key]); all_labs.extend(d["labs"])
                if len(set(all_labs)) > 1:
                    row.append(f"{roc_auc_score(all_labs, all_scores):.3f}")
                else:
                    row.append("undef")
            w.writerow(row)

        # Aggregate FPR on Group A failures (success-head)
        row = ["FPR_success @0.5 (Group A pooled)"]
        for model in ["baseline", "loss1", "loss2"]:
            scores, labs = [], []
            for src in GROUP_A:
                d = per_src[model].get(src, {"sp": [], "sm": [], "labs": []})
                scores.extend(d["sm"]); labs.extend(d["labs"])
            v = fpr_at(scores, labs)
            row.append(f"{v:.3f}" if v is not None else "undef")
        w.writerow(row)

        # Aggregate FPR on Sim failures
        row = ["FPR_success @0.5 (Sim families pooled)"]
        for model in ["baseline", "loss1", "loss2"]:
            scores, labs = [], []
            for src in SIM:
                d = per_src[model].get(src, {"sp": [], "sm": [], "labs": []})
                scores.extend(d["sm"]); labs.extend(d["labs"])
            v = fpr_at(scores, labs)
            row.append(f"{v:.3f}" if v is not None else "undef")
        w.writerow(row)

        # Mean P(success) on sim failures
        row = ["mean P(success) on Sim failures"]
        for model in ["baseline", "loss1", "loss2"]:
            scores = []
            for src in SIM:
                d = per_src[model].get(src, {"sp": [], "sm": [], "labs": []})
                for s, l in zip(d["sm"], d["labs"]):
                    if l == 0:
                        scores.append(s)
            row.append(f"{np.mean(scores):.4f}" if scores else "undef")
        w.writerow(row)

    # Figures
    plt.style.use("seaborn-v0_8-whitegrid")

    # Fig 5: per-source FPR (3 bars per source)
    sources_to_plot = [s for s in sources if any(per_src[m].get(s, {"labs": []})["labs"] for m in ["baseline","loss1","loss2"])]
    short = lambda s: s.replace("robometer_frames_", "")
    x = np.arange(len(sources_to_plot))
    width = 0.27
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, model in enumerate(["baseline", "loss1", "loss2"]):
        vals = []
        for src in sources_to_plot:
            d = per_src[model].get(src, {"sm": [], "labs": []})
            v = fpr_at(d["sm"], d["labs"])
            vals.append(v if v is not None else 0)
        ax.bar(x + (i - 1) * width, vals, width, label=LABELS[model], color=COLORS[model])
    ax.axhline(0.05, color="green", linestyle="--", linewidth=0.8, label="deploy bar (5%)")
    ax.set_xticks(x)
    ax.set_xticklabels([short(s) for s in sources_to_plot], rotation=30, ha="right")
    ax.set_ylabel("FPR @ τ=0.5  (success-head)")
    ax.set_title("Held-out test set — FPR per data source\n[lower is better; deploy bar = 5%]")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_5_test_set_fpr.png", dpi=150)
    plt.close()

    # Fig 6: per-source AUC (Group A only — where AUC is defined)
    group_a_present = [s for s in sources_to_plot if s in GROUP_A]
    x = np.arange(len(group_a_present))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, model in enumerate(["baseline", "loss1", "loss2"]):
        vals = []
        for src in group_a_present:
            d = per_src[model].get(src, {"sm": [], "labs": []})
            if len(set(d["labs"])) > 1:
                vals.append(roc_auc_score(d["labs"], d["sm"]))
            else:
                vals.append(0)
        ax.bar(x + (i - 1) * width, vals, width, label=LABELS[model], color=COLORS[model])
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([short(s) for s in group_a_present], rotation=30, ha="right")
    ax.set_ylabel("AUC (success-head)")
    ax.set_ylim(0, 1)
    ax.set_title("Held-out test set — AUC per Group A source\n[higher is better; sim families omitted (failure-only)]")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_6_test_set_auc.png", dpi=150)
    plt.close()

    print(f"\nWritten: {OUT}/table_4_test_set_3way.csv")
    print(f"Written: {OUT}/table_5_test_set_headline.csv")
    print(f"Written: {OUT}/fig_5_test_set_fpr.png")
    print(f"Written: {OUT}/fig_6_test_set_auc.png")


if __name__ == "__main__":
    main()
