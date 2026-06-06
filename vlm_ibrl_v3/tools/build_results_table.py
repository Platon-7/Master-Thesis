"""Build the OOD results table from per-checkpoint inference parquets.

Consumes the per-episode parquets emitted by ``tools/inference_eval_set.py``
(one row per episode of canonical ``robometer/rbm-1m-ood``) and produces a
markdown + CSV table comparing checkpoints on:

  * ROC-AUC                       — discrimination quality (higher = better)
  * TPR @ FPR = 0                 — fraction of successes detected with
                                    *zero* false positives among failures
  * TPR @ FPR = 0.05              — fraction at the 5% FPR operating point
  * mean p(success | label=0)     — what the model says on negatives
                                    (lower = better; this is the FPR story)

Label policy: ``label`` is already binarized in the inference parquets
(``1`` iff quality_label == "successful"; ``suboptimal`` rolled into 0).

Per-source family AUC is also reported for the 5 families that carry
both classes (usc_xarm, utd_so101_clean_top, utd_so101_clean_wrist,
usc_trossen, utd_so101_policy_ranking). Single-class families only
contribute to the overall positive pool.

Example::

    python tools/build_results_table.py \\
        --inputs robometer_4b=/scratch-shared/$USER/vlm_ibrl_results_table/baseline_4b.parquet \\
                 robometer_ft_run1=/scratch-shared/$USER/vlm_ibrl_results_table/run1_step3000.parquet \\
        --out-md  /scratch-shared/$USER/vlm_ibrl_results_table/table.md \\
        --out-csv /scratch-shared/$USER/vlm_ibrl_results_table/table.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pyarrow.parquet as pq


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via rank-sum (Mann-Whitney U). Handles ties exactly."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    # rank all scores together (ties averaged)
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    all_scores = np.concatenate([pos, neg])[order]
    all_labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])[order]
    # average ranks for ties
    ranks = np.empty_like(all_scores, dtype=np.float64)
    n = len(all_scores)
    i = 0
    while i < n:
        j = i
        while j < n and all_scores[j] == all_scores[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0  # 1-indexed
        ranks[i:j] = avg_rank
        i = j
    n_pos = len(pos)
    n_neg = len(neg)
    rank_sum_pos = ranks[all_labels == 1].sum()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, max_fpr: float) -> float:
    """Highest TPR achievable while keeping FPR <= max_fpr.

    Sweeps the threshold over the unique negative-class scores so that the
    realized FPR is exactly representable. With small ``len(neg)`` (e.g. 116
    in rbm-1m-ood), ``max_fpr=0`` is well-defined as "threshold > max(neg)".
    """
    if len(np.unique(labels)) < 2:
        return float("nan")
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    if max_fpr <= 0.0:
        # Strict: threshold must be strictly greater than the max negative.
        # TPR = fraction of positives strictly above that max.
        thr = float(np.max(neg))
        return float(np.mean(pos > thr))

    # General case: sort negatives descending; the threshold equal to the
    # k-th largest negative yields FPR = k / n_neg (strict >) plus ties.
    # We pick the smallest threshold such that FPR_strict_gt <= max_fpr.
    n_neg = len(neg)
    k_max = int(np.floor(max_fpr * n_neg))  # number of FPs allowed
    neg_sorted_desc = np.sort(neg)[::-1]
    # If k_max == 0 → strict threshold; if k_max == n_neg → accept all.
    if k_max == 0:
        thr = float(neg_sorted_desc[0])
        return float(np.mean(pos > thr))
    if k_max >= n_neg:
        return 1.0
    # threshold = the (k_max)-th largest negative (1-indexed). Any pos
    # strictly greater than this clears k_max FPs.
    thr = float(neg_sorted_desc[k_max])
    return float(np.mean(pos > thr))


def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    return dict(
        n_pos=int(len(pos)),
        n_neg=int(len(neg)),
        roc_auc=roc_auc(scores, labels),
        tpr_at_fpr0=tpr_at_fpr(scores, labels, 0.0),
        tpr_at_fpr5=tpr_at_fpr(scores, labels, 0.05),
        mean_p_neg=float(np.mean(neg)) if len(neg) else float("nan"),
        mean_p_pos=float(np.mean(pos)) if len(pos) else float("nan"),
    )


# Families that carry both classes in canonical rbm-1m-ood.
DUAL_CLASS_FAMILIES = [
    "usc_xarm",
    "utd_so101_clean_top",
    "utd_so101_clean_wrist",
    "usc_trossen",
    "utd_so101_policy_ranking",
]


def load_parquet(path: str):
    tbl = pq.read_table(path,
                        columns=["data_source", "label", "success_prob_last"])
    src = np.array(tbl["data_source"].to_pylist())
    lbl = np.array(tbl["label"].to_pylist(), dtype=np.int64)
    scr = np.array(tbl["success_prob_last"].to_pylist(), dtype=np.float64)
    return src, lbl, scr


def fmt(v: float) -> str:
    if v != v:  # NaN
        return "  —  "
    return f"{v:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="One or more <name>=<parquet_path>")
    parser.add_argument("--out-md", default=None,
                        help="Markdown table output path")
    parser.add_argument("--out-csv", default=None,
                        help="CSV table output path")
    args = parser.parse_args()

    inputs: List[tuple[str, str]] = []
    for spec in args.inputs:
        if "=" not in spec:
            print(f"ERROR: --inputs entry {spec!r} is not <name>=<path>", file=sys.stderr)
            return 2
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"ERROR: parquet not found: {path}", file=sys.stderr)
            return 2
        inputs.append((name, path))

    # Per-model metrics
    rows: List[Dict] = []
    per_family: Dict[str, Dict[str, float]] = {}  # model_name -> {fam: auc}
    for name, path in inputs:
        src, lbl, scr = load_parquet(path)
        overall = compute_metrics(scr, lbl)
        overall["model"] = name
        rows.append(overall)

        fam_aucs: Dict[str, float] = {}
        for fam in DUAL_CLASS_FAMILIES:
            mask = src == fam
            if mask.sum() == 0:
                fam_aucs[fam] = float("nan")
                continue
            fam_aucs[fam] = roc_auc(scr[mask], lbl[mask])
        per_family[name] = fam_aucs

    # Render headline table
    headline_cols = [
        ("ROC-AUC",           "roc_auc"),
        ("TPR@FPR=0",         "tpr_at_fpr0"),
        ("TPR@FPR=5%",        "tpr_at_fpr5"),
        ("mean p(s|fail)",    "mean_p_neg"),
        ("mean p(s|succ)",    "mean_p_pos"),
    ]
    header = ["model", "n+", "n-"] + [c[0] for c in headline_cols]
    md_lines = []
    md_lines.append("# OOD Results — robometer/rbm-1m-ood (782 episodes)\n")
    md_lines.append("Label: 1 = `successful`, 0 = `failure` ∪ `suboptimal`. "
                    f"Pos {rows[0]['n_pos']} / Neg {rows[0]['n_neg']}.\n")
    md_lines.append("Score: `success_prob_last` (last-frame success-head output, "
                    "single forward pass over 16 linspace-subsampled frames per ep).\n")

    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        row_vals = [r["model"], str(r["n_pos"]), str(r["n_neg"])]
        for _, key in headline_cols:
            row_vals.append(fmt(r[key]))
        md_lines.append("| " + " | ".join(row_vals) + " |")
    md_lines.append("")

    # Per-family AUC (5 families with both classes)
    md_lines.append("## Per-family ROC-AUC (only families with both classes)\n")
    fam_header = ["model"] + DUAL_CLASS_FAMILIES
    md_lines.append("| " + " | ".join(fam_header) + " |")
    md_lines.append("|" + "|".join(["---"] * len(fam_header)) + "|")
    for name, _ in inputs:
        cells = [name] + [fmt(per_family[name][f]) for f in DUAL_CLASS_FAMILIES]
        md_lines.append("| " + " | ".join(cells) + " |")
    md_lines.append("")

    out_md = "\n".join(md_lines)
    print(out_md)

    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
        with open(args.out_md, "w") as f:
            f.write(out_md)
        print(f"\nWrote {args.out_md}")

    if args.out_csv:
        import csv
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "n_pos", "n_neg", "roc_auc", "tpr_at_fpr0",
                        "tpr_at_fpr5", "mean_p_neg", "mean_p_pos"]
                       + [f"auc__{fam}" for fam in DUAL_CLASS_FAMILIES])
            for r in rows:
                fam_aucs = per_family[r["model"]]
                w.writerow([r["model"], r["n_pos"], r["n_neg"],
                            r["roc_auc"], r["tpr_at_fpr0"], r["tpr_at_fpr5"],
                            r["mean_p_neg"], r["mean_p_pos"]]
                           + [fam_aucs[fam] for fam in DUAL_CLASS_FAMILIES])
        print(f"Wrote {args.out_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
