"""
Diagnostic: what does the asymmetric C51 + asymmetric BCE loss do to the
prediction distribution over training?

Parses wandb-logged `policy_ranking_samples` eval tables (per-trajectory
final-frame predictions for success vs failure trajectories) and produces:

  1. Per-source curves vs training step:
       - mean (positive trajectories)
       - mean (negative trajectories)
       - pos-mean - neg-mean separation
       - AUC
       - TPR @ FPR=0%
       - TPR @ FPR=5%

  2. Histograms of pos vs neg predictions at a few key checkpoints, per source.

The question we want to answer: does the asymmetric loss compress BOTH pos and
neg distributions toward 0 (which would explain why TPR @ low FPR is bad even
when AUC stays decent), or does it only push neg down?
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WANDB_RUN_DIR = Path(
    "/gpfs/home3/pkarageorgis1/Master-Thesis/Qwen35-FT/wandb/run-20260515_001800-4d12sa7k"
)
TABLES_DIR = WANDB_RUN_DIR / "files/media/table/policy_ranking_samples"
OUT_DIR = Path("/gpfs/home3/pkarageorgis1/Master-Thesis/Qwen35-FT/scripts/_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILE_RE = re.compile(
    r"^robometer_frames_eval_(?P<source>[a-z]+)_(?P<step>\d+)_[a-f0-9]+\.table\.json$"
)

# Per-row cells like:  successful:[0.15, 0.13, 0.13],failure:[0.12, 0.12, 0.1]
CELL_RE = re.compile(
    r"successful:\[(?P<succ>[^\]]*)\]\s*,\s*failure:\[(?P<fail>[^\]]*)\]"
)


def parse_cell(s: str) -> tuple[list[float], list[float]]:
    m = CELL_RE.search(s)
    if not m:
        return [], []
    def to_floats(x: str) -> list[float]:
        x = x.strip()
        if not x:
            return []
        return [float(v) for v in x.split(",") if v.strip()]
    return to_floats(m.group("succ")), to_floats(m.group("fail"))


def auc_score(pos: np.ndarray, neg: np.ndarray) -> float:
    """ROC-AUC via Mann-Whitney; no sklearn dependency."""
    n_p, n_n = len(pos), len(neg)
    if n_p == 0 or n_n == 0:
        return float("nan")
    # rank-sum of positives within the combined sample
    combined = np.concatenate([pos, neg])
    order = combined.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1)
    # handle ties via average rank
    unique, inv, counts = np.unique(combined, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        # cumulative ranks, then group-average
        sorted_idx = combined.argsort()
        sorted_vals = combined[sorted_idx]
        sorted_ranks = np.arange(1, len(combined) + 1, dtype=float)
        i = 0
        while i < len(sorted_vals):
            j = i + 1
            while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
                j += 1
            if j > i + 1:
                avg = sorted_ranks[i:j].mean()
                sorted_ranks[i:j] = avg
            i = j
        ranks = np.empty_like(sorted_idx, dtype=float)
        ranks[sorted_idx] = sorted_ranks
    rank_sum_pos = ranks[:n_p].sum()
    u = rank_sum_pos - n_p * (n_p + 1) / 2.0
    return float(u / (n_p * n_n))


def tpr_at_fpr(pos: np.ndarray, neg: np.ndarray, target_fpr: float) -> tuple[float, float]:
    """Largest TPR achievable with FPR ≤ target_fpr. Returns (tpr, threshold)."""
    if len(pos) == 0 or len(neg) == 0:
        return (float("nan"), float("nan"))
    # threshold = each unique score; for tau, predict pos iff score >= tau
    candidates = np.unique(np.concatenate([pos, neg]))
    # also include +inf so we can hit tpr=0 case cleanly
    best_tpr, best_tau = 0.0, float("inf")
    for tau in candidates[::-1]:  # high to low
        fpr = (neg >= tau).mean()
        tpr = (pos >= tau).mean()
        if fpr <= target_fpr and tpr > best_tpr:
            best_tpr, best_tau = float(tpr), float(tau)
    return best_tpr, best_tau


def main() -> None:
    rows = []
    raw_per_ckpt: dict[tuple[str, int], dict[str, np.ndarray]] = {}

    for path in sorted(TABLES_DIR.iterdir()):
        m = FILE_RE.match(path.name)
        if not m:
            continue
        source = m.group("source")
        step = int(m.group("step"))
        data = json.loads(path.read_text())
        # Columns: task, quality_and_rews_last, quality_and_rews_avg, quality_and_rews_sum, avg_differences
        cols = data["columns"]
        idx_last = cols.index("quality_and_rews_last")
        pos_all, neg_all = [], []
        for row in data["data"]:
            p, n = parse_cell(str(row[idx_last]))
            pos_all.extend(p)
            neg_all.extend(n)
        pos = np.asarray(pos_all, dtype=float)
        neg = np.asarray(neg_all, dtype=float)
        if len(pos) == 0 or len(neg) == 0:
            continue
        raw_per_ckpt[(source, step)] = {"pos": pos, "neg": neg}
        auc = auc_score(pos, neg)
        tpr0, thr0 = tpr_at_fpr(pos, neg, 0.0)
        tpr5, thr5 = tpr_at_fpr(pos, neg, 0.05)
        tpr10, _ = tpr_at_fpr(pos, neg, 0.10)
        rows.append({
            "source": source,
            "step": step,
            "n_pos": len(pos),
            "n_neg": len(neg),
            "pos_mean": float(pos.mean()),
            "pos_max": float(pos.max()),
            "pos_min": float(pos.min()),
            "neg_mean": float(neg.mean()),
            "neg_max": float(neg.max()),
            "neg_min": float(neg.min()),
            "separation": float(pos.mean() - neg.mean()),
            "auc": auc,
            "tpr@fpr=0": tpr0,
            "thr@fpr=0": thr0,
            "tpr@fpr=5": tpr5,
            "thr@fpr=5": thr5,
            "tpr@fpr=10": tpr10,
        })
    df = pd.DataFrame(rows).sort_values(["source", "step"]).reset_index(drop=True)
    csv_path = OUT_DIR / "loss_distribution_per_checkpoint.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}  ({len(df)} rows)")
    print()
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)))

    # ---- step-curves figure ------------------------------------------------
    sources = ["droid", "robometer", "metaworld", "failsafe"]
    fig, axes = plt.subplots(3, len(sources), figsize=(4.0 * len(sources), 9.0), sharex=True)

    for j, src in enumerate(sources):
        sub = df[df["source"] == src].sort_values("step")
        if sub.empty:
            for i in range(3):
                axes[i, j].set_title(f"{src}\n(no data)")
                axes[i, j].axis("off")
            continue
        steps = sub["step"].to_numpy()
        # Row 0: pos vs neg means + range bands
        ax0 = axes[0, j]
        ax0.plot(steps, sub["pos_mean"], color="tab:blue", label="pos mean", lw=2)
        ax0.fill_between(steps, sub["pos_min"], sub["pos_max"], color="tab:blue", alpha=0.15)
        ax0.plot(steps, sub["neg_mean"], color="tab:red", label="neg mean", lw=2)
        ax0.fill_between(steps, sub["neg_min"], sub["neg_max"], color="tab:red", alpha=0.15)
        ax0.set_ylim(-0.05, 1.05)
        ax0.set_title(src)
        if j == 0:
            ax0.set_ylabel("prediction value\n(pos vs neg trajectories)")
            ax0.legend(loc="upper right", fontsize=8)
        ax0.grid(alpha=0.3)

        # Row 1: AUC + separation
        ax1 = axes[1, j]
        ax1.plot(steps, sub["auc"], color="tab:purple", label="AUC", lw=2)
        ax1.plot(steps, sub["separation"], color="tab:orange", label="pos_mean - neg_mean", lw=2)
        ax1.axhline(0.5, color="grey", linestyle=":", alpha=0.5)
        ax1.set_ylim(0, 1.0)
        if j == 0:
            ax1.set_ylabel("AUC and separation")
            ax1.legend(loc="lower right", fontsize=8)
        ax1.grid(alpha=0.3)

        # Row 2: TPR @ FPR=0%, 5%, 10%
        ax2 = axes[2, j]
        ax2.plot(steps, sub["tpr@fpr=0"], color="tab:green", label="TPR@FPR=0%", lw=2)
        ax2.plot(steps, sub["tpr@fpr=5"], color="tab:olive", label="TPR@FPR=5%", lw=2)
        ax2.plot(steps, sub["tpr@fpr=10"], color="tab:cyan", label="TPR@FPR=10%", lw=2)
        ax2.set_ylim(-0.02, 1.02)
        ax2.set_xlabel("training step")
        if j == 0:
            ax2.set_ylabel("TPR at fixed FPR\n(operating points)")
            ax2.legend(loc="lower right", fontsize=8)
        ax2.grid(alpha=0.3)

    fig.suptitle(
        "Qwen3.5-FT run4 (icl, asymmetric C51 + asymmetric BCE, λ=0.3)\n"
        "What does the loss do to the prediction distribution?",
        fontsize=13,
    )
    fig.tight_layout()
    out_pdf = OUT_DIR / "run4_loss_distributions_step_curves.pdf"
    out_png = OUT_DIR / "run4_loss_distributions_step_curves.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=150)
    print(f"\nwrote {out_pdf}")
    print(f"wrote {out_png}")

    # ---- histogram-of-predictions figure at key checkpoints ----------------
    # Pick early, mid, late checkpoints; show pos vs neg histograms per source.
    pick_steps = [501, 2501, 5001]
    fig2, axes2 = plt.subplots(len(pick_steps), len(sources), figsize=(4.0 * len(sources), 3.0 * len(pick_steps)))
    bins = np.linspace(0, 1, 31)
    for i, st in enumerate(pick_steps):
        for j, src in enumerate(sources):
            ax = axes2[i, j]
            key = (src, st)
            if key not in raw_per_ckpt:
                ax.set_title(f"{src} step {st}\n(no data)")
                ax.axis("off")
                continue
            pos = raw_per_ckpt[key]["pos"]
            neg = raw_per_ckpt[key]["neg"]
            ax.hist(neg, bins=bins, color="tab:red", alpha=0.55, label=f"neg n={len(neg)}")
            ax.hist(pos, bins=bins, color="tab:blue", alpha=0.55, label=f"pos n={len(pos)}")
            ax.set_xlim(0, 1)
            ax.set_title(f"{src}  step {st}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    fig2.suptitle("Prediction distributions: positives vs negatives at 3 checkpoints", fontsize=13)
    fig2.tight_layout()
    out_pdf2 = OUT_DIR / "run4_pos_neg_histograms.pdf"
    out_png2 = OUT_DIR / "run4_pos_neg_histograms.png"
    fig2.savefig(out_pdf2)
    fig2.savefig(out_png2, dpi=150)
    print(f"wrote {out_pdf2}")
    print(f"wrote {out_png2}")


if __name__ == "__main__":
    main()
