"""
Sparse-RL FPR/TPR comparison across reward models on CoffeePush release demos.

For each model, computes from the `robometer_offline_cm.py --full-clip` dump:
  - ROC AUC
  - TPR at fixed FPR operating points {0%, 5%, 10%, 20%}
  - The threshold that achieves each operating point

Each clip is a 0-to-end_t prefix, GT=1 iff env reward at frame end_t == 1.
This matches the IBRL `reward_at_truncation=1` regime exactly — episode-end
single scoring, env's ground-truth reward as the label.

Outputs two files in this folder:
  - sparse_rl_fpr_tpr.csv (full sweep)
  - sparse_rl_fpr_tpr_summary.md (concise table for the slide)
  - sparse_rl_roc.{pdf,png} (ROC curve overlay)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CM_BASE = Path("/scratch-shared/pkarageorgis1/vlm_ibrl_cm")
OUT_DIR = Path("/gpfs/home3/pkarageorgis1/Master-Thesis/loss-debug")


def find_latest(substr: str, task: str = "CoffeePush", scorer: str = "ft") -> Path | None:
    """Latest CM dump whose model_path contains substr AND has any non-NaN
    success_prob. Skips dumps where the bf16 cold-start NaN bug fired."""
    if not CM_BASE.exists():
        return None
    matches = []
    for jobdir in CM_BASE.iterdir():
        if not jobdir.is_dir():
            continue
        for p in jobdir.glob(f"cm_robometer_{scorer}_{task}.json"):
            try:
                d = json.loads(p.read_text())
                if substr not in d.get("args", {}).get("model_path", ""):
                    continue
                # Skip all-NaN dumps (numerical pathology, not a real measurement)
                rows = d.get("results", [])
                if not rows: continue
                n_real = sum(1 for r in rows if r["success_prob"] == r["success_prob"])
                if n_real == 0: continue
                # Also prefer the largest dump (300 > 60 clips). Sort key: (n_rows, mtime).
                matches.append((len(rows), p.stat().st_mtime, p))
            except Exception:
                continue
    if not matches:
        return None
    matches.sort()
    return matches[-1][2]


def compute_metrics(path: Path) -> dict:
    d = json.loads(path.read_text())
    rows = d["results"]
    sp = np.array([r["success_prob"] for r in rows], float)
    gt = np.array([r["gt"] for r in rows], int)
    # Drop any NaN predictions (some checkpoints produce NaN through the success
    # head on full-clip CoffeePush — bf16 underflow on out-of-dist frames).
    nan_mask = np.isnan(sp)
    if nan_mask.any():
        n_nan = int(nan_mask.sum())
        if n_nan == len(sp):
            return {"_error": f"all {n_nan} predictions are NaN"}
        sp = sp[~nan_mask]; gt = gt[~nan_mask]
    pos = sp[gt == 1]; neg = sp[gt == 0]
    if len(pos) == 0 or len(neg) == 0:
        return {}

    # ROC: at each unique threshold, what's (FPR, TPR)?
    thresholds = np.unique(np.concatenate([pos, neg, [-np.inf, np.inf]]))
    tprs, fprs, taus = [], [], []
    for tau in thresholds[::-1]:
        tprs.append((pos >= tau).mean())
        fprs.append((neg >= tau).mean())
        taus.append(float(tau))
    tprs = np.array(tprs); fprs = np.array(fprs); taus = np.array(taus)

    # AUC (Mann-Whitney)
    arr = np.concatenate([pos, neg]); order = arr.argsort()
    sorted_vals = arr[order]; sorted_ranks = np.arange(1, len(arr)+1, dtype=float)
    i = 0
    while i < len(sorted_vals):
        j = i + 1
        while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]: j += 1
        if j > i + 1: sorted_ranks[i:j] = sorted_ranks[i:j].mean()
        i = j
    ranks = np.empty_like(order, dtype=float); ranks[order] = sorted_ranks
    auc = float((ranks[:len(pos)].sum() - len(pos)*(len(pos)+1)/2.0) / (len(pos)*len(neg)))

    def tpr_at(target):
        # Largest TPR achievable with FPR <= target
        mask = fprs <= target
        if not mask.any():
            return float("nan"), float("nan")
        idx = np.argmax(tprs * mask)
        return float(tprs[idx]), float(taus[idx])

    # Sparse-RL ECE: predictions are P(at-goal-now); labels are per-frame at-goal.
    edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    n = len(sp)
    for k in range(10):
        lo, hi = edges[k], edges[k+1]
        mask = (sp >= lo) & (sp <= hi) if k == 9 else (sp >= lo) & (sp < hi)
        if mask.sum() == 0: continue
        conf = float(sp[mask].mean())
        acc = float(gt[mask].mean())
        ece += (mask.sum() / n) * abs(conf - acc)

    # Bootstrap 95% CIs on AUC + ECE
    rng = np.random.default_rng(42)
    boot_auc = []; boot_ece = []
    for _ in range(2000):
        p = rng.choice(pos, size=len(pos), replace=True)
        n_ = rng.choice(neg, size=len(neg), replace=True)
        arr = np.concatenate([p, n_]); order = arr.argsort()
        sv = arr[order]; sr = np.arange(1, len(arr)+1, dtype=float)
        i = 0
        while i < len(sv):
            j = i + 1
            while j < len(sv) and sv[j] == sv[i]: j += 1
            if j > i + 1: sr[i:j] = sr[i:j].mean()
            i = j
        r = np.empty_like(order, dtype=float); r[order] = sr
        boot_auc.append(float((r[:len(p)].sum() - len(p)*(len(p)+1)/2.0)/(len(p)*len(n_))))
        # ECE on resampled
        sp_b = np.concatenate([p, n_])
        gt_b = np.concatenate([np.ones_like(p), np.zeros_like(n_)])
        ece_b = 0.0
        for k in range(10):
            mask = (sp_b >= edges[k]) & (sp_b <= edges[k+1]) if k == 9 else (sp_b >= edges[k]) & (sp_b < edges[k+1])
            if mask.sum() == 0: continue
            ece_b += (mask.sum() / len(sp_b)) * abs(float(sp_b[mask].mean()) - float(gt_b[mask].mean()))
        boot_ece.append(ece_b)
    boot_auc = np.array(boot_auc); boot_ece = np.array(boot_ece)

    res = {
        "n_pos": len(pos), "n_neg": len(neg),
        "pos_mean": float(pos.mean()), "neg_mean": float(neg.mean()),
        "sp_min": float(sp.min()), "sp_max": float(sp.max()),
        "auc": auc,
        "auc_lo": float(np.quantile(boot_auc, 0.025)),
        "auc_hi": float(np.quantile(boot_auc, 0.975)),
        "ece": ece,
        "ece_lo": float(np.quantile(boot_ece, 0.025)),
        "ece_hi": float(np.quantile(boot_ece, 0.975)),
        "fprs": fprs, "tprs": tprs, "taus": taus,
    }
    for fpr_target, label in [(0.0, "tpr_at_fpr0"), (0.05, "tpr_at_fpr5"),
                              (0.10, "tpr_at_fpr10"), (0.20, "tpr_at_fpr20")]:
        tpr, tau = tpr_at(fpr_target)
        res[label] = tpr
        res[f"tau_for_{label}"] = tau
    return res


def main():
    # (display label, model_path substring, scorer kind)
    MODELS = [
        ("Robometer-4B (post-fix baseline)",  "Robometer-4B",                         "4b"),
        ("Robometer-FT step-3000",            "Robometer_FT_consolidated/run1_icl_ours_step3000", "ft"),
        ("Robometer-FT step-4000",            "Robometer_FT_consolidated/run1_icl_ours_step4000", "ft"),
        ("Robometer-FT step-5000",            "Robometer_FT_consolidated/run1_icl_ours_step5000", "ft"),
        ("Qwen3.5-FT step-3000",              "Qwen35_FT_consolidated/run4_step3000",  "ft"),
        ("Qwen3.5-FT step-4000",              "Qwen35_FT_consolidated/run4_step4000",  "ft"),
        ("Qwen3.5-FT step-5000",              "Qwen35_FT_consolidated/run4_step5000",  "ft"),
    ]

    rows, rocs = [], []
    for label, substr, scorer in MODELS:
        p = find_latest(substr, scorer=scorer)
        if p is None:
            print(f"[skip] {label}: no dump found yet ({substr})", file=sys.stderr)
            continue
        m = compute_metrics(p)
        if "_error" in m:
            print(f"[skip] {label}: {m['_error']}", file=sys.stderr)
            continue
        if not m:
            continue
        rocs.append((label, m["fprs"], m["tprs"], m["auc"]))
        rows.append({"model": label,
                     **{k: v for k, v in m.items() if not isinstance(v, np.ndarray)}})

    if not rows:
        print("no models with dumps yet — nothing to do", file=sys.stderr)
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "sparse_rl_fpr_tpr.csv", index=False)
    print(f"wrote {OUT_DIR/'sparse_rl_fpr_tpr.csv'}  ({len(df)} models)")

    # ---- markdown summary table for the slide
    lines = [
        "# Sparse-RL FPR/TPR comparison (CoffeePush, `reward_at_truncation=1`)",
        "",
        "Each row = one reward model.",
        "Setup: 60 clips per model (15 pre-success + 45 post-success per release demo).",
        "GT = env reward at the clip's end frame (per-frame, non-sticky).",
        "",
        "| Model | n_pos | n_neg | AUC | TPR@0%FPR | TPR@5%FPR | TPR@10%FPR | TPR@20%FPR | τ for TPR@0%FPR |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        def f(x): return f"{x:.3f}" if isinstance(x, float) and not np.isnan(x) else "—"
        lines.append(
            f"| {r.model} | {int(r.n_pos)} | {int(r.n_neg)} | "
            f"{f(r.auc)} | {f(r.tpr_at_fpr0)} | {f(r.tpr_at_fpr5)} | "
            f"{f(r.tpr_at_fpr10)} | {f(r.tpr_at_fpr20)} | "
            f"{f(r.tau_for_tpr_at_fpr0)} |"
        )
    (OUT_DIR / "sparse_rl_fpr_tpr_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_DIR/'sparse_rl_fpr_tpr_summary.md'}")

    # ---- ROC overlay plot
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    colors = ["#7f7f7f", "#e377c2", "#bcbd22", "#17becf",
              "#d62728", "#9467bd", "#8c564b"]
    for (label, fprs, tprs, auc), c in zip(rocs, colors[:len(rocs)]):
        # ROC: sort by FPR for proper line plot
        order = np.argsort(fprs)
        ax.plot(fprs[order], tprs[order], color=c, lw=2, marker=".", ms=4,
                label=f"{label}  (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="lightgrey", linestyle=":", label="chance")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Sparse-RL ROC: CoffeePush, prefix clips, env GT at last frame")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sparse_rl_roc.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "sparse_rl_roc.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_DIR/'sparse_rl_roc.pdf'}")
    print(f"wrote {OUT_DIR/'sparse_rl_roc.png'}")

    # ---- terminal-friendly summary
    print()
    print("=== sparse-RL summary (60-clip CoffeePush, post-fix) ===")
    print(f"{'model':45s} {'AUC [95% CI]':>22s} {'ECE [95% CI]':>22s} {'TPR@5%':>8s}")
    for _, r in df.iterrows():
        auc_str = f"{r.auc:.3f} [{r.auc_lo:.2f},{r.auc_hi:.2f}]"
        ece_str = f"{r.ece:.3f} [{r.ece_lo:.2f},{r.ece_hi:.2f}]"
        print(f"{r.model:45s} {auc_str:>22s} {ece_str:>22s} {r.tpr_at_fpr5:>8.3f}")


if __name__ == "__main__":
    main()
