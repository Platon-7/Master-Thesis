"""
2x2 ablation deliverables for `loss-debug/`.

Compares the asymmetric loss (our recipe) against the paper-standard loss
across two base models on the in-distribution eval set, using the
policy_ranking_samples dumps saved at training time.

Produces three figures:
  fig1_per_source.{pdf,png}    : 2 (base) x 4 (eval source) grid of pos/neg
                                 mean trajectories vs training step.
  fig2_aggregated.{pdf,png}    : single 1 x 2 panel pooling ALL eval sources;
                                 the clean "headline" figure.
  fig3_ece_and_separation.{pdf,png} : 2 x 2 grid showing ECE and pos-neg
                                      separation vs training step (asks
                                      explicitly whether ECE was also tricked
                                      by the compression artifact).

Plus a CSV with the underlying per-(run, source, step) numbers.

Run mapping (confirmed via wandb_info.json):
  Qwen3.5-FT run5 (asymmetric, no ICL)  = wandb w0otbkig  = run5_noicl_ours_22813871
  Qwen3.5-FT run6 (standard,   no ICL)  = wandb u9u7seky  = run6_noicl_standard_22813873
  Robometer-FT run2 (asymmetric, no ICL)                  = run2_noicl_ours_22786984
  Robometer-FT run3 (standard,   no ICL)                  = run3_noicl_standard_22786985
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUNS = [
    ("qwen35-FT run5 (ours)",
     Path("/projects/prjs1958/Qwen35_FT_weights/run5_noicl_ours_22813871/qwen35_ft_run5_noicl_ours/policy_ranking_samples"),
     "ours (asymmetric, λ=0.3)", "Qwen3.5-VL-4B"),
    ("qwen35-FT run6 (paper)",
     Path("/projects/prjs1958/Qwen35_FT_weights/run6_noicl_standard_22813873/qwen35_ft_run6_noicl_standard/policy_ranking_samples"),
     "paper-standard", "Qwen3.5-VL-4B"),
    ("robometer-FT run2 (ours)",
     Path("/projects/prjs1958/Robometer_FT_weights/run2_noicl_ours_22786984/robometer_ft_run2_noicl_ours/policy_ranking_samples"),
     "ours (asymmetric, λ=0.3)", "Robometer-4B"),
    ("robometer-FT run3 (paper)",
     Path("/projects/prjs1958/Robometer_FT_weights/run3_noicl_standard_22786985/robometer_ft_run3_noicl_standard/policy_ranking_samples"),
     "paper-standard", "Robometer-4B"),
]

OUT_DIR = Path("/gpfs/home3/pkarageorgis1/Master-Thesis/loss-debug")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_RE_S = re.compile(r"successful:\[(?P<succ>[^\]]*)\]\s*,?\s*failure:\[(?P<fail>[^\]]*)\]")
CELL_RE_F = re.compile(r"failure:\[(?P<fail>[^\]]*)\]\s*,?\s*successful:\[(?P<succ>[^\]]*)\]")

# Visual style — keep it formal
LOSS_COLOR = {"ours (asymmetric, λ=0.3)": "#d62728", "paper-standard": "#1f77b4"}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def parse_cell(s):
    m = CELL_RE_S.search(s) or CELL_RE_F.search(s)
    if not m: return [], []
    def to_floats(x):
        x = x.strip()
        if not x: return []
        return [float(v) for v in x.split(",") if v.strip()]
    return to_floats(m.group("succ")), to_floats(m.group("fail"))


def load_rows(path):
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return [r["quality_and_rews_last"] for r in raw]
    cols = raw["columns"]
    idx = cols.index("quality_and_rews_last")
    return [row[idx] for row in raw["data"]]


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    arr = np.concatenate([pos, neg]); order = arr.argsort()
    sorted_vals = arr[order]; sorted_ranks = np.arange(1, len(arr)+1, dtype=float)
    i = 0
    while i < len(sorted_vals):
        j = i + 1
        while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]: j += 1
        if j > i + 1: sorted_ranks[i:j] = sorted_ranks[i:j].mean()
        i = j
    ranks = np.empty_like(order, dtype=float); ranks[order] = sorted_ranks
    return float((ranks[:len(pos)].sum() - len(pos)*(len(pos)+1)/2.0) / (len(pos)*len(neg)))


def ece(preds: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width bins on [0, 1]."""
    if len(preds) == 0: return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    n = len(preds)
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        if k == n_bins - 1:
            mask = (preds >= lo) & (preds <= hi)
        else:
            mask = (preds >= lo) & (preds < hi)
        if mask.sum() == 0: continue
        conf = preds[mask].mean()
        acc = labels[mask].mean()
        total += (mask.sum() / n) * abs(conf - acc)
    return float(total)


def collect_run(label, root):
    rows = []
    if not root.exists():
        print(f"[skip] {label}: {root} missing"); return pd.DataFrame()
    for step_dir in sorted(root.iterdir()):
        m = re.match(r"step_(\d+)", step_dir.name)
        if not m: continue
        step = int(m.group(1))
        # Per-source AND pool across all sources for this step
        pooled_pos, pooled_neg = [], []
        for jf in step_dir.glob("*.json"):
            source = jf.stem.replace("robometer_frames_eval_", "")
            try: cells = load_rows(jf)
            except Exception as e: print(f"[skip] {jf}: {e}"); continue
            p_all, n_all = [], []
            for c in cells:
                p, n = parse_cell(str(c))
                p_all.extend(p); n_all.extend(n)
            pos = np.asarray(p_all, float); neg = np.asarray(n_all, float)
            if len(pos) == 0 or len(neg) == 0: continue
            all_preds = np.concatenate([pos, neg])
            all_labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
            rows.append({
                "run": label, "source": source, "step": step,
                "n_pos": len(pos), "n_neg": len(neg),
                "pos_mean": float(pos.mean()), "neg_mean": float(neg.mean()),
                "separation": float(pos.mean() - neg.mean()),
                "auc": auc(pos, neg),
                "ece": ece(all_preds, all_labels),
                "pos_max": float(pos.max()), "neg_max": float(neg.max()),
            })
            pooled_pos.extend(pos); pooled_neg.extend(neg)
        if pooled_pos and pooled_neg:
            pp = np.asarray(pooled_pos, float); nn = np.asarray(pooled_neg, float)
            all_preds = np.concatenate([pp, nn])
            all_labels = np.concatenate([np.ones_like(pp), np.zeros_like(nn)])
            rows.append({
                "run": label, "source": "__pooled__", "step": step,
                "n_pos": len(pp), "n_neg": len(nn),
                "pos_mean": float(pp.mean()), "neg_mean": float(nn.mean()),
                "separation": float(pp.mean() - nn.mean()),
                "auc": auc(pp, nn),
                "ece": ece(all_preds, all_labels),
                "pos_max": float(pp.max()), "neg_max": float(nn.max()),
            })
    return pd.DataFrame(rows)


def main():
    all_df = []
    for label, root, loss, base in RUNS:
        df = collect_run(label, root)
        if df.empty: continue
        df["loss"] = loss; df["base"] = base
        all_df.append(df)
        print(f"{label}: {df['step'].nunique()} checkpoints x {df['source'].nunique()} sources/+1 pooled")
    big = pd.concat(all_df, ignore_index=True).sort_values(["run","source","step"])
    csv = OUT_DIR / "loss_2x2_ablation.csv"
    big.to_csv(csv, index=False)
    print(f"wrote {csv}")

    sources = ["droid", "robometer", "metaworld", "failsafe"]
    bases = ["Robometer-4B", "Qwen3.5-VL-4B"]

    # ---------- fig1: per-source ----------
    fig, axes = plt.subplots(2, len(sources), figsize=(4.2*len(sources), 7.0), sharex=False)
    for i, base in enumerate(bases):
        for j, src in enumerate(sources):
            ax = axes[i, j]
            sub = big[(big.base == base) & (big.source == src)]
            for loss in ["ours (asymmetric, λ=0.3)", "paper-standard"]:
                s = sub[sub.loss == loss].sort_values("step")
                if s.empty: continue
                c = LOSS_COLOR[loss]
                ax.plot(s.step, s.pos_mean, color=c, marker="o", lw=2, ms=4,
                        label=f"{loss}: successful")
                ax.plot(s.step, s.neg_mean, color=c, linestyle="--", marker="o",
                        mfc="white", lw=1.5, ms=4, label=f"{loss}: failure")
                ax.fill_between(s.step, s.neg_mean, s.pos_mean, color=c, alpha=0.10)
            ax.set_ylim(-0.02, 0.75)
            ax.set_title(f"{base}  ·  {src}", fontsize=10.5)
            ax.grid(alpha=0.25)
            if i == len(bases)-1: ax.set_xlabel("Training step")
            if j == 0:
                ax.set_ylabel("Mean P(success)")
    # Single shared legend below the figure
    handles = [plt.Line2D([0], [0], color=LOSS_COLOR["ours (asymmetric, λ=0.3)"], lw=2, marker="o", label="asymmetric loss (ours) — successful"),
               plt.Line2D([0], [0], color=LOSS_COLOR["ours (asymmetric, λ=0.3)"], lw=2, marker="o", mfc="white", linestyle="--", label="asymmetric loss (ours) — failure"),
               plt.Line2D([0], [0], color=LOSS_COLOR["paper-standard"], lw=2, marker="o", label="paper-standard loss — successful"),
               plt.Line2D([0], [0], color=LOSS_COLOR["paper-standard"], lw=2, marker="o", mfc="white", linestyle="--", label="paper-standard loss — failure")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Predicted P(success) over training. Asymmetric loss (red) compresses both classes toward zero; "
        "standard loss (blue) keeps them more separated.",
        fontsize=10.5, y=0.99,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_per_source.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig1_per_source.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote fig1_per_source")

    # ---------- fig2: aggregated (pool all sources) ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for i, base in enumerate(bases):
        ax = axes[i]
        sub = big[(big.base == base) & (big.source == "__pooled__")]
        for loss in ["ours (asymmetric, λ=0.3)", "paper-standard"]:
            s = sub[sub.loss == loss].sort_values("step")
            if s.empty: continue
            c = LOSS_COLOR[loss]
            ax.plot(s.step, s.pos_mean, color=c, marker="o", lw=2.2, ms=5,
                    label=f"{loss}: successful")
            ax.plot(s.step, s.neg_mean, color=c, linestyle="--", marker="o",
                    mfc="white", lw=1.7, ms=5, label=f"{loss}: failure")
            ax.fill_between(s.step, s.neg_mean, s.pos_mean, color=c, alpha=0.10)
        ax.set_ylim(-0.02, 0.65)
        ax.set_title(f"{base}  (pooled across all 4 eval sources)", fontsize=11)
        ax.grid(alpha=0.25)
        ax.set_xlabel("Training step")
        if i == 0:
            ax.set_ylabel("Mean P(success)")
    handles = [plt.Line2D([0], [0], color=LOSS_COLOR["ours (asymmetric, λ=0.3)"], lw=2.2, marker="o", label="asymmetric loss (ours) — successful"),
               plt.Line2D([0], [0], color=LOSS_COLOR["ours (asymmetric, λ=0.3)"], lw=2.2, marker="o", mfc="white", linestyle="--", label="asymmetric loss (ours) — failure"),
               plt.Line2D([0], [0], color=LOSS_COLOR["paper-standard"], lw=2.2, marker="o", label="paper-standard loss — successful"),
               plt.Line2D([0], [0], color=LOSS_COLOR["paper-standard"], lw=2.2, marker="o", mfc="white", linestyle="--", label="paper-standard loss — failure")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "Asymmetric loss (red) suppresses BOTH classes; standard loss (blue) separates them more.",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_aggregated.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig2_aggregated.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote fig2_aggregated")

    # ---------- fig3: ECE / AUC / separation summary ----------
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.0))
    for i, base in enumerate(bases):
        ax_ece = axes[i, 0]; ax_auc = axes[i, 1]; ax_sep = axes[i, 2]
        sub = big[(big.base == base) & (big.source == "__pooled__")]
        for loss in ["ours (asymmetric, λ=0.3)", "paper-standard"]:
            s = sub[sub.loss == loss].sort_values("step")
            if s.empty: continue
            c = LOSS_COLOR[loss]
            ax_ece.plot(s.step, s.ece,        color=c, marker="o", lw=2.2, ms=5, label=loss)
            ax_auc.plot(s.step, s.auc,        color=c, marker="o", lw=2.2, ms=5, label=loss)
            ax_sep.plot(s.step, s.separation, color=c, marker="o", lw=2.2, ms=5, label=loss)
        ax_ece.set_ylim(0.0, 0.5); ax_auc.set_ylim(0.5, 1.0); ax_sep.set_ylim(-0.05, 0.3)
        ax_ece.set_title(f"{base}  ·  ECE (lower = better)")
        ax_auc.set_title(f"{base}  ·  AUC (higher = better)")
        ax_sep.set_title(f"{base}  ·  separation (higher = better)")
        for ax in (ax_ece, ax_auc, ax_sep): ax.grid(alpha=0.25)
        if i == len(bases)-1:
            for ax in (ax_ece, ax_auc, ax_sep): ax.set_xlabel("Training step")
        ax_ece.set_ylabel("ECE"); ax_auc.set_ylabel("AUC"); ax_sep.set_ylabel("Separation")
    handles = [plt.Line2D([0], [0], color=LOSS_COLOR["ours (asymmetric, λ=0.3)"], lw=2.2, marker="o", label="asymmetric loss (ours)"),
               plt.Line2D([0], [0], color=LOSS_COLOR["paper-standard"], lw=2.2, marker="o", label="paper-standard loss")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "ECE is HIGHER (worse) for the asymmetric loss on trajectory labels — opposite of what the LoRA per-frame plot suggested.",
        fontsize=11, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_ece_and_separation.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig3_ece_and_separation.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote fig3_ece_and_separation")

    # ---------- printed summary table for the deck ----------
    print()
    print("=== headline numbers at last checkpoint (pooled across sources) ===")
    for base in bases:
        print(f"\n  {base}:")
        for loss in ["ours (asymmetric, λ=0.3)", "paper-standard"]:
            s = big[(big.base == base) & (big.source == "__pooled__") & (big.loss == loss)].sort_values("step")
            if s.empty: continue
            last = s.iloc[-1]
            print(f"    {loss:35s}  step={int(last.step):>5}  "
                  f"pos={last.pos_mean:.3f}  neg={last.neg_mean:.3f}  "
                  f"sep={last.separation:+.3f}  AUC={last.auc:.3f}  ECE={last.ece:.3f}")


if __name__ == "__main__":
    main()
