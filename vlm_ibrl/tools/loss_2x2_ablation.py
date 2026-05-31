"""
2x2 ablation: track per-checkpoint prediction distributions on the eval set.

  Rows: {Robometer-4B base, Qwen3.5 base}
  Cols: {asymmetric loss (ours), paper-standard loss}

For each (base, loss) cell, plot pos_mean and neg_mean trajectories vs training
step across the 4 eval sources. The question: does the asymmetric loss
actually push neg predictions DOWN, or do BOTH pos+neg drift up together?

Reads the saved policy_ranking_samples/step_*/<source>.json dumps under each
run's training output dir. No GPU needed.

User-confirmed mapping:
  Qwen3.5-FT run5 (asymmetric noicl)  = wandb w0otbkig  =  22813871 dir
  Qwen3.5-FT run6 (standard   noicl)  = wandb u9u7seky  =  22813873 dir
  Robometer-FT run2 (asymmetric noicl) =                =  22786984 dir
  Robometer-FT run3 (standard noicl)   =                =  22786985 dir
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# (label, root, loss, base)
RUNS = [
    ("qwen35-FT run5 asymmetric (w0otbkig)",
     Path("/projects/prjs1958/Qwen35_FT_weights/run5_noicl_ours_22813871/qwen35_ft_run5_noicl_ours/policy_ranking_samples"),
     "asymmetric", "qwen3.5"),
    ("qwen35-FT run6 standard   (u9u7seky)",
     Path("/projects/prjs1958/Qwen35_FT_weights/run6_noicl_standard_22813873/qwen35_ft_run6_noicl_standard/policy_ranking_samples"),
     "standard", "qwen3.5"),
    ("robometer-FT run2 asymmetric",
     Path("/projects/prjs1958/Robometer_FT_weights/run2_noicl_ours_22786984/robometer_ft_run2_noicl_ours/policy_ranking_samples"),
     "asymmetric", "robometer-4b"),
    ("robometer-FT run3 standard",
     Path("/projects/prjs1958/Robometer_FT_weights/run3_noicl_standard_22786985/robometer_ft_run3_noicl_standard/policy_ranking_samples"),
     "standard", "robometer-4b"),
]

OUT_DIR = Path("/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cell formats observed: "successful:[...],failure:[...]" OR "failure:[...],successful:[...]"
CELL_RE_S = re.compile(r"successful:\[(?P<succ>[^\]]*)\]\s*,?\s*failure:\[(?P<fail>[^\]]*)\]")
CELL_RE_F = re.compile(r"failure:\[(?P<fail>[^\]]*)\]\s*,?\s*successful:\[(?P<succ>[^\]]*)\]")


def parse_cell(s):
    m = CELL_RE_S.search(s) or CELL_RE_F.search(s)
    if not m:
        return [], []
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
    idx_last = cols.index("quality_and_rews_last")
    return [row[idx_last] for row in raw["data"]]


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


def tpr_at_fpr(pos, neg, target):
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    cands = np.unique(np.concatenate([pos, neg]))
    best = 0.0
    for tau in cands[::-1]:
        if (neg >= tau).mean() <= target:
            tpr = (pos >= tau).mean()
            if tpr > best: best = float(tpr)
    return best


def collect_run(label, root):
    rows = []
    if not root.exists():
        print(f"[skip] {label}: {root} not found"); return pd.DataFrame()
    for step_dir in sorted(root.iterdir()):
        m = re.match(r"step_(\d+)", step_dir.name)
        if not m: continue
        step = int(m.group(1))
        for jf in step_dir.glob("*.json"):
            source = jf.stem.replace("robometer_frames_eval_", "")
            try:
                cells = load_rows(jf)
            except Exception as e:
                print(f"[skip] {jf}: {e}"); continue
            pos_all, neg_all = [], []
            for c in cells:
                p, n = parse_cell(str(c))
                pos_all.extend(p); neg_all.extend(n)
            pos = np.asarray(pos_all, float); neg = np.asarray(neg_all, float)
            if len(pos) == 0 or len(neg) == 0: continue
            rows.append({
                "run": label, "source": source, "step": step,
                "n_pos": len(pos), "n_neg": len(neg),
                "pos_mean": float(pos.mean()), "pos_max": float(pos.max()),
                "neg_mean": float(neg.mean()), "neg_max": float(neg.max()),
                "separation": float(pos.mean() - neg.mean()),
                "auc": auc(pos, neg),
                "tpr@0": tpr_at_fpr(pos, neg, 0.0),
                "tpr@5": tpr_at_fpr(pos, neg, 0.05),
            })
    return pd.DataFrame(rows)


def main():
    all_df = []
    for label, root, loss, base in RUNS:
        df = collect_run(label, root)
        if df.empty:
            print(f"[skip] {label}: no checkpoints found"); continue
        df["loss"] = loss; df["base"] = base
        all_df.append(df)
        n_steps = sorted(df["step"].unique())
        sources = sorted(df["source"].unique())
        print(f"{label}: steps={n_steps} sources={sources}")
    big = pd.concat(all_df, ignore_index=True).sort_values(["run","source","step"])
    csv = OUT_DIR / "loss_2x2_ablation.csv"
    big.to_csv(csv, index=False)
    print(f"\nwrote {csv}  ({len(big)} rows)")

    # Plot pos vs neg mean trajectories, 2 rows x 4 cols
    sources = ["droid", "robometer", "metaworld", "failsafe"]
    bases = ["robometer-4b", "qwen3.5"]
    fig, axes = plt.subplots(2, len(sources), figsize=(4.0*len(sources), 7.5))
    for i, base in enumerate(bases):
        for j, src in enumerate(sources):
            ax = axes[i, j]
            sub = big[(big.base == base) & (big.source == src)]
            for loss, col in [("asymmetric", "tab:red"), ("standard", "tab:blue")]:
                s = sub[sub.loss == loss].sort_values("step")
                if s.empty: continue
                ax.plot(s.step, s.pos_mean, color=col, marker="o", lw=2, ms=3, label=f"{loss} pos_mean")
                ax.plot(s.step, s.neg_mean, color=col, linestyle="--", marker="x", lw=1.5, ms=3, label=f"{loss} neg_mean")
            ax.set_ylim(-0.02, 0.7)
            ax.set_title(f"{base}  /  {src}")
            ax.grid(alpha=0.3)
            if i == 1: ax.set_xlabel("training step")
            if j == 0:
                ax.set_ylabel("pred value\n(pos solid, neg dashed)")
                ax.legend(loc="best", fontsize=7)
    fig.suptitle("Asymmetric (red) vs paper-standard (blue): do NEG predictions actually go DOWN?", fontsize=12)
    fig.tight_layout()
    out_pdf = OUT_DIR / "loss_2x2_pos_neg_trajectories.pdf"
    out_png = OUT_DIR / "loss_2x2_pos_neg_trajectories.png"
    fig.savefig(out_pdf); fig.savefig(out_png, dpi=150)
    print(f"wrote {out_pdf}"); print(f"wrote {out_png}")

    # Also: per-(base,source) printed table at first / mid / last steps
    print()
    print("=== headline numbers at first / mid / last available step ===")
    for base in bases:
        for src in sources:
            sub = big[(big.base == base) & (big.source == src)].copy()
            if sub.empty: continue
            for loss in ["asymmetric", "standard"]:
                s = sub[sub.loss == loss].sort_values("step")
                if s.empty: continue
                first = s.iloc[0]; mid = s.iloc[len(s)//2]; last = s.iloc[-1]
                print(f"  {base:14s} {src:10s} {loss:10s}: "
                      f"step {int(first.step):>4} pos={first.pos_mean:.3f} neg={first.neg_mean:.3f} sep={first.separation:+.3f} | "
                      f"step {int(mid.step):>4} pos={mid.pos_mean:.3f} neg={mid.neg_mean:.3f} sep={mid.separation:+.3f} | "
                      f"step {int(last.step):>4} pos={last.pos_mean:.3f} neg={last.neg_mean:.3f} sep={last.separation:+.3f}")


if __name__ == "__main__":
    main()
