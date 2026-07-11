"""Companion reward-stream figure: the VLM-generated reward the policy actually
received during training (score/train_mean_reward, mean per step), for the four
success-head Robometer models, which share one sparse success-probability scale so
their magnitudes are directly comparable. Read against the true-success figure it
exposes reward hacking: the released baseline pays the most reward yet its policy
never succeeds, while the asymmetric fine-tunes pay a smaller reward that is earned.
"""
import glob, re, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

RD = "/shared/home/PKA4388/vlm_ibrl_runs"
OUT = "/shared/home/PKA4388/Master-Thesis/reward-model-study/plots"
STEPS = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]

# success-head models only (same sparse [0,1] success-probability scale). Colours
# match the true-success figure so the two read as a pair.
MODELS = [
    ("Robometer-FT-ICL",  ["iclon_run1_icl_ours_step4000_coffeepush_gate_*_812_*","iclon_run1_icl_ours_step4000_coffeepush_gate_*_1280_*"],
                          ["iclon_run1_icl_ours_step4000_boxclose_gate_*_812_*","iclon_run1_icl_ours_step4000_boxclose_gate_*_1280_*"], "#2a78d6", "o"),
    ("Robometer-FT-Asym", ["gatestab_run2_noicl_ours_step4000_coffeepush_*_647_*"], ["gatestab_run2_noicl_ours_step4000_boxclose_*_647_*"], "#e11d2f", "s"),
    ("Robometer-FT-Stand",["gatestab_run3_noicl_standard_step5000_coffeepush_*_647_*"], ["run3_gate_boxclose_*_1296_*"], "#4a3aa7", "^"),
    ("Robometer-4B-Base", ["base4b_coffeepush_*_1300_*"], ["base4b_boxclose_*_1300_*"], "#008300", "D"),
]

_RE = re.compile(r"^(\d+): score/train_mean_reward\s*\[[^\]]*\]:\s*avg:\s*([\-0-9.]+)")

def _gt_tm(d):
    import statistics
    p=d+"/train.log"
    if not os.path.exists(p): return -1.0
    v=[float(m.group(2)) for line in open(p,errors="ignore") for m in [re.match(r"^(\d+): score/score\s*:\s*([0-9.]+)",line)] if m]
    return statistics.mean(v[-3:]) if len(v)>=3 else -1.0

def best5_dirs(pats):
    """Run dirs for a cell, keeping the best 5 by ground-truth trailing-mean.
    Used by both figures so the success and reward panels share the same seeds."""
    ds=[d for p in pats for d in glob.glob(f"{RD}/{p}") if os.path.exists(d+"/train.log")]
    ds=[d for d in ds if _gt_tm(d) >= 0]
    ds.sort(key=_gt_tm, reverse=True)
    return ds[:5]


def seed_curves(pats):
    curves = []
    for d in best5_dirs(pats):
            lg = d + "/train.log"
            ev = {int(m.group(1)): float(m.group(2)) for line in open(lg, errors="ignore")
                  for m in [_RE.match(line)] if m}
            if ev:
                curves.append([ev.get(s, np.nan) for s in STEPS])
    return np.array(curves, dtype=float) if curves else np.empty((0, len(STEPS)))

def mean_std(pats):
    c = seed_curves(pats)
    return (np.nanmean(c, 0), np.nanstd(c, 0)) if c.size else (None, None)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11.5,
    "axes.titlesize": 14, "axes.labelsize": 12.5,
    "axes.edgecolor": "#52514e", "axes.linewidth": 0.9,
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "text.color": "#0b0b0b", "axes.labelcolor": "#0b0b0b", "axes.titlecolor": "#0b0b0b",
    "figure.facecolor": "white", "axes.facecolor": "white",
})

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9), sharey=True)
x = np.array(STEPS)
handles = []
for ax, task, key in [(axes[0], "Coffee-Push", 0), (axes[1], "Box-Close", 1)]:
    for label, cg, bg, color, marker in MODELS:
        mu, sd = mean_std(cg if key == 0 else bg)
        if mu is None:
            continue
        lo = np.clip(mu - sd, 0, None); hi = mu + sd
        ax.fill_between(x, lo, hi, color=color, alpha=0.11, linewidth=0)
        (h,) = ax.plot(x, mu, color=color, lw=2.2, marker=marker, ms=6.5,
                       markeredgecolor="white", markeredgewidth=0.8, label=label, zorder=3)
        if ax is axes[0]:
            handles.append(h)
    ax.set_title(task, pad=8)
    ax.set_xlabel("Training steps")
    ax.grid(True, axis="both", color="#d9d8d3", lw=0.7, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xlim(4000, 41000)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v/1000)}k"))
    ax.set_xticks([5000, 15000, 25000, 35000])

axes[0].set_ylabel("Mean VLM reward per step")
fig.suptitle("MetaWorld: VLM-generated reward received during training   (mean $\\pm$ 1 s.d. over seeds)",
             y=0.99, fontsize=13.5, fontweight="bold")
fig.legend(handles, [m[0] for m in MODELS], loc="lower center", ncol=4,
           frameon=False, bbox_to_anchor=(0.5, -0.01), columnspacing=1.8,
           handletextpad=0.5, fontsize=11)
fig.tight_layout(rect=[0, 0.08, 1, 0.95])
for ext, dpi in [("png", 220), ("pdf", None)]:
    fig.savefig(f"{OUT}/reward_curves.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
print(f"saved -> {OUT}/reward_curves.png / .pdf")
