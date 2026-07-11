"""Robomimic IBRL (PickPlace-Can): policy success vs training steps, one line per VLM reward
model. Styled to match the MetaWorld baseline figure (same per-model colors/markers, fonts,
bottom legend). Reads each run's train.log live (GT eval = score/score). 1 seed per model."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

RUNS = "/shared/home/PKA4388/vlm_ibrl_runs"
OUT = "/shared/home/PKA4388/Master-Thesis/reward-model-study/plots/can_baseline_curves.png"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.linewidth": 1.1,
    "axes.edgecolor": "#333333",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# label, run-dir, color (MetaWorld palette), marker, legend-order
MODELS = [
    ("Robometer-FT (ours)",   "can_run2_thr0.80_minep120_750k",        "#e31a1c", "s"),  # red (hero)
    ("Robometer-FT-ICL",      "can_run1icl_auto_thr0.90_gate120_750k", "#1f78b4", "o"),  # blue
    ("Robometer-FT-Standard", "can_run3std_auto_thr0.35_gate120_750k", "#6a3d9a", "^"),  # purple
    ("Robometer-4B-Base",     "can_4Bbase_auto_thr0.80_gate120_750k",  "#33a02c", "D"),  # green
    ("RoboDopamine",          "can_robodopamine4b_scorer_750k",        "#d4a017", "v"),  # gold
    ("RoboReward",            "can_roboreward4b_scorer_750k",          "#8c564b", "P"),  # brown
    ("LRM-TRI",               "can_lrm8b_scorer_750k",                 "#000000", "X"),  # black
]
# legend order to mirror MetaWorld's 4-column layout (row-major)
LEGEND_ORDER = [1, 2, 4, 6, 0, 3, 5]


def load_curve(run_dir):
    f = os.path.join(RUNS, run_dir, "train.log")
    if not os.path.isfile(f):
        return None, None
    lines = open(f, "rb").read().decode("utf-8", "ignore").splitlines()
    st = [int(l.split(":")[-1]) for l in lines if "other/step" in l and l.strip().endswith(tuple("0123456789"))]
    sc = [float(l.split(":")[-1]) for l in lines if "score/score" in l]
    n = min(len(st), len(sc))
    return (np.array(st[:n]), np.array(sc[:n])) if n else (None, None)


def smooth(y, w=7):
    y = np.asarray(y, float)
    if len(y) < 3:
        return y
    half = w // 2
    return np.array([y[max(0, i - half):min(len(y), i + half + 1)].mean() for i in range(len(y))])


fig, ax = plt.subplots(figsize=(11, 5.6), dpi=200)
handles = {}
for i, (label, d, color, marker) in enumerate(MODELS):
    st, sc = load_curve(d)
    if st is None:
        continue
    me = max(1, len(st) // 14)
    (h,) = ax.plot(st, smooth(sc), color=color, lw=2.4, marker=marker, ms=6.5,
                   markevery=me, markeredgecolor="white", markeredgewidth=0.6,
                   label=label, zorder=5 if i == 0 else 3, alpha=0.95)
    handles[label] = h

# behavior-cloning reference (subtle)
ax.axhline(0.56, color="#9a9a9a", ls=(0, (2, 3)), lw=1.1, zorder=1)
ax.text(752000, 0.565, "behavior cloning", color="#8a8a8a", fontsize=10.5, va="bottom", ha="right")

ax.set_title("Robomimic IBRL (PickPlace-Can): policy success by VLM reward model",
             fontsize=17, fontweight="bold", pad=14)
ax.set_xlabel("Training steps", fontsize=15)
ax.set_ylabel("Task success rate (ground truth)", fontsize=15)
ax.set_xlim(0, 762000)
ax.set_ylim(-0.02, 0.92)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x else "0"))
ax.tick_params(labelsize=12.5, length=5, width=1.0)
ax.grid(True, color="#d9d9d9", lw=0.8, alpha=0.9)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(True)
    ax.spines[s].set_color("#cccccc")

ordered = [handles[MODELS[i][0]] for i in LEGEND_ORDER if MODELS[i][0] in handles]
ax.legend(handles=ordered, loc="upper center", bbox_to_anchor=(0.5, -0.13),
          ncol=4, frameon=False, fontsize=12.5, handlelength=2.4,
          columnspacing=1.8, handletextpad=0.6)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"saved {OUT}")
for label, d, _, _ in MODELS:
    st, sc = load_curve(d)
    print(f"  {label:24s} " + (f"{st[-1]/1000:.0f}k, peak {sc.max():.2f}" if st is not None else "PENDING"))
