"""Deck figures from FULL_METRICS.csv (zero compute).

  fig1_progress_head_collapse — THE headline: VOC-pearson by model, asymmetric
                                vs paper-standard, OOD + in-dist. Asymmetric ≈ 0.
  fig2_specialization         — success AUC in-dist vs OOD per model (FT wins
                                in-dist, loses OOD; baseline opposite).
  fig3_ece_flat               — dense-ECE by model (shows it does NOT separate).
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RES = Path("/gpfs/home3/pkarageorgis1/Master-Thesis/reward-model-study/results")
FIG = Path("/gpfs/home3/pkarageorgis1/Master-Thesis/reward-model-study/figures")
FIG.mkdir(exist_ok=True)

# model -> (display label, family: 'baseline'|'asym'|'paper')
MODELS = {
    "baseline":   ("Robometer-4B\n(baseline)", "baseline"),
    "run1_s5000": ("Robometer-FT\nasym+ICL", "asym"),
    "run2_s5000": ("Robometer-FT\nasym", "asym"),
    "run3_s5000": ("Robometer-FT\npaper-std", "paper"),
    "run4_s6500": ("Qwen3.5-FT\nasym+ICL", "asym"),
    "run5_s6500": ("Qwen3.5-FT\nasym", "asym"),
    "run6_s6500": ("Qwen3.5-FT\npaper-std", "paper"),
}
ORDER = ["baseline", "run3_s5000", "run6_s6500", "run1_s5000", "run2_s5000", "run4_s6500", "run5_s6500"]
COLOR = {"baseline": "#444444", "asym": "#e0473b", "paper": "#1f6feb"}

data = defaultdict(dict)  # (model,cell) -> metrics
for r in csv.DictReader(open(RES / "FULL_METRICS.csv")):
    data[(r["model"], r["cell"])] = r


def val(m, cell, key):
    r = data.get((m, cell))
    if not r or r[key] in ("", "nan"):
        return np.nan
    return float(r[key])


def bars(ax, cell, key, title, ylabel, ylim=None, hline=None):
    xs = np.arange(len(ORDER))
    vals = [val(m, cell, key) for m in ORDER]
    cols = [COLOR[MODELS[m][1]] for m in ORDER]
    ax.bar(xs, vals, color=cols, edgecolor="#222", linewidth=0.5)
    for x, v in zip(xs, vals):
        if not np.isnan(v):
            ax.text(x, v + (0.02 if v >= 0 else -0.05), f"{v:.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([MODELS[m][0] for m in ORDER], fontsize=7.5)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)
    if ylim: ax.set_ylim(*ylim)
    if hline is not None: ax.axhline(hline, color="#888", ls="--", lw=0.8)
    ax.grid(axis="y", ls=":", alpha=0.4)


# fig1 — progress-head collapse (THE figure)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
bars(axes[0], "ood", "prog_VOCpearson", "OOD", "VOC-pearson (progress vs GT)", ylim=(-0.15, 1.0), hline=0)
bars(axes[1], "indist_icloff", "prog_VOCpearson", "In-distribution", "VOC-pearson (progress vs GT)", ylim=(-0.15, 1.0), hline=0)
fig.suptitle("Asymmetric loss destroys the progress head (red ≈ 0; blue/grey intact)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "fig1_progress_head_collapse.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote fig1_progress_head_collapse.png")

# fig2 — specialization tradeoff (success AUC in-dist vs OOD)
fig, ax = plt.subplots(figsize=(8, 5.5))
for m in ORDER:
    x = val(m, "ood", "succ_AUC"); y = val(m, "indist_icloff", "succ_AUC")
    ax.scatter(x, y, s=90, color=COLOR[MODELS[m][1]], edgecolor="#222", zorder=3)
    ax.annotate(MODELS[m][0].replace("\n", " "), (x, y), fontsize=7.5,
                xytext=(5, 4), textcoords="offset points")
ax.axhline(0.5, color="#ccc", ls=":"); ax.axvline(0.5, color="#ccc", ls=":")
ax.plot([0.4, 0.9], [0.4, 0.9], color="#ddd", ls="--", lw=0.8)
ax.set_xlabel("OOD success AUC (generalization)"); ax.set_ylabel("In-dist success AUC (specialization)")
ax.set_title("Specialization tradeoff: FT wins in-dist, loses OOD\n(baseline is the opposite)", fontsize=12)
ax.set_xlim(0.4, 0.9); ax.set_ylim(0.4, 0.9); ax.grid(ls=":", alpha=0.4)
fig.tight_layout(); fig.savefig(FIG / "fig2_specialization.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote fig2_specialization.png")

# fig3 — dense-ECE is flat (NOT the differentiator)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
bars(axes[0], "ood", "succ_denseECE", "OOD", "dense-ECE (lower=better)", ylim=(0, 0.8))
bars(axes[1], "indist_icloff", "succ_denseECE", "In-distribution", "dense-ECE (lower=better)", ylim=(0, 0.8))
fig.suptitle("dense-ECE does NOT separate the models — it is not the differentiator", fontsize=13)
fig.tight_layout(); fig.savefig(FIG / "fig3_ece_flat.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote fig3_ece_flat.png")
print("done")
