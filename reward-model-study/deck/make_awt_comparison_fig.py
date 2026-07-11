"""Matched-checkpoint C51 comparison: does the bimodal hedge survive the target fix?

Three panels, all step-4000 asymmetric-loss checkpoints scored on the same 30 solved
task-28 training clips (final frame): baseline (reference shape), run2 with the
window-relative targets (absolute_first_frame), run2 with the fixed whole-episode
targets (absolute_wrt_total_frames). Failure-side means quoted in each panel.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DECK = "/shared/home/PKA4388/Master-Thesis/reward-model-study/deck"
FIGS = os.path.join(DECK, "figs_libero")
P_MAIN = json.load(open("/shared/home/PKA4388/robometer-policy-learning/plots/train_replay_probe.json"))
P_AWT = json.load(open("/shared/home/PKA4388/robometer-policy-learning/plots/train_replay_probe_awt.json"))

INK = "#1F3864"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.edgecolor": "#c8c8c8", "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.7,
    "axes.axisbelow": True, "figure.facecolor": "white",
})

def bins_mean(probe, model, cls="success"):
    rows = [r for r in probe[model][cls] if r.get("bins")]
    return np.mean(np.array([r["bins"] for r in rows], float), axis=0)

def mean_val(probe, model, cls, key):
    return float(np.mean([r[key] for r in probe[model][cls]]))

import sys
WHICH = sys.argv[1] if len(sys.argv) > 1 else "run2"
if WHICH == "run3":
    panels = [
        ("baseline", P_MAIN, "Robometer-4B (baseline)", "#2a78d6"),
        ("run3_std", P_MAIN, "run3, window targets (step 5000)", "#4a3aa7"),
        ("run3_awt_s4500", P_AWT, "run3, fixed targets (step 4500)", "#2e2373"),
    ]
else:
    panels = [
        ("baseline", P_MAIN, "Robometer-4B (baseline)", "#2a78d6"),
        ("run2_asym", P_MAIN, "run2, window targets (step 4000)", "#eb6834"),
        ("run2_awt_s4000", P_AWT, "run2, fixed targets (step 4000)", "#c2451e"),
    ]
centers = np.linspace(0, 1, 10)
fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.3), sharey=True)
for ax, (key, probe, title, c) in zip(axes, panels):
    b = bins_mean(probe, key)
    ax.bar(centers, b, width=0.085, color=c)
    ev = float((b * centers).sum())
    ax.axvline(ev, color=INK, lw=1.6, ls="--")
    ax.annotate(f"mean readout = {ev:.2f}", xy=(ev, max(b) * 0.9), xytext=(0.36, max(b) * 0.97),
                fontsize=10, color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    fs = mean_val(probe, key, "failure", "prog_full")
    ss = mean_val(probe, key, "success", "prog_full")
    ax.text(0.03, max(b) * 0.62, f"solved read {ss:.2f}\nfailed read {fs:.2f}",
            fontsize=9.5, color="#555555")
    ax.set_title(title, fontsize=11.5, color=c)
    ax.set_xlabel("progress value (10 bins)")
    ax.set_xticks([0, 0.5, 1.0])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("probability mass")
fig.suptitle("Same loss, same step, same solved training clips: only the progress-target scheme differs",
             fontsize=13, color=INK, y=1.04)
fig.tight_layout()
out = f"{FIGS}/fig8_awt_comparison.png" if WHICH == "run2" else f"{FIGS}/fig9_awt_run3_comparison.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
