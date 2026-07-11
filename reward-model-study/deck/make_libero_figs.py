"""Figures for the LIBERO progress-head diagnosis deck.

Palette (validated categorical slots): baseline blue #2a78d6, run2 orange #eb6834,
run3 violet #4a3aa7; failure/gray #9a9992; ink #1F3864 to match the deck.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DECK = "/shared/home/PKA4388/Master-Thesis/reward-model-study/deck"
FIGS = os.path.join(DECK, "figs_libero")
os.makedirs(FIGS, exist_ok=True)
PROBE = json.load(open("/shared/home/PKA4388/robometer-policy-learning/plots/train_replay_probe.json"))
CURVES = json.load(open("/tmp/claude-883810755/-shared-home-PKA4388/5d810315-34a8-4488-8367-319e3d8eac45/scratchpad/deck_curves.json"))

INK = "#1F3864"
C_BASE, C_RUN2, C_RUN3, C_GRAY = "#2a78d6", "#eb6834", "#4a3aa7", "#9a9992"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.edgecolor": "#c8c8c8", "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.7,
    "axes.axisbelow": True, "figure.facecolor": "white",
})

def m(model, cls, key):
    rows = PROBE[model][cls]
    return float(np.mean([r[key] for r in rows]))

def bins_mean(model):
    rows = [r for r in PROBE[model]["success"] if r.get("bins")]
    return np.mean(np.array([r["bins"] for r in rows], float), axis=0)

def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# ---------------------------------------------------------------- fig 1: heads
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
models = [("baseline", "Robometer-4B\n(baseline)", C_BASE),
          ("run3_std", "run3\n(standard loss)", C_RUN3),
          ("run2_asym", "run2\n(asymmetric loss)", C_RUN2)]
for ax, key, title in [(axes[0], "prog_full", "Progress head, final-frame reading"),
                       (axes[1], "succ_full", "Success head, final-frame reading")]:
    x = np.arange(3)
    sv = [m(k, "success", key) for k, _, _ in models]
    fv = [m(k, "failure", key) for k, _, _ in models]
    b1 = ax.bar(x - 0.19, sv, 0.34, color=[c for _, _, c in models], label="solved episodes")
    b2 = ax.bar(x + 0.19, fv, 0.34, color=C_GRAY, label="failed episodes")
    for r in list(b1) + list(b2):
        ax.annotate(f"{r.get_height():.2f}", (r.get_x() + r.get_width()/2, r.get_height()),
                    ha="center", va="bottom", fontsize=10.5)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n, _ in models], fontsize=10.5)
    ax.set_ylim(0, 1.12); ax.set_title(title, fontsize=12.5, color=INK, pad=30)
    style(ax)
axes[0].set_ylabel("model output")
axes[1].legend(frameon=False, fontsize=10, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
axes[0].annotate("reads its own training\ndemos at 0.25", xy=(2.0, 0.30), xytext=(1.05, 0.80),
                 fontsize=10.5, color=C_RUN2, ha="center",
                 arrowprops=dict(arrowstyle="->", color=C_RUN2, lw=1.4))
fig.suptitle("Scoring each model on the 'close the drawer' training clips it has seen", fontsize=13.5, color=INK, y=1.08)
fig.tight_layout()
fig.savefig(f"{FIGS}/fig1_heads.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------- fig 2: C51 distributions
fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.1), sharey=True)
centers = np.linspace(0, 1, 10)
for ax, (k, name, c) in zip(axes, models):
    b = bins_mean(k)
    ax.bar(centers, b, width=0.085, color=c)
    ev = float((b * centers).sum())
    ax.axvline(ev, color=INK, lw=1.6, ls="--")
    ax.annotate(f"mean readout\n= {ev:.2f}", xy=(ev, max(b)*0.92), xytext=(0.38, max(b)*0.8),
                fontsize=10, color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    ax.set_title(name.replace("\n", " "), fontsize=12, color=c)
    ax.set_xlabel("progress value (10 bins)")
    ax.set_xticks([0, 0.5, 1.0]); style(ax)
axes[0].set_ylabel("probability mass")
fig.suptitle("What the progress head actually outputs on a solved frame: a distribution over 10 progress bins",
             fontsize=13.5, color=INK, y=1.04)
fig.tight_layout()
fig.savefig(f"{FIGS}/fig2_c51.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------ fig 3: window renormalization
fig, ax = plt.subplots(figsize=(9.8, 3.3))
n = 20; frame = 12
ax.set_xlim(-0.8, n + 4.4); ax.set_ylim(-0.6, 3.6); ax.axis("off")
for row, (a, b, tgt, lab) in enumerate([
        (0, 19, (frame + 1) / 20, "window = whole episode"),
        (8, 16, (frame - 8) / (16 - 8), "window starts at frame 8"),
        (12, 19, 0.0, "window starts at frame 12")]):
    y = 2.6 - row * 1.15
    for i in range(n):
        inside = a <= i <= b
        col = "#dce7f7" if inside else "#f2f2f0"
        edge = "#7ba7dd" if inside else "#d8d8d4"
        if i == frame:
            col = C_RUN2; edge = C_RUN2
        ax.add_patch(plt.Rectangle((i, y), 0.86, 0.62, facecolor=col, edgecolor=edge, lw=1.0))
    ax.text(n + 0.6, y + 0.31, f"target for frame {frame}:  {tgt:.2f}",
            va="center", fontsize=12.5, color=INK,
            fontweight="bold" if row else "normal")
    ax.text(a, y + 0.86, lab, fontsize=10.5, color="#555555")
ax.text(frame + 0.43, 3.45, f"the same frame {frame}\n(drawer 60 percent closed)", ha="center",
        fontsize=10.5, color=C_RUN2)
fig.tight_layout()
fig.savefig(f"{FIGS}/fig3_windows.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------- fig 4: landscape + RL curves
fig = plt.figure(figsize=(10.8, 3.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.25)
# left: reward landscape dot plot
ax = fig.add_subplot(gs[0])
land = [("baseline", C_BASE, 0.25, 0.77, 0.90),
        ("run2", C_RUN2, 0.03, 0.20, 0.25)]
for i, (name, c, start, hover, solved) in enumerate(land):
    y = 1 - i
    ax.plot([start, solved], [y, y], color=c, lw=2.4, alpha=0.35, zorder=1)
    ax.scatter([start], [y], s=90, color="white", edgecolor=c, lw=2, zorder=3)
    ax.scatter([hover], [y], s=90, color=c, alpha=0.45, zorder=3)
    ax.scatter([solved], [y], s=110, color=c, zorder=3)
    gap = solved - hover
    ax.annotate("", xy=(solved, y + 0.22), xytext=(hover, y + 0.22),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
    ax.text((hover + solved) / 2, y + 0.3, f"margin {gap:.2f}", ha="center", fontsize=10.5, color=INK)
    ax.text(-0.03, y, name, ha="right", va="center", fontsize=12, color=c, fontweight="bold")
ax.scatter([], [], s=90, color="white", edgecolor=INK, lw=2, label="episode start")
ax.scatter([], [], s=90, color=INK, alpha=0.45, label="hovering at the drawer")
ax.scatter([], [], s=110, color=INK, label="drawer closed")
ax.legend(frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(0.0, 1.22), ncol=3, columnspacing=0.8)
ax.set_xlim(-0.28, 1.0); ax.set_ylim(-0.5, 1.75)
ax.set_yticks([]); ax.set_xlabel("progress reward (mean readout)")
style(ax); ax.grid(axis="y", visible=False)
# right: RL curves (real clips, mean readout)
ax = fig.add_subplot(gs[1])
cb = CURVES["accum_baseline"][0]; cr = CURVES["accum_run2"][0]
ax.plot(np.array(cb["steps"]) / 1000, cb["succ"], color=C_BASE, lw=2.2, label="baseline")
ax.plot(np.array(cr["steps"]) / 1000, cr["succ"], color=C_RUN2, lw=2.2, label="run2")
ax.annotate("holds 100 percent,\nthen decays", xy=(75, 50), xytext=(48, 20),
            fontsize=10.5, color=C_RUN2, arrowprops=dict(arrowstyle="->", color=C_RUN2, lw=1.2))
ax.set_xlabel("training steps (thousands)"); ax.set_ylabel("evaluation success rate (percent)")
ax.set_ylim(-4, 104); ax.legend(frameon=False, fontsize=10.5, loc="lower right")
ax.set_title("Reinforcement learning on task 28, correct video input, one seed", fontsize=12, color=INK)
style(ax)
fig.tight_layout()
fig.savefig(f"{FIGS}/fig4_hacking.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# -------------------------------------------------- fig 5: frame bug effect bars + RL
fig = plt.figure(figsize=(10.8, 3.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.28)
ax = fig.add_subplot(gs[0])
x = np.arange(2)
real = [m("baseline", "success", "succ_full"), m("run2_asym", "success", "succ_full")]
tile = [m("baseline", "success", "succ_tile"), m("run2_asym", "success", "succ_tile")]
b1 = ax.bar(x - 0.19, real, 0.34, color=[C_BASE, C_RUN2], label="real 16-frame clip")
b2 = ax.bar(x + 0.19, tile, 0.34, color=[C_BASE, C_RUN2], alpha=0.35, hatch="//", label="one frame repeated 16 times")
for r in list(b1) + list(b2):
    ax.annotate(f"{r.get_height():.2f}", (r.get_x() + r.get_width()/2, r.get_height()),
                ha="center", va="bottom", fontsize=10.5)
ax.axhline(0.65, color=INK, lw=1.2, ls=":")
ax.text(0.5, 0.665, "termination threshold 0.65", fontsize=9.5, color=INK, va="bottom", ha="center")
ax.set_xticks(x); ax.set_xticklabels(["baseline", "run2"], fontsize=11.5)
ax.set_ylim(0, 1.1); ax.set_ylabel("success head on solved clips")
ax.legend(frameon=False, fontsize=9.5, loc="upper right", bbox_to_anchor=(1.02, 1.18))
style(ax)
# right: single-frame vs real-clip baseline RL
ax = fig.add_subplot(gs[1])
grid = np.arange(5000, 100001, 5000)
mats = []
for c in CURVES["singleframe_baseline"]:
    v = np.interp(grid, c["steps"], c["succ"], left=0, right=np.nan)
    v[grid > max(c["steps"]) + 1] = np.nan
    mats.append(v)
mat = np.vstack(mats)
mean, std = np.nanmean(mat, 0), np.nanstd(mat, 0)
ax.plot(grid/1000, mean, color=C_BASE, lw=2.2, label="repeated single frame (5 seeds)")
ax.fill_between(grid/1000, np.clip(mean-std, 0, 100), np.clip(mean+std, 0, 100), color=C_BASE, alpha=0.15)
cb = CURVES["accum_baseline"][0]
ax.plot(np.array(cb["steps"])/1000, cb["succ"], color=C_BASE, lw=2.0, ls="--", label="correct video input (1 seed)")
ax.set_xlabel("training steps (thousands)"); ax.set_ylabel("evaluation success rate (percent)")
ax.set_ylim(-4, 104); ax.legend(frameon=False, fontsize=10, loc="lower right")
ax.set_title("Baseline reinforcement learning, with and without the bug", fontsize=12, color=INK)
style(ax)
fig.tight_layout()
fig.savefig(f"{FIGS}/fig5_framebug.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------ fig 6: readout comparison
fig = plt.figure(figsize=(10.8, 3.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.3)
ax = fig.add_subplot(gs[0])
b = bins_mean("run2_asym")
ax.bar(centers, b, width=0.085, color=C_RUN2)
ev = float((b*centers).sum())
ax.axvline(ev, color=INK, lw=1.6, ls="--")
ax.annotate(f"mean readout = {ev:.2f}\n(a value the model gives\n3 percent probability)",
            xy=(ev, 0.42), xytext=(0.33, 0.44), fontsize=10, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
ax.annotate(f"top-bin readout = {b[-1]:.2f}\n('probability the task\nis complete')",
            xy=(0.985, b[-1]), xytext=(0.47, 0.06), fontsize=10, color=C_RUN2,
            arrowprops=dict(arrowstyle="->", color=C_RUN2, lw=1.2))
ax.set_xlabel("progress value (10 bins)"); ax.set_ylabel("probability mass")
ax.set_title("run2 on a solved frame: two readouts of the same output", fontsize=12, color=INK)
style(ax)
ax = fig.add_subplot(gs[1])
labels = ["mean readout", "top-bin readout"]
solved = [m("run2_asym", "success", "prog_full"), 0.216]
failed = [m("run2_asym", "failure", "prog_full"), 0.066]
x = np.arange(2)
b1 = ax.bar(x - 0.19, solved, 0.34, color=C_RUN2, label="solved episodes")
b2 = ax.bar(x + 0.19, failed, 0.34, color=C_GRAY, label="failed episodes")
for r in list(b1) + list(b2):
    ax.annotate(f"{r.get_height():.2f}", (r.get_x() + r.get_width()/2, r.get_height()),
                ha="center", va="bottom", fontsize=10.5)
ax.text(0, 0.30, "1.8 : 1", ha="center", fontsize=11.5, color=INK, fontweight="bold")
ax.text(1, 0.30, "3.3 : 1", ha="center", fontsize=11.5, color=INK, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11.5)
ax.set_ylim(0, 0.4); ax.set_ylabel("progress reading")
ax.legend(frameon=False, fontsize=10, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2)
ax.set_title("Separation of solved versus failed", fontsize=12, color=INK, pad=28)
style(ax)
fig.tight_layout()
fig.savefig(f"{FIGS}/fig6_readout.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------ fig 7: early downstream results
fig, ax = plt.subplots(figsize=(9.6, 3.4))
cr = CURVES["accum_run2"][0]
ax.plot(np.array(cr["steps"])/1000, cr["succ"], color=C_GRAY, lw=2.0, label="run2, mean readout (decays)")
cp = CURVES["ptop_run2_partial"][0]
ax.plot(np.array(cp["steps"])/1000, cp["succ"], color=C_RUN2, lw=2.2, label="run2, top-bin readout (interrupted at 65 thousand)")
ct = CURVES["term2_run2_partial"][0]
ax.plot(np.array(ct["steps"])/1000, ct["succ"], color=C_RUN3, lw=2.2,
        label="run2, success-head termination, calibrated threshold (interrupted at 30 thousand)")
for c, col in [(cp, C_RUN2), (ct, C_RUN3)]:
    ax.scatter([max(c["steps"])/1000], [c["succ"][-1]], marker="x", s=70, color=col, zorder=4)
ax.set_xlabel("training steps (thousands)"); ax.set_ylabel("evaluation success rate (percent)")
ax.set_ylim(-4, 108); ax.set_xlim(0, 100)
ax.legend(frameon=False, fontsize=9.5, loc="lower right")
style(ax)
fig.tight_layout()
fig.savefig(f"{FIGS}/fig7_early.png", dpi=200, bbox_inches="tight"); plt.close(fig)

print("figures written to", FIGS)
for f in sorted(os.listdir(FIGS)):
    print(" ", f)
