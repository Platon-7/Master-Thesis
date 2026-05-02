"""Generate publication-quality figures for the RoboRef presentation.

All figures sized for the 13.33×7.50 in (1920×1080) UvA slide canvas.
"""
from __future__ import annotations

import os
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from PIL import Image

UVA_RED = "#BC0031"
UVA_INK = "#1F1D21"
UVA_ORANGE = "#E98300"
UVA_GREEN = "#257835"
UVA_YELLOW = "#BEB511"
UVA_PURPLE = "#751B68"
UVA_BLUE = "#004E92"
UVA_CYAN = "#2AA5D0"
SAND = "#F4F1EE"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.edgecolor": UVA_INK,
    "axes.labelcolor": UVA_INK,
    "xtick.color": UVA_INK,
    "ytick.color": UVA_INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})

ROOT = Path("/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer-LoRA/presentation")
FIG_DIR = ROOT / "figures"
ASSETS = ROOT / "assets"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def wrap(s, w=48):
    """Hard-wrap each newline-separated paragraph at width w."""
    return "\n".join(textwrap.fill(p, w) for p in s.split("\n"))


def card(ax, color, label, title, body, body_wrap=46):
    """Render a SAND-coloured card with a coloured left stripe."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
                boxstyle="round,pad=0.012,rounding_size=0.025",
                linewidth=0, facecolor=SAND))
    ax.add_patch(Rectangle((0.0, 0.0), 0.022, 1.0, facecolor=color, linewidth=0))
    ax.text(0.05, 0.92, label.upper(), fontsize=9, fontweight="bold",
            color="#666", va="top")
    ax.text(0.05, 0.82, title, fontsize=12.5, fontweight="bold", color=color, va="top")
    ax.text(0.05, 0.65, wrap(body, body_wrap), fontsize=10, color=UVA_INK, va="top")


def _score_color(s):
    return {1: "#9CA3AF", 2: UVA_YELLOW, 3: UVA_ORANGE,
            4: UVA_BLUE, 5: UVA_GREEN}.get(s, "#9CA3AF")


# --------------------------------------------------------------------------
# Slide 3 — Robometer overview
# --------------------------------------------------------------------------
def fig_robometer_overview():
    fig = plt.figure(figsize=(13.33, 6.5))
    fig.suptitle("Robometer  —  the current state of the art for general-purpose reward modelling",
                 fontsize=18, fontweight="bold", color=UVA_INK, x=0.025, ha="left", y=0.97)
    fig.text(0.025, 0.91,
             "Aliang et al., 2026  ·  4-billion-parameter VLM reward model  ·  trained on RBM-1M (~1.7M trajectories).",
             fontsize=11.5, color="#555")
    gs = fig.add_gridspec(2, 3, left=0.03, right=0.97, top=0.86, bottom=0.05,
                          hspace=0.35, wspace=0.20)
    cards = [
        ("Backbone", "Qwen3-VL-4B-Instruct", UVA_BLUE,
         "Multi-image vision–language transformer.\nInputs: trajectory frames + task instruction. "
         "Outputs are pooled per frame into a per-frame hidden state."),
        ("Three task heads", "progress  ·  success  ·  preference", UVA_RED,
         "Progress: 10-bin C51 distribution over [0,1]. Success: per-frame binary logit. "
         "Preference: pairwise ranking score for trajectory comparison."),
        ("Training signal", "Frame-level + trajectory-level", UVA_GREEN,
         "Progress on successes via t/T heuristic. Failures supervised only by the preference head "
         "via paired comparisons. No dense per-frame failure labels."),
        ("Dataset (RBM-1M)", "~1.7M trajectories  ·  93 archives", UVA_PURPLE,
         "Humanoid, human-hand, and standard arms across diverse embodiments and tasks. "
         "Successes 8× more abundant than failures — heavy success bias."),
        ("Evaluation paradigm", "Reward alignment  +  policy ranking", UVA_ORANGE,
         "Per-frame Spearman / Kendall on labelled progress. Ranking accuracy on held-out "
         "trajectory pairs across embodiments."),
        ("Open question for downstream RL", "False positives  →  reward hacking", UVA_CYAN,
         "Reward overestimation on failure trajectories breaks RL fine-tuning. Robometer's "
         "preference-only failure supervision leaves this unaddressed."),
    ]
    for k, (lbl, t, c, body) in enumerate(cards):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        card(ax, c, lbl, t, body, body_wrap=44)
    fig.savefig(FIG_DIR / "fig_robometer_overview.png")
    plt.close(fig); print("✓ fig_robometer_overview.png")


# --------------------------------------------------------------------------
# Slide 4 — Dataset breakdown
# --------------------------------------------------------------------------
def fig_dataset_breakdown():
    families = ["Humanoid", "Human / human-hand", "Standard robot arms"]
    archives = [8, 11, 74]
    successes = [551_147, 366_699, 558_476]
    failures = [0, 0, 215_537]
    totals = [s + f for s, f in zip(successes, failures)]

    fig = plt.figure(figsize=(13.33, 6.5))
    fig.suptitle("Robometer's dataset  —  RBM-1M composition by embodiment family",
                 fontsize=18, fontweight="bold", color=UVA_INK, x=0.025, ha="left", y=0.97)
    fig.text(0.025, 0.91,
             "1.69M trajectories from 93 archives  —  successes dominate; failures concentrate in standard arms.",
             fontsize=11.5, color="#555")

    gs = fig.add_gridspec(1, 5, left=0.03, right=0.98, top=0.84, bottom=0.10, wspace=0.45)
    ax_bar = fig.add_subplot(gs[0, :3])
    ax_kpi = fig.add_subplot(gs[0, 3:]); ax_kpi.axis("off")

    y = np.arange(len(families))
    bar_h = 0.55
    ax_bar.barh(y, successes, bar_h, color=UVA_BLUE, label="Successful trajectories")
    ax_bar.barh(y, failures, bar_h, left=successes, color=UVA_RED, label="Failure trajectories")
    for i, (s, f, t, a) in enumerate(zip(successes, failures, totals, archives)):
        ax_bar.text(t + 30_000, i + 0.05, f"{t/1000:,.0f}k episodes",
                    va="center", fontsize=11, color=UVA_INK, fontweight="bold")
        ax_bar.text(t + 30_000, i - 0.15, f"{a} archives",
                    va="center", fontsize=10, color="#666")
        if s > 0:
            ax_bar.text(s/2, i, f"{s/1000:,.0f}k", ha="center", va="center",
                        color="white", fontsize=11, fontweight="bold")
        if f > 0:
            ax_bar.text(s + f/2, i, f"{f/1000:,.0f}k", ha="center", va="center",
                        color="white", fontsize=11, fontweight="bold")

    ax_bar.set_yticks(y); ax_bar.set_yticklabels(families, fontsize=12.5)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Episodes")
    ax_bar.set_xlim(0, max(totals) * 1.55)
    ax_bar.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{int(x/1000):,}k"))
    ax_bar.legend(loc="upper right", frameon=False, fontsize=11,
                  bbox_to_anchor=(1.0, -0.08), ncol=2)
    ax_bar.grid(axis="x", alpha=0.25, linestyle=":")

    kpis = [
        ("1.69M", "Total episodes"),
        ("1.48M", "Successful trajectories"),
        ("215.5k", "Failure trajectories"),
        ("93", "Archives scanned"),
        ("68.9k", "ICL pairs (failure → demo)"),
    ]
    ax_kpi.set_xlim(0, 1); ax_kpi.set_ylim(0, 1)
    h = 0.165; pad = 0.025
    for i, (val, lbl) in enumerate(kpis):
        y0 = 1 - (i + 1) * (h + pad)
        ax_kpi.add_patch(FancyBboxPatch((0.02, y0), 0.96, h,
                        boxstyle="round,pad=0.012,rounding_size=0.018",
                        linewidth=0, facecolor=SAND))
        ax_kpi.text(0.06, y0 + h/2, val, fontsize=20, fontweight="bold",
                    color=UVA_RED, va="center")
        ax_kpi.text(0.42, y0 + h/2, lbl, fontsize=11, color=UVA_INK, va="center")
    ax_kpi.text(0.0, 1.02, "AT A GLANCE", fontsize=10, fontweight="bold",
                color="#666", va="bottom")

    fig.savefig(FIG_DIR / "fig_dataset_breakdown.png")
    plt.close(fig); print("✓ fig_dataset_breakdown.png")


# --------------------------------------------------------------------------
# Slide 5 — Contribution
# --------------------------------------------------------------------------
def fig_contribution():
    fig = plt.figure(figsize=(13.33, 6.5))
    fig.suptitle("Contribution  —  dense per-frame failure annotation at scale",
                 fontsize=20, fontweight="bold", color=UVA_INK, x=0.025, ha="left", y=0.97)
    fig.text(0.025, 0.91,
             "Two complementary annotation channels populate failures with the same ordinal scale that successes already enjoy.",
             fontsize=12, color="#555")

    gs = fig.add_gridspec(1, 2, left=0.03, right=0.98, top=0.84, bottom=0.06, wspace=0.06)

    def panel(ax, color, badge_label, headline, bullets, footer):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.01, 0.0), 0.98, 1.0,
                    boxstyle="round,pad=0.012,rounding_size=0.025",
                    linewidth=0, facecolor=SAND))
        ax.add_patch(Rectangle((0.01, 0.92), 0.98, 0.08, facecolor=color, linewidth=0))
        ax.text(0.5, 0.96, badge_label, fontsize=11, fontweight="bold",
                color="white", ha="center", va="center")
        ax.text(0.05, 0.83, headline, fontsize=14, fontweight="bold",
                color=color, va="top")
        for i, b in enumerate(bullets):
            ax.text(0.07, 0.71 - 0.10*i, "•", fontsize=12, color=color, va="top")
            ax.text(0.10, 0.71 - 0.10*i, wrap(b, 60), fontsize=10.3,
                    color=UVA_INK, va="top")
        for i, f in enumerate(footer):
            ax.text(0.05, 0.18 - 0.07*i, f, fontsize=10.5, color="#444",
                    style="italic", va="top")

    ax = fig.add_subplot(gs[0, 0])
    panel(ax, UVA_BLUE,
          "SIMULATOR-DERIVED   (Failsafe / MetaWorld)",
          "Procedural failure curriculum",
          [
              "Hand-crafted rubric injects 27 distinct failure modes per task.",
              "Three tasks (pick / push / stack) and three viewpoints (front / side / wrist).",
              "Per-frame label derived from simulator state (gripper pose, object-to-goal distance).",
              "Cleanest possible labels — used as the deciding evaluation split.",
          ],
          [
              "≈ 2,900 episodes  ·  3 tasks  ·  3 cameras",
              "Labels: {1, 2, 3, 4, 5}  —  no progress  →  success",
          ])

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, UVA_RED,
          "VLM + LLM   (DROID, Robometer Group A)",
          "Two-stage neural annotation pipeline",
          [
              "Stage 1 — Qwen3-VL describes each frame (objects, gripper state, sub-step status).",
              "Stage 2 — Qwen3-LLM scores progress on the rubric using descriptions + task prompt.",
              "Decoupling vision from reasoning suppresses hallucinated rewards.",
              "Scales to ~10,500 real-world failure trajectories without human labelling.",
          ],
          [
              "≈ 5,500 DROID  +  ≈ 5,000 Robometer Group A",
              "Labels: {1, 2, 3, 4}  —  failures only, paired with success demos",
          ])

    fig.savefig(FIG_DIR / "fig_contribution.png")
    plt.close(fig); print("✓ fig_contribution.png")


# --------------------------------------------------------------------------
# Slide 6 — Failsafe example
# --------------------------------------------------------------------------
def fig_failsafe_example():
    folder = ASSETS / "failsafe_example"
    n = 16
    scores = [1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1]
    files = sorted(folder.glob("frame_*.jpg"))
    imgs = [Image.open(p).convert("RGB") for p in files[:n]]

    fig = plt.figure(figsize=(13.33, 7.0))
    fig.suptitle("Failure annotation in simulation  —  Failsafe (ManiSkill, FailStackCube-v1)",
                 fontsize=18, fontweight="bold", color=UVA_INK, x=0.025, ha="left", y=0.97)
    fig.text(0.025, 0.92, "Task:  Pick up the red cube and stack it on top of the green cube.",
             fontsize=12, color=UVA_INK)
    fig.text(0.025, 0.89,
             "Failure scenario: grasp + carry away without dropping  ·  16 keyframes  ·  per-frame label from simulator state.",
             fontsize=11, color="#555")

    gs = fig.add_gridspec(3, 8, left=0.025, right=0.985, top=0.85, bottom=0.10,
                          hspace=0.30, wspace=0.10, height_ratios=[1.0, 1.0, 0.7])

    for k, (img, s) in enumerate(zip(imgs, scores)):
        r, c = k // 8, k % 8
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(_score_color(s)); sp.set_linewidth(3)
        ax.set_title(f"frame {k:02d}", fontsize=9.5, color="#444", pad=3)
        ax.add_patch(Rectangle((0.78, 0.78), 0.20, 0.20, transform=ax.transAxes,
                               facecolor=_score_color(s), edgecolor="white", linewidth=1.5))
        ax.text(0.88, 0.88, str(s), transform=ax.transAxes,
                ha="center", va="center", color="white", fontsize=12, fontweight="bold")

    ax_tr = fig.add_subplot(gs[2, :])
    ax_tr.plot(range(n), scores, color=UVA_RED, lw=2.5, marker="o", markersize=8,
               markerfacecolor=UVA_RED, markeredgecolor="white")
    ax_tr.fill_between(range(n), 0, scores, color=UVA_RED, alpha=0.10)
    ax_tr.set_xlim(-0.3, n - 0.7); ax_tr.set_ylim(0.5, 5.5)
    ax_tr.set_yticks([1, 2, 3, 4, 5])
    ax_tr.set_yticklabels(
        ["1  no progress", "2  approach", "3  grasp",
         "4  near completion", "5  success"], fontsize=10)
    ax_tr.set_xticks(range(n))
    ax_tr.set_xticklabels([f"{i:02d}" for i in range(n)], fontsize=9, color="#555")
    ax_tr.set_xlabel("frame index", fontsize=10)
    ax_tr.grid(axis="y", alpha=0.25, linestyle=":")
    ax_tr.set_title("Per-frame ground-truth ordinal label",
                    fontsize=11.5, loc="left", pad=6)

    fig.savefig(FIG_DIR / "fig_failsafe_example.png")
    plt.close(fig); print("✓ fig_failsafe_example.png")


# --------------------------------------------------------------------------
# Slide 7 — VLM + LLM example
# --------------------------------------------------------------------------
def fig_vlm_llm_example():
    folder = ASSETS / "droid_example"
    files = sorted(folder.glob("frame_*.jpg"))
    keep = [0, 2, 4, 6, 8, 10, 12, 14]
    scores = [1, 2, 2, 2, 3, 4, 3, 3]
    imgs = [Image.open(files[i]).convert("RGB") for i in keep]

    fig = plt.figure(figsize=(13.33, 7.0))
    fig.suptitle("Failure annotation in the wild  —  DROID, two-stage VLM + LLM pipeline",
                 fontsize=18, fontweight="bold", color=UVA_INK, x=0.025, ha="left", y=0.97)
    fig.text(0.025, 0.92, "Task:  Pack towel into container.", fontsize=12, color=UVA_INK)
    fig.text(0.025, 0.89,
             "Pipeline:  Qwen3-VL describes each frame  →  Qwen3-LLM scores progress 1–4 on the rubric using description + task prompt.",
             fontsize=11, color="#555")

    gs = fig.add_gridspec(2, 8, left=0.025, right=0.985, top=0.85, bottom=0.30,
                          hspace=0.30, wspace=0.10, height_ratios=[1.0, 0.7])

    for k, (img, s) in enumerate(zip(imgs, scores)):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(_score_color(s)); sp.set_linewidth(3)
        ax.set_title(f"frame {keep[k]:02d}", fontsize=9.5, color="#444", pad=3)
        ax.add_patch(Rectangle((0.78, 0.78), 0.20, 0.20, transform=ax.transAxes,
                               facecolor=_score_color(s), edgecolor="white", linewidth=1.5))
        ax.text(0.88, 0.88, str(s), transform=ax.transAxes, ha="center", va="center",
                color="white", fontsize=12, fontweight="bold")

    ax_tr = fig.add_subplot(gs[1, :])
    xs = list(range(len(scores)))
    ax_tr.plot(xs, scores, color=UVA_RED, lw=2.5, marker="o", markersize=9,
               markerfacecolor=UVA_RED, markeredgecolor="white")
    ax_tr.fill_between(xs, 0, scores, color=UVA_RED, alpha=0.10)
    ax_tr.set_xticks(xs); ax_tr.set_xticklabels([f"{keep[i]:02d}" for i in xs], fontsize=10)
    ax_tr.set_yticks([1, 2, 3, 4]); ax_tr.set_yticklabels(["1", "2", "3", "4"], fontsize=10)
    ax_tr.set_ylim(0.5, 4.5); ax_tr.set_xlabel("frame index", fontsize=10)
    ax_tr.grid(axis="y", alpha=0.25, linestyle=":")
    ax_tr.annotate("regression", xy=(6, 3), xytext=(6.2, 4.2),
                   fontsize=10, color=UVA_RED, ha="left",
                   arrowprops=dict(arrowstyle="->", color=UVA_RED))
    ax_tr.set_title("LLM-assigned ordinal progress score per keyframe",
                    fontsize=11.5, loc="left", pad=6)

    # Sample VLM + LLM trace at the bottom (single representative frame)
    fig.text(0.025, 0.225, "Representative pipeline trace  (frame 10, score 4)",
             fontsize=11.5, fontweight="bold", color=UVA_INK)
    fig.text(0.025, 0.18,
             "VLM description:  \"1. Black container on table, white towel in container, robot arm with gripper. "
             "2. Gripper is holding the towel above the container. "
             "3. The towel has been grasped and is being placed into the container.\"",
             fontsize=9.8, color="#444", wrap=False)
    fig.text(0.025, 0.06,
             "LLM verdict:  ANSWER: 4 because the robot has grasped the towel and is in the process of placing it "
             "into the container, completing most sub-steps of the task.",
             fontsize=9.8, color=UVA_RED)

    fig.savefig(FIG_DIR / "fig_vlm_llm_example.png")
    plt.close(fig); print("✓ fig_vlm_llm_example.png")


# --------------------------------------------------------------------------
# Slide 8 — Loss comparison
# --------------------------------------------------------------------------
def fig_loss_comparison():
    fig = plt.figure(figsize=(13.33, 6.7))
    fig.suptitle("Three candidate objective functions",
                 fontsize=20, fontweight="bold", color=UVA_INK, x=0.025, ha="left", y=0.98)
    fig.text(0.025, 0.93,
             "Same backbone, same data, same LoRA setup  —  the only thing that varies is the loss applied to the heads.",
             fontsize=12, color="#555")

    gs = fig.add_gridspec(2, 3, left=0.03, right=0.98, top=0.86, bottom=0.06,
                          height_ratios=[1.4, 1.0], hspace=0.45, wspace=0.20)

    titles = ["Robometer  (released baseline)",
              "Loss 1  ·  Asymmetric ordinal CORN",
              "Loss 2  ·  Asymmetric C51 + asymmetric BCE"]
    subtitles = [
        "Heads: progress (C51, 10 bins) + binary success + pairwise preference. "
        "Failures supervised only via the preference head.",
        "Single 4-logit cumulative head replaces all three. Penalises over-prediction "
        "with a class-graded weight α_k = 1 + c·(k−2).",
        "Keeps Robometer's progress and success heads. Adds a per-element asymmetric "
        "weight λ that damps under-confident negatives.",
    ]
    colors = ["#7C7C7C", UVA_RED, UVA_BLUE]

    def head_diagram(ax, mode, color):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        if mode == "robometer":
            heads = [("progress\nC51 × 10", UVA_BLUE),
                     ("success\nbinary", UVA_GREEN),
                     ("preference\npairwise", UVA_PURPLE)]
            for j, (lbl, col) in enumerate(heads):
                cx = 0.05 + j * 0.32
                ax.add_patch(FancyBboxPatch((cx, 0.10), 0.27, 0.78,
                            boxstyle="round,pad=0.005,rounding_size=0.04",
                            linewidth=2, edgecolor=col, facecolor="white"))
                ax.text(cx + 0.135, 0.49, lbl, fontsize=10, color=col,
                        ha="center", va="center", fontweight="bold")
        elif mode == "corn":
            ax.add_patch(FancyBboxPatch((0.20, 0.10), 0.60, 0.78,
                        boxstyle="round,pad=0.005,rounding_size=0.04",
                        linewidth=2.2, edgecolor=color, facecolor="white"))
            ax.text(0.50, 0.49, "CORN ordinal head\n4 logits  ·  P(y ≥ k)",
                    fontsize=11, color=color, ha="center", va="center", fontweight="bold")
        else:
            heads = [("progress\nC51 × 10\n(asym. CE)", UVA_BLUE),
                     ("success\nbinary\n(asym. BCE)", UVA_GREEN)]
            for j, (lbl, col) in enumerate(heads):
                cx = 0.10 + j * 0.42
                ax.add_patch(FancyBboxPatch((cx, 0.10), 0.36, 0.78,
                            boxstyle="round,pad=0.005,rounding_size=0.04",
                            linewidth=2, edgecolor=col, facecolor="white"))
                ax.text(cx + 0.18, 0.49, lbl, fontsize=10, color=col,
                        ha="center", va="center", fontweight="bold")

    modes = ["robometer", "corn", "two-head"]
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
                    boxstyle="round,pad=0.012,rounding_size=0.025",
                    linewidth=0, facecolor=SAND))
        ax.text(0.04, 0.95, titles[i], fontsize=13, fontweight="bold", color=colors[i], va="top")
        ax.text(0.04, 0.85, wrap(subtitles[i], 42), fontsize=10, color=UVA_INK, va="top")
        # head diagram occupies the bottom 30% of the card; transform via parent bbox
        bbox = ax.get_position()
        sub = fig.add_axes([bbox.x0 + 0.010, bbox.y0 + 0.012,
                            bbox.width - 0.020, bbox.height * 0.28])
        head_diagram(sub, modes[i], colors[i])

    formulas = [
        r"$L = L_{\mathrm{progress}}^{\mathrm{C51}} + L_{\mathrm{success}}^{\mathrm{BCE}} + L_{\mathrm{pref}}^{\mathrm{rank}}$",
        r"$L = -\sum_{t,k}\,\beta_k\,b_{t,k}\log\sigma(z_{t,k}) + \alpha_k\,(1-b_{t,k})\log(1-\sigma(z_{t,k}))$",
        r"$L_{\mathrm{prog}} = w\cdot\mathrm{CE}(p,p^*),\;\; w = \mathbf{1}[\hat p > p^*] + \lambda\,\mathbf{1}[\hat p \leq p^*]$",
    ]
    notes = [
        "•  Symmetric C51 / BCE.\n•  No supervision on failure progress.\n•  Fits demos; brittle as RL reward.",
        "•  Cumulative thresholds — ordinal-aware.\n•  α_k > β_k punishes over-prediction.\n•  Single head → calibrated P(success) = σ(z_5).",
        "•  Pretrained heads kept, no re-init.\n•  λ damps the under-prediction direction.\n•  Progress + success summed unweighted.",
    ]
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.text(0.0, 0.92, formulas[i], fontsize=10.5 if i == 1 else 12,
                color=colors[i], va="top")
        ax.text(0.0, 0.55, notes[i], fontsize=10.3, color=UVA_INK, va="top")

    fig.savefig(FIG_DIR / "fig_loss_comparison.png")
    plt.close(fig); print("✓ fig_loss_comparison.png")


# --------------------------------------------------------------------------
# Slide 9 — Training strategy
# --------------------------------------------------------------------------
def fig_training_strategy():
    fig = plt.figure(figsize=(13.33, 6.8))
    fig.suptitle("Training strategy",
                 fontsize=20, fontweight="bold", color=UVA_INK, x=0.025, ha="left", y=0.97)
    fig.text(0.025, 0.92,
             "LoRA fine-tune of Robometer-4B on ~18,900 balanced (failure | success) ICL pairs.",
             fontsize=12, color="#555")

    gs = fig.add_gridspec(2, 3, left=0.03, right=0.98, top=0.86, bottom=0.05,
                          hspace=0.40, wspace=0.20)
    panels = [
        ("In-context learning  ·  per-example coin flip",
         "Each sample independently draws ICL on (prepend a success demo) or ICL off "
         "(query only) with p = 0.5. The demo defines what 'progress' means for this task.",
         UVA_RED),
        ("Balanced batches  ·  50/50 success–failure",
         "Failure-query and success-query examples are constructed in equal numbers (~9.4k each). "
         "A stratified sampler enforces exact 50/50 within each batch after a short warmup.",
         UVA_BLUE),
        ("LoRA adapters on Robometer-4B",
         "rank 32, α = 64, dropout 0.05.  Adapters on q/k/v/o + MLP gate/up/down. "
         "Backbone frozen, heads trained fully.  bf16 forward, fp32 adapters & heads.",
         UVA_GREEN),
        ("Optimisation schedule",
         "AdamW · lr 1e-4 (adapters) · lr 5e-5 (heads) · weight decay 0.01.\n"
         "Linear warmup over 5% of steps → cosine decay to 10% of peak.\n"
         "7,500 steps · batch 8 · grad-clip 1.0 · seed 42.",
         UVA_PURPLE),
        ("Two-phase warmup",
         "First N steps draw failure-only batches to bootstrap the ordinal head before "
         "exposing the model to successes. Loss 1: N = 2,000. Loss 2: N = 1,000.",
         UVA_ORANGE),
        ("KL rehearsal anchor  ·  planned, full fine-tune",
         "FIFO buffer of past failure logits. On each success step sample one and add "
         "λ_KL · KL(P_old ∥ P_new) to prevent failure-prediction drift during success-heavy phases.",
         UVA_CYAN),
    ]
    for k, (title, body, c) in enumerate(panels):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        card(ax, c, "", title, body, body_wrap=46)

    fig.savefig(FIG_DIR / "fig_training_strategy.png")
    plt.close(fig); print("✓ fig_training_strategy.png")


if __name__ == "__main__":
    print("starting figure generation...")
    fig_robometer_overview()
    fig_dataset_breakdown()
    fig_contribution()
    fig_failsafe_example()
    fig_vlm_llm_example()
    fig_loss_comparison()
    fig_training_strategy()
    print("All figures written to:", FIG_DIR)
