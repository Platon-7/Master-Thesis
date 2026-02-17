"""
Plot training results for DSRL + RoboReward VLM experiments.

Parses .out log files and generates:
  1. Training curve (ep_rew_mean vs steps) — GT baseline vs VLM
  2. Success rate curve (Figure 2 style) — Sim Success % vs steps
  3. VLM reward analysis — VLM Score, VLM vs Sim agreement
  4. Loss curves — actor/critic loss comparison
"""

import re
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DSRL_DIR = SCRIPT_DIR.parent
OUT_DIR = SCRIPT_DIR

VLM_LOG = DSRL_DIR / "dsrl_vlm_91539.out"
GT_LOG = DSRL_DIR / "dsrl_gt_91452.out"


# ── Parsers ────────────────────────────────────────────────────────────────────

def parse_sb3_metrics(filepath):
    """Parse SB3-style rollout/train log blocks from a .out file."""
    data = {
        "total_timesteps": [],
        "ep_rew_mean": [],
        "episodes": [],
        "actor_loss": [],
        "critic_loss": [],
        "ent_coef": [],
        "noise_critic_loss": [],
    }
    text = filepath.read_text()

    # Match metric lines like: |    ep_rew_mean       | -264     |
    pattern = re.compile(r"\|\s+([\w/]+)\s+\|\s+([\d.e+\-]+)\s+\|")

    current_block = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("---"):
            # End of block — flush if we have timesteps
            if "total_timesteps" in current_block:
                ts = current_block["total_timesteps"]
                for key in data:
                    if key in current_block:
                        data[key].append(current_block[key])
                    elif key == "total_timesteps":
                        pass  # already handled
                    else:
                        data[key].append(np.nan)
            current_block = {}
            continue

        m = pattern.match(line)
        if m:
            name = m.group(1).strip()
            value = float(m.group(2))
            current_block[name] = value

    # Convert to numpy
    for key in data:
        data[key] = np.array(data[key])

    return data


def parse_vlm_evals(filepath):
    """Parse VLM evaluation lines: [Step X] VLM Score: ..., VLM Success: ..., Sim Success: ..."""
    steps = []
    vlm_scores = []
    vlm_success = []
    sim_success = []

    pattern = re.compile(
        r"\[Step (\d+)\] VLM Score: ([\d.]+), VLM Success: ([\d.]+)%, Sim Success: ([\d.]+)%"
    )

    text = filepath.read_text()
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            steps.append(int(m.group(1)))
            vlm_scores.append(float(m.group(2)))
            vlm_success.append(float(m.group(3)))
            sim_success.append(float(m.group(4)))

    return {
        "step": np.array(steps),
        "vlm_score": np.array(vlm_scores),
        "vlm_success": np.array(vlm_success),
        "sim_success": np.array(sim_success),
    }


def smooth(y, window=20):
    """Simple moving average for smoothing noisy curves."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    # Pad edges to avoid shrinking
    padded = np.concatenate([np.full(window // 2, y[0]), y, np.full(window // 2, y[-1])])
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(y)]


# ── Load data ──────────────────────────────────────────────────────────────────

print("Parsing VLM run (91539)...")
vlm_sb3 = parse_sb3_metrics(VLM_LOG)
vlm_eval = parse_vlm_evals(VLM_LOG)

print("Parsing GT run (91452)...")
gt_sb3 = parse_sb3_metrics(GT_LOG)

print(f"  VLM: {len(vlm_sb3['total_timesteps'])} SB3 blocks, {len(vlm_eval['step'])} VLM evals")
print(f"  GT:  {len(gt_sb3['total_timesteps'])} SB3 blocks")

# Use total_timesteps as x-axis (RL steps, comparable across runs)
vlm_steps = vlm_sb3["total_timesteps"]
gt_steps = gt_sb3["total_timesteps"]

# ── Plot style ─────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
PURPLE = "#9467bd"


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Training Curve (ep_rew_mean vs steps)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

# GT baseline — limit to same step range for fair comparison
gt_mask = gt_steps <= 250000
ax.plot(gt_steps[gt_mask] / 1000, gt_sb3["ep_rew_mean"][gt_mask],
        alpha=0.15, color=BLUE, linewidth=0.5)
ax.plot(gt_steps[gt_mask] / 1000, smooth(gt_sb3["ep_rew_mean"][gt_mask]),
        color=BLUE, linewidth=2, label="GT Rewards (baseline)")

# VLM run — plot on secondary y-axis since scales differ
ax2 = ax.twinx()
ax2.plot(vlm_steps / 1000, vlm_sb3["ep_rew_mean"],
         alpha=0.15, color=ORANGE, linewidth=0.5)
ax2.plot(vlm_steps / 1000, smooth(vlm_sb3["ep_rew_mean"]),
         color=ORANGE, linewidth=2, label="VLM Rewards (RoboReward-8B)")

ax.set_xlabel("Training Steps (x1000)")
ax.set_ylabel("Episode Reward (GT)", color=BLUE)
ax2.set_ylabel("Episode Reward (VLM)", color=ORANGE)
ax.tick_params(axis="y", labelcolor=BLUE)
ax2.tick_params(axis="y", labelcolor=ORANGE)

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

ax.set_title("Training Curves: GT vs VLM Rewards")
ax.set_xlim(0, 250)
fig.tight_layout()
fig.savefig(OUT_DIR / "training_curves.png", bbox_inches="tight")
fig.savefig(OUT_DIR / "training_curves.pdf", bbox_inches="tight")
print("Saved: training_curves.png/pdf")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Success Rate vs Steps (reproducing paper's Figure 2)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

eval_steps = vlm_eval["step"]
# Normalize eval steps to start from 0
eval_steps_norm = eval_steps - eval_steps[0]

# Convert to "episodes" scale (each eval is every 4 episodes = 1200 sim steps)
# Map eval_steps to total_timesteps: the eval happens every 300 RL steps
# Each eval corresponds to ~4 episodes (n_envs=4, ep_len=75)
n_evals = len(eval_steps)
episodes_axis = np.arange(n_evals) * 4  # 4 episodes per eval checkpoint

# Sim Success (ground truth)
ax.plot(episodes_axis, vlm_eval["sim_success"],
        alpha=0.15, color=BLUE, linewidth=0.5)
ax.plot(episodes_axis, smooth(vlm_eval["sim_success"], window=30),
        color=BLUE, linewidth=2, label="Sim Success Rate")

# VLM Success
ax.plot(episodes_axis, vlm_eval["vlm_success"],
        alpha=0.15, color=ORANGE, linewidth=0.5)
ax.plot(episodes_axis, smooth(vlm_eval["vlm_success"], window=30),
        color=ORANGE, linewidth=2, label="VLM Success Rate")

ax.set_xlabel("Episodes")
ax.set_ylabel("Success Rate (%)")
ax.set_title("Success Rate During Training (cf. RoboReward Fig. 2)")
ax.set_ylim(-5, 105)
ax.legend(loc="upper left")
ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_DIR / "success_rate.png", bbox_inches="tight")
fig.savefig(OUT_DIR / "success_rate.pdf", bbox_inches="tight")
print("Saved: success_rate.png/pdf")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: VLM Reward Analysis
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top: VLM Score over time
ax = axes[0]
ax.plot(episodes_axis, vlm_eval["vlm_score"],
        alpha=0.15, color=PURPLE, linewidth=0.5)
ax.plot(episodes_axis, smooth(vlm_eval["vlm_score"], window=30),
        color=PURPLE, linewidth=2, label="VLM Score (1-5)")
ax.set_ylabel("VLM Score")
ax.set_ylim(0.5, 5.5)
ax.set_title("VLM Reward Signal Analysis")
ax.legend(loc="upper right")
ax.axhline(y=3, color="gray", linestyle="--", alpha=0.3, label="Midpoint (3/5)")

# Bottom: VLM vs Sim agreement scatter
ax = axes[1]
# Compute rolling agreement (both say success or both say failure)
vlm_binary = (vlm_eval["vlm_success"] >= 50).astype(float)
sim_binary = (vlm_eval["sim_success"] >= 50).astype(float)
agreement = (vlm_binary == sim_binary).astype(float) * 100

ax.plot(episodes_axis, smooth(agreement, window=30),
        color=GREEN, linewidth=2, label="VLM-Sim Agreement")
ax.fill_between(episodes_axis, 0, smooth(agreement, window=30),
                alpha=0.15, color=GREEN)
ax.set_ylabel("Agreement (%)")
ax.set_xlabel("Episodes")
ax.set_ylim(0, 105)
ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3, label="Random chance")
ax.legend(loc="upper right")

fig.tight_layout()
fig.savefig(OUT_DIR / "vlm_analysis.png", bbox_inches="tight")
fig.savefig(OUT_DIR / "vlm_analysis.pdf", bbox_inches="tight")
print("Saved: vlm_analysis.png/pdf")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Loss Curves Comparison
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Critic loss
ax = axes[0]
gt_mask_valid = gt_mask & ~np.isnan(gt_sb3["critic_loss"])
vlm_valid = ~np.isnan(vlm_sb3["critic_loss"])
ax.plot(gt_steps[gt_mask_valid] / 1000, gt_sb3["critic_loss"][gt_mask_valid],
        alpha=0.15, color=BLUE, linewidth=0.5)
ax.plot(gt_steps[gt_mask_valid] / 1000, smooth(gt_sb3["critic_loss"][gt_mask_valid]),
        color=BLUE, linewidth=2, label="GT")
ax.plot(vlm_steps[vlm_valid] / 1000, vlm_sb3["critic_loss"][vlm_valid],
        alpha=0.15, color=ORANGE, linewidth=0.5)
ax.plot(vlm_steps[vlm_valid] / 1000, smooth(vlm_sb3["critic_loss"][vlm_valid]),
        color=ORANGE, linewidth=2, label="VLM")
ax.set_xlabel("Training Steps (x1000)")
ax.set_ylabel("Critic Loss")
ax.set_title("Critic Loss")
ax.set_yscale("log")
ax.legend()

# Actor loss
ax = axes[1]
gt_actor_valid = gt_mask & ~np.isnan(gt_sb3["actor_loss"])
vlm_actor_valid = ~np.isnan(vlm_sb3["actor_loss"])
ax.plot(gt_steps[gt_actor_valid] / 1000, gt_sb3["actor_loss"][gt_actor_valid],
        alpha=0.15, color=BLUE, linewidth=0.5)
ax.plot(gt_steps[gt_actor_valid] / 1000, smooth(gt_sb3["actor_loss"][gt_actor_valid]),
        color=BLUE, linewidth=2, label="GT")
ax.plot(vlm_steps[vlm_actor_valid] / 1000, vlm_sb3["actor_loss"][vlm_actor_valid],
        alpha=0.15, color=ORANGE, linewidth=0.5)
ax.plot(vlm_steps[vlm_actor_valid] / 1000, smooth(vlm_sb3["actor_loss"][vlm_actor_valid]),
        color=ORANGE, linewidth=2, label="VLM")
ax.set_xlabel("Training Steps (x1000)")
ax.set_ylabel("Actor Loss")
ax.set_title("Actor Loss")
ax.legend()

fig.suptitle("Loss Curves: GT vs VLM Rewards", y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "loss_curves.png", bbox_inches="tight")
fig.savefig(OUT_DIR / "loss_curves.pdf", bbox_inches="tight")
print("Saved: loss_curves.png/pdf")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5: VLM Score Distribution
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: histogram of VLM scores
ax = axes[0]
scores = vlm_eval["vlm_score"]
# Scores are averages of 4 envs, so they can be 1.0, 1.25, 1.5, ..., 5.0
bins = np.arange(0.875, 5.375, 0.25)
ax.hist(scores, bins=bins, color=PURPLE, alpha=0.7, edgecolor="black", linewidth=0.5)
ax.set_xlabel("VLM Score")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of VLM Scores")
ax.axvline(x=np.mean(scores), color=RED, linestyle="--", linewidth=1.5,
           label=f"Mean: {np.mean(scores):.2f}")
ax.legend()

# Right: confusion matrix — VLM success vs Sim success
ax = axes[1]
# Count the 4 cases
vlm_bool = vlm_binary.astype(bool)
sim_bool = sim_binary.astype(bool)
tp = np.sum(vlm_bool & sim_bool)        # Both success
tn = np.sum(~vlm_bool & ~sim_bool)      # Both failure
fp = np.sum(vlm_bool & ~sim_bool)       # VLM says success, sim says failure
fn = np.sum(~vlm_bool & sim_bool)       # VLM says failure, sim says success

conf = np.array([[tn, fp], [fn, tp]])
total = conf.sum()
im = ax.imshow(conf / total * 100, cmap="Blues", vmin=0, vmax=60)

# Labels
for i in range(2):
    for j in range(2):
        val = conf[i, j]
        pct = val / total * 100
        ax.text(j, i, f"{int(val)}\n({pct:.1f}%)",
                ha="center", va="center", fontsize=11,
                color="white" if pct > 30 else "black")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Failure", "Success"])
ax.set_yticklabels(["Failure", "Success"])
ax.set_xlabel("VLM Prediction")
ax.set_ylabel("Simulator (Ground Truth)")
ax.set_title("VLM vs Simulator Agreement")
plt.colorbar(im, ax=ax, label="% of evaluations")

fig.tight_layout()
fig.savefig(OUT_DIR / "vlm_score_distribution.png", bbox_inches="tight")
fig.savefig(OUT_DIR / "vlm_score_distribution.pdf", bbox_inches="tight")
print("Saved: vlm_score_distribution.png/pdf")
plt.close(fig)


# ── Summary statistics ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
print(f"\nVLM Run (91539): {len(vlm_sb3['total_timesteps'])} log blocks, "
      f"{vlm_sb3['total_timesteps'][-1]:,.0f} total steps")
print(f"GT Run  (91452): {len(gt_sb3['total_timesteps'])} log blocks, "
      f"{gt_sb3['total_timesteps'][-1]:,.0f} total steps")

print(f"\nVLM Score: mean={np.mean(scores):.2f}, std={np.std(scores):.2f}, "
      f"min={np.min(scores):.2f}, max={np.max(scores):.2f}")
print(f"VLM Success Rate: mean={np.mean(vlm_eval['vlm_success']):.1f}%")
print(f"Sim Success Rate: mean={np.mean(vlm_eval['sim_success']):.1f}%")
print(f"VLM-Sim Agreement: {np.mean(agreement):.1f}%")

# Correlation
corr = np.corrcoef(vlm_eval["vlm_success"], vlm_eval["sim_success"])[0, 1]
print(f"VLM-Sim Correlation (Pearson): {corr:.3f}")

print(f"\nConfusion Matrix (N={total:.0f}):")
print(f"  True Negatives  (both fail):    {tn:.0f} ({tn/total*100:.1f}%)")
print(f"  True Positives  (both success): {tp:.0f} ({tp/total*100:.1f}%)")
print(f"  False Positives (VLM hallucinates success): {fp:.0f} ({fp/total*100:.1f}%)")
print(f"  False Negatives (VLM misses success):       {fn:.0f} ({fn/total*100:.1f}%)")
