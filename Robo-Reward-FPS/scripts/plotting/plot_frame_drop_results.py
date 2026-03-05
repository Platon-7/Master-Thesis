#!/usr/bin/env python3
"""
Plots for the Frame Drop Experiment.

Figure 1: Accuracy + Avg Score vs Drop Rate (dual-axis line chart)
Figure 2: Confusion matrix breakdown (stacked bar) + Label flip rate
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# ── Data (from frame_drop_91706.txt summary) ──────────────────────────────
conditions   = ['Original\n(0%)', 'Drop 20%', 'Drop 40%', 'Drop 60%']
accuracy     = [78.5, 69.2, 67.7, 49.2]
avg_score    = [3.62, 3.49, 3.29, 3.02]
avg_qframes  = [592.8, 519.6, 389.7, 259.7]

tp = [41, 35, 31, 19]
fn = [ 9, 15, 19, 31]
fp = [ 5,  5,  2,  2]
tn = [10, 10, 13, 13]

label_flips  = [0, 14, 17, 25]   # vs original; original=0 by definition
n_videos     = 65

output_dir = Path("/home/pkarageo/master-thesis/Robo-Reward-FPS/figures")
output_dir.mkdir(exist_ok=True)

x = np.arange(len(conditions))

# ============================================================================
# FIGURE 1 — Accuracy & Avg Score vs Frame Count
# ============================================================================
fig, ax1 = plt.subplots(figsize=(8, 5))

color_acc   = '#2E86AB'
color_score = '#E84855'
color_frames = '#A0A0A0'

ax1.set_xlabel('Condition (frames removed)', fontsize=12)
ax1.set_ylabel('Accuracy (%)', color=color_acc, fontsize=12)
ax1.plot(x, accuracy, 'o-', color=color_acc, linewidth=2.5, markersize=8, label='Accuracy (%)')
ax1.tick_params(axis='y', labelcolor=color_acc)
ax1.set_xticks(x)
ax1.set_xticklabels(conditions, fontsize=11)
ax1.set_ylim(40, 90)
ax1.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random chance')

# Annotate accuracy values
for i, (xi, yi) in enumerate(zip(x, accuracy)):
    ax1.annotate(f'{yi:.1f}%', (xi, yi), textcoords="offset points",
                 xytext=(0, 10), ha='center', color=color_acc, fontsize=10, fontweight='bold')

ax2 = ax1.twinx()
ax2.set_ylabel('Avg RoboReward Score', color=color_score, fontsize=12)
ax2.plot(x, avg_score, 's--', color=color_score, linewidth=2.5, markersize=8, label='Avg Score')
ax2.tick_params(axis='y', labelcolor=color_score)
ax2.set_ylim(2.5, 4.2)
ax2.axhline(3.5, color=color_score, linestyle=':', linewidth=1, alpha=0.4, label='Decision threshold (3.5)')

# Annotate score values
for i, (xi, yi) in enumerate(zip(x, avg_score)):
    ax2.annotate(f'{yi:.2f}', (xi, yi), textcoords="offset points",
                 xytext=(0, -18), ha='center', color=color_score, fontsize=10)

# Annotate avg qwen frames below x-axis labels
for i, (xi, qf) in enumerate(zip(x, avg_qframes)):
    ax1.annotate(f'~{int(qf)} frames', (xi, 41.5), ha='center',
                 fontsize=8.5, color=color_frames, style='italic')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

plt.title('RoboReward: Effect of Dropping Frames on Performance\n(DROID Pillow Case, 65 videos, 60 FPS)', fontsize=12)
plt.tight_layout()
out1 = output_dir / 'frame_drop_accuracy_score.png'
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out1}')

# ============================================================================
# FIGURE 2 — Stacked confusion bar + label flip rate
# ============================================================================
fig, (ax_bar, ax_flip) = plt.subplots(1, 2, figsize=(12, 5))

# --- Left: stacked bar of TP / FN / FP / TN ---
bar_w = 0.5
colors = {'TP': '#2ecc71', 'FN': '#e74c3c', 'FP': '#e67e22', 'TN': '#95a5a6'}

bottom_fn = np.array(tp, dtype=float)
bottom_fp = bottom_fn + np.array(fn, dtype=float)
bottom_tn = bottom_fp + np.array(fp, dtype=float)

bars_tp = ax_bar.bar(x, tp, bar_w, label='TP (correct success)', color=colors['TP'])
bars_fn = ax_bar.bar(x, fn, bar_w, bottom=tp, label='FN (missed success)', color=colors['FN'])
bars_fp = ax_bar.bar(x, fp, bar_w, bottom=bottom_fp, label='FP (false alarm)', color=colors['FP'])
bars_tn = ax_bar.bar(x, tn, bar_w, bottom=bottom_tn, label='TN (correct failure)', color=colors['TN'])

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(conditions, fontsize=11)
ax_bar.set_ylabel('Number of Videos', fontsize=12)
ax_bar.set_title('Confusion Matrix Breakdown', fontsize=11)
ax_bar.legend(fontsize=9, loc='lower left')
ax_bar.set_ylim(0, 80)

# Annotate each segment if large enough
for i in range(len(x)):
    def label_seg(ax, xi, bot, val, col):
        if val >= 2:
            ax.text(xi, bot + val/2, str(val), ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')
    label_seg(ax_bar, i, 0,              tp[i], colors['TP'])
    label_seg(ax_bar, i, tp[i],          fn[i], colors['FN'])
    label_seg(ax_bar, i, bottom_fp[i],   fp[i], colors['FP'])
    label_seg(ax_bar, i, bottom_tn[i],   tn[i], colors['TN'])

# --- Right: label flip rate ---
flip_pct = [f / n_videos * 100 for f in label_flips]
bar_colors = ['#bdc3c7', '#f39c12', '#e67e22', '#c0392b']
ax_flip.bar(x, flip_pct, bar_w, color=bar_colors, edgecolor='white')
ax_flip.set_xticks(x)
ax_flip.set_xticklabels(conditions, fontsize=11)
ax_flip.set_ylabel('Videos where prediction changed (%)', fontsize=11)
ax_flip.set_title('Label Flip Rate vs Original Prediction', fontsize=11)
ax_flip.set_ylim(0, 50)

for i, (xi, yv) in enumerate(zip(x, flip_pct)):
    ax_flip.text(xi, yv + 1, f'{yv:.1f}%\n({label_flips[i]} videos)',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('RoboReward Sensitivity to Video Length — DROID Pillow Case', fontsize=12, y=1.01)
plt.tight_layout()
out2 = output_dir / 'frame_drop_confusion_flips.png'
plt.savefig(out2, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {out2}')
