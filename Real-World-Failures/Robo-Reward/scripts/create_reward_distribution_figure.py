#!/usr/bin/env python3
"""
Create professional figure showing RoboReward dataset reward distribution.
Data source: inspect_all_datasets_91520.txt

This figure visualizes the distribution of reward labels (1-5) across
the entire RoboReward dataset (45,072 samples).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set professional style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Output directory
output_dir = Path("/home/pkarageo/master-thesis/Robo-Reward-FPS/figures")
output_dir.mkdir(exist_ok=True)

# ============================================================================
# Data from inspect_all_datasets_91520.txt
# ============================================================================

# Reward distribution (45,072 total samples)
rewards = [1, 2, 3, 4, 5]
counts = [13648, 8577, 7850, 6572, 8425]
percentages = [30.3, 19.0, 17.4, 14.6, 18.7]
total_samples = 45072

# ============================================================================
# Create Reward Distribution Bar Plot
# ============================================================================

def create_reward_distribution_figure():
    """Create professional bar plot of reward distribution."""

    fig, ax = plt.subplots(figsize=(12, 8))

    # Color scheme: gradient from red (low rewards) to green (high rewards)
    colors = ['#d62728', '#ff7f0e', '#ffdd57', '#90c290', '#2ca02c']

    # Create bars
    bars = ax.bar(rewards, counts, width=0.7, color=colors,
                  edgecolor='black', linewidth=1.5, alpha=0.85)

    # Add value labels on top of bars
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        height = bar.get_height()
        # Count
        ax.text(bar.get_x() + bar.get_width()/2., height + 200,
                f'{count:,}',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
        # Percentage
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{pct}%',
                ha='center', va='center', fontsize=12, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.6))

    # Labels and title
    ax.set_xlabel('Reward Label', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Number of Samples', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('RoboReward Dataset: Reward Distribution\nTotal Samples: 45,072',
                 fontsize=16, fontweight='bold', pad=20)

    # Set x-axis
    ax.set_xticks(rewards)
    ax.set_xticklabels([f'Reward {r}' for r in rewards], fontsize=12)

    # Format y-axis with comma separators
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax.tick_params(axis='y', labelsize=11)

    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    # Set y-axis limits with more padding to avoid overlap
    ax.set_ylim([0, max(counts) * 1.15])

    # Add combined legend and statistics box on the right
    combined_text = (
        f'Dataset Statistics:\n'
        f'──────────────────────\n'
        f'Total Samples:   {total_samples:,}\n'
        f'Failures (1-3):  {sum(counts[:3]):,} ({sum(percentages[:3]):.1f}%)\n'
        f'Successes (4-5): {sum(counts[3:]):,} ({sum(percentages[3:]):.1f}%)\n'
        f'\n'
        f'Reward Scale:\n'
        f'──────────────────────\n'
        f'1-3: Failure\n'
        f'4-5: Success'
    )
    ax.text(0.98, 0.98, combined_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4, pad=0.8),
            family='monospace')

    plt.tight_layout()

    # Save
    output_path = output_dir / "reward_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")

    output_path_pdf = output_dir / "reward_distribution.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path_pdf}")

    plt.close()


# ============================================================================
# Alternative: Pie Chart Version
# ============================================================================

def create_reward_distribution_pie():
    """Create alternative pie chart visualization."""

    fig, ax = plt.subplots(figsize=(10, 8))

    # Colors
    colors = ['#d62728', '#ff7f0e', '#ffdd57', '#90c290', '#2ca02c']

    # Explode the slices slightly for emphasis
    explode = (0.05, 0.02, 0.02, 0.02, 0.05)

    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=[f'Reward {r}' for r in rewards],
        autopct='%1.1f%%',
        colors=colors,
        explode=explode,
        startangle=90,
        textprops={'fontsize': 12, 'weight': 'bold'},
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}
    )

    # Enhance autopct text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_weight('bold')

    # Add count labels
    for i, (wedge, count) in enumerate(zip(wedges, counts)):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = 1.3 * np.cos(np.radians(angle))
        y = 1.3 * np.sin(np.radians(angle))
        ax.text(x, y, f'n={count:,}', ha='center', va='center',
                fontsize=10, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='gray', alpha=0.8))

    ax.set_title('RoboReward Dataset: Reward Distribution\nTotal Samples: 45,072',
                 fontsize=16, fontweight='bold', pad=20)

    # Add legend
    legend_text = (
        'Reward Scale:\n'
        '1-3: Failure\n'
        '4-5: Success'
    )
    ax.text(-0.5, -1.4, legend_text,
            fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4, pad=0.6),
            family='monospace')

    plt.tight_layout()

    # Save
    output_path = output_dir / "reward_distribution_pie.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")

    output_path_pdf = output_dir / "reward_distribution_pie.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path_pdf}")

    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("📊 Creating RoboReward reward distribution figures...")
    print("=" * 70)
    print(f"Data source: inspect_all_datasets_91520.txt")
    print(f"Total samples: {total_samples:,}")
    print("=" * 70)

    create_reward_distribution_figure()
    create_reward_distribution_pie()

    print("=" * 70)
    print(f"✅ All figures saved to: {output_dir}")
    print("\nGenerated files:")
    print("  1. reward_distribution.png/pdf - Bar chart")
    print("  2. reward_distribution_pie.png/pdf - Pie chart")
    print("\nReward breakdown:")
    for r, c, p in zip(rewards, counts, percentages):
        print(f"  Reward {r}: {c:,} ({p}%)")
