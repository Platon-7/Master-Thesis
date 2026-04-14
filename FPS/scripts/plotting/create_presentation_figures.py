#!/usr/bin/env python3
"""
Create professional figures for FPS sensitivity analysis presentation.
Generates:
1. Confusion matrix heatmap (best FPS: 60.0)
2. Comparison across all FPS settings
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
# Figure 1: Confusion Matrix for FPS 60.0 (Best Performance)
# ============================================================================

def create_confusion_matrix_figure():
    """Create professional confusion matrix heatmap."""

    # Data for FPS 60.0 (best performing)
    confusion_matrix = np.array([
        [40, 10],  # Actual Success: 40 TP, 10 FN
        [6, 9]     # Actual Failure: 6 FP, 9 TN
    ])

    # Metrics
    accuracy = 75.38
    precision = 86.96
    recall = 80.00
    f1 = 83.33
    specificity = 60.00

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar_kws={'label': 'Count'},
        square=True,
        linewidths=2,
        linecolor='white',
        ax=ax,
        annot_kws={'size': 18, 'weight': 'bold'}
    )

    # Labels
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Actual Label', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('Confusion Matrix: RoboReward-8B at 60 FPS\n(Best Performance)',
                 fontsize=16, fontweight='bold', pad=20)

    # Set tick labels
    ax.set_xticklabels(['Success', 'Failure'], fontsize=12)
    ax.set_yticklabels(['Success', 'Failure'], fontsize=12, rotation=0)

    # Add metrics text box
    metrics_text = (
        f'Performance Metrics:\n'
        f'─────────────────\n'
        f'Accuracy:    {accuracy:.1f}%\n'
        f'Precision:   {precision:.1f}%\n'
        f'Recall:      {recall:.1f}%\n'
        f'F1 Score:    {f1:.1f}%\n'
        f'Specificity: {specificity:.1f}%'
    )

    # Position text box
    ax.text(
        2.6, 0.5,
        metrics_text,
        transform=ax.transData,
        fontsize=11,
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, pad=1),
        family='monospace'
    )

    plt.tight_layout()

    # Save
    output_path = output_dir / "confusion_matrix_fps60.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")

    # Also save as PDF for presentations
    output_path_pdf = output_dir / "confusion_matrix_fps60.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path_pdf}")

    plt.close()


# ============================================================================
# Figure 2: Comparison Across All FPS Settings
# ============================================================================

def create_fps_comparison_figure():
    """Create professional comparison chart across FPS settings."""

    # Data from the experiment
    fps_labels = ['default\n(2.0)', '2.0', '5.0', '10.0', '15.0', '20.0', '30.0', '60.0']
    fps_values = [2.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 60.0]

    accuracy = [23.08, 23.08, 26.15, 26.15, 29.23, 44.62, 56.92, 75.38]
    precision = [0.00, 0.00, 100.00, 100.00, 100.00, 100.00, 92.31, 86.96]
    recall = [0.00, 0.00, 4.00, 4.00, 8.00, 28.00, 48.00, 80.00]
    f1_score = [0.00, 0.00, 7.69, 7.69, 14.81, 43.75, 63.16, 83.33]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    x = np.arange(len(fps_labels))
    width = 0.2

    # ========== Subplot 1: Main Metrics (Accuracy, F1, Recall) ==========
    bars1 = ax1.bar(x - width, accuracy, width, label='Accuracy',
                    color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x, f1_score, width, label='F1 Score',
                    color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars3 = ax1.bar(x + width, recall, width, label='Recall',
                    color='#F18F01', alpha=0.8, edgecolor='black', linewidth=1.2)

    ax1.set_xlabel('Frames Per Second (FPS)', fontsize=13, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Score (%)', fontsize=13, fontweight='bold', labelpad=10)
    ax1.set_title('RoboReward-8B Performance vs FPS: Primary Metrics',
                  fontsize=15, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(fps_labels, fontsize=11)
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 105])

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    # ========== Subplot 2: Precision ==========
    bars4 = ax2.bar(x, precision, width*1.5, label='Precision',
                    color='#06A77D', alpha=0.8, edgecolor='black', linewidth=1.2)

    ax2.set_xlabel('Frames Per Second (FPS)', fontsize=13, fontweight='bold', labelpad=10)
    ax2.set_ylabel('Precision (%)', fontsize=13, fontweight='bold', labelpad=10)
    ax2.set_title('RoboReward-8B Precision vs FPS',
                  fontsize=15, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(fps_labels, fontsize=11)
    ax2.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 105])

    # Add value labels
    for bar in bars4:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add annotation for best FPS
    ax1.annotate('Best\nF1: 83.33%',
                xy=(7, f1_score[7]),
                xytext=(6.2, 90),
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                               lw=2, color='red'))

    plt.tight_layout()

    # Save
    output_path = output_dir / "fps_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")

    # Also save as PDF
    output_path_pdf = output_dir / "fps_comparison.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path_pdf}")

    plt.close()


# ============================================================================
# Alternative: Combined Line Plot
# ============================================================================

def create_fps_comparison_lineplot():
    """Create line plot version for alternative visualization."""

    # Data
    fps_values = [2.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 60.0]
    accuracy = [23.08, 23.08, 26.15, 26.15, 29.23, 44.62, 56.92, 75.38]
    precision = [0.00, 0.00, 100.00, 100.00, 100.00, 100.00, 92.31, 86.96]
    recall = [0.00, 0.00, 4.00, 4.00, 8.00, 28.00, 48.00, 80.00]
    f1_score = [0.00, 0.00, 7.69, 7.69, 14.81, 43.75, 63.16, 83.33]

    # Use unique FPS values for x-axis
    unique_fps = [2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 60.0]
    unique_acc = [23.08, 26.15, 26.15, 29.23, 44.62, 56.92, 75.38]
    unique_prec = [0.00, 100.00, 100.00, 100.00, 100.00, 92.31, 86.96]
    unique_rec = [0.00, 4.00, 4.00, 8.00, 28.00, 48.00, 80.00]
    unique_f1 = [0.00, 7.69, 7.69, 14.81, 43.75, 63.16, 83.33]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot lines
    ax.plot(unique_fps, unique_acc, marker='o', linewidth=2.5, markersize=8,
            label='Accuracy', color='#2E86AB')
    ax.plot(unique_fps, unique_f1, marker='s', linewidth=2.5, markersize=8,
            label='F1 Score', color='#A23B72')
    ax.plot(unique_fps, unique_rec, marker='^', linewidth=2.5, markersize=8,
            label='Recall', color='#F18F01')
    ax.plot(unique_fps, unique_prec, marker='D', linewidth=2.5, markersize=8,
            label='Precision', color='#06A77D')

    ax.set_xlabel('Frames Per Second (FPS)', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('RoboReward-8B: FPS Sensitivity Analysis\nTask: Rearrange Pillows on Sofa',
                 fontsize=16, fontweight='bold', pad=20)

    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([-5, 105])
    ax.set_xscale('log')
    ax.set_xticks(unique_fps)
    ax.set_xticklabels([f'{int(fps)}' for fps in unique_fps], fontsize=11)

    # Highlight best performance
    ax.axvline(x=60, color='red', linestyle='--', linewidth=1.5, alpha=0.5,
               label='Optimal FPS')
    ax.text(60, 90, '← Best: 60 FPS\nF1: 83.33%', fontsize=11,
            fontweight='bold', ha='right',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()

    # Save
    output_path = output_dir / "fps_comparison_lineplot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")

    output_path_pdf = output_dir / "fps_comparison_lineplot.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path_pdf}")

    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("🎨 Creating professional presentation figures...")
    print("=" * 60)

    create_confusion_matrix_figure()
    create_fps_comparison_figure()
    create_fps_comparison_lineplot()

    print("=" * 60)
    print(f"✅ All figures saved to: {output_dir}")
    print("\nGenerated files:")
    print("  1. confusion_matrix_fps60.png/pdf")
    print("  2. fps_comparison.png/pdf")
    print("  3. fps_comparison_lineplot.png/pdf")
