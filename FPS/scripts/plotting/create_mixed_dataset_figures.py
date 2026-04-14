#!/usr/bin/env python3
"""
Create professional figures for RoboReward MIXED dataset FPS sensitivity analysis.
This analyzes RoboReward's performance on data from the dataset it was trained on.

Generates:
1. Confusion matrix heatmap (best FPS: 10.0)
2. Line plot comparison across all FPS settings
3. Complete results table
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
# Data from fps_confusion_roboreward_mixed_91519.txt
# ============================================================================

# FPS settings tested
fps_labels = ['default\n(2.0)', '2.0', '5.0', '10.0']
fps_values = [2.0, 2.0, 5.0, 10.0]

# Performance metrics
accuracy = [52.50, 52.50, 70.00, 87.50]
precision = [100.00, 100.00, 100.00, 94.12]
recall = [5.00, 5.00, 40.00, 80.00]
f1_score = [9.52, 9.52, 57.14, 86.49]
specificity = [100.00, 100.00, 100.00, 95.00]

# Confusion matrices [TP, FN, FP, TN]
confusion_data = {
    'default (2.0)': {'TP': 1, 'FN': 19, 'FP': 0, 'TN': 20},
    '2.0': {'TP': 1, 'FN': 19, 'FP': 0, 'TN': 20},
    '5.0': {'TP': 8, 'FN': 12, 'FP': 0, 'TN': 20},
    '10.0': {'TP': 16, 'FN': 4, 'FP': 1, 'TN': 19},  # Best performance
}

# ============================================================================
# Figure 1: Confusion Matrix for FPS 10.0 (Best Performance)
# ============================================================================

def create_confusion_matrix_fps10():
    """Create professional confusion matrix heatmap for best FPS (10.0)."""

    # Data for FPS 10.0 (best performing)
    confusion_matrix = np.array([
        [16, 4],   # Actual Success: 16 TP, 4 FN
        [1, 19]    # Actual Failure: 1 FP, 19 TN
    ])

    # Metrics
    acc = 87.50
    prec = 94.12
    rec = 80.00
    f1 = 86.49
    spec = 95.00

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
    ax.set_title('Confusion Matrix: RoboReward-8B at 10 FPS\n(Best Performance - Mixed Dataset)',
                 fontsize=16, fontweight='bold', pad=20)

    # Set tick labels
    ax.set_xticklabels(['Success', 'Failure'], fontsize=12)
    ax.set_yticklabels(['Success', 'Failure'], fontsize=12, rotation=0)

    # Add metrics text box
    metrics_text = (
        f'Performance Metrics:\n'
        f'─────────────────\n'
        f'Accuracy:    {acc:.2f}%\n'
        f'Precision:   {prec:.2f}%\n'
        f'Recall:      {rec:.2f}%\n'
        f'F1 Score:    {f1:.2f}%\n'
        f'Specificity: {spec:.2f}%'
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
    output_path = output_dir / "confusion_matrix_fps10_mixed.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")

    # Also save as PDF for presentations
    output_path_pdf = output_dir / "confusion_matrix_fps10_mixed.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path_pdf}")

    plt.close()


# ============================================================================
# Figure 2: FPS Comparison Line Plot
# ============================================================================

def create_fps_comparison_lineplot():
    """Create line plot version for FPS comparison."""

    # Use unique FPS values for x-axis (skip duplicate 2.0)
    unique_fps = [2.0, 5.0, 10.0]
    unique_acc = [52.50, 70.00, 87.50]
    unique_prec = [100.00, 100.00, 94.12]
    unique_rec = [5.00, 40.00, 80.00]
    unique_f1 = [9.52, 57.14, 86.49]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot lines
    ax.plot(unique_fps, unique_acc, marker='o', linewidth=2.5, markersize=10,
            label='Accuracy', color='#2E86AB')
    ax.plot(unique_fps, unique_f1, marker='s', linewidth=2.5, markersize=10,
            label='F1 Score', color='#A23B72')
    ax.plot(unique_fps, unique_rec, marker='^', linewidth=2.5, markersize=10,
            label='Recall', color='#F18F01')
    ax.plot(unique_fps, unique_prec, marker='D', linewidth=2.5, markersize=10,
            label='Precision', color='#06A77D')

    ax.set_xlabel('Frames Per Second (FPS)', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('RoboReward-8B: FPS Sensitivity Analysis (Mixed Dataset)\nData from Training Distribution',
                 fontsize=16, fontweight='bold', pad=20)

    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 105])
    ax.set_xticks(unique_fps)
    ax.set_xticklabels([f'{int(fps)}' for fps in unique_fps], fontsize=11)

    # Highlight best performance
    ax.axvline(x=10, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.text(10, 92, '← Best: 10 FPS\nF1: 86.49%', fontsize=11,
            fontweight='bold', ha='right',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()

    # Save
    output_path = output_dir / "fps_comparison_lineplot_mixed.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")

    output_path_pdf = output_dir / "fps_comparison_lineplot_mixed.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path_pdf}")

    plt.close()


# ============================================================================
# Figure 3: Complete Results Table
# ============================================================================

def create_fps_results_table():
    """Create comprehensive results table with all metrics."""

    # Data from experiment
    data = [
        ['default (2.0)', 52.50, 100.00, 5.00, 9.52, 100.00, 1, 19, 0, 20],
        ['2.0', 52.50, 100.00, 5.00, 9.52, 100.00, 1, 19, 0, 20],
        ['5.0', 70.00, 100.00, 40.00, 57.14, 100.00, 8, 12, 0, 20],
        ['10.0 (BEST)', 87.50, 94.12, 80.00, 86.49, 95.00, 16, 4, 1, 19],
    ]

    # Column headers
    columns = ['FPS', 'Accuracy\n(%)', 'Precision\n(%)', 'Recall\n(%)',
               'F1 Score\n(%)', 'Specificity\n(%)', 'TP', 'FN', 'FP', 'TN']

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=data, colLabels=columns, cellLoc='center',
                     loc='center', bbox=[0, 0, 1, 1])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Color scheme
    header_color = '#2E86AB'
    row_colors = ['#f1f1f2', 'white']
    best_row_color = '#90EE90'  # Light green for best performance

    # Style header
    for j, col in enumerate(columns):
        cell = table[(0, j)]
        cell.set_facecolor(header_color)
        cell.set_text_props(weight='bold', color='white', fontsize=12)
        cell.set_edgecolor('white')
        cell.set_linewidth(2)

    # Style data rows
    for i in range(len(data)):
        # Highlight best row (10.0 FPS - last row)
        if i == len(data) - 1:
            row_color = best_row_color
        else:
            row_color = row_colors[i % 2]

        for j in range(len(columns)):
            cell = table[(i + 1, j)]
            cell.set_facecolor(row_color)
            cell.set_edgecolor('#cccccc')
            cell.set_linewidth(0.5)

            # Bold the best row
            if i == len(data) - 1:
                cell.set_text_props(weight='bold')

            # Right-align numbers
            if j > 0:
                cell.set_text_props(ha='center')

    # Add title
    title = 'RoboReward-8B FPS Sensitivity Analysis: Mixed Dataset Results\nData from Training Distribution'
    plt.title(title, fontsize=16, fontweight='bold', pad=20, y=0.98)

    # Add footer notes
    notes = 'Total Samples: 40 (20 Successes, 20 Failures)  |  Best Performance: 10 FPS (highlighted in green)'

    plt.figtext(0.5, 0.02, notes, ha='center', fontsize=10,
                style='italic', color='#555555', weight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.2))

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])

    # Save
    output_path = output_dir / "fps_results_table_mixed.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")

    output_path_pdf = output_dir / "fps_results_table_mixed.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path_pdf}")

    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("🎨 Creating professional presentation figures for MIXED dataset...")
    print("=" * 70)
    print("Dataset: RoboReward training distribution (mixed tasks)")
    print("Source: fps_confusion_roboreward_mixed_91519.txt")
    print("=" * 70)

    create_confusion_matrix_fps10()
    create_fps_comparison_lineplot()
    create_fps_results_table()

    print("=" * 70)
    print(f"✅ All figures saved to: {output_dir}")
    print("\nGenerated files:")
    print("  1. confusion_matrix_fps10_mixed.png/pdf - Best performing FPS (10.0)")
    print("  2. fps_comparison_lineplot_mixed.png/pdf - Performance across FPS settings")
    print("  3. fps_results_table_mixed.png/pdf - Complete results table")
    print("\nNote: Best performance at 10 FPS (F1: 86.49%), not 60 FPS as in DROID dataset")
