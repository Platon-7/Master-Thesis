#!/usr/bin/env python3
"""
Create professional table figure showing complete FPS experiment results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set professional style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# Output directory
output_dir = Path("/home/pkarageo/master-thesis/Robo-Reward-FPS/figures")
output_dir.mkdir(exist_ok=True)

# ============================================================================
# Complete Results Table
# ============================================================================

def create_results_table():
    """Create comprehensive results table with all metrics."""

    # Data from experiment
    data = [
        ['default (2.0)', 23.08, 0.00, 0.00, 0.00, 60.00, 0, 50, 0, 15],
        ['2.0', 23.08, 0.00, 0.00, 0.00, 60.00, 0, 50, 0, 15],
        ['5.0', 26.15, 100.00, 4.00, 7.69, 100.00, 2, 48, 0, 15],
        ['10.0', 26.15, 100.00, 4.00, 7.69, 100.00, 2, 48, 0, 15],
        ['15.0', 29.23, 100.00, 8.00, 14.81, 100.00, 4, 46, 0, 15],
        ['20.0', 44.62, 100.00, 28.00, 43.75, 100.00, 14, 36, 0, 15],
        ['30.0', 56.92, 92.31, 48.00, 63.16, 86.67, 24, 26, 2, 13],
        ['60.0', 75.38, 86.96, 80.00, 83.33, 60.00, 40, 10, 6, 9],
    ]

    # Column headers
    columns = ['FPS', 'Accuracy\n(%)', 'Precision\n(%)', 'Recall\n(%)',
               'F1 Score\n(%)', 'Specificity\n(%)', 'TP', 'FN', 'FP', 'TN']

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=data, colLabels=columns, cellLoc='center',
                     loc='center', bbox=[0, 0, 1, 1])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)

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
        # Highlight best row (60.0 FPS - last row)
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
    title = 'RoboReward-8B FPS Sensitivity Analysis: Complete Results\nTask: Rearrange Pillows on Sofa (DROID Dataset)'
    plt.title(title, fontsize=16, fontweight='bold', pad=20, y=0.98)

    # Add footer notes
    notes = 'Total Samples: 65 (50 Successes, 15 Failures)  |  Best Performance: 60 FPS (highlighted in green)'

    plt.figtext(0.5, 0.02, notes, ha='center', fontsize=10,
                style='italic', color='#555555', weight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.2))

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])

    # Save
    output_path = output_dir / "fps_results_table.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")

    output_path_pdf = output_dir / "fps_results_table.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path_pdf}")

    plt.close()


# ============================================================================
# Alternative: More Compact Table with Better Formatting
# ============================================================================

def create_compact_results_table():
    """Create a more compact version with better visual hierarchy."""

    # Data
    data = [
        ['default (2.0)', '23.08', '0.00', '0.00', '0.00', '60.00', '0', '50', '0', '15'],
        ['2.0', '23.08', '0.00', '0.00', '0.00', '60.00', '0', '50', '0', '15'],
        ['5.0', '26.15', '100.00', '4.00', '7.69', '100.00', '2', '48', '0', '15'],
        ['10.0', '26.15', '100.00', '4.00', '7.69', '100.00', '2', '48', '0', '15'],
        ['15.0', '29.23', '100.00', '8.00', '14.81', '100.00', '4', '46', '0', '15'],
        ['20.0', '44.62', '100.00', '28.00', '43.75', '100.00', '14', '36', '0', '15'],
        ['30.0', '56.92', '92.31', '48.00', '63.16', '86.67', '24', '26', '2', '13'],
        ['60.0 ⭐', '75.38', '86.96', '80.00', '83.33', '60.00', '40', '10', '6', '9'],
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.axis('off')

    # Create main table
    table = plt.table(
        cellText=data,
        colLabels=['FPS', 'Accuracy\n(%)', 'Precision\n(%)', 'Recall\n(%)',
                   'F1 Score\n(%)', 'Specificity\n(%)', 'TP', 'FN', 'FP', 'TN'],
        cellLoc='center',
        loc='center',
        colWidths=[0.12, 0.10, 0.10, 0.10, 0.10, 0.12, 0.08, 0.08, 0.08, 0.08]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.8)

    # Header styling
    header_color = '#1f4788'
    for i in range(10):
        cell = table[(0, i)]
        cell.set_facecolor(header_color)
        cell.set_text_props(weight='bold', color='white', fontsize=13)
        cell.set_edgecolor('white')
        cell.set_linewidth(3)

    # Add separators between metric groups
    metric_separators = [0, 1, 6]  # After FPS, after Specificity%, after TN

    # Row styling
    for i in range(len(data)):
        # Alternate colors
        if i % 2 == 0:
            base_color = '#f8f9fa'
        else:
            base_color = 'white'

        # Highlight best row
        if i == len(data) - 1:
            base_color = '#90EE90'

        for j in range(10):
            cell = table[(i + 1, j)]
            cell.set_facecolor(base_color)
            cell.set_edgecolor('#dddddd')
            cell.set_linewidth(1)

            # Bold best row
            if i == len(data) - 1:
                cell.set_text_props(weight='bold', fontsize=12)

            # Add vertical separators
            if j in [1, 6]:
                cell.set_edgecolor('#888888')
                cell.set_linewidth(2)

    # Title
    plt.suptitle('FPS Sensitivity Analysis: RoboReward-8B Performance Metrics',
                 fontsize=18, fontweight='bold', y=0.98)
    plt.title('Task: Rearrange Pillows on Sofa | Dataset: DROID | Samples: 65 (50 Success, 15 Failure)',
              fontsize=12, style='italic', pad=15, color='#555555')

    # Legend
    legend_text = (
        '📊 Confusion Matrix Legend:\n'
        '  TP (True Positive): Model correctly predicts Success\n'
        '  FN (False Negative): Model misses Success (predicts Failure)\n'
        '  FP (False Positive): Model incorrectly predicts Success\n'
        '  TN (True Negative): Model correctly predicts Failure\n\n'
        '⭐ Best Performance: 60 FPS achieves highest F1 Score (83.33%)'
    )

    plt.figtext(0.5, -0.02, legend_text, ha='center', fontsize=10,
                family='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='#fffacd',
                         alpha=0.8, edgecolor='#cccccc', linewidth=2))

    plt.tight_layout(rect=[0, 0.12, 1, 0.94])

    # Save
    output_path = output_dir / "fps_results_table_compact.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")

    output_path_pdf = output_dir / "fps_results_table_compact.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path_pdf}")

    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("📋 Creating professional results table...")
    print("=" * 60)

    create_results_table()
    create_compact_results_table()

    print("=" * 60)
    print(f"✅ Tables saved to: {output_dir}")
    print("\nGenerated files:")
    print("  1. fps_results_table.png/pdf - Standard format")
    print("  2. fps_results_table_compact.png/pdf - Compact format with legend")
