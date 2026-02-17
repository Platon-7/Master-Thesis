"""
Plot training curves from DSRL + VLM training logs.

Usage:
    python plot_training.py /var/scratch/pkarageo/dsrl_logs/training_log_lift.csv
    python plot_training.py /var/scratch/pkarageo/dsrl_logs/training_log_lift.csv --output plots/

Requires: matplotlib, pandas
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_training_curves(csv_path: str, output_dir: str = None, show: bool = True):
    """
    Generate training plots from CSV log.
    
    Creates:
    1. Success Rate vs Steps (both VLM and simulator)
    2. VLM Score vs Steps
    3. Loss curves (actor, critic)
    4. Combined summary plot
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Filter out empty rows and convert types
    df = df.dropna(subset=['timestep'])
    df['timestep'] = pd.to_numeric(df['timestep'], errors='coerce')
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.dirname(csv_path) or '.'
    os.makedirs(output_dir, exist_ok=True)
    
    task_name = os.path.basename(csv_path).replace('training_log_', '').replace('.csv', '')
    
    # Style settings
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {
        'vlm': '#2ecc71',      # Green
        'sim': '#3498db',       # Blue
        'actor': '#e74c3c',     # Red
        'critic': '#9b59b6',    # Purple
        'score': '#f39c12',     # Orange
    }
    
    # =========================================================
    # Plot 1: Success Rate vs Steps
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Training data
    train_mask = df['train_vlm_success_rate'].notna()
    if train_mask.any():
        ax.plot(df.loc[train_mask, 'timestep'], 
                df.loc[train_mask, 'train_vlm_success_rate'] * 100,
                color=colors['vlm'], linewidth=2, label='Train VLM Success', alpha=0.8)
        ax.plot(df.loc[train_mask, 'timestep'], 
                df.loc[train_mask, 'train_sim_success_rate'] * 100,
                color=colors['sim'], linewidth=2, linestyle='--', label='Train Sim Success', alpha=0.8)
    
    # Eval data
    eval_mask = df['eval_vlm_success_rate'].notna()
    if eval_mask.any():
        ax.scatter(df.loc[eval_mask, 'timestep'], 
                   df.loc[eval_mask, 'eval_vlm_success_rate'] * 100,
                   color=colors['vlm'], s=50, marker='o', label='Eval VLM Success', zorder=5)
        ax.scatter(df.loc[eval_mask, 'timestep'], 
                   df.loc[eval_mask, 'eval_sim_success_rate'] * 100,
                   color=colors['sim'], s=50, marker='s', label='Eval Sim Success', zorder=5)
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title(f'Success Rate vs Training Steps - {task_name.capitalize()}', fontsize=14)
    ax.legend(loc='best')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{task_name}_success_rate.png'), dpi=150)
    print(f"Saved: {task_name}_success_rate.png")
    
    # =========================================================
    # Plot 2: VLM Score vs Steps
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if train_mask.any() and 'train_vlm_score_raw' in df.columns:
        ax.plot(df.loc[train_mask, 'timestep'], 
                df.loc[train_mask, 'train_vlm_score_raw'],
                color=colors['score'], linewidth=2, label='Train VLM Score', alpha=0.8)
    
    if eval_mask.any() and 'eval_vlm_score_raw' in df.columns:
        ax.scatter(df.loc[eval_mask, 'timestep'], 
                   df.loc[eval_mask, 'eval_vlm_score_raw'],
                   color=colors['score'], s=50, marker='o', label='Eval VLM Score', zorder=5)
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('VLM Score (1-5)', fontsize=12)
    ax.set_title(f'VLM Score vs Training Steps - {task_name.capitalize()}', fontsize=14)
    ax.legend(loc='best')
    ax.set_ylim(0.5, 5.5)
    ax.axhline(y=4, color='gray', linestyle=':', alpha=0.5, label='Success Threshold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{task_name}_vlm_score.png'), dpi=150)
    print(f"Saved: {task_name}_vlm_score.png")
    
    # =========================================================
    # Plot 3: Loss Curves
    # =========================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    loss_mask = df['train_actor_loss'].notna()
    if loss_mask.any():
        ax1.plot(df.loc[loss_mask, 'timestep'], 
                 df.loc[loss_mask, 'train_actor_loss'],
                 color=colors['actor'], linewidth=2, label='Actor Loss')
        ax1.set_xlabel('Training Steps', fontsize=12)
        ax1.set_ylabel('Actor Loss', fontsize=12)
        ax1.set_title('Actor Loss', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(df.loc[loss_mask, 'timestep'], 
                 df.loc[loss_mask, 'train_critic_loss'],
                 color=colors['critic'], linewidth=2, label='Critic Loss')
        ax2.set_xlabel('Training Steps', fontsize=12)
        ax2.set_ylabel('Critic Loss', fontsize=12)
        ax2.set_title('Critic Loss', fontsize=14)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{task_name}_losses.png'), dpi=150)
    print(f"Saved: {task_name}_losses.png")
    
    # =========================================================
    # Plot 4: Combined Summary (2x2)
    # =========================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: VLM Success Rate
    ax = axes[0, 0]
    if train_mask.any():
        ax.plot(df.loc[train_mask, 'timestep'], 
                df.loc[train_mask, 'train_vlm_success_rate'] * 100,
                color=colors['vlm'], linewidth=2, label='Train')
    if eval_mask.any():
        ax.scatter(df.loc[eval_mask, 'timestep'], 
                   df.loc[eval_mask, 'eval_vlm_success_rate'] * 100,
                   color=colors['vlm'], s=50, marker='o', label='Eval', zorder=5)
    ax.set_xlabel('Steps')
    ax.set_ylabel('VLM Success Rate (%)')
    ax.set_title('VLM Success Rate')
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    
    # Top-right: Simulator Success Rate
    ax = axes[0, 1]
    if train_mask.any():
        ax.plot(df.loc[train_mask, 'timestep'], 
                df.loc[train_mask, 'train_sim_success_rate'] * 100,
                color=colors['sim'], linewidth=2, label='Train')
    if eval_mask.any():
        ax.scatter(df.loc[eval_mask, 'timestep'], 
                   df.loc[eval_mask, 'eval_sim_success_rate'] * 100,
                   color=colors['sim'], s=50, marker='s', label='Eval', zorder=5)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Sim Success Rate (%)')
    ax.set_title('Simulator Success Rate')
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: VLM Score
    ax = axes[1, 0]
    if train_mask.any() and 'train_vlm_score_raw' in df.columns:
        ax.plot(df.loc[train_mask, 'timestep'], 
                df.loc[train_mask, 'train_vlm_score_raw'],
                color=colors['score'], linewidth=2, label='Train')
    if eval_mask.any() and 'eval_vlm_score_raw' in df.columns:
        ax.scatter(df.loc[eval_mask, 'timestep'], 
                   df.loc[eval_mask, 'eval_vlm_score_raw'],
                   color=colors['score'], s=50, marker='o', label='Eval', zorder=5)
    ax.set_xlabel('Steps')
    ax.set_ylabel('VLM Score (1-5)')
    ax.set_title('VLM Score')
    ax.legend()
    ax.set_ylim(0.5, 5.5)
    ax.axhline(y=4, color='gray', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: Losses
    ax = axes[1, 1]
    if loss_mask.any():
        ax.plot(df.loc[loss_mask, 'timestep'], 
                df.loc[loss_mask, 'train_actor_loss'],
                color=colors['actor'], linewidth=2, label='Actor')
        ax.plot(df.loc[loss_mask, 'timestep'], 
                df.loc[loss_mask, 'train_critic_loss'],
                color=colors['critic'], linewidth=2, label='Critic')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'DSRL + RoboReward Training Summary - {task_name.capitalize()}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{task_name}_summary.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {task_name}_summary.png")
    
    if show:
        plt.show()
    
    plt.close('all')
    
    # Print summary statistics
    print("\n" + "="*50)
    print(f"Training Summary: {task_name}")
    print("="*50)
    if train_mask.any():
        print(f"Total training steps: {df['timestep'].max():,.0f}")
        final_vlm = df.loc[train_mask, 'train_vlm_success_rate'].iloc[-1] * 100
        final_sim = df.loc[train_mask, 'train_sim_success_rate'].iloc[-1] * 100
        print(f"Final VLM success rate: {final_vlm:.1f}%")
        print(f"Final Sim success rate: {final_sim:.1f}%")
    if eval_mask.any():
        best_vlm = df.loc[eval_mask, 'eval_vlm_success_rate'].max() * 100
        best_sim = df.loc[eval_mask, 'eval_sim_success_rate'].max() * 100
        print(f"Best eval VLM success: {best_vlm:.1f}%")
        print(f"Best eval Sim success: {best_sim:.1f}%")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Plot DSRL + VLM training curves')
    parser.add_argument('csv_path', help='Path to training CSV log')
    parser.add_argument('--output', '-o', default=None, help='Output directory for plots')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots')
    args = parser.parse_args()
    
    plot_training_curves(args.csv_path, args.output, show=not args.no_show)


if __name__ == '__main__':
    main()
