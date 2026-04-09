#!/usr/bin/env python3
"""
Quick verification: decode a successful trajectory, apply perturbations,
generate failure videos through Ctrl-World, and save frames for comparison.

Saves everything under PlayWorld/verification/:
  original/    — decoded frames from the real successful trajectory
  z_offset/    — gripper raised +4cm (misses grasp)
  temporal_delay/ — actions shifted forward by 5 steps (arrives late)
  magnitude/   — movement deltas scaled to 60% (undershoots)
  gaussian/    — small Gaussian noise (std=0.1) on positions
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

CTRLWORLD_DIR = os.environ.get(
    "CTRLWORLD_DIR", "/home/pkarageorgis/DAS-5/Master-Thesis/Ctrl-World"
)
sys.path.insert(0, CTRLWORLD_DIR)

import torch


def decode_and_save(agent, latents, out_dir, label=""):
    """Decode single-view latents to RGB and save as JPGs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = agent.decode_latents_to_rgb(latents)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(str(out_dir / f"frame_{i:03d}.jpg"), quality=95)
    print(f"  [{label}] Saved {len(frames)} frames to {out_dir}")
    return frames


def perturb_z_offset(states, offset=0.04):
    """Raise gripper Z position by offset (meters). Gripper closes on air."""
    perturbed = states.copy()
    perturbed[:, 2] += offset  # dim 2 = Z
    return perturbed


def perturb_temporal_delay(states, delay_steps=5):
    """Shift actions forward in time: robot arrives late to every waypoint."""
    perturbed = states.copy()
    # Pad beginning with initial pose, drop last delay_steps
    pad = np.tile(states[0:1], (delay_steps, 1))
    perturbed = np.concatenate([pad, states[:-delay_steps]], axis=0)
    return perturbed


def perturb_magnitude(states, scale=0.6):
    """Scale movement deltas: robot undershoots every target position."""
    perturbed = states.copy()
    # Compute deltas from first frame, scale them down
    origin = states[0:1]
    deltas = states - origin
    perturbed = origin + deltas * scale
    return perturbed


def perturb_gaussian(states, noise_std=0.02):
    """Add small Gaussian noise to XYZ positions (dims 0-2)."""
    perturbed = states.copy()
    noise = np.random.randn(len(states), 3) * noise_std
    perturbed[:, 0:3] += noise
    return perturbed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-id", type=str, default="0")
    parser.add_argument("--n-interact", type=int, default=10,
                        help="World model interaction steps")
    parser.add_argument("--checkpoint", type=str,
                        default="/scratch-shared/pkarageorgis/playworld_data/checkpoint-100000.pt")
    parser.add_argument("--data-dir", type=str,
                        default="/scratch-shared/pkarageorgis/playworld_data/v0_2025_12_28_2000")
    parser.add_argument("--svd-path", type=str,
                        default="/scratch-shared/pkarageorgis/hf_cache/stable-video-diffusion-img2vid")
    parser.add_argument("--clip-path", type=str,
                        default="/scratch-shared/pkarageorgis/hf_cache/clip-vit-base-patch32")
    parser.add_argument("--data-stat-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str,
                        default="/home/pkarageorgis/DAS-5/Master-Thesis/PlayWorld/verification")
    args = parser.parse_args()

    # Auto-detect data stat path
    if args.data_stat_path is None:
        candidates = [
            Path(CTRLWORLD_DIR) / "dataset_meta_info" / "droid" / "stat.json",
            Path(CTRLWORLD_DIR) / "dataset_meta_info" / "droid_subset" / "stat.json",
            Path(args.data_dir) / "stat.json",
        ]
        for c in candidates:
            if c.exists():
                args.data_stat_path = str(c)
                print(f"Found data stats: {c}")
                break
        if args.data_stat_path is None:
            print("ERROR: Could not find stat.json")
            sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import PlayWorldAgent from generate_failures.py
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_failures import PlayWorldAgent, load_latent_views

    # Load world model
    agent = PlayWorldAgent(args)

    # Load source trajectory
    ann_path = Path(args.data_dir) / "annotation" / "train" / f"{args.traj_id}.json"
    with open(ann_path) as f:
        ann = json.load(f)

    base_states = np.array(ann["states"])  # (T, 7) at 5Hz
    text = ann.get("texts", ["Play data demonstration"])[0]
    print(f"\nTrajectory {args.traj_id}: {len(base_states)} states, task='{text}'")
    print(f"State ranges: X=[{base_states[:,0].min():.3f},{base_states[:,0].max():.3f}] "
          f"Y=[{base_states[:,1].min():.3f},{base_states[:,1].max():.3f}] "
          f"Z=[{base_states[:,2].min():.3f},{base_states[:,2].max():.3f}]")

    # Load latent views
    latent_views = load_latent_views(args.data_dir, args.traj_id)
    if latent_views is None:
        print("ERROR: Latent videos not found")
        sys.exit(1)
    print(f"Latent views: {len(latent_views)} views, shape {latent_views[0].shape}")

    # ── Step 1: Decode original trajectory (ground truth frames) ──
    print(f"\n{'='*60}")
    print("Decoding ORIGINAL trajectory (ground truth)")
    print(f"{'='*60}")
    # Decode every 10th frame of view 0 to show the full original trajectory
    original_latents = latent_views[0].to(agent.device).to(agent.dtype)
    # Sample ~10 evenly spaced frames
    n_orig = original_latents.shape[0]
    orig_indices = np.linspace(0, n_orig - 1, min(10, n_orig), dtype=int)
    orig_sample = original_latents[orig_indices]
    decode_and_save(agent, orig_sample, output_dir / "original", "original")

    # ── Step 2: Generate with UNPERTURBED actions (baseline) ──
    print(f"\n{'='*60}")
    print("Generating with ORIGINAL actions (replay baseline)")
    print(f"{'='*60}")
    replay_latents = agent.generate_trajectory(
        latent_views, base_states, text=text, n_interact=args.n_interact,
    )
    n_replay = len(replay_latents)
    replay_indices = np.linspace(0, n_replay - 1, min(10, n_replay), dtype=int)
    replay_sample = torch.stack([replay_latents[i] for i in replay_indices])
    decode_and_save(agent, replay_sample, output_dir / "replay", "replay")

    # ── Step 3: Perturbation experiments ──
    perturbations = {
        "z_offset": ("Z-offset +4cm", perturb_z_offset(base_states, offset=0.04)),
        "temporal_delay": ("Temporal delay 5 steps", perturb_temporal_delay(base_states, delay_steps=5)),
        "magnitude": ("Magnitude scale 0.6x", perturb_magnitude(base_states, scale=0.6)),
        "gaussian": ("Gaussian noise std=0.02", perturb_gaussian(base_states, noise_std=0.02)),
    }

    for name, (desc, perturbed_states) in perturbations.items():
        print(f"\n{'='*60}")
        print(f"Generating: {desc}")
        print(f"{'='*60}")

        # Show how much states changed
        diff = np.abs(perturbed_states - base_states)
        print(f"  Mean absolute perturbation per dim: {diff.mean(axis=0)[:3]}")

        try:
            perturbed_latents = agent.generate_trajectory(
                latent_views, perturbed_states, text=text,
                n_interact=args.n_interact,
            )
            n_pert = len(perturbed_latents)
            pert_indices = np.linspace(0, n_pert - 1, min(10, n_pert), dtype=int)
            pert_sample = torch.stack([perturbed_latents[i] for i in pert_indices])
            decode_and_save(agent, pert_sample, output_dir / name, name)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"All done! Check {output_dir}/ for:")
    print(f"  original/        — ground truth decoded frames")
    print(f"  replay/          — world model replay with original actions")
    print(f"  z_offset/        — gripper raised +4cm")
    print(f"  temporal_delay/  — actions delayed by 5 steps")
    print(f"  magnitude/       — movements scaled to 60%")
    print(f"  gaussian/        — small position noise (std=0.02)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
