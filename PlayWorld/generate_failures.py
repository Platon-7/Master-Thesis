#!/usr/bin/env python3
"""
Generate failure trajectories using PlayWorld's world model (Ctrl-World).

Takes initial frames from the PlayWorld dataset and feeds random/perturbed
actions through the world model to generate realistic failure videos.
Extracts 8 keyframes per generated trajectory.

Works directly with pre-encoded latent videos (no raw video decoding needed).

Requires: Ctrl-World repo cloned, PlayWorld checkpoint + data downloaded,
          SVD and CLIP models available.

Usage:
    python generate_failures.py --num-episodes 100 --noise-scale 0.5
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

# Add Ctrl-World to path
CTRLWORLD_DIR = os.environ.get(
    "CTRLWORLD_DIR", "/home/pkarageorgis/DAS-5/Master-Thesis/Ctrl-World"
)
sys.path.insert(0, CTRLWORLD_DIR)

import torch
import einops


# ============================================================
# World model wrapper
# ============================================================

class PlayWorldAgent:
    """Wraps Ctrl-World model for failure trajectory generation.

    Follows the exact same pattern as Ctrl-World's rollout_replay_traj.py:
    - 3 camera views concatenated along height: (1, 4, 72, 40)
    - History buffer with indices [0, 0, -8, -6, -4, -2]
    - Actions normalized using dataset statistics
    """

    def __init__(self, args):
        from models.ctrl_world import CrtlWorld
        from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
        from config import wm_args

        # Build config with our paths
        self.cfg = wm_args()
        self.cfg.svd_model_path = args.svd_path
        self.cfg.clip_model_path = args.clip_path
        self.cfg.val_model_path = args.checkpoint
        self.cfg.ckpt_path = args.checkpoint
        self.cfg.data_stat_path = args.data_stat_path

        self.device = torch.device(args.device)
        self.dtype = self.cfg.dtype  # bfloat16

        # Load model
        print(f"Loading world model from {args.checkpoint}...")
        self.model = CrtlWorld(self.cfg)
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).to(self.dtype)
        self.model.eval()

        self.pipeline = self.model.pipeline
        self.vae = self.pipeline.vae

        # Load normalization stats
        with open(args.data_stat_path) as f:
            data_stat = json.load(f)
        self.state_p01 = np.array(data_stat["state_01"])[None, :]
        self.state_p99 = np.array(data_stat["state_99"])[None, :]

        # Key model parameters (from config.py)
        self.num_frames = self.cfg.num_frames      # 5 (prediction horizon)
        self.num_history = self.cfg.num_history     # 6 (history conditioning)
        self.action_dim = self.cfg.action_dim       # 7
        self.pred_step = self.cfg.pred_step         # 5

        print(f"World model loaded. num_frames={self.num_frames}, "
              f"num_history={self.num_history}, action_dim={self.action_dim}")

    def normalize_bound(self, data, clip_min=-1, clip_max=1, eps=1e-8):
        """Normalize actions to [-1, 1] using dataset statistics."""
        ndata = 2 * (data - self.state_p01) / (self.state_p99 - self.state_p01 + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def decode_latents_to_rgb(self, latents):
        """Decode VAE latents to RGB frames.

        Args:
            latents: (N, 4, 24, 40) tensor — single-view latents

        Returns:
            frames: list of (192, 320, 3) uint8 numpy arrays
        """
        frames = []
        chunk_size = self.cfg.decode_chunk_size
        for i in range(0, latents.shape[0], chunk_size):
            chunk = latents[i:i + chunk_size] / self.vae.config.scaling_factor
            with torch.no_grad():
                decoded = self.vae.decode(
                    chunk.to(self.dtype), num_frames=chunk.shape[0]
                ).sample
            frames.append(decoded)

        video = torch.cat(frames, dim=0)
        video = ((video / 2.0 + 0.5).clamp(0, 1) * 255)
        video = video.detach().to(torch.float32).cpu().numpy()
        video = video.transpose(0, 2, 3, 1).astype(np.uint8)  # (N, H, W, 3)
        return [video[i] for i in range(video.shape[0])]

    def forward_wm(self, action_cond, current_latent, his_cond=None, text=None):
        """Run one world model step. Follows rollout_replay_traj.py exactly.

        Args:
            action_cond: (num_history + num_frames, 7) numpy array
            current_latent: (1, 4, 72, 40) — current frame (3 views stacked)
            his_cond: (1, num_history, 4, 72, 40) history latents
            text: optional text instruction

        Returns:
            predicted_latents: (3, num_frames, 4, 24, 40) per-view latents
            predicted_latents_stacked: (num_frames, 4, 72, 40) stacked latents
        """
        from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline

        # Normalize and tensorize actions
        action_normed = self.normalize_bound(action_cond)
        action_tensor = torch.tensor(action_normed).unsqueeze(0).to(self.device).to(self.dtype)

        # Encode action + text
        with torch.no_grad():
            if text is not None:
                text_token = self.model.action_encoder(
                    action_tensor, text,
                    self.model.tokenizer, self.model.text_encoder
                )
            else:
                text_token = self.model.action_encoder(action_tensor)

            # Run diffusion pipeline (3 views stacked in height)
            _, latents = CtrlWorldDiffusionPipeline.__call__(
                self.pipeline,
                image=current_latent,
                text=text_token,
                width=self.cfg.width,
                height=int(self.cfg.height * 3),
                num_frames=self.cfg.num_frames,
                history=his_cond,
                num_inference_steps=self.cfg.num_inference_steps,
                decode_chunk_size=self.cfg.decode_chunk_size,
                max_guidance_scale=self.cfg.guidance_scale,
                fps=self.cfg.fps,
                motion_bucket_id=self.cfg.motion_bucket_id,
                mask=None,
                output_type="latent",
                return_dict=False,
                frame_level_cond=True,
            )

        # latents: (1, num_frames, 4, 72, 40) — split into 3 views
        # rearrange: (b f c (m h) (n w) -> (b m n) f c h w, m=3, n=1)
        per_view = einops.rearrange(
            latents, "b f c (m h) (n w) -> (b m n) f c h w", m=3, n=1
        )  # (3, num_frames, 4, 24, 40)

        # Also keep stacked form for history buffer
        stacked = latents.squeeze(0)  # (num_frames, 4, 72, 40)

        return per_view, stacked

    def generate_trajectory(self, latent_views, states, text=None, n_interact=10):
        """Generate a full trajectory from initial latents and action sequence.

        Args:
            latent_views: list of 3 tensors, each (T, 4, 24, 40) for each camera view
            states: (T, 7) numpy array of cartesian poses (5Hz)
            text: optional text instruction
            n_interact: number of world model interaction steps

        Returns:
            all_view0_latents: list of (4, 24, 40) tensors for view 0
        """
        # Stack 3 views along height: each view[0] is (4, 24, 40),
        # cat along dim=1 (H) → (4, 72, 40), then unsqueeze → (1, 4, 72, 40)
        first_latent = torch.cat(
            [v[0].to(self.device).to(self.dtype) for v in latent_views], dim=1
        ).unsqueeze(0)  # (1, 4, 72, 40) — 3 views × 24 = 72

        # Initialize history buffers (following reference code exactly)
        his_cond = [first_latent.clone() for _ in range(self.num_history * 4)]
        his_eef = [states[0:1] for _ in range(self.num_history * 4)]

        all_view0_latents = [latent_views[0][0].to(self.device).to(self.dtype)]
        history_idx = [0, 0, -8, -6, -4, -2]

        pred_step = self.pred_step

        for step in range(n_interact):
            start_id = int(step * (pred_step - 1))
            end_id = start_id + pred_step

            if end_id > len(states):
                break

            cartesian_pose = states[start_id:end_id]  # (pred_step, 7)

            # Build action conditioning: history poses + current chunk
            his_pose = np.concatenate(
                [his_eef[idx] for idx in history_idx], axis=0
            )  # (6, 7)
            action_cond = np.concatenate([his_pose, cartesian_pose], axis=0)
            # (num_history + num_frames, 7)

            # Build history latent conditioning
            his_cond_input = torch.cat(
                [his_cond[idx] for idx in history_idx], dim=0
            ).unsqueeze(0)  # (1, num_history, 4, 72, 40)

            current_latent = his_cond[-1]  # (1, 4, 72, 40)

            # Forward world model
            per_view_latents, stacked_latents = self.forward_wm(
                action_cond, current_latent,
                his_cond=his_cond_input,
                text=text if self.cfg.text_cond else None,
            )

            # Update history buffers
            his_eef.append(cartesian_pose[pred_step - 1:pred_step])
            # stacked_latents is (num_frames, 4, 72, 40), take last pred frame
            # and add batch dim → (1, 4, 72, 40)
            his_cond.append(stacked_latents[pred_step - 1:pred_step])

            # Collect view-0 latents (skip first frame which overlaps)
            for f in range(1, pred_step):
                if f < per_view_latents.shape[1]:
                    all_view0_latents.append(per_view_latents[0, f])

        return all_view0_latents


# ============================================================
# Action generation strategies
# ============================================================

def generate_perturbed_actions(base_actions, noise_scale=0.5, seed=None):
    """Perturb real trajectory actions with noise to create failures."""
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.randn(*base_actions.shape) * noise_scale
    return base_actions + noise


def generate_zero_actions(n_steps, action_dim=7):
    """Robot does nothing — guaranteed failure for any task."""
    return np.zeros((n_steps, action_dim))


def generate_reversed_actions(base_actions):
    """Reverse the action sequence — robot undoes progress."""
    return base_actions[::-1].copy()


def score_keyframes(strategy, kf_indices, n_generated_frames,
                    base_actions, failure_actions, noise_scale=0.5):
    """Score each keyframe 1-4 based on how the failure actions deviate from success.

    We know:
    - base_actions: the original successful trajectory
    - failure_actions: what we actually fed the world model
    - strategy: how we corrupted the actions
    - kf_indices: which generated frames are keyframes

    Scoring logic:
    - zero: all frames score 1 (no action = no progress)
    - reverse: frame score based on how far backwards we've gone
      early frames may still be near start → score 1-2, later → score 1
    - perturb: cumulative action fidelity at each keyframe
      high fidelity = closer to success = higher score
    """
    n_kf = len(kf_indices)
    pred_step = 5  # frames per interaction step (minus overlap)

    if strategy == "zero":
        # No actions at all → no progress ever
        return [1] * n_kf

    elif strategy == "reverse":
        # Reversed actions: robot undoes the task.
        # Early frames still show initial state (score 1),
        # later frames show active regression (still score 1).
        # Give score 2 to first few frames where the robot hasn't
        # moved much yet, then 1 as it actively goes wrong.
        scores = []
        for kf_idx in kf_indices:
            progress = kf_idx / max(n_generated_frames - 1, 1)
            if progress < 0.15:
                scores.append(2)  # just started, near initial state
            else:
                scores.append(1)  # actively going backwards
        return scores

    elif strategy == "perturb":
        # Perturbed actions: compare to original at each keyframe.
        # Map each keyframe to an action timestep, compute cumulative
        # cosine similarity of actions up to that point.
        scores = []
        for kf_idx in kf_indices:
            # Map frame index to action index (each interaction step
            # produces pred_step-1 new frames)
            action_idx = min(kf_idx, len(base_actions) - 1)

            if action_idx == 0:
                scores.append(1)
                continue

            # Cumulative action similarity up to this point
            orig = base_actions[:action_idx]
            fail = failure_actions[:action_idx]
            # Cosine similarity per timestep, averaged
            dot = np.sum(orig * fail, axis=1)
            norm_orig = np.linalg.norm(orig, axis=1) + 1e-8
            norm_fail = np.linalg.norm(fail, axis=1) + 1e-8
            cos_sim = dot / (norm_orig * norm_fail)
            avg_sim = np.mean(cos_sim)

            # Also consider L2 deviation as fraction of original magnitude
            l2_dev = np.mean(np.linalg.norm(fail - orig, axis=1))
            l2_orig = np.mean(np.linalg.norm(orig, axis=1)) + 1e-8
            deviation_ratio = l2_dev / l2_orig

            # Combined score: high similarity + low deviation → higher score
            # noise_scale 0.5 typically gives deviation_ratio ~0.5-1.5
            if avg_sim > 0.9 and deviation_ratio < 0.3:
                scores.append(4)  # very close to original
            elif avg_sim > 0.7 and deviation_ratio < 0.7:
                scores.append(3)  # moderate fidelity
            elif avg_sim > 0.3 and deviation_ratio < 1.5:
                scores.append(2)  # some resemblance
            else:
                scores.append(1)  # totally diverged
        return scores

    else:
        return [1] * n_kf


# ============================================================
# Data loading
# ============================================================

def load_annotations(data_dir, split="train"):
    """Load trajectory annotations that have matching latent videos."""
    ann_dir = Path(data_dir) / "annotation" / split
    lat_dir = Path(data_dir) / "latent_videos" / split

    # Only load annotations that have corresponding latent videos
    lat_ids = set(p.name for p in lat_dir.iterdir() if p.is_dir()) if lat_dir.exists() else set()

    annotations = []
    for json_file in sorted(ann_dir.glob("*.json"), key=lambda p: int(p.stem)):
        traj_id = json_file.stem
        if traj_id not in lat_ids:
            continue
        with open(json_file) as f:
            ann = json.load(f)
        ann["_file"] = str(json_file)
        ann["_id"] = traj_id
        annotations.append(ann)

    print(f"  ({len(annotations)} trajectories with latent videos out of "
          f"{len(list(ann_dir.glob('*.json')))} total annotations)")
    return annotations


def load_latent_views(data_dir, traj_id, split="train"):
    """Load all 3 camera view latents for a trajectory.

    Returns: list of 3 tensors, each (T, 4, 24, 40)
    """
    lat_dir = Path(data_dir) / "latent_videos" / split / str(traj_id)
    views = []
    for view_idx in range(3):
        lat_path = lat_dir / f"{view_idx}.pt"
        if not lat_path.exists():
            return None
        views.append(torch.load(str(lat_path), map_location="cpu"))
    return views


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate PlayWorld failure videos")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--noise-scale", type=float, default=0.5,
                        help="Noise scale for action perturbation")
    parser.add_argument("--n-interact", type=int, default=10,
                        help="World model interaction steps per episode")
    parser.add_argument("--strategy", choices=["perturb", "zero", "reverse", "all"],
                        default="all")
    parser.add_argument("--output-dir", type=str,
                        default="/home/pkarageorgis/DAS-5/Master-Thesis/PlayWorld/output")
    parser.add_argument("--checkpoint", type=str,
                        default="/scratch-shared/pkarageorgis/playworld_data/checkpoint-100000.pt")
    parser.add_argument("--data-dir", type=str,
                        default="/scratch-shared/pkarageorgis/playworld_data/v0_2025_12_28_2000")
    parser.add_argument("--data-stat-path", type=str, default=None)
    parser.add_argument("--svd-path", type=str,
                        default="/scratch-shared/pkarageorgis/hf_cache/stable-video-diffusion-img2vid")
    parser.add_argument("--clip-path", type=str,
                        default="/scratch-shared/pkarageorgis/hf_cache/clip-vit-base-patch32")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-keyframes", type=int, default=8)
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
            print("ERROR: Could not find stat.json. Specify --data-stat-path")
            sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keyframes_dir = output_dir / "keyframes"
    keyframes_dir.mkdir(exist_ok=True)

    print(f"Strategy: {args.strategy}")
    print(f"Noise scale: {args.noise_scale}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output: {output_dir}")

    # Load world model
    agent = PlayWorldAgent(args)

    # Load trajectory annotations
    annotations = load_annotations(args.data_dir, split="train")
    print(f"Loaded {len(annotations)} source trajectories")

    # Determine strategies
    if args.strategy == "all":
        strategies = ["perturb", "zero", "reverse"]
    else:
        strategies = [args.strategy]

    eps_per_strategy = max(1, args.num_episodes // len(strategies))
    all_results = []
    ep_count = 0

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy} ({eps_per_strategy} episodes)")
        print(f"{'='*60}")

        for src_idx in range(min(eps_per_strategy, len(annotations))):
            ann = annotations[src_idx]
            traj_id = ann["_id"]
            text = ann.get("texts", ["robot manipulation"])[0]
            base_states = np.array(ann["states"])  # (T, 7) cartesian poses at 5Hz

            print(f"\n  [{ep_count+1}/{args.num_episodes}] src={traj_id} "
                  f"strategy={strategy} task={text[:50]}")

            # Load pre-encoded latent views
            latent_views = load_latent_views(args.data_dir, traj_id)
            if latent_views is None:
                print(f"    Skipping - latent videos not found")
                continue

            # Generate failure actions
            if strategy == "perturb":
                actions = generate_perturbed_actions(
                    base_states, noise_scale=args.noise_scale, seed=ep_count
                )
            elif strategy == "zero":
                actions = generate_zero_actions(len(base_states), agent.action_dim)
            elif strategy == "reverse":
                actions = generate_reversed_actions(base_states)
            else:
                actions = base_states

            # Generate trajectory through world model
            try:
                view0_latents = agent.generate_trajectory(
                    latent_views, actions, text=text,
                    n_interact=args.n_interact,
                )
            except Exception as e:
                import traceback
                print(f"    ERROR: {e}")
                traceback.print_exc()
                continue

            n_frames = len(view0_latents)
            if n_frames < args.n_keyframes:
                print(f"    Too few frames ({n_frames}), skipping")
                continue

            # Extract keyframe indices
            kf_indices = np.linspace(0, n_frames - 1, args.n_keyframes, dtype=int)
            kf_latents = torch.stack([view0_latents[i] for i in kf_indices])

            # Decode keyframe latents to RGB
            kf_frames = agent.decode_latents_to_rgb(kf_latents)

            # Save keyframe images
            ep_id = f"playworld_{strategy}_{traj_id}"
            ep_dir = keyframes_dir / ep_id
            ep_dir.mkdir(parents=True, exist_ok=True)

            # Score keyframes based on action corruption
            scores = score_keyframes(
                strategy, kf_indices, n_frames,
                base_states, actions, noise_scale=args.noise_scale,
            )
            # Enforce monotonicity: scores should not decrease
            for i in range(1, len(scores)):
                if scores[i] < scores[i - 1]:
                    scores[i] = scores[i - 1]

            frame_records = []
            for f_idx, (kf_idx, kf_img) in enumerate(zip(kf_indices, kf_frames)):
                timestamp = float(kf_idx) / max(n_frames - 1, 1)
                img_path = ep_dir / f"frame_{f_idx}_{timestamp:.2f}s.jpg"
                Image.fromarray(kf_img).save(str(img_path), quality=95)

                frame_records.append({
                    "frame_idx": f_idx,
                    "step": int(kf_idx),
                    "timestamp": timestamp,
                    "score": scores[f_idx],
                })

            final_score = scores[-1]

            result = {
                "episode_id": ep_id,
                "task": text,
                "source": "playworld",
                "strategy": strategy,
                "src_traj_id": traj_id,
                "noise_scale": args.noise_scale if strategy == "perturb" else None,
                "total_generated_frames": n_frames,
                "n_keyframes": len(kf_frames),
                "final_score": final_score,
                "frames": frame_records,
            }
            all_results.append(result)
            ep_count += 1
            print(f"    Generated {n_frames} frames, saved {len(kf_frames)} keyframes, "
                  f"scores={scores}, final={final_score}")

            if ep_count >= args.num_episodes:
                break
        if ep_count >= args.num_episodes:
            break

    # Write JSONL
    jsonl_path = output_dir / "playworld_failures.jsonl"
    with open(jsonl_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    print(f"\n{'='*60}")
    print(f"Done! {len(all_results)} failure episodes generated")
    print(f"Keyframes: {keyframes_dir}")
    print(f"Metadata: {jsonl_path}")
    strat_counts = {s: sum(1 for r in all_results if r["strategy"] == s) for s in strategies}
    print(f"Strategies: {strat_counts}")

    # Score distribution
    from collections import Counter
    score_counts = Counter(r["final_score"] for r in all_results)
    print(f"Score distribution: " + ", ".join(
        f"s{s}={score_counts.get(s,0)}" for s in sorted(score_counts.keys())
    ))
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
