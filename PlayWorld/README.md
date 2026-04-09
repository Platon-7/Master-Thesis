# PlayWorld Failure Dataset

## What is this?

This pipeline generates **labeled failure trajectories** for real-robot manipulation tasks using [Ctrl-World](https://github.com/yunyikristy/Ctrl-World) — a learned **video diffusion world model** trained on 30 hours of autonomous robot play data (DROID robot, real-world tabletop scenes).

The goal is the same as MetaWorld: produce failure episodes with **progress scores 1–4** to finetune RoboMeter. The key difference is that PlayWorld uses a **learned simulator** (the world model) rather than a physics engine, so the generated videos look like real robot footage.

---

## What is Ctrl-World?

Ctrl-World is an **action-conditioned video diffusion model** based on Stable Video Diffusion (SVD). Given:
1. An initial frame (encoded as a VAE latent)
2. A sequence of Cartesian end-effector poses (XYZ + orientation + gripper)

It predicts what the scene will look like after executing those actions, generating photorealistic video.

### Key architecture details
- **Backbone**: Stable Video Diffusion (SVD) — temporal U-Net + VAE
- **Action conditioning**: 7-dim Cartesian state vectors at 5Hz, fed through an action encoder
- **Prediction horizon**: 5 frames per world model call
- **3 camera views**: left, right, wrist cameras concatenated along height dimension: `(1, 4, 72, 40)` latents (3 × 24 height)
- **VAE compression**: 8× spatial (192×320 RGB → 24×40×4 latent)
- **Checkpoint**: `checkpoint-100000.pt` (9.3GB, trained for 100K steps)

---

## Dataset

The PlayWorld dataset (`tennyyyin/playworld_dataset_preview`) contains ~2034 training trajectories of autonomous robot play. Each trajectory has:

- **Annotation JSON**: states (100, 7) Cartesian poses at 5Hz, raw actions (300, 8) at 15Hz, text label (all say "Play data demonstration")
- **Latent videos**: 3 × `(T, 4, 24, 40)` pre-encoded VAE latents, one per camera view
- **No task labels**: all `success: 1` (dummy value — the data is unlabeled play)

**Download location**: `/scratch-shared/pkarageorgis/playworld_data/v0_2025_12_28_2000/`

To download missing trajectories:
```bash
python download_latents.py  # requires internet (login node only)
```

---

## How failures are generated

### Base trajectory selection

We select a **successful trajectory** (visually confirmed by watching decoded MP4s) as the starting point. This gives us:
- A real initial frame to condition the world model
- A real action sequence that produces a clean manipulation

### Action perturbation strategies

We perturb the expert's **Cartesian state sequence** in structured ways before feeding it to the world model. Since Ctrl-World is a learned simulator (not a real robot), it renders whatever actions we give it:

| Strategy | How | Effect |
|---|---|---|
| **Z-offset** | Add +0.03–0.05m to Z position | Gripper closes above the object (miss) |
| **Temporal delay** | Shift action sequence forward by N steps | Robot arrives late at every waypoint |
| **Magnitude scaling** | Scale movement deltas from start by 0.5–0.8× | Robot undershoots every target |
| **Gaussian noise** | Add small noise (std=0.02–0.05) to XYZ | Subtle jitter, slight misalignment |

> **Note**: The PlayWorld paper uses language instruction perturbation to create failures on a real robot. We use action perturbation instead because: (a) the world model is action-conditioned, not language-conditioned; (b) the dataset has no meaningful language labels (all say "Play data demonstration"). Action perturbation is the correct approach at world model inference time.

---

## Scoring (1–4)

Unlike MetaWorld (which has ground-truth task signals), PlayWorld has no semantic labels. Scores are assigned based on **how much the perturbed actions deviate from the original**:

| Strategy | Score logic |
|---|---|
| **z_offset** | All keyframes score 1 (any Z offset causes a miss) |
| **temporal_delay** | Early frames score 2 (close to start), later frames score 1 |
| **magnitude** | Cosine similarity + L2 deviation of perturbed vs. original actions, mapped to 1–4 |
| **gaussian** | Same as magnitude — deviation from original determines score |

For magnitude/gaussian perturbations:
- High cosine similarity + low L2 deviation → score 4 (barely perturbed, near-success)
- Low similarity + high deviation → score 1 (totally diverged)

Scores are monotonically non-decreasing across the 8 keyframes.

---

## Output format

```
output/
  keyframes/
    playworld_{strategy}_{traj_id}/
      frame_0_0.00s.jpg    ← keyframe 1 of 8
      ...
      frame_7_1.00s.jpg
  playworld_failures.jsonl
```

Each line in the JSONL:
```json
{
  "episode_id": "playworld_z_offset_105",
  "task": "Play data demonstration",
  "source": "playworld",
  "strategy": "z_offset",
  "src_traj_id": "105",
  "noise_scale": 0.04,
  "total_generated_frames": 41,
  "n_keyframes": 8,
  "final_score": 1,
  "frames": [
    {"frame_idx": 0, "step": 0, "timestamp": 0.0, "score": 1},
    ...
  ]
}
```

---

## Running

### Decode all base trajectories as MP4 (visual inspection):
```bash
sbatch jobs/decode_all_videos.job
# Output: videos/traj_{id}.mp4
```

### Verify perturbation strategies on one trajectory:
```bash
sbatch jobs/verify_perturbations.job
# Output: verification/{original,replay,z_offset,temporal_delay,magnitude,gaussian}/
```

### Generate full failure dataset:
```bash
sbatch jobs/generate.job
```

### Environment:
- Conda env: `ctrl-world`
- Cluster: DAS-5 GPU node (`gpu_a100` partition, ≥24GB VRAM)
- Model load time: ~2 minutes
- Speed: ~30s per episode on A100 (5 interaction steps)

---

## Key design decisions

- **Why not language perturbation (as in the paper)?** The PlayWorld paper uses a VLM to generate diverse language instructions that cause a VLA policy to naturally produce failures on a real robot. This is not applicable here: (1) the world model takes actions, not language; (2) the dataset has no task-specific labels to perturb.
- **Why use the world model at all, not real failures?** The world model generates photorealistic robot footage that looks like real data. This lets us create controlled failure modes (exact Z offset, exact delay) with known severity — something hard to guarantee with real rollouts.
- **Why only 163 (now 2034) trajectories have latents?** The dataset was partially downloaded initially due to HuggingFace authentication issues on compute nodes (no internet). After downloading with a token from the login node, all 2034 trajectories are available.
- **Why view 0 only?** The left camera (view 0) provides the clearest frontal view of manipulation. Views 1 (right) and 2 (wrist) are available in the latent files but not used for keyframe extraction.
