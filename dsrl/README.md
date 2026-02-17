<div align="center">

# DSRL + VLM Rewards

**Diffusion Steering via Reinforcement Learning** with RoboReward-8B VLM reward functions.

[[DSRL Paper](https://arxiv.org/pdf/2506.15799)] [[DSRL Website](https://diffusion-steering.github.io)]

</div>

## Overview

This fork extends [DSRL](https://github.com/ajwagen/dsrl) with VLM-based reward functions using [RoboReward-8B](https://huggingface.co/teetone/RoboReward-8B). Instead of ground-truth simulator rewards, RoboReward scores robot episodes on a 1-5 discrete progress scale from rendered frames.

**Two training modes:**
- **Ground Truth (GT):** Standard DSRL with simulator rewards (`train_dsrl.py`)
- **VLM Rewards:** DSRL with RoboReward-8B (`train_dsrl_vlm.py`)

See [DSRL_VLM_IMPLEMENTATION.md](DSRL_VLM_IMPLEMENTATION.md) for detailed design notes.

## Prerequisites

- CUDA 12.x compatible GPU(s) with 24GB+ VRAM
- [Apptainer/Singularity](https://apptainer.org/) for containerized execution
- Pretrained diffusion policy checkpoints in `dppo/log/` ([download](https://drive.google.com/drive/folders/1kzC49RRFOE7aTnJh_7OvJ1K5XaDmtuh1?usp=share_link))
- RoboReward-8B model weights cached via HuggingFace (`teetone/RoboReward-8B`)

## Container Build

```bash
apptainer build $SCRATCH/dsrl_roboreward.sif dsrl_env.def
```

The container includes all Python dependencies (PyTorch, MuJoCo, transformers, bitsandbytes).

## Running

**Supported tasks:** `lift`, `can`, `square` (each has its own config with task-specific hyperparameters).

### Ground Truth Rewards (single GPU)

```bash
# Local (replace dsrl_lift with dsrl_can or dsrl_square for other tasks)
python train_dsrl.py --config-path=cfg/robomimic --config-name=dsrl_lift.yaml

# Snellius
sbatch jobs/snellius_gt_train.job
```

### VLM Rewards (single or multi-GPU)

```bash
# Local (replace dsrl_lift_vlm with dsrl_can_vlm or dsrl_square_vlm)
python train_dsrl_vlm.py --config-path=cfg/robomimic --config-name=dsrl_lift_vlm.yaml

# Snellius (edit --config-name inside the job script for other tasks)
sbatch jobs/snellius_vlm_train.job
```

### Single-GPU Mode

Set `vlm_device: "cuda:0"` in `cfg/robomimic/dsrl_lift_vlm.yaml`, or leave as `auto` (auto-detects GPU count). An A100 80GB fits both DSRL and RoboReward-8B 4-bit.

## Configuration

Key settings in `cfg/robomimic/dsrl_lift_vlm.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vlm_device` | `auto` | VLM GPU (`auto`: cuda:1 if 2+ GPUs, else cuda:0) |
| `use_vlm_reward` | `True` | Toggle VLM vs simulator rewards |
| `log_dir` | `./logs` | Output directory (override for scratch storage) |
| `total_timesteps` | `250000` | Training budget |
| `train.utd` | `30` | Gradient steps per environment step |
| `train.action_magnitude` | `1.5` | Noise action space magnitude |
| `env.n_envs` | `4` | Parallel training environments |

## Key Metrics

| Metric | Source | Description |
|--------|--------|-------------|
| `vlm/reward` | WandB | Mean RoboReward score per episode (1-5) |
| `vlm/success_rate` | WandB | Fraction of episodes with VLM score >= 4 |
| `sim/success_rate` | WandB | Ground-truth success rate (trustworthy metric) |
| `eval/sim_success_rate` | WandB | Evaluation success using simulator rewards |

## Downloading RoboReward-8B

If the model isn't already in your HuggingFace cache, download it on the cluster login node:

```bash
# 1. Set your permanent scratch directory
export SCRATCH="/scratch-shared/$USER"
export HF_HOME="$SCRATCH/hf_cache"

# 2. Load the standard Snellius Python module
module load 2023
module load Python

# 3. Create a quick virtual environment just for downloading
python -m venv "$SCRATCH/hf_download_env"
source "$SCRATCH/hf_download_env/bin/activate"

# 4. Install the Hugging Face tool
pip install huggingface_hub

# 5. Download the model
huggingface-cli download teetone/RoboReward-8B

# 6. Align the folder structure for the container's expected cache layout
mv "$HF_HOME/hub/models--teetone--RoboReward-8B" "$HF_HOME/"
```

## Citation

```
@article{wagenmaker2025steering,
  author    = {Wagenmaker, Andrew and Nakamoto, Mitsuhiko and Zhang, Yunchu and Park, Seohong and Yagoub, Waleed and Nagabandi, Anusha and Gupta, Abhishek and Levine, Sergey},
  title     = {Steering Your Diffusion Policy with Latent Space Reinforcement Learning},
  journal   = {Conference on Robot Learning (CoRL)},
  year      = {2025},
}
```
