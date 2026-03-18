# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Master's thesis project that combines **DSRL (Diffusion Steering via Reinforcement Learning)** with **VLM-based reward functions** for robotic manipulation tasks. The repository contains three main components:

1. **dsrl/** - Modified DSRL implementation with VLM reward integration
2. **RL-VLM-F/** - RL-VLM-F framework adapted for Qwen3-VL instead of Gemini API
3. **Robo-Reward-FPS/** - RoboReward-8B model integration utilities

## Environment Setup

This project runs on an HPC cluster using **Apptainer/Singularity containers** for dependency isolation and MuJoCo rendering compatibility.

### Building Containers

```bash
# DSRL container (includes RoboReward-8B dependencies)
cd master-thesis/dsrl
apptainer build /var/scratch/$USER/dsrl_roboreward.sif dsrl_env.def

# RL-VLM-F container (includes Qwen3-VL-8B dependencies)
cd master-thesis/RL-VLM-F
apptainer build /var/scratch/$USER/rl_vlm_container.sif complete_env.def
```

**Important**: Containers are stored in `/var/scratch/$USER/` due to size (~8-10GB each).

## Running Experiments

All experiments are submitted via SLURM job scripts located in `job_files/`.

### DSRL Training

**Baseline (Ground Truth Rewards):**
```bash
sbatch job_files/dsrl_train.job
```
- Uses simulator rewards from Robomimic environment
- Single GPU (TitanRTX)
- Output: `dsrl/dsrl_gt_<JOBID>.out`
- Config: `dsrl/cfg/robomimic/dsrl_lift.yaml`

**VLM Rewards (RoboReward-8B):**
```bash
sbatch job_files/dsrl_vlm_train.job
```
- Uses RoboReward-8B for reward computation
- Two GPUs: GPU 0 for RL agent, GPU 1 for RoboReward model
- Output: `dsrl/dsrl_vlm_<JOBID>.out`
- Config: `dsrl/cfg/robomimic/dsrl_lift_vlm.yaml`

### RL-VLM-F Training

```bash
cd RL-VLM-F
sbatch --export=TASK_ENV=metaworld_sweep-into-v2,SERVER_PORT=8001 jobs/run_qwen_container.job
```
- Two GPUs: GPU 0 for SAC agent, GPU 1 for Qwen3-VL server
- Server runs on configurable port (default 8000, use different ports for concurrent runs)
- Outputs: `RL-VLM-F/exp/<experiment_name>/`

### Checking Training Progress

```bash
# View live training output
tail -f dsrl/dsrl_gt_<JOBID>.out

# Check job status
squeue -u $USER

# Monitor GPU usage on allocated node
ssh node206  # Replace with your SLURM_NODELIST
nvidia-smi
```

### Key Training Metrics

**DSRL metrics** (from `.out` files):
- `ep_rew_mean`: Episode reward (negative values, higher is better)
- `total_timesteps`: Training progress (target: 1,000,000)
- `actor_loss`, `critic_loss`: Policy/value network losses
- `eval episode X at timestep Y`: Evaluation runs every 3,000 steps

**RL-VLM-F metrics** (from `eval.csv`):
- `success_rate`: Task completion percentage
- `episode_reward`: Cumulative reward per episode

## Code Architecture

### DSRL Integration (dsrl/)

**Training Entry Points:**
- `train_dsrl.py` - Standard DSRL with ground truth rewards
- `train_dsrl_vlm.py` - DSRL with VLM rewards (RoboReward-8B)

**Key Components:**
- `env_utils.py` - Environment wrappers for Robomimic
  - `DiffusionPolicyEnvWrapper`: Converts noise space to action space via diffusion policy
  - `ObservationWrapperRobomimic`: Normalizes observations using precomputed statistics
  - `ActionChunkWrapper`: Handles action chunking (4-step sequences)

- `vlm_reward_wrapper.py` - VLM reward computation wrapper
  - Replaces simulator rewards with RoboReward-8B predictions
  - Logs both VLM and ground truth rewards for analysis
  - Caches observations per episode for end-of-episode reward queries

- `roboreward_wrapper.py` - RoboReward-8B model interface
  - Loads RoboReward-8B-Instruct with 4-bit quantization
  - Processes (observation, instruction, next_observation) tuples
  - Returns binary success predictions (0 or 1)

- `utils.py` - Utility functions
  - `load_base_policy()`: Loads pretrained diffusion policy checkpoints
  - `load_offline_data()`: Loads demonstration data for replay buffer initialization
  - `collect_rollouts()`: Runs initial rollouts before training starts

**Modified Dependencies:**
- `stable-baselines3/` - Fork with DSRL algorithm implementation
- `dppo/` - Fork with diffusion policy utilities

### RL-VLM-F Architecture (RL-VLM-F/)

**Two-Process Architecture:**

```
┌─────────────────────────┐         ┌─────────────────────────┐
│     GPU 0 (RL Agent)    │         │  GPU 1 (VLM Server)     │
│  train_PEBBLE.py        │  HTTP   │  local_server.py        │
│  reward_model.py        │────────▶│  Qwen3-VL-8B            │
│  SAC agent              │  POST   │  (4-bit quantized)      │
└─────────────────────────┘         └─────────────────────────┘
```

**Key Files:**
- `train_PEBBLE.py` - Main training loop with PEBBLE reward learning
- `reward_model.py` - Learns reward function from VLM preferences
- `local_server.py` - FastAPI server hosting Qwen3-VL-8B
- `vlms/gemini_infer.py` - VLM query interface (filename preserved from original Gemini implementation)
  - `gemini_query_1()`: Single-step VLM query
  - `gemini_query_2()`: Two-step preference extraction (vision → text reasoning)

**Environment Variables:**
- `SERVER_HOST`: VLM server hostname (default: localhost)
- `SERVER_PORT`: VLM server port (default: 8000)
- `ARTIFACT_ROOT`: Redirect heavy outputs to scratch storage
- `LOCAL_MODEL_PATH`: Use cached model weights for offline clusters

### Configuration System (dsrl/cfg/)

Configs use Hydra for hierarchical composition:

```yaml
# dsrl/cfg/robomimic/dsrl_lift.yaml
total_timesteps: 1000000  # Training budget
algorithm: dsrl_na        # DSRL-NA (Noise Action) variant

env:
  n_envs: 4              # Parallel environments
  max_episode_steps: 300

train:
  utd: 30                # Gradient steps per environment step
  action_magnitude: 1.5  # Noise action space magnitude
  layer_size: 2048       # Actor/critic network width
  init_rollout_steps: 1501  # Initial data collection

model:
  denoising_steps: 20    # Full diffusion steps
  ddim_steps: 8          # DDIM sampling steps
```

**Critical Hyperparameters:**
- `utd` (30): Update-to-data ratio - number of gradient steps per rollout step
- `action_magnitude` (1.5): Controls exploration in noise space
- `init_rollout_steps` (1501): Collects initial data before training starts

## Visualization

### DSRL Learning Curves
```bash
cd dsrl
python plot_training.py
```
Generates plots from WandB logs (project: `dsrl_groundtruth` or `dsrl_vlm`).

### RL-VLM-F Figures
```bash
cd RL-VLM-F

# Figure 4: Success rate over time
python scripts/plot_figure4.py exp/<exp_name>/eval.csv --output results/figure4.png

# Figure 6: VLM accuracy vs. query difficulty
python scripts/plot_figure6.py exp/<exp_name>/vlm_query_log.csv --output results/figure6.png
```

## Common Issues

### Container Build Failures
- Ensure `/var/scratch/$USER/` exists and has sufficient space (~50GB)
- Use `apptainer build --sandbox` for debugging
- Check CUDA driver compatibility (requires CUDA 12.x)

### MuJoCo Rendering Errors
- Containers use `MUJOCO_GL=glfw` (NOT egl) for compatibility with cluster hardware
- Xvfb is launched inside containers for headless rendering
- Display number is derived from job ID to avoid conflicts: `DISPLAY=:$((SLURM_JOB_ID % 50 + 100))`

### VLM Inference Timeouts (RL-VLM-F)
- Qwen3-VL takes 5-10 minutes to load at startup
- Job script waits 10 minutes before starting training
- If training fails with connection errors, increase wait time in `run_qwen_container.job`

### DSRL Training Speed
- Expected: ~156,000 steps/day on single TitanRTX (ground truth rewards)
- VLM version: ~50-70k steps/day (RoboReward adds overhead)
- Speed bottleneck is MuJoCo simulation, not GPU compute

### Out of Memory (VLM Jobs)
- Ensure GPUs have at least 24GB VRAM (TitanRTX, A6000, etc.)
- RoboReward-8B uses 4-bit quantization (~8GB VRAM)
- Qwen3-VL uses 4-bit quantization (~8GB VRAM)
- If OOM occurs, reduce `utd` or `batch_size` in config

## File Naming Conventions

- `dsrl_gt_*.out` - Ground truth reward training logs
- `dsrl_vlm_*.out` - VLM reward training logs
- `*_<JOBID>.out` - SLURM job output (JOBID from submission)
- `wandb/run-*/` - WandB run directories with detailed metrics

## WandB Integration

Projects are logged to WandB (entity: `nlp-squad`):
- `dsrl_groundtruth` - Baseline DSRL experiments
- `dsrl_vlm` - VLM reward experiments
- `rl_vlm_f_qwen` - RL-VLM-F experiments (if configured)

Access dashboard: `https://wandb.ai/nlp-squad/<project_name>`

## Important Notes

- **Never commit WandB API keys** - They are in job scripts for convenience but should be environment variables in production
- **Scratch directory cleanup** - `/var/scratch/` may be purged periodically, sync important results to home directory
- **Concurrent runs** - Use different `SERVER_PORT` values when running multiple RL-VLM-F jobs simultaneously
- **Container modifications** - Rebuild containers after editing `.def` files (30-60 minutes)
- **Git submodules** - `dppo/` and `stable-baselines3/` are submodules with local modifications, don't update blindly
