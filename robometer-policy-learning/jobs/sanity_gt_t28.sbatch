#!/bin/bash
#SBATCH --job-name=rpl_sanity_gt
#SBATCH --output=/shared/home/PKA4388/robometer-policy-learning/logs/sanity_gt_%j.out
#SBATCH --error=/shared/home/PKA4388/robometer-policy-learning/logs/sanity_gt_%j.err
#SBATCH --time=00:30:00
#SBATCH --account=rob-tme-gaia
#SBATCH --partition=spot-g6-1gpu-l4
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G

# Sanity probe for the robometer-policy-learning LIBERO reproduction on the AWS
# research-partner-cluster. Runs a DELIBERATELY TINY ground-truth-reward SAC run
# on libero_90 task 28 -- just enough to confirm the cluster env works end to end:
#   - uv .venv imports complete on a GPU (unsloth needs a real accelerator)
#   - LIBERO task-28 assets load + the env constructs
#   - robosuite/MuJoCo EGL offscreen render works
#   - the SAC online loop + one eval run without crashing
# No reward model here (GT rewards) -> isolates env/sim from the 4B-model path.

set -uo pipefail
cd /shared/home/PKA4388/robometer-policy-learning

# --- env quirks discovered during setup (see memory: robometer-policy-learning-repro) ---
export PATH="$HOME/.local/bin:$PATH"              # uv
export MUJOCO_GL=egl                              # headless robosuite render
export LD_LIBRARY_PATH="$PWD/.ffmpeg7/lib:${LD_LIBRARY_PATH:-}"  # ffmpeg shared libs for torchcodec
export HF_HUB_OFFLINE=0                            # HF online (dinov2 pre-cached, but allow fetch)
export TOKENIZERS_PARALLELISM=false

echo "==================== rpl sanity (GT, task 28) ===================="
echo "HOST       : $(hostname)"
echo "PARTITION  : ${SLURM_JOB_PARTITION:-?}"
echo "LD_LIB     : $LD_LIBRARY_PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "=================================================================="

uv run python scripts/train.py \
  --config-path=../robometer_policy_learning/configs \
  --config-name=libero_online_rl \
  env.env_name=libero_90 \
  env.task_id=28 \
  env.use_gt_rewards=true \
  algorithm@online_algorithm=sac \
  alg.online_alg_name=sac \
  training.num_rollouts=500 \
  training.seed=100 \
  online_algorithm.learning_starts=100 \
  eval.eval_freq=400 \
  eval.eval_num_episodes=2 \
  eval.eval_record_video=false \
  logging.wandb_offline=true \
  logging.wandb_name=sanity_gt_t28

echo "==================== sanity train.py EXIT=$? ===================="
