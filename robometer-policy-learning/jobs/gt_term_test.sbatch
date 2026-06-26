#!/bin/bash
#SBATCH --job-name=rpl_gtterm
#SBATCH --output=/shared/home/PKA4388/robometer-policy-learning/logs/gtterm_%x_%j.out
#SBATCH --error=/shared/home/PKA4388/robometer-policy-learning/logs/gtterm_%x_%j.err
#SBATCH --time=14:00:00
#SBATCH --account=rob-tme-gaia
#SBATCH --partition=ondemand-g6-1gpu-l4
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# DECISIVE EXPERIMENT: does the authors' LIBERO RL rely on the GT-success termination?
# Robometer-4B reward (their model, their code) on libero_90 task 28, 60k steps.
# Two knobs passed via env (set per submission with --export):
#   ADD_EST       : reward_model.add_estimated_reward (true = RM+GT-sparse [their default], false = RM only)
#   SUPPRESS_DONE : RPL_SUPPRESS_GT_DONE (0 = GT-success terminates [their default], 1 = timeout-only)
# 2x2 over {ADD_EST} x {SUPPRESS_DONE}; here we run the their-default reward row first.

set -uo pipefail
cd /shared/home/PKA4388/robometer-policy-learning

export PATH="$HOME/.local/bin:$PATH"
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="$PWD/.ffmpeg7/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export RPL_SUPPRESS_GT_DONE="${SUPPRESS_DONE:-0}"

ADD_EST="${ADD_EST:-true}"
SDET="${SDET:-false}"
TAG="add${ADD_EST}_suppress${RPL_SUPPRESS_GT_DONE}_sdet${SDET}"

echo "==================== GT-termination test [$TAG] ===================="
echo "HOST          : $(hostname)"
echo "ADD_EST       : $ADD_EST   (true=RM+GTsparse, false=RM-only)"
echo "SUPPRESS_DONE : $RPL_SUPPRESS_GT_DONE   (0=GT terminates, 1=timeout-only)"
echo "SDET          : $SDET   (true=VLM success-head termination)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "===================================================================="

uv run python scripts/train.py \
  --config-path=../robometer_policy_learning/configs \
  --config-name=libero_online_rl \
  reward_model=robometer \
  reward_model.model_path=${MODEL:-/shared/home/PKA4388/checkpoints/Robometer-4B} \
  reward_model.add_estimated_reward=${ADD_EST} \
  reward_model.use_success_detection=${SDET} \
  env.env_name=libero_90 \
  env.task_id=28 \
  env.use_gt_rewards=false \
  algorithm@online_algorithm=sac \
  alg.online_alg_name=sac \
  training.num_rollouts=${STEPS:-60000} \
  training.seed=100 \
  online_algorithm.learning_starts=5000 \
  online_algorithm.num_critic_updates_per_actor_update=1 \
  online_algorithm.critic_optimizer_lr=1e-5 \
  online_algorithm.actor_optimizer_lr=1e-5 \
  eval.eval_freq=5000 \
  eval.eval_num_episodes=10 \
  eval.eval_record_video=false \
  logging.wandb_offline=true \
  logging.wandb_name=gtterm_${TAG}

echo "==================== EXIT=$? [$TAG] ===================="
