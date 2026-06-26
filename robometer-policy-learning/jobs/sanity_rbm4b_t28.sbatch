#!/bin/bash
#SBATCH --job-name=rpl_sanity_rbm
#SBATCH --output=/shared/home/PKA4388/robometer-policy-learning/logs/sanity_rbm_%j.out
#SBATCH --error=/shared/home/PKA4388/robometer-policy-learning/logs/sanity_rbm_%j.err
#SBATCH --time=00:45:00
#SBATCH --account=rob-tme-gaia
#SBATCH --partition=ondemand-g6-1gpu-l4
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# Reward-model sanity: tiny online run on libero_90 task 28 using the published
# Robometer-4B as the reward signal. Validates the last unexercised path --
# load_model_from_hf(4B) under unsloth + per-rollout reward scoring on the GPU --
# and MEASURES rollout throughput with the 4B model in the loop (to size the full
# 100k-step reproduction). Eval is skipped (eval doesn't use the reward model).

set -uo pipefail
cd /shared/home/PKA4388/robometer-policy-learning

export PATH="$HOME/.local/bin:$PATH"
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="$PWD/.ffmpeg7/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=0                 # tokenizer/processor may fetch from HF (offline tokenizer is broken)
export TOKENIZERS_PARALLELISM=false

echo "==================== rpl sanity (Robometer-4B, task 28) ===================="
echo "HOST       : $(hostname)"
echo "PARTITION  : ${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "==========================================================================="

uv run python scripts/train.py \
  --config-path=../robometer_policy_learning/configs \
  --config-name=libero_online_rl \
  reward_model=robometer \
  reward_model.model_path=/shared/home/PKA4388/checkpoints/Robometer-4B \
  reward_model.add_estimated_reward=true \
  reward_model.use_success_detection=false \
  env.env_name=libero_90 \
  env.task_id=28 \
  env.use_gt_rewards=false \
  algorithm@online_algorithm=sac \
  alg.online_alg_name=sac \
  training.num_rollouts=200 \
  training.seed=100 \
  online_algorithm.learning_starts=100 \
  eval.eval_freq=100000 \
  eval.eval_on_first_step=false \
  logging.wandb_offline=true \
  logging.wandb_name=sanity_rbm4b_t28

echo "==================== sanity rbm train.py EXIT=$? ===================="
