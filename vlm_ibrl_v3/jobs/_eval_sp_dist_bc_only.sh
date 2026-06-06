#!/bin/bash
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
HERE="$(pwd)"
ENV_NAME="${IBRL_CONDA_ENV:-demo2reward}"
ROBOMETER_DIR="${ROBOMETER_DIR:-$HERE/../Robometer}"
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load 2025 Anaconda3/2025.06-1
export PYTHONNOUSERSITE=1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
source set_env.sh
export PYTHONPATH="$PYTHONPATH:$ROBOMETER_DIR"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
if [[ -n "${ROBOMETER_ICL_DEMO_PATH:-}" ]]; then
    export ROBOMETER_ICL_DEMO_PATH ROBOMETER_ICL_DEMO_IDX ROBOMETER_ICL_FRAMES
    echo "ICL_DEMO_PATH = $ROBOMETER_ICL_DEMO_PATH"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1
python tools/eval_sp_distribution.py --run-dir "$RUN_DIR" --num-episodes 200 --seed 12345 --out-csv "$OUT_CSV" --bc-only
