#!/bin/bash
# PullCube-v1, 300k steps, 5 seeds x {run2, base}. The headline comparison.
#
#   bash jobs/snellius_pullcube_sweep.sh            # submit
#   DRY=1 bash jobs/snellius_pullcube_sweep.sh      # print only
#
# Everything else (canonical recipe, episode logging, paths) is baked into
# jobs/snellius_maniskill_sac.job.
set -euo pipefail
cd "$(dirname "$0")/.."

TASK="${TASK:-PullCube-v1}"
STEPS="${STEPS:-300000}"
SEEDS="${SEEDS:-0 1 2 3 4}"
MODELS="${MODELS:-run2 base}"

for MODEL in $MODELS; do
  for SEED in $SEEDS; do
    CMD=(sbatch --export=ALL,ARM=dense,TASK="$TASK",MODEL="$MODEL",SEED="$SEED",STEPS="$STEPS"
         jobs/snellius_maniskill_sac.job)
    if [[ "${DRY:-0}" == "1" ]]; then
      printf '%s\n' "${CMD[*]}"
    else
      "${CMD[@]}"
    fi
  done
done
