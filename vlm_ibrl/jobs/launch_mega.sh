#!/bin/bash
# FULL factorial autonomous-RL sweep (no GT). Every combination of the levers we
# identified, x3 seeds. Submit: bash jobs/launch_mega.sh go
#
# Axes:
#   model            : robometer_ft, robometer_4b                 (2)
#   detect_head      : success, progress                          (2)   [NEW]
#   detect_threshold : success{0.80,0.85,0.90} progress{0.85,0.90,0.95} (3 each)
#   reward beta      : 0.0 (success head), 1.0 (progress head)    (2)
#   reward timing rt : 1 (sparse/at-end), 0 (dense/per-step)      (2)
#   debounce consec  : 1 (none), 3 (Christian's robust end)       (2)
#   seed             : 1,2,3                                       (3)
#  => 2*2*3*2*2*2*3 = 288 runs. (beta=0.5 mix + reward-binarization left out to
#     keep it ~20h; add later if a region looks promising.)
set -uo pipefail
cd "$(dirname "$0")/.."
CL=research-partner-cluster
# ondemand first (preemption-safe over a long run), spot as overflow; all L40S, packed via cons_tres
PARTS="ondemand-g6e-1gpu-l40s,ondemand-g6e-4gpu-l40s,ondemand-g6e-8gpu-l40s,spot-g6e-1gpu-l40s,spot-g6e-4gpu-l40s,spot-g6e-8gpu-l40s"
JOB=jobs/ibrl_sweep_robometer_aws.job
MAN=/shared/home/PKA4388/vlm_ibrl_runs/mega_manifest.tsv
GO="${1:-dry}"
COMMON="AUTONOMOUS_SUCCESS=1,V3_CORNER2_ZOOM=1,ROBOMETER_REWARD_CAMERA=corner2_default,EPISODE_LENGTH=100,NUM_TRAIN_STEP=40000,USE_WB=0,TRAIN_TIMEOUT=27000,ROBOMETER_THRESHOLD=0.0,ROBOMETER_REWARD_SCALE=1.0"
FT=/shared/home/PKA4388/checkpoints/Robometer_FT_consolidated/run1_icl_ours_step4000
B4=/shared/home/PKA4388/checkpoints/Robometer-4B
PATHS="ROBOMETER_FT_PATH=$FT,ROBOMETER_4B_PATH=$B4"

echo -e "# mega\tjobid\tvlm\thead\tthr\tbeta\trt\tconsec\tseed" > "$MAN"
n=0
for vlm in robometer_ft robometer_4b; do
  for head in success progress; do
    if [ "$head" = "success" ]; then THRS="0.80 0.85 0.90"; else THRS="0.85 0.90 0.95"; fi
    for thr in $THRS; do
      for beta in 0.0 1.0; do
        for rt in 1 0; do
          for consec in 1 3; do
            for seed in 1 2 3; do
              EXP="ALL,${COMMON},${PATHS},VLM_NAME=${vlm},BC_POLICY=coffeepush,ROBOMETER_DETECT_HEAD=${head},ROBOMETER_SUCCESS_THRESHOLD=${thr},ROBOMETER_BETA=${beta},REWARD_AT_TRUNC=${rt},ROBOMETER_SUCCESS_CONSECUTIVE=${consec},SEED=${seed}"
              name="${vlm#robometer_}_d${head:0:4}${thr}_b${beta}_rt${rt}_c${consec}_s${seed}"
              n=$((n+1))
              if [ "$GO" = "go" ]; then
                jid=$(sbatch -M "$CL" --parsable --job-name="$name" --partition="$PARTS" --export="$EXP" "$JOB" 2>&1 | grep -oE "^[0-9]+")
                echo -e "${name}\t${jid}\t${vlm}\t${head}\t${thr}\t${beta}\t${rt}\t${consec}\t${seed}" >> "$MAN"
              fi
            done
          done
        done
      done
    done
  done
done
echo "total: $n runs ($GO). manifest: $MAN"
