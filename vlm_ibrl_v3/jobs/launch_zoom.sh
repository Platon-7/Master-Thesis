#!/bin/bash
# Zoom-in around the lone survivor ft_anchor_c (FT, det=0.85, b0, rt1, continuous,
# autonomous). Replicate + map the neighborhood for BOTH models. Submit: bash jobs/launch_zoom.sh go
set -uo pipefail
cd "$(dirname "$0")/.."
CL=research-partner-cluster
PARTS="ondemand-g6e-1gpu-l40s,ondemand-g6e-4gpu-l40s,ondemand-g6e-8gpu-l40s,spot-g6e-1gpu-l40s,spot-g6e-4gpu-l40s,spot-g6e-8gpu-l40s"
JOB=jobs/ibrl_sweep_robometer_aws.job
MAN=/shared/home/PKA4388/vlm_ibrl_runs/zoom_manifest.tsv
GO="${1:-dry}"
COMMON="AUTONOMOUS_SUCCESS=1,V3_CORNER2_ZOOM=1,ROBOMETER_REWARD_CAMERA=corner2_default,EPISODE_LENGTH=100,NUM_TRAIN_STEP=40000,USE_WB=0,TRAIN_TIMEOUT=27000"
FT=/shared/home/PKA4388/checkpoints/Robometer_FT_consolidated/run1_icl_ours_step4000
B4=/shared/home/PKA4388/checkpoints/Robometer-4B
PATHS="ROBOMETER_FT_PATH=$FT,ROBOMETER_4B_PATH=$B4"

echo -e "# zoom\tjobid\tlabel\tvlm\tbeta\trt\tdet\tconsec\tseed" > "$MAN"
n=0
sub(){ # label vlm beta rthr rt det consec seed
  local label=$1 vlm=$2 beta=$3 rthr=$4 rt=$5 det=$6 consec=$7 seed=$8
  local EXP="ALL,${COMMON},${PATHS},VLM_NAME=${vlm},BC_POLICY=coffeepush,ROBOMETER_BETA=${beta},ROBOMETER_THRESHOLD=${rthr},REWARD_AT_TRUNC=${rt},ROBOMETER_SUCCESS_THRESHOLD=${det},ROBOMETER_SUCCESS_CONSECUTIVE=${consec},SEED=${seed}"
  local name="${vlm#robometer_}_${label}_s${seed}"
  n=$((n+1))
  if [ "$GO" = "go" ]; then
    local jid=$(sbatch -M "$CL" --parsable --job-name="$name" --partition="$PARTS" --export="$EXP" "$JOB" 2>&1 | grep -oE "^[0-9]+")
    echo -e "${name}\t${jid}\t${label}\t${vlm}\t${beta}\t${rt}\t${det}\t${consec}\t${seed}" >> "$MAN"
    echo "submitted $name -> $jid"
  else
    echo "[dry] $name : vlm=$vlm beta=$beta rt=$rt det=$det consec=$consec seed=$seed"
  fi
}

# Group 1: replication + threshold (b0,rt1,continuous) x {ft,4b} x det{0.80,0.85,0.90} x seed{1,2,3}
for vlm in robometer_ft robometer_4b; do
  for det in 0.80 0.85 0.90; do
    for seed in 1 2 3; do sub "thr${det}" "$vlm" 0.0 0.0 1 "$det" 1 "$seed"; done
  done
done
# Group 2: debounce at det=0.80 x {ft,4b} x consec{2,3} x seed{1,2}
for vlm in robometer_ft robometer_4b; do
  for consec in 2 3; do
    for seed in 1 2; do sub "deb${consec}_thr0.80" "$vlm" 0.0 0.0 1 0.80 "$consec" "$seed"; done
  done
done
# Group 3: reward shape at det=0.85 (FT only): dense-success (rt0,b0), dense-progress (rt0,b1) x seed{1,2}
for seed in 1 2; do
  sub "dsucc_thr0.85" robometer_ft 0.0 0.0 0 0.85 1 "$seed"
  sub "dprog_thr0.85" robometer_ft 1.0 0.0 0 0.85 1 "$seed"
done
echo "total: $n runs ($GO). manifest: $MAN"
