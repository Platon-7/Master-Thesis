#!/bin/bash
# Handpicked 20 FAIR autonomous-RL runs (no GT in the loop) — test whether
# EITHER model (FT s4000 / baseline Robometer-4B) stays alive when the MODEL
# does success detection. 10 cells per model spanning the sparse<->dense reward
# spectrum x a few detection thresholds. Single seed (handpick); multi-seed the
# survivors later. Submit: bash jobs/launch_handpick.sh go
set -uo pipefail
cd "$(dirname "$0")/.."
CL=research-partner-cluster
PARTS="ondemand-g6e-1gpu-l40s,ondemand-g6e-4gpu-l40s,ondemand-g6e-8gpu-l40s"   # preemption-safe, packed via cons_tres
JOB=jobs/ibrl_sweep_robometer_aws.job
MANIFEST=/shared/home/PKA4388/vlm_ibrl_runs/handpick_manifest.tsv
GO="${1:-dry}"

# common (fair autonomous): model detects success, GT only logged; advisor episode_length=100
COMMON="AUTONOMOUS_SUCCESS=1,V3_CORNER2_ZOOM=1,ROBOMETER_REWARD_CAMERA=corner2_default,EPISODE_LENGTH=100,NUM_TRAIN_STEP=40000,USE_WB=0,TRAIN_TIMEOUT=21600"
FT=/shared/home/PKA4388/checkpoints/Robometer_FT_consolidated/run1_icl_ours_step4000
B4=/shared/home/PKA4388/checkpoints/Robometer-4B

# label                vlm           beta rthr rt det  seed   (rthr=reward-binarize thr; det=ROBOMETER_SUCCESS_THRESHOLD)
CELLS="
anchor        ROBVLM 0.0 0.0 1 0.6  1
dsucc         ROBVLM 0.0 0.0 0 0.6  1
dprog         ROBVLM 1.0 0.0 0 0.6  1
dprog_c       ROBVLM 1.0 0.0 0 0.85 1
anchor_c      ROBVLM 0.0 0.0 1 0.85 1
sparse        ROBVLM 0.0 0.6 1 0.6  1
dmix          ROBVLM 0.5 0.0 0 0.6  1
dsucc_c       ROBVLM 0.0 0.0 0 0.85 1
progsparse    ROBVLM 1.0 0.0 1 0.6  1
anchor_lo     ROBVLM 0.0 0.0 1 0.5  1
"

echo -e "# handpick launch\tjobid\tlabel\tvlm\tbeta\trthr\trt\tdet\tseed" > "$MANIFEST"
n=0
for VLM in robometer_ft robometer_4b; do
  EXTRA="ROBOMETER_FT_PATH=$FT,ROBOMETER_4B_PATH=$B4"
  while read -r label _vlm beta rthr rt det seed; do
    [ -z "${label:-}" ] && continue
    case "$label" in \#*) continue;; esac
    EXP="ALL,${COMMON},${EXTRA},VLM_NAME=${VLM},BC_POLICY=coffeepush,ROBOMETER_BETA=${beta},ROBOMETER_THRESHOLD=${rthr},REWARD_AT_TRUNC=${rt},ROBOMETER_SUCCESS_THRESHOLD=${det},SEED=${seed}"
    name="${VLM#robometer_}_${label}"
    n=$((n+1))
    if [ "$GO" = "go" ]; then
      jid=$(sbatch -M "$CL" --parsable --job-name="$name" --partition="$PARTS" --export="$EXP" "$JOB" 2>&1)
      jid=$(echo "$jid" | grep -oE "^[0-9]+")
      echo -e "${name}\t${jid}\t${label}\t${VLM}\t${beta}\t${rthr}\t${rt}\t${det}\t${seed}" >> "$MANIFEST"
      echo "submitted $name -> $jid"
    else
      echo "[dry] $name : beta=$beta rthr=$rthr rt=$rt det=$det seed=$seed"
    fi
  done <<< "$CELLS"
done
echo "total: $n runs ($GO).  manifest: $MANIFEST"
