#!/bin/bash
# Phase-1 watcher: wait until concat (48) finishes and BOTH preprocess jobs
# (49 full, 50 no_extras) are RUNNING, then capture early progress and exit.
# Auto-resubmits concat if it dies a non-requeued death. Writes a status log.
CONCAT=48 FULL=49 NOX=50
STEP1_DIR=/fsx/$USER/robometer_frames_hf_full
LOG=/shared/home/PKA4388/Master-Thesis/Robometer-LoRA/logs/step2_watch.log
cd /shared/home/PKA4388/Master-Thesis/Robometer-LoRA
st(){ squeue -h -j "$1" -o "%T" 2>/dev/null; }      # queue state ("" if gone)
fin(){ sacct -nX -j "$1" -o State 2>/dev/null | head -1 | tr -d ' '; }  # terminal state
echo "=== watch start $(date -u +%H:%M:%S) ===" >> "$LOG"
for i in $(seq 1 240); do            # up to ~2h of 30s polls
  cs=$(st $CONCAT); fs=$(st $FULL); ns=$(st $NOX)
  echo "$(date -u +%H:%M:%S) concat=${cs:-_/$(fin $CONCAT)} full=${fs:-_} nox=${ns:-_}" >> "$LOG"

  # concat finished? -> train_raw should exist
  if [ -z "$cs" ]; then
    if [ -d "$STEP1_DIR/train_raw/robometer_frames_train" ]; then
      : # good, deps will release
    else
      cstate=$(fin $CONCAT)
      case "$cstate" in
        COMPLETED) ;;  # marked done but dir missing -> let it surface below
        FAILED|CANCELLED|NODE_FAIL|TIMEOUT|OUT_OF_MEMORY)
          echo "CONCAT_DIED state=$cstate -> resubmitting concat + redirecting deps" >> "$LOG"
          NC=$(sbatch --parsable jobs/concat_train_aws.job)
          scancel $FULL $NOX 2>/dev/null
          NF=$(sbatch --parsable --dependency=afterok:$NC jobs/step2_full_preprocess_aws.job)
          NN=$(sbatch --parsable --dependency=afterok:$NC jobs/step2_no_extras_aws.job)
          CONCAT=$NC; FULL=$NF; NOX=$NN
          echo "RESUBMIT concat=$NC full=$NF nox=$NN" >> "$LOG"
          ;;
      esac
    fi
  fi

  # both preprocess jobs running? capture early progress and finish phase 1
  if [ "$fs" = "RUNNING" ] && [ "$ns" = "RUNNING" ]; then
    echo "BOTH_RUNNING full=$FULL nox=$NOX — sampling progress in 120s" >> "$LOG"
    sleep 120
    echo "--- full ($FULL) tail ---" >> "$LOG"
    tr '\r' '\n' < logs/step2_full_${FULL}.out 2>/dev/null | grep -vE "^\s*$" | tail -6 >> "$LOG"
    tr '\r' '\n' < logs/step2_full_${FULL}.err 2>/dev/null | grep -oE "[0-9]+/[0-9]+ \[[0-9:]+<[0-9:?]+" | tail -1 >> "$LOG"
    echo "--- nox ($NOX) tail ---" >> "$LOG"
    tr '\r' '\n' < logs/step2_noextras_${NOX}.out 2>/dev/null | grep -vE "^\s*$" | tail -6 >> "$LOG"
    echo "PHASE1_OK full=$FULL nox=$NOX" >> "$LOG"
    echo "PHASE1_OK full=$FULL nox=$NOX"
    exit 0
  fi
  sleep 30
done
echo "PHASE1_TIMEOUT" >> "$LOG"; echo "PHASE1_TIMEOUT"; exit 2
