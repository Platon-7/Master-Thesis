#!/bin/bash
# Deterministic watchdog: monitors the robometer_frame_dataset rclone restore,
# kills+cools-down+restarts on a detected hang, and on confirmed completion
# chains into step1 (build_splits.py) + step2 (5x preprocess_split.job).
#
# Runs as a background daemon (nohup'd by the caller). Everything is plain
# bash so behavior can't drift across a long unattended run.

set -u
LOG="/gpfs/home4/pkarageorgis/DAS-5/Master-Thesis/Robometer-LoRA/rclone_restore_frame_dataset.log"
WLOG="/gpfs/home4/pkarageorgis/DAS-5/Master-Thesis/Robometer-LoRA/watchdog.log"
DST="/scratch-shared/tmp.cwkV8vOvfY/robometer_frame_dataset"
TOTAL_FILES=4148
TOTAL_BYTES=295656707149
CHECK_INTERVAL=600      # 10 min routine check
HANG_PROBE_GAP=90       # seconds between the two samples used to detect a hang
COOLDOWN=1800           # 30 min after a detected hang before restarting

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$WLOG"; }

launch_rclone() {
    nohup rclone copy myonedrive:robometer_frame_dataset "$DST" \
        --progress --transfers=2 --checkers=4 --stats=30s \
        --low-level-retries=5 --contimeout=30s --timeout=60s \
        >> "$LOG" 2>&1 &
    disown
    log "launched rclone pid=$!"
}

last_transferred() {
    grep -oP 'Transferred:\s+\K[0-9.]+ GiB(?= /)' "$LOG" 2>/dev/null | tail -1
}

is_complete() {
    local nfiles nbytes
    nfiles=$(find "$DST" -type f 2>/dev/null | wc -l)
    nbytes=$(du -sb "$DST" 2>/dev/null | cut -f1)
    [[ "${nfiles:-0}" -ge "$TOTAL_FILES" && "${nbytes:-0}" -ge "$TOTAL_BYTES" ]]
}

log "watchdog started"

while true; do
    PID=$(pgrep -f "rclone copy myonedrive:robometer_frame_dataset" | head -1)

    if [[ -n "$PID" ]]; then
        s1=$(last_transferred)
        sleep "$HANG_PROBE_GAP"
        s2=$(last_transferred)
        if [[ -n "$s1" && "$s1" == "$s2" ]]; then
            log "HANG detected (stuck at $s1 for >${HANG_PROBE_GAP}s) — killing pid=$PID"
            kill "$PID" 2>/dev/null
            sleep 3
            kill -9 "$PID" 2>/dev/null
            nfiles=$(find "$DST" -type f 2>/dev/null | wc -l)
            log "landed so far: $nfiles files. Cooling down ${COOLDOWN}s before restart."
            sleep "$COOLDOWN"
            continue
        else
            nfiles=$(find "$DST" -type f 2>/dev/null | wc -l)
            log "healthy: $s2 transferred, $nfiles files landed"
            sleep "$CHECK_INTERVAL"
            continue
        fi
    fi

    # no process running
    if is_complete; then
        nfiles=$(find "$DST" -type f 2>/dev/null | wc -l)
        nbytes=$(du -sb "$DST" 2>/dev/null | cut -f1)
        log "DOWNLOAD COMPLETE: $nfiles files, $nbytes bytes"
        break
    else
        nfiles=$(find "$DST" -type f 2>/dev/null | wc -l)
        log "not running, incomplete ($nfiles files) — relaunching"
        launch_rclone
        sleep "$CHECK_INTERVAL"
        continue
    fi
done

# --- cross-check against the remote listing before trusting local completion ---
remote_n=$(rclone lsf myonedrive:robometer_frame_dataset -R --files-only 2>/dev/null | wc -l)
local_n=$(find "$DST" -type f 2>/dev/null | wc -l)
log "cross-check: remote=$remote_n local=$local_n"
if [[ "$local_n" -lt "$remote_n" ]]; then
    log "MISMATCH: local file count < remote count. NOT proceeding to step1/step2. Manual check needed."
    exit 1
fi

# --- chain: step1 (build_val_splits.py -- the FULL-POOL splitter, NOT the LoRA
# bake-off's build_splits.py. "No artificial train caps. Train pool = pairs_unified
# MINUS eval queries+partners. All ~860k non-eval rows kept." Patched earlier in
# this session to also drop LIBERO-90 (matches build_splits.py's exclusion). ---
SPLITS_DIR=/scratch-shared/tmp.cwkV8vOvfY/robometer_frames_splits_full_v2
STEP1_DIR=/scratch-shared/tmp.cwkV8vOvfY/robometer_frames_hf_full_v2
HF_OUT_DIR=/scratch-shared/tmp.cwkV8vOvfY/robometer_frames_hf_full_step2_v2
RAW_DATA=/scratch-shared/tmp.cwkV8vOvfY/robometer_frame_dataset

log "starting step1: build_val_splits.py (full pool, LIBERO-90 excluded)"
cd /gpfs/home4/pkarageorgis/DAS-5/Master-Thesis/Robometer-LoRA || { log "cd to Robometer-LoRA failed"; exit 1; }
/home/pkarageorgis/.conda/envs/robometer_gpu/bin/python scripts/build_val_splits.py \
    --pairs-jsonl "$RAW_DATA/pairs_unified.jsonl" \
    --output-dir "$SPLITS_DIR" \
    >> "$WLOG" 2>&1
rc=$?
if [[ $rc -ne 0 ]]; then
    log "build_val_splits.py FAILED with exit code $rc — NOT submitting preprocess jobs. Manual check needed."
    exit 1
fi
log "step1 (build_val_splits.py) succeeded"

# --- shard the (large, full-pool) train pairs_index for parallel step1 builds ---
log "sharding pairs_index_train.jsonl into 16 parts"
/home/pkarageorgis/.conda/envs/robometer_gpu/bin/python scripts/shard_pairs_index.py \
    --splits-dir "$SPLITS_DIR" --split train --n-shards 16 \
    >> "$WLOG" 2>&1
rc=$?
if [[ $rc -ne 0 ]]; then
    log "shard_pairs_index.py FAILED with exit code $rc — NOT submitting preprocess jobs. Manual check needed."
    exit 1
fi

# --- submit train array (16 parallel shards, step1 only) ---
log "submitting train array (16 shards)"
TRAIN_ARRAY_JID=$(sbatch --parsable --array=0-15 \
    --export=ALL,SPLIT=train,N_SHARDS=16,SPLITS_DIR=$SPLITS_DIR,ROBOMETER_PROCESSED_DATASETS_PATH=$STEP1_DIR,ROBOMETER_DATASET_PATH=$RAW_DATA \
    jobs/preprocess_split_array.job)
log "train array submitted -> job $TRAIN_ARRAY_JID (16 tasks)"

# --- submit the 4 eval splits (single-task each, step1 only) ---
declare -A EVAL_JIDS
for SPLIT in eval_droid eval_metaworld eval_failsafe eval_robometer; do
    jid=$(sbatch --parsable --array=0-0 \
        --export=ALL,SPLIT=$SPLIT,N_SHARDS=1,SPLITS_DIR=$SPLITS_DIR,ROBOMETER_PROCESSED_DATASETS_PATH=$STEP1_DIR,ROBOMETER_DATASET_PATH=$RAW_DATA \
        jobs/preprocess_split_array.job)
    EVAL_JIDS[$SPLIT]=$jid
    log "submitted SPLIT=$SPLIT step1 -> job $jid"
done

# --- chain step2 via Slurm dependencies (afterok on an array JID waits for ALL
# tasks in that array to succeed) -- no polling needed, Slurm sequences this. ---
log "submitting step2 for train (dependency=afterok:$TRAIN_ARRAY_JID)"
jid=$(sbatch --parsable --dependency=afterok:$TRAIN_ARRAY_JID \
    --export=ALL,SPLIT=train,N_SHARDS=16,SPLITS_DIR=$SPLITS_DIR,STEP1_DIR=$STEP1_DIR,ROBOMETER_PROCESSED_DATASETS_PATH=$HF_OUT_DIR,ROBOMETER_DATASET_PATH=$RAW_DATA \
    jobs/step2_after_array.job)
log "step2 train submitted -> job $jid"

for SPLIT in eval_droid eval_metaworld eval_failsafe eval_robometer; do
    dep=${EVAL_JIDS[$SPLIT]}
    jid=$(sbatch --parsable --dependency=afterok:$dep \
        --export=ALL,SPLIT=$SPLIT,SPLITS_DIR=$SPLITS_DIR,STEP1_DIR=$STEP1_DIR,ROBOMETER_PROCESSED_DATASETS_PATH=$HF_OUT_DIR,ROBOMETER_DATASET_PATH=$RAW_DATA \
        jobs/step2_after_array.job)
    log "step2 $SPLIT submitted -> job $jid (dependency=afterok:$dep)"
done

log "watchdog DONE — full pipeline submitted: train array ($TRAIN_ARRAY_JID, 16 tasks) + 4 eval step1 jobs + 5 step2 jobs (Slurm dependency-chained, no further polling needed)"
