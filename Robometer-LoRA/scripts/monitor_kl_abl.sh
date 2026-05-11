#!/bin/bash
# Monitor 3 KL ablation jobs. Emit one line per state change, per first-KL-fire, per first-error.
set -u
JOBS="22636650 22637786 22637789"
LOG_DIR=/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer-LoRA/logs
STATEDIR=/tmp/monitor_kl_abl_$$
mkdir -p "$STATEDIR"
for j in $JOBS; do
    echo "INIT" > "$STATEDIR/state_$j"
    echo "0"    > "$STATEDIR/kl_fired_$j"
    echo "0"    > "$STATEDIR/err_seen_$j"
done

while true; do
    finished=0
    for j in $JOBS; do
        st=$(sacct -j "$j" --noheader -X -o State 2>/dev/null | head -1 | awk '{print $1}')
        # Transient sacct hiccup: treat empty/UNKNOWN as "no info, keep prev state"
        # rather than emitting a spurious INIT -> UNKNOWN -> RUNNING flapping.
        if [[ -z "$st" || "$st" == "UNKNOWN" ]]; then
            st=$(cat "$STATEDIR/state_$j" 2>/dev/null)
            [[ -z "$st" ]] && st="UNKNOWN"
        fi
        prev=$(cat "$STATEDIR/state_$j" 2>/dev/null)
        if [[ "$prev" != "$st" ]]; then
            ts=$(date '+%H:%M:%S')
            echo "[$ts] job $j state change: $prev -> $st"
            echo "$st" > "$STATEDIR/state_$j"
        fi
        case "$st" in
            COMPLETED|FAILED|TIMEOUT|CANCELLED|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL)
                finished=$((finished+1))
                ;;
        esac

        # First KL fire detection
        if [[ "$(cat $STATEDIR/kl_fired_$j)" == "0" ]]; then
            LOG_O="$LOG_DIR/bakeoff_${j}_rbm_bakeoff.out"
            LOG_E="$LOG_DIR/bakeoff_${j}_rbm_bakeoff.err"
            if grep -qm1 "loss/failure_kl_buffer_fill\|loss/failure_kl" "$LOG_O" "$LOG_E" 2>/dev/null; then
                ts=$(date '+%H:%M:%S')
                echo "[$ts] job $j KL anchor FIRED (failure_kl metric logged)"
                echo "1" > "$STATEDIR/kl_fired_$j"
            fi
        fi

        # First error/traceback detection
        if [[ "$(cat $STATEDIR/err_seen_$j)" == "0" ]]; then
            LOG_E="$LOG_DIR/bakeoff_${j}_rbm_bakeoff.err"
            if [[ -f "$LOG_E" ]] && grep -qm1 -E "Traceback|RuntimeError|ValueError|OutOfMemoryError|CUDA out of memory" "$LOG_E"; then
                errtype=$(grep -m1 -oE "Traceback|RuntimeError|ValueError|OutOfMemoryError|CUDA out of memory" "$LOG_E" | head -1)
                ts=$(date '+%H:%M:%S')
                echo "[$ts] job $j ERROR detected: $errtype"
                echo "1" > "$STATEDIR/err_seen_$j"
            fi
        fi
    done

    if [[ $finished -ge 3 ]]; then
        ts=$(date '+%H:%M:%S')
        echo "[$ts] all 3 jobs terminal -- monitor exiting"
        break
    fi
    sleep 120
done
