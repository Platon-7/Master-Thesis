#!/usr/bin/env bash
# Inspect a submitted Robometer-FT GPU smoke. Reports queue/run state, then
# (if finished) summarizes the .out file: trainer return code, finite-loss
# check on the 2 logged steps, and SMOKE PASSED/FAILED marker.
#
# Usage:  bash scripts/check_gpu_smoke.sh <JID> [<JID> ...]

set -uo pipefail
HERE="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$HERE"

GREEN=$'\033[32m'; RED=$'\033[31m'; YELLOW=$'\033[33m'; DIM=$'\033[2m'; RESET=$'\033[0m'

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <JID> [<JID> ...]" >&2
    exit 2
fi

for JID in "$@"; do
    echo "=================== job $JID ==================="
    state=$(sacct -j "$JID" --format=State --noheader --parsable2 2>/dev/null | head -1 | tr -d ' ')
    if [[ -z "$state" ]]; then
        # Still in queue or just submitted — sacct lags squeue.
        sq=$(squeue -j "$JID" --noheader -o '%T %R' 2>/dev/null || true)
        echo "${YELLOW}state${RESET}: ${sq:-<unknown — sacct empty, squeue empty>}"
        echo
        continue
    fi
    echo "state: $state"

    OUT="logs/smoke_${JID}.out"
    if [[ ! -f "$OUT" ]]; then
        echo "${YELLOW}(no output file yet at $OUT)${RESET}"
        echo
        continue
    fi

    # Trainer return code line (printed at end of job).
    rc_line=$(grep '^trainer return code:' "$OUT" | tail -1 || true)
    pass_line=$(grep -E '^SMOKE (PASSED|FAILED)' "$OUT" | tail -1 || true)

    echo "log size: $(wc -l < "$OUT") lines"
    [[ -n "$rc_line" ]] && echo "$rc_line" || echo "${YELLOW}(no return-code line — job may not have finished)${RESET}"

    # Finite-loss check on the 2 training steps the smoke logs.
    # Trainer logs lines like:  {'loss': 1.234, 'grad_norm': 0.5, 'learning_rate': 2e-05, ...}
    loss_lines=$(grep -E "'loss': [0-9]" "$OUT" | head -3 || true)
    if [[ -n "$loss_lines" ]]; then
        echo "${DIM}--- training step loss lines ---${RESET}"
        echo "$loss_lines" | sed 's/^/  /'
        # Reject NaN/Inf in any of the logged losses.
        if echo "$loss_lines" | grep -qE "'loss': (nan|inf|-inf)"; then
            echo "${RED}FINITE-LOSS CHECK FAILED — NaN/Inf detected${RESET}"
        else
            echo "${GREEN}finite-loss check OK${RESET}"
        fi
    else
        echo "${YELLOW}(no 'loss': lines found — trainer may not have reached step 1)${RESET}"
    fi

    if [[ "$pass_line" == "SMOKE PASSED"* ]]; then
        echo "${GREEN}>>> $pass_line <<<${RESET}"
    elif [[ "$pass_line" == "SMOKE FAILED"* ]]; then
        echo "${RED}>>> $pass_line <<<${RESET}"
        echo "${DIM}--- last 30 lines of stderr ---${RESET}"
        tail -30 "logs/smoke_${JID}.err" 2>/dev/null | sed 's/^/  /'
    fi
    echo
done
