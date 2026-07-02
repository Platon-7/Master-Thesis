#!/bin/bash
# Completion watcher: exit when both step-2 jobs (49 full, 50 nox) reach a
# terminal state, then report final progress + error counts + chain status.
FULL=49 NOX=50
cd /shared/home/PKA4388/Master-Thesis/Robometer-LoRA
LOG=logs/step2_watch.log
running(){ squeue -h -j "$1" -o "%T" 2>/dev/null; }
for i in $(seq 1 360); do          # up to ~12h of 2-min polls
  f=$(running $FULL); n=$(running $NOX)
  if [ -z "$f" ] && [ -z "$n" ]; then
    echo "STEP2_TERMINAL $(date -u +%H:%M:%S)" >> "$LOG"
    echo "=== STEP2 DONE ==="
    sacct -j $FULL,$NOX --format=JobID,JobName%16,State,ExitCode,Elapsed -P 2>/dev/null | grep -vE "\.batch|\.extern"
    echo "errors full(49): $(tr '\r' '\n' < logs/step2_full_${FULL}.out 2>/dev/null | grep -c 'Error reading')"
    echo "errors nox(50):  $(tr '\r' '\n' < logs/step2_noextras_${NOX}.out 2>/dev/null | grep -c 'Error reading')"
    echo "--- downstream chain (reencode 51 / topups 52,53) ---"
    squeue -u "$USER" -o "%.10i %.10T %.30E %R" 2>/dev/null
    exit 0
  fi
  sleep 120
done
echo "WATCH_TIMEOUT"; exit 2
