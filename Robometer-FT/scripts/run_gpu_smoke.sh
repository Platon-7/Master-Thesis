#!/usr/bin/env bash
# Submit GPU smokes for both losses in parallel and report job IDs + tail commands.
# Use scripts/check_gpu_smoke.sh <JID> to inspect status afterward.

set -euo pipefail
HERE="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$HERE"

mkdir -p logs

# Submit Loss 1.
JID1=$(sbatch --parsable --export=ALL,LOSS=1 jobs/smoke.job)
echo "submitted: smoke LOSS=1 → job $JID1"

# Submit Loss 2.
JID2=$(sbatch --parsable --export=ALL,LOSS=2 jobs/smoke.job)
echo "submitted: smoke LOSS=2 → job $JID2"

cat <<EOF

both smokes submitted. monitor with:
    squeue -u \$USER --jobs=$JID1,$JID2
    tail -F logs/smoke_${JID1}.out logs/smoke_${JID2}.out

quick health-check after they finish:
    bash scripts/check_gpu_smoke.sh $JID1 $JID2
EOF
