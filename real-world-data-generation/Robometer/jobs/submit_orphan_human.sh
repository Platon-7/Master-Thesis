#!/bin/bash
# Submit one SLURM job per archive for humanoid + human_hand orphan-success
# extraction. All jobs run concurrently — wall clock = slowest archive.
#
# Usage:
#   bash submit_orphan_human.sh
set -e

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB="${JOB_DIR}/extract_orphan_human.job"

# (group, archive) — order is by est. wall time (split + huge first, so they
# finish around the same time as the small singles dispatched later).
JOBS=(
    # human_hand split-tar archives (the biggest)
    "human_hand jesbu1_epic_rfm_epic"                     # 206 GB, 37k eps
    "human_hand jesbu1_egodex_rfm_egodex_part1"           # 135 GB, 45k eps
    "human_hand anqil_rh20t_subset_rfm_rh20t_human"       # 80 GB,  14k eps

    # humanoid split-tar archives
    "humanoid jesbu1_galaxea_rfm_galaxea_part3_r1_lite"   # 78 GB, 25k eps
    "humanoid jesbu1_galaxea_rfm_galaxea_part2_r1_lite"   # 72 GB, 25k eps
    "humanoid jesbu1_galaxea_rfm_galaxea_part1_r1_lite"   # 61 GB, 22k eps
    "humanoid jesbu1_galaxea_rfm_galaxea_part4_r1_lite"   # 60 GB, 22k eps
    "humanoid jesbu1_galaxea_rfm_galaxea_part5_r1_lite"   # 46 GB, 15k eps

    # Single-tar archives (fast — seekable)
    "humanoid jesbu1_humanoid_everyday_rfm_humanoid_everyday_rfm"  # 23G, 9k
    "human_hand jesbu1_h2r_rfm_h2r"                                # 11G, 2k
    "human_hand jesbu1_egodex_rfm_egodex_test"                     # 8.5G, 3k
    "human_hand jesbu1_usc_koch_human_robot_paired_usc_koch_human_robot_paired_human"  # 126M
    # hand_paired_human (37 MB, 9 eps) already extracted in smoke test — skip
)

SUBMITTED=()
for entry in "${JOBS[@]}"; do
    GROUP="${entry%% *}"
    ARCHIVE="${entry##* }"
    JOBID=$(sbatch --parsable \
        --export="GROUP=${GROUP},ARCHIVE=${ARCHIVE}" \
        --job-name="orph_${GROUP}_${ARCHIVE: -20}" \
        "$JOB")
    echo "submitted $JOBID  group=$GROUP  $ARCHIVE"
    SUBMITTED+=("$JOBID")
done

echo ""
echo "Submitted ${#SUBMITTED[@]} extraction jobs."
echo "Watch with: squeue -u \$USER -o '%i %j %T %M %R'"
echo "Job IDs: ${SUBMITTED[*]}"
