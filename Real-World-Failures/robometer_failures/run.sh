#!/bin/bash
# ============================================================
# RoboMeter Failure Scoring Pipeline — Main Entry Point
#
# Usage:
#   bash run.sh <archive_name>           # process one dataset
#   bash run.sh all                       # process all single-file archives
#   bash run.sh all real_robot            # process all real-robot archives
#   bash run.sh <archive_name> --stage 1  # VLM descriptions only
#   bash run.sh <archive_name> --stage 2  # LLM grading only (after stage 1)
#
# Required env vars:
#   HF_TOKEN   — HuggingFace token (for model + dataset downloads)
#
# Optional env vars (have sensible defaults):
#   DATA_DIR          — where to write keyframes + scores  (default: /data/robometer_failures)
#   VLM_MODEL_ID      — VLM model to use                  (default: Qwen/Qwen3.5-35B-A3B)
#   LLM_MODEL_ID      — LLM model to use                  (default: Qwen/Qwen3-32B)
#   VLM_BATCH_SIZE    — VLM inference batch size           (default: 32)
#   HF_HOME           — HuggingFace cache dir             (default: /data/hf_cache)
# ============================================================

set -e

ARCHIVE="${1:-all}"
CATEGORY="${2:-}"        # optional: real_robot | simulated | human_egocentric | eval_benchmark
STAGE_ARG="${3:-}"       # optional: --stage 1 or --stage 2

DATA_DIR="${DATA_DIR:-/data/robometer_failures}"
VLM_MODEL_ID="${VLM_MODEL_ID:-Qwen/Qwen3.5-35B-A3B}"
LLM_MODEL_ID="${LLM_MODEL_ID:-Qwen/Qwen3-32B}"
HF_HOME="${HF_HOME:-/data/hf_cache}"

export HF_TOKEN VLM_MODEL_ID LLM_MODEL_ID VLM_BATCH_SIZE HF_HOME
export PYTHONPATH="/pipeline:${PYTHONPATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "RoboMeter Failure Scoring Pipeline"
echo "  Archive:  ${ARCHIVE}"
echo "  Category: ${CATEGORY:-all}"
echo "  Data dir: ${DATA_DIR}"
echo "  VLM:      ${VLM_MODEL_ID}"
echo "  LLM:      ${LLM_MODEL_ID}"
echo "  Stage:    ${STAGE_ARG:-both (1+2)}"
echo "  Date:     $(date)"
echo "============================================================"

if [ -z "${HF_TOKEN}" ]; then
    echo "ERROR: HF_TOKEN is not set. Export it before running:"
    echo "  export HF_TOKEN=hf_xxxxxxxxxxxx"
    exit 1
fi

# ── Step 1: List available datasets (informational) ──────────────────────────
if [ "${ARCHIVE}" = "list" ]; then
    python3 "${SCRIPT_DIR}/extract_failures.py" --list-datasets
    exit 0
fi

# ── Step 2: Extract keyframes from HuggingFace archives ─────────────────────
echo ""
echo "=== STEP 1: Extracting failure keyframes ==="

EXTRACT_ARGS="--archive ${ARCHIVE} --output-dir ${DATA_DIR} --hf-token ${HF_TOKEN} --hf-cache-dir ${HF_HOME}"
if [ -n "${CATEGORY}" ]; then
    EXTRACT_ARGS="${EXTRACT_ARGS} --category ${CATEGORY}"
fi

python3 "${SCRIPT_DIR}/extract_failures.py" ${EXTRACT_ARGS}

echo ""
echo "=== STEP 2: VLM + LLM scoring ==="

SCORE_ARGS="--archive ${ARCHIVE} --data-dir ${DATA_DIR}"
if [ -n "${CATEGORY}" ]; then
    # If category filtering was applied in extraction, 'all' in scoring
    # will naturally only find manifests for those archives
    :
fi
if [ -n "${STAGE_ARG}" ]; then
    SCORE_ARGS="${SCORE_ARGS} ${STAGE_ARG}"
fi

python3 "${SCRIPT_DIR}/score_robometer_failures.py" ${SCORE_ARGS}

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "Results:"
echo "  Scores:      ${DATA_DIR}/scores/"
echo "  VLM desc:    ${DATA_DIR}/vlm_descriptions/"
echo "  Manifests:   ${DATA_DIR}/manifests/"
echo "  Keyframes:   ${DATA_DIR}/keyframes/"
echo "============================================================"
