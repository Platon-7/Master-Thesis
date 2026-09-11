#!/bin/bash
# ============================================================
# RoboMeter Failure Scoring Pipeline — Main Entry Point
#
# Usage:
#   bash run.sh <archive_name>
#
# Source archives — set ONE of:
#   LOCAL_ARCHIVE_DIR  path to the root that contains single/ and split/
#                      subdirectories (e.g. /projects/prjs1958/robometer_full_dataset/raw_archives)
#                      OR a directory with a Google Drive download layout
#   HF_TOKEN           HuggingFace access token (used when LOCAL_ARCHIVE_DIR is unset)
#
# Required:
#   DATA_DIR           where to write keyframes + scores
#
# Optional:
#   VLM_MODEL_ID       Stage 1 model  (default: Qwen/Qwen3.5-35B-A3B)
#   LLM_MODEL_ID       Stage 2 model  (default: Qwen/Qwen3-32B)
#   VLM_BATCH_SIZE     VLM batch size (default: 32; set to 16 if GPU OOM)
#   HF_HOME            HuggingFace cache dir (default: /data/hf_cache)
#   MAX_EPISODES       stop after N episodes (testing only)
# ============================================================

set -e

ARCHIVE="${1:?Usage: bash run.sh <archive_name>}"

DATA_DIR="${DATA_DIR:-/data/robometer_failures}"
VLM_MODEL_ID="${VLM_MODEL_ID:-Qwen/Qwen3.5-35B-A3B}"
LLM_MODEL_ID="${LLM_MODEL_ID:-Qwen/Qwen3-32B}"
VLM_BATCH_SIZE="${VLM_BATCH_SIZE:-32}"
HF_HOME="${HF_HOME:-/data/hf_cache}"

export VLM_MODEL_ID LLM_MODEL_ID VLM_BATCH_SIZE HF_HOME
export PYTHONPATH="$(dirname "${BASH_SOURCE[0]}"):${PYTHONPATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "RoboMeter Failure Scoring Pipeline"
echo "  Archive:  ${ARCHIVE}"
echo "  Data dir: ${DATA_DIR}"
echo "  VLM:      ${VLM_MODEL_ID}"
echo "  LLM:      ${LLM_MODEL_ID}"
echo "  Date:     $(date)"
echo "============================================================"

# ── Resolve archive source ────────────────────────────────────────────────────
EXTRACT_ARGS="--archive ${ARCHIVE} --output-dir ${DATA_DIR}"

if [ -n "${MAX_EPISODES}" ]; then
    EXTRACT_ARGS="${EXTRACT_ARGS} --max-episodes ${MAX_EPISODES}"
fi

if [ -n "${LOCAL_ARCHIVE_DIR}" ]; then
    SINGLE_TAR="${LOCAL_ARCHIVE_DIR}/single/${ARCHIVE}.tar"
    SPLIT_DIR="${LOCAL_ARCHIVE_DIR}/split/${ARCHIVE}"

    if [ -f "${SINGLE_TAR}" ]; then
        echo "  Source: local single-file tar  (${SINGLE_TAR})"
        EXTRACT_ARGS="${EXTRACT_ARGS} --local-tar ${SINGLE_TAR}"
    elif [ -d "${SPLIT_DIR}" ]; then
        echo "  Source: local split parts dir  (${SPLIT_DIR})"
        EXTRACT_ARGS="${EXTRACT_ARGS} --local-tar-dir ${SPLIT_DIR}"
    else
        echo "ERROR: archive '${ARCHIVE}' not found under LOCAL_ARCHIVE_DIR=${LOCAL_ARCHIVE_DIR}"
        echo "  Looked for: ${SINGLE_TAR}"
        echo "           or: ${SPLIT_DIR}/"
        exit 1
    fi
elif [ -n "${HF_TOKEN}" ]; then
    echo "  Source: HuggingFace (token set)"
    EXTRACT_ARGS="${EXTRACT_ARGS} --hf-token ${HF_TOKEN}"
else
    echo "ERROR: set LOCAL_ARCHIVE_DIR (local files) or HF_TOKEN (HuggingFace download)"
    exit 1
fi

# ── Step 1: Extract keyframes ─────────────────────────────────────────────────
echo ""
echo "=== STEP 1: Extracting failure keyframes ==="
python3 "${SCRIPT_DIR}/selective_download.py" ${EXTRACT_ARGS}

# ── Step 2: VLM + LLM scoring ────────────────────────────────────────────────
echo ""
echo "=== STEP 2: VLM + LLM scoring ==="
python3 "${SCRIPT_DIR}/score_robometer_failures.py" \
    --archive "${ARCHIVE}" \
    --data-dir "${DATA_DIR}"

echo ""
echo "============================================================"
echo "Pipeline complete for: ${ARCHIVE}"
echo "  Scores:    ${DATA_DIR}/scores/${ARCHIVE}_scored.jsonl"
echo "  Manifests: ${DATA_DIR}/manifests/${ARCHIVE}.jsonl"
echo "  Keyframes: ${DATA_DIR}/keyframes/${ARCHIVE}/  (delete after scoring)"
echo "============================================================"
