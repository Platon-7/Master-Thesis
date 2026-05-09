#!/usr/bin/env bash
# Convert one family of the keyframe dataset into Robometer's HF dataset cache.
# Usage: FAMILY=metaworld scripts/preprocess.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")"/.. && pwd)"
ROBOMETER_DIR="${ROBOMETER_DIR:-$HERE/../Robometer}"
FAMILY="${FAMILY:-metaworld}"

export ROBOMETER_DATASET_PATH="/projects/prjs1958/robometer_frame_dataset"
export ROBOMETER_PROCESSED_DATASETS_PATH="${ROBOMETER_PROCESSED_DATASETS_PATH:-/scratch-shared/$USER/robometer_frames_hf}"

cd "$ROBOMETER_DIR"
uv run python -m dataset_upload.generate_hf_dataset \
  --config_path "$HERE/configs/dataset_frames.yaml" \
  --dataset.dataset_name "robometer_frames_${FAMILY}" \
  --dataset.family "$FAMILY" \
  --output.output_dir "$ROBOMETER_PROCESSED_DATASETS_PATH/${FAMILY}_raw"

uv run python -m robometer.data.scripts.preprocess_datasets \
  --config "$HERE/configs/preprocess_${FAMILY}.yaml" \
  --cache_dir "$ROBOMETER_PROCESSED_DATASETS_PATH"
