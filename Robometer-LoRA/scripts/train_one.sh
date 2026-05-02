#!/usr/bin/env bash
# Run a single LoRA training with a given loss variant id.
# Usage: LOSS_ID=disc4 scripts/train_one.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")"/.. && pwd)"
ROBOMETER_DIR="${ROBOMETER_DIR:-$HERE/../Robometer}"
LOSS_ID="${LOSS_ID:?set LOSS_ID (e.g. l1|l2|disc4|disc10)}"

export ROBOMETER_PROCESSED_DATASETS_PATH="${ROBOMETER_PROCESSED_DATASETS_PATH:-/scratch-shared/$USER/robometer_frames_hf}"

# launch_sweep.py materialises per-variant overrides into this file.
OVERRIDES_FILE="$HERE/configs/_resolved_${LOSS_ID}.overrides"
if [[ ! -f "$OVERRIDES_FILE" ]]; then
  echo "missing $OVERRIDES_FILE — run launch_sweep.py or create it manually" >&2
  exit 2
fi

cd "$ROBOMETER_DIR"
# shellcheck disable=SC2046
uv run python train.py \
  $(tr '\n' ' ' < "$OVERRIDES_FILE") \
  training.output_dir="$HERE/logs/${LOSS_ID}" \
  training.exp_name="robometer_lora_${LOSS_ID}"
