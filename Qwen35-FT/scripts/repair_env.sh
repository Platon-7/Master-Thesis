#!/usr/bin/env bash
# Repair the robometer_qwen35_gpu env back to a known-good state.
#
# Symptoms this fixes:
#   * peft / torchvision import errors after a flash-linear-attention 0.5.0 install
#     (PyPI version yanks torch up to 2.11+cu13 and breaks the cu124 wheels).
#   * libcudnn.so.9 / nvidia-* missing (an over-aggressive cleanup uninstalled them).
#
# Strategy:
#   * NEVER uninstall nvidia-* packages by name — `pip install --force-reinstall torch`
#     pulls them back as deps automatically.
#   * Pin torch 2.6.0+cu124 + matching torchvision 0.21.0+cu124.
#   * Reinstall flash-linear-attention with --no-deps so it can't re-yank torch.
#   * flash-attn + causal-conv1d need nvcc — defer to jobs/install_flash_attn.job
#     (login nodes have no CUDA toolchain, sdist builds will fail with NameError).
#
# Usage (login node, ~5 min):
#     bash scripts/repair_env.sh
# Then verify:
#     bash scripts/preflight.sh
# Then (optional, on a GPU node) install flash-attn:
#     sbatch jobs/install_flash_attn.job

set -euo pipefail
PIP="/home/pkarageorgis1/.conda/envs/robometer_qwen35_gpu/bin/pip"
PY="/home/pkarageorgis1/.conda/envs/robometer_qwen35_gpu/bin/python"

echo "[$(date)] uninstall the high-level packages we want to swap (NEVER touch nvidia-*)"
$PIP uninstall -y flash-attn flash-linear-attention fla-core causal-conv1d 2>&1 || true

echo "[$(date)] force-reinstall pinned torch 2.6.0+cu124 — pulls all needed nvidia-* deps"
$PIP install --force-reinstall --index-url https://download.pytorch.org/whl/cu124 'torch==2.6.0+cu124' 'torchvision==0.21.0+cu124'

echo "[$(date)] force-reinstall peft + accelerate against torch 2.6"
$PIP install --force-reinstall --no-deps peft accelerate

echo "[$(date)] flash-linear-attention WITH --no-deps to keep torch 2.6"
$PIP install --no-deps -U flash-linear-attention fla-core einops

echo "[$(date)] verify imports (no flash_attn/causal_conv1d yet — those are GPU-job)"
$PY -c "
import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
import torchvision; print('torchvision:', torchvision.__version__)
import peft; print('peft:', peft.__version__)
import accelerate; print('accelerate:', accelerate.__version__)
import transformers; print('transformers:', transformers.__version__)
from transformers import Qwen3_5ForConditionalGeneration; print('Qwen3_5 OK')
import fla; print('fla: imported')
"

echo "[$(date)] REPAIR_DONE — flash-attn next: sbatch jobs/install_flash_attn.job"
