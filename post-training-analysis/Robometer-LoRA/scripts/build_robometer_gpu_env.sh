#!/bin/bash
# Build the robometer_gpu conda env on Snellius (this account).
# Uses explicit $PIP/$PYTHON paths so conda-activate quirks don't matter.

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/home/pkarageorgis/.conda/envs/robometer_gpu}"
REPO_ROOT="/home/pkarageorgis/DAS-5/Master-Thesis"

echo "=== loading Anaconda module ==="
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load 2025 Anaconda3/2025.06-1

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    echo "=== creating env: $ENV_PREFIX (python 3.10, conda-forge only) ==="
    mkdir -p "$(dirname "$ENV_PREFIX")"
    conda create -n robometer_gpu -c conda-forge --override-channels python=3.10 -y
else
    echo "=== env $ENV_PREFIX already exists; reusing ==="
fi

PYTHON="$ENV_PREFIX/bin/python"
PIP="$PYTHON -m pip"

# Keep pip cache off $HOME (quota tight) — use scratch
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch-shared/tmp.cwkV8vOvfY/pip_cache}"
mkdir -p "$PIP_CACHE_DIR"

echo "python:    $($PYTHON --version)"
echo "pip:       $($PIP --version)"
echo "pip cache: $PIP_CACHE_DIR"

echo "=== upgrading pip in env ==="
$PIP install --upgrade pip

echo "=== installing torch 2.8.0 + torchvision 0.23.0 (cu128) ==="
$PIP install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 torchvision==0.23.0

$PYTHON -c "import torch; print('torch:', torch.__version__, 'cuda_available:', torch.cuda.is_available())"

echo "=== installing pinned deps ==="
$PIP install \
    accelerate==1.13.0 \
    transformers==4.57.2 \
    datasets==4.3.0 \
    peft==0.19.1 \
    safetensors==0.7.0 \
    huggingface_hub==0.36.2 \
    sentence-transformers==5.4.1 \
    'qwen-vl-utils[decord]==0.0.14' \
    decord==0.6.0 \
    xformers==0.0.32.post2 \
    bitsandbytes==0.49.2 \
    numpy==2.2.6 \
    einops==0.8.2 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    'pyrallis>=0.3.0' \
    'rich>=14.0.0' \
    'loguru>=0.7.3' \
    'termcolor>=3.1.0' \
    'codetiming>=1.4.0' \
    matplotlib pillow h5py scipy pyyaml tqdm \
    opencv-python-headless \
    wandb \
    pandas \
    av \
    fastapi uvicorn requests \
    moviepy imageio \
    evaluate \
    ipdb \
    hf_transfer \
    unsloth==2026.4.8

echo "=== verifying imports ==="
$PYTHON -c "
import sys
print('python:', sys.version.split()[0])
for mod in ['torch', 'transformers', 'datasets', 'accelerate', 'peft', 'safetensors',
            'numpy', 'sentence_transformers', 'hydra', 'omegaconf', 'einops',
            'xformers', 'bitsandbytes', 'torchvision', 'decord', 'qwen_vl_utils']:
    try:
        m = __import__(mod)
        print(f'  {mod}: {getattr(m, \"__version__\", \"?\")}')
    except Exception as e:
        print(f'  {mod}: MISSING ({type(e).__name__}: {e})')
"

echo "=== installing Robometer in editable mode (--no-deps) ==="
cd "$REPO_ROOT/Robometer"
$PIP install -e . --no-deps

echo ""
echo "=== DONE ==="
echo "env path: $ENV_PREFIX"
echo "python:   $PYTHON"
