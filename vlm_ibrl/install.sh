#!/usr/bin/env bash
# Install dependencies for Demo2Reward.
#
# Prereqs:
#   - MuJoCo 2.1 binaries unpacked at $HOME/.mujoco/mujoco210
#     (or wherever set_env.sh points). See README.
#   - A Python 3.9 conda env created and activated, e.g.:
#       conda create -n demo2reward python=3.9 -y
#       conda activate demo2reward
#
# Run from the repo root:
#   bash install.sh

set -euo pipefail

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "WARNING: no conda env appears to be active. Continue? [y/N]"
    read -r reply
    [[ "$reply" =~ ^[Yy]$ ]] || exit 1
fi

# 0) Fetch git submodules (pybind11). No-op if cloned with --recursive.
git submodule update --init --recursive

# 1) PyTorch + CUDA 12.1
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 2) Hugging Face stack required by Qwen3-VL. transformers from main because
#    the Qwen3-VL classes are not yet in a tagged release.
pip install git+https://github.com/huggingface/transformers
pip install accelerate
pip install qwen-vl-utils==0.0.14

# 3) flash-attn build prerequisites.
pip install packaging ninja

# 4) flash-attn. --no-build-isolation reuses the torch+ninja installed above
#    instead of pulling them again into a build sandbox.
python -m pip install flash-attn==2.7.3 --no-build-isolation --no-cache-dir

# 5) Remaining project dependencies (includes the correct triton pin for torch 2.4).
pip install -r requirements.txt

# 6) Compile the C++ replay-buffer extension.
make -C common_utils

echo
echo "Install complete. Remember to 'source set_env.sh' once per shell before running anything."
