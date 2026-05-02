#!/bin/bash
# Create the 'metaworld' conda env with pinned versions matching the user's
# previous cluster setup.
#
# Versions:
#   metaworld : Farama-Foundation/Metaworld @ 58e32b4d (2025-04-03)
#               (last commit before `gym.vector.AutoresetMode` usage was
#                introduced in #relase branch 182673d2 on 2025-04-08, which
#                requires gymnasium >= 1.0. This commit still has -v3 tasks
#                + the modern gymnasium/mujoco bindings.)
#   gymnasium : 0.29.1
#   mujoco    : 3.1.1
#   pyopengl  : 3.1.7
#   numpy     : <2
#   cython    : <3
#   torch     : CUDA 12.1 build

set -euo pipefail

CONDA_BASE=/gpfs/scratch1/shared/pkarageorgis1/miniforge3
source $CONDA_BASE/etc/profile.d/conda.sh

ENV_NAME=metaworld
REPO_DIR=/home/pkarageorgis1/Master-Thesis/MetaWorld/metaworld_repo

# --- Create env ---------------------------------------------------------
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Env '$ENV_NAME' already exists — skipping create"
else
    echo "Creating conda env '$ENV_NAME' (python=3.10)..."
    conda create -y -n $ENV_NAME python=3.10
fi

conda activate $ENV_NAME

# --- Clean up stale installs (e.g. pre-2023 metaworld with mujoco_py) ---
echo "Uninstalling stale metaworld/mujoco_py if present..."
pip uninstall -y metaworld mujoco-py 2>/dev/null || true

# --- PyTorch CUDA 12.1 --------------------------------------------------
echo "Installing PyTorch (CUDA 12.1)..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# --- Core pins ----------------------------------------------------------
echo "Installing core pinned deps..."
pip install \
    "numpy<2" \
    "cython<3" \
    "gymnasium==0.29.1" \
    "mujoco==3.1.1" \
    "pyopengl==3.1.7" \
    "pillow" \
    "imageio" \
    "pyyaml"

# --- Metaworld repo (pinned to pre-2023-06-01 commit) -------------------
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "Cloning Metaworld..."
    git clone https://github.com/Farama-Foundation/Metaworld.git "$REPO_DIR"
fi

cd "$REPO_DIR"
git fetch --all --tags

# Pin to 58e32b4d (2025-04-03) — last commit before `gym.vector.AutoresetMode`
# usage was added (which requires gymnasium >= 1.0). Still has -v3 tasks and
# the modern gymnasium+mujoco bindings.
PIN_COMMIT=58e32b4d1f5f68f2396499bb16fc4cf7d4110786
echo "Checking out metaworld pin $PIN_COMMIT"
git checkout --detach "$PIN_COMMIT"
echo "Pinned metaworld to $PIN_COMMIT ($(git log -1 --format=%ci))"

echo "Installing metaworld (editable, skip dependency resolution so mujoco_py stays out)..."
pip install -e . --no-deps
# Metaworld's runtime deps beyond what we already pinned:
pip install scipy

# --- Verify -------------------------------------------------------------
echo
echo "=== Verification ==="
python -c "
import sys, torch, numpy, gymnasium, mujoco, metaworld
print(f'python   : {sys.version.split()[0]}')
print(f'torch    : {torch.__version__}  cuda={torch.cuda.is_available()}')
print(f'numpy    : {numpy.__version__}')
print(f'gymnasium: {gymnasium.__version__}')
print(f'mujoco   : {mujoco.__version__}')
print(f'metaworld: loaded from {metaworld.__file__}')
print(f'MT1(\"pick-place-v3\") builds: ', end='')
mt1 = metaworld.MT1('pick-place-v3', seed=42)
print(f'{len(mt1.train_tasks)} train_tasks')
"

echo
echo "=== Env '$ENV_NAME' ready ==="
