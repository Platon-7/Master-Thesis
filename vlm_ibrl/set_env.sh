# Source this file once per shell before running any script from this repo:
#   source set_env.sh
#
# Override the defaults below by exporting the variables before sourcing,
# e.g. to point at a different conda env or MuJoCo install:
#   IBRL_CONDA_ENV=my_env MUJOCO_PATH=/opt/mujoco210 source set_env.sh

# Add repo root to PYTHONPATH so `import env`, `import rl`, etc. resolve.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Conda env. Defaults to "demo2reward"; override via IBRL_CONDA_ENV.
conda activate "${IBRL_CONDA_ENV:-demo2reward}"

# MuJoCo 2.1. Override via MUJOCO_PATH if you installed it elsewhere.
: "${MUJOCO_PATH:=$HOME/.mujoco/mujoco210}"
export MUJOCO_PY_MUJOCO_PATH="$MUJOCO_PATH"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/lib/nvidia:$MUJOCO_PATH/bin"

# Required for robomimic multi-process evaluation to behave.
export OMP_NUM_THREADS=1
