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

# Bundled NVIDIA libs from the pip-installed torch (cu121) wheels. The C++
# replay-buffer extension built via `make -C common_utils` links against
# libnvrtc.so.12 etc., which torch's pip install drops into
# site-packages/nvidia/*/lib/ but never adds to LD_LIBRARY_PATH. Without this,
# `import common_utils.rela` raises `ImportError: libnvrtc.so.12: cannot open
# shared object file`.
if [ -n "${CONDA_PREFIX:-}" ]; then
    _NVIDIA_LIBS=$(find "$CONDA_PREFIX"/lib/python*/site-packages/nvidia -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':' | sed 's/:$//')
    [ -n "$_NVIDIA_LIBS" ] && export LD_LIBRARY_PATH="$_NVIDIA_LIBS:$LD_LIBRARY_PATH"
    unset _NVIDIA_LIBS
fi

# Required for robomimic multi-process evaluation to behave.
export OMP_NUM_THREADS=1
