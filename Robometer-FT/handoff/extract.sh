#!/usr/bin/env bash
# One-time extract of the conda-packed Robometer env. Runs on the login node,
# no GPU/SLURM allocation needed. Takes ~3 min and ~3-5 GB of disk.
set -euo pipefail

TARBALL="${TARBALL:-/projects/prjs1958/envs/robometer_gpu.tar.gz}"
ENV_PREFIX="${ENV_PREFIX:-/projects/prjs1958/envs/robometer_gpu}"

[[ -f "$TARBALL" ]] || { echo "ERROR: tarball not at $TARBALL" >&2; exit 1; }

if [[ -d "$ENV_PREFIX" && -n "$(ls -A "$ENV_PREFIX" 2>/dev/null)" ]]; then
    echo "WARNING: $ENV_PREFIX is non-empty — refusing to overwrite." >&2
    echo "  Either delete it or set ENV_PREFIX=<other path> before re-running." >&2
    exit 1
fi

mkdir -p "$ENV_PREFIX"
echo "[extract] unpacking $TARBALL into $ENV_PREFIX..."
tar -xzf "$TARBALL" -C "$ENV_PREFIX"

echo "[extract] running conda-unpack (rewrites shebangs to new prefix)..."
"$ENV_PREFIX/bin/conda-unpack"

echo
echo "[extract] DONE — env ready at $ENV_PREFIX"
echo
echo "Next: submit the env smoke (replaces <YOUR_ACCOUNT> with your SLURM account):"
echo "    sbatch --account=<YOUR_ACCOUNT> \\"
echo "           --export=ALL,ENV_PREFIX=$ENV_PREFIX \\"
echo "           env_smoke.job"
echo
echo "Then: watch the .out for either '>>> ENV SMOKE PASSED' or '>>> ENV SMOKE FAILED'."
