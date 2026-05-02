#!/usr/bin/env bash
# Run smoke tests 00–07 in order. Halts on first failure.
#
# Usage: from Qwen35-FT/:
#     bash scripts/run_all_smoke_tests.sh
#
# Each test prints "PASS" on success. The runner re-runs through 07 even if a heavy
# test (03/05) takes minutes — order matters because 04 caches discovery JSON that
# 07 consumes.

set -uo pipefail

HERE="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$HERE"

PYTHON="${QWEN35_PYTHON:-/home/pkarageorgis1/.conda/envs/robometer_qwen35_gpu/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: $PYTHON not found — env not built?" >&2
    echo "  see logs/env_build.log" >&2
    exit 1
fi

# Vendored robometer/ must win on import path (only matters for tests 06/07).
export PYTHONPATH="$HERE:${PYTHONPATH:-}"

TESTS=(
    smoke_test_00_env.py
    smoke_test_01_config.py
    smoke_test_02_processor.py
    smoke_test_03_model_load.py
    smoke_test_04_module_names.py
    smoke_test_05_forward.py
    smoke_test_06_collator.py
    smoke_test_07_lora_targets.py
    smoke_test_08_setup_utils.py
)

for t in "${TESTS[@]}"; do
    echo
    echo "================ $t ================"
    if ! "$PYTHON" "scripts/$t"; then
        echo "FAIL: $t — halting" >&2
        exit 1
    fi
done
echo
echo "================ all smoke tests passed ================"
