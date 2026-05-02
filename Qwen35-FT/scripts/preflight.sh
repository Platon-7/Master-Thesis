#!/usr/bin/env bash
# Pre-flight checks before `sbatch jobs/train_loss{1,2}.job`. Catches the dumb
# misconfiguration cases (missing env vars, missing deps, dataset paths not visible)
# in <30 seconds, on the login node.
#
# Usage: from Qwen35-FT/:
#     bash scripts/preflight.sh
#
# Exit code 0 = ready to sbatch. Non-zero = fix the printed issue first.

set -uo pipefail
HERE="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$HERE"

FAIL=0
note() { echo "  $*"; }
ok()   { echo "  [OK] $*"; }
warn() { echo "  [WARN] $*"; }
err()  { echo "  [FAIL] $*"; FAIL=1; }

PY="${QWEN35_PYTHON:-/home/pkarageorgis1/.conda/envs/robometer_qwen35_gpu/bin/python}"
ACC="${QWEN35_ACCEL:-/home/pkarageorgis1/.conda/envs/robometer_qwen35_gpu/bin/accelerate}"

echo "== preflight: Qwen35-FT =="

# 1. Env interpreters
[[ -x "$PY"  ]] && ok "python: $PY" || err "python not executable: $PY"
[[ -x "$ACC" ]] && ok "accelerate: $ACC" || err "accelerate not executable: $ACC"

# 2. Vendored package wins on import
out="$($PY -c "import sys; sys.path.insert(0, '$HERE'); import robometer; import os; print(os.path.realpath(os.path.dirname(robometer.__file__)))" 2>/dev/null)"
if [[ "$out" == "$HERE/robometer" ]]; then
    ok "robometer pkg resolves to vendored copy"
else
    err "robometer pkg resolves to: $out (expected $HERE/robometer)"
fi

# 3. Critical packages
for pkg in transformers torch peft accelerate; do
    ver="$($PY -c "import $pkg; print($pkg.__version__)" 2>/dev/null)"
    [[ -n "$ver" ]] && ok "$pkg $ver" || err "$pkg not importable"
done

# 4. Qwen3.5 class
$PY -c "from transformers import Qwen3_5ForConditionalGeneration" 2>/dev/null && \
    ok "Qwen3_5ForConditionalGeneration importable" || \
    err "Qwen3_5ForConditionalGeneration NOT importable — transformers >=5.7?"

# 5. Vendored setup_utils sanity
PYTHONPATH="$HERE:${PYTHONPATH:-}" $PY -c "
import robometer.utils.setup_utils as su
assert su.HAS_QWEN35
assert su._is_qwen35('Qwen/Qwen3.5-4B')
assert not su._is_qwen35('Qwen/Qwen3-VL-4B-Instruct')
" 2>/dev/null && ok "vendored setup_utils dispatch correct" || err "vendored setup_utils broken"

# 6. Flash-attention family (optional — warning only)
$PY -c "import flash_attn" 2>/dev/null && ok "flash_attn installed" || warn "flash_attn missing — fast path off, training will be slower"
$PY -c "import fla" 2>/dev/null && ok "flash-linear-attention (fla) installed" || warn "fla missing — Qwen3.5 linear_attn slow path"
$PY -c "import causal_conv1d" 2>/dev/null && ok "causal_conv1d installed" || warn "causal_conv1d missing — fla conv path slow"

# 7. WandB credentials
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    ok "WANDB_API_KEY is set"
else
    warn "WANDB_API_KEY not set — runs will go anonymous or fail at logger init"
fi

# 8. Dataset paths visible
HF_DIR="${ROBOMETER_PROCESSED_DATASETS_PATH:-/projects/prjs1958/robometer_frames_hf}"
[[ -d "$HF_DIR" ]] && ok "dataset dir: $HF_DIR" || err "dataset dir missing: $HF_DIR"

ICL_PAIRS="/scratch-shared/$USER/robometer_frames_splits/pairs_index_train.jsonl"
[[ -f "$ICL_PAIRS" ]] && ok "ICL pairs: $ICL_PAIRS" || warn "ICL pairs index missing — set ++data.use_icl=false or build splits"

# 9. WEIGHTS_DIR writable
WD="${WEIGHTS_DIR:-/projects/prjs1958/Qwen35_FT_weights}"
mkdir -p "$WD" 2>/dev/null && ok "weights dir writable: $WD" || err "cannot create $WD"

# 10. SLURM account
SACCT_INFO="$(sacctmgr -np show user $USER 2>/dev/null | head -1)"
echo "$SACCT_INFO" | grep -q gusei17535 && ok "SLURM account gusei17535 attached" || warn "user not on gusei17535 — train_loss*.job may fail at submit"

# 11. Configs sed-flatten cleanly (catch list-syntax breaks)
n_overrides=$(grep -hv '^[[:space:]]*#' configs/train_base.yaml configs/loss2_c51.yaml 2>/dev/null | sed -e 's/[[:space:]]*#.*//' | grep -v '^[[:space:]]*$' | wc -l)
[[ "$n_overrides" -gt 5 ]] && ok "loss2 sed-flatten: $n_overrides overrides" || err "loss2 sed-flatten produced too few lines: $n_overrides"

n_overrides=$(grep -hv '^[[:space:]]*#' configs/train_base.yaml configs/loss1_corn.yaml 2>/dev/null | sed -e 's/[[:space:]]*#.*//' | grep -v '^[[:space:]]*$' | wc -l)
[[ "$n_overrides" -gt 5 ]] && ok "loss1 sed-flatten: $n_overrides overrides" || err "loss1 sed-flatten produced too few lines: $n_overrides"

echo
if [[ $FAIL -eq 0 ]]; then
    echo "== preflight PASS — ready to sbatch =="
    exit 0
else
    echo "== preflight FAIL — fix the [FAIL] items above =="
    exit 1
fi
