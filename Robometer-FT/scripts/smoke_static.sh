#!/usr/bin/env bash
# Static smoke for Robometer-FT — runs on login node, no GPU needed.
#
# Layers (in order, halts on first failure):
#   T01  YAML lint              — every config parses as YAML
#   T02  Sed-flatten works      — train_base + lossN flatten cleanly to ++key=val
#   T03  Override precedence    — loss preset wins on duplicate keys (last-wins)
#   T04  Path existence         — Robometer dir, train.py, fsdp.yaml, dataset cache,
#                                 pairs index, conda env python+accelerate
#   T05  Hydra dataclass parse  — merged overrides build a valid ExperimentConfig
#                                 (catches typo'd keys, type mismatches)
#
# Usage:
#     bash scripts/smoke_static.sh
#
# Exit 0 on full pass, non-zero on first failure.

set -uo pipefail

HERE="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$HERE"

PYTHON="${ROBOMETER_PYTHON:-/home/pkarageorgis1/.conda/envs/robometer_gpu/bin/python}"
ROBOMETER_DIR="${ROBOMETER_DIR:-$HERE/../Robometer}"
HF_DIR="${ROBOMETER_PROCESSED_DATASETS_PATH:-/projects/prjs1958/robometer_frames_hf}"
PAIRS_INDEX="${PAIRS_INDEX:-/scratch-shared/pkarageorgis1/robometer_frames_splits/pairs_index_train.jsonl}"

GREEN=$'\033[32m'; RED=$'\033[31m'; DIM=$'\033[2m'; RESET=$'\033[0m'
pass() { echo "${GREEN}[PASS]${RESET} $*"; }
fail() { echo "${RED}[FAIL]${RESET} $*" >&2; exit 1; }
info() { echo "${DIM}       $*${RESET}"; }

echo "=== Robometer-FT static smoke ==="
echo "HERE          : $HERE"
echo "ROBOMETER_DIR : $ROBOMETER_DIR"
echo "PYTHON        : $PYTHON"
echo

# ---------- T01: YAML lint ----------
echo "[T01] YAML lint"
for f in configs/train_base.yaml configs/loss1_corn.yaml configs/loss2_c51.yaml configs/distributed/fsdp.yaml; do
    "$PYTHON" -c "import yaml,sys; yaml.safe_load(open('$f'))" 2>&1 || fail "T01: $f did not parse as YAML"
    info "$f OK"
done
pass "T01: all 4 configs parse as YAML"
echo

# ---------- T02: Sed-flatten produces well-formed Hydra overrides ----------
echo "[T02] Sed-flatten — train_base + lossN → Hydra ++key=value"
flatten() {
    local base="$1" preset="$2"
    cat "$base" "$preset" \
      | grep -v '^[[:space:]]*#' \
      | sed -e 's/[[:space:]]*#.*//' \
      | grep -v '^[[:space:]]*$' \
      | sed -e 's/: */=/' -e 's/^/++/'
}

for preset in configs/loss1_corn.yaml configs/loss2_c51.yaml; do
    out=$(flatten configs/train_base.yaml "$preset")
    n=$(echo "$out" | wc -l)
    [[ "$n" -lt 20 ]] && fail "T02: only $n overrides for $preset — flatten broke"

    # Every line MUST be of the form ++key=value (no stray whitespace, no
    # block-list survivors).
    bad=$(echo "$out" | grep -v '^++[A-Za-z_][A-Za-z0-9_.]*=' || true)
    if [[ -n "$bad" ]]; then
        echo "$bad" >&2
        fail "T02: malformed override lines for $preset"
    fi
    info "$preset → $n overrides, all well-formed"
done
pass "T02: sed-flatten produces well-formed ++key=value overrides"
echo

# ---------- T03: Override precedence — loss preset wins on duplicate keys ----------
echo "[T03] Override precedence — loss preset must come AFTER train_base"
# train_base sets train_progress_head=true, train_success_head=true.
# loss1_corn overrides train_success_head=false.
# loss2_c51 keeps both true (no actual override on success_head, but still
# checks the duplicate-key path).
out1=$(flatten configs/train_base.yaml configs/loss1_corn.yaml)
last_succ=$(echo "$out1" | grep '^++model.train_success_head=' | tail -1)
[[ "$last_succ" == "++model.train_success_head=false" ]] \
    || fail "T03: loss1 should leave train_success_head=false as the LAST occurrence (got: $last_succ)"
info "loss1: last occurrence of model.train_success_head = false  ✓"

# loss2 overrides loss.progress_loss_type=c51_asymmetric (train_base default
# is whatever the parent config.yaml has). Verify the value at the last
# occurrence is the loss-preset's value.
out2=$(flatten configs/train_base.yaml configs/loss2_c51.yaml)
last_ploss=$(echo "$out2" | grep '^++loss.progress_loss_type=' | tail -1)
[[ "$last_ploss" == "++loss.progress_loss_type=c51_asymmetric" ]] \
    || fail "T03: loss2 should leave progress_loss_type=c51_asymmetric as the LAST occurrence (got: $last_ploss)"
info "loss2: last occurrence of loss.progress_loss_type = c51_asymmetric  ✓"
pass "T03: loss preset wins on duplicate keys (cat order: base → preset)"
echo

# ---------- T04: Path existence ----------
echo "[T04] Path existence"
for p in \
    "$ROBOMETER_DIR/train.py" \
    "$ROBOMETER_DIR/robometer/configs/config.yaml" \
    "$HERE/configs/distributed/fsdp.yaml" \
    "$PYTHON" \
    "${PYTHON%/*}/accelerate" \
    "$HF_DIR" \
    "$PAIRS_INDEX"; do
    if [[ ! -e "$p" ]]; then fail "T04: missing $p"; fi
    info "$p"
done
pass "T04: all required paths exist"
echo

# ---------- T05: Hydra dataclass parse ----------
echo "[T05] Hydra dataclass parse — merged overrides build ExperimentConfig"
# Pick loss1 (the more architecturally-divergent preset — fresh CORN head,
# different progress_head_mode). If it parses, loss2 will too.
overrides=$(flatten configs/train_base.yaml configs/loss1_corn.yaml | tr '\n' ' ')

# Build a tiny driver that runs Hydra's compose API with the same overrides
# the SLURM job will pass. No model load, no data load — just config parse.
PYTHONPATH="$ROBOMETER_DIR:${PYTHONPATH:-}" "$PYTHON" - <<EOF || fail "T05: Hydra dataclass parse failed"
import os, sys
os.chdir("$ROBOMETER_DIR")
sys.path.insert(0, "$ROBOMETER_DIR")

from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.core.config_store import ConfigStore
from robometer.configs.experiment_configs import (
    ExperimentConfig, ModelConfig, PEFTConfig, DataConfig, TrainingConfig,
    LossConfig, LoggingConfig, SaveBestConfig,
)

# Mirror train.py's ConfigStore registration so the defaults entry
# (base_config, _self_) resolves. Without this, Hydra's defaults resolver
# looks for a base_config.yaml file and fails with MissingConfigException.
cs = ConfigStore.instance()
cs.store(name="base_config", node=ExperimentConfig)
cs.store(group="model", name="model_config", node=ModelConfig)
cs.store(group="peft", name="peft_config", node=PEFTConfig)
cs.store(group="data", name="data_config", node=DataConfig)
cs.store(group="training", name="training_config", node=TrainingConfig)
cs.store(group="loss", name="loss_config", node=LossConfig)
cs.store(group="logging", name="logging_config", node=LoggingConfig)
cs.store(group="logging/save_best", name="save_best_config", node=SaveBestConfig)

GlobalHydra.instance().clear()
overrides = [o for o in """$overrides""".split() if o.strip()]

with initialize_config_dir(config_dir=os.path.abspath("robometer/configs"), version_base=None):
    cfg = compose(config_name="config", overrides=overrides)
    # Sentinel keys that the loss1 preset MUST have set.
    assert cfg.model.use_peft is False, f"use_peft should be False, got {cfg.model.use_peft}"
    assert cfg.model.use_unsloth is False, f"use_unsloth should be False, got {cfg.model.use_unsloth}"
    assert cfg.model.progress_head_mode == "corn", f"progress_head_mode should be 'corn', got {cfg.model.progress_head_mode}"
    assert cfg.loss.progress_loss_type == "corn_asymmetric", f"progress_loss_type should be 'corn_asymmetric', got {cfg.loss.progress_loss_type}"
    assert cfg.training.load_from_checkpoint == "robometer/Robometer-4B", f"load_from_checkpoint should be 'robometer/Robometer-4B', got {cfg.training.load_from_checkpoint}"
    assert cfg.data.use_icl is True, "use_icl should be True"
    print(f"compose OK: {len(overrides)} overrides applied, all sentinel keys correct")
EOF
pass "T05: merged overrides compose into a valid Robometer config"
echo

echo "${GREEN}=== ALL STATIC SMOKE TESTS PASSED ===${RESET}"
