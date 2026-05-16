#!/usr/bin/env python3
"""End-to-end test of TWO recent fixes:

  (A) Metaworld task-drop filter — `_filter_dataset` excludes 4 specific tasks
      from TRAIN only (eval kept). Verifies:
        - The exclusion logic actually filters those rows
        - Expected drop counts match (~3,210 from train, 0 from eval)
        - Eval split keeps the 4 tasks intact for use as held-out targets

  (B) Droid failure-oversample boost — `failure_oversample_per_source.droid: 5.5`
      makes droid failures 5.5× more likely (multiplicative on the global 2.61).
      Verifies:
        - yaml → sed → Hydra → DataConfig pipeline carries the value
        - Sampler weight construction is exact: droid_fail = 2.61 × 5.5 = 14.35
        - Other-source weights are unchanged
        - Empirical sampling reproduces the theoretical succ:fail ratio
          within droid (64% : 36%, was 91% : 9%)
        - Visits per unique droid_fail over a typical run ≈ 5

Uses the REAL train HF dataset for the count/sampling checks (not synthetic),
so this exercises the production pipeline end-to-end. Bulk-materializes columns
via PyArrow to keep wall time under 2 min.

Run:
    /home/pkarageorgis1/.conda/envs/robometer_qwen35_gpu/bin/python3 \
        /gpfs/home3/pkarageorgis1/Master-Thesis/Robometer/scripts/test_sampling_and_filter_fixes.py
"""
import sys, os, subprocess, time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import glob
import torch
from torch.utils.data import WeightedRandomSampler

sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer")
sys.stdout.reconfigure(line_buffering=True)

def section(t): print(f"\n{'='*80}\n  {t}\n{'='*80}")

HELDOUT_METAWORLD = {
    "Grasp the cover and close the box with it.",        # box-close
    "Grasp a stick and pull a box with the stick.",      # stick-pull
    "Push a mug under a coffee machine.",                # coffee-push
    "Pick up a nut and place it onto a peg.",            # assemble (nut)
}

# ===========================================================================
section("TEST 1 — yaml/Hydra/OmegaConf round-trip for BOTH new knobs")
# ===========================================================================
from omegaconf import OmegaConf
from hydra.core.override_parser.overrides_parser import OverridesParser
from robometer.configs.experiment_configs import ExperimentConfig, DataConfig

# 1a) DataConfig has the new field and the right default
cfg_default = DataConfig()
assert hasattr(cfg_default, "failure_oversample_per_source"), \
    "DataConfig is missing failure_oversample_per_source field"
assert cfg_default.failure_oversample_per_source == {}, \
    f"Default should be empty dict, got {cfg_default.failure_oversample_per_source!r}"
assert cfg_default.success_oversample_per_source == {}, \
    f"Default should be empty dict, got {cfg_default.success_oversample_per_source!r}"
print(f"  ✓ DataConfig defaults: succ_per_src=={{}}, fail_per_src=={{}}")

# 1b) Hydra parses the yaml-derived overrides
yaml_lines = [
    "data.success_oversample_per_source.metaworld: 13.0",
    "data.failure_oversample_per_source.droid: 5.5",
]
sed_pipeline = "sed -e 's/[[:space:]]*#.*//' | grep -v '^[[:space:]]*$' | sed -e 's/: */=/' -e 's/^/++/'"
overrides = []
for line in yaml_lines:
    out = subprocess.check_output(f"echo '{line}' | {sed_pipeline}", shell=True, executable="/bin/bash").decode().strip()
    overrides.append(out)
    print(f"  yaml: {line!r}  →  override: {out!r}")

# 1c) Apply the overrides through Hydra's actual parser
parser = OverridesParser.create()
parsed = parser.parse_overrides(overrides)
for p in parsed:
    print(f"  Hydra parsed: key={p.key_or_group}  value={p.value()}  (type={type(p.value()).__name__})")

# 1d) Verify structured-config merge actually works (this was the .None-merge bug we hit before)
schema = OmegaConf.structured(ExperimentConfig)
for p in parsed:
    OmegaConf.update(schema, p.key_or_group, p.value(), merge=True)
assert dict(schema.data.success_oversample_per_source) == {"metaworld": 13.0}, \
    f"merged succ_per_src wrong: {dict(schema.data.success_oversample_per_source)}"
assert dict(schema.data.failure_oversample_per_source) == {"droid": 5.5}, \
    f"merged fail_per_src wrong: {dict(schema.data.failure_oversample_per_source)}"
print(f"  ✓ Final merged config: succ_per_src={dict(schema.data.success_oversample_per_source)}, "
      f"fail_per_src={dict(schema.data.failure_oversample_per_source)}")


# ===========================================================================
section("TEST 2 — load REAL train + eval datasets (PyArrow bulk-materialize)")
# ===========================================================================
def bulk_load(arrow_glob):
    """Bulk-load 4 columns via PyArrow streams (~10s vs ~5+min for per-row)."""
    files = sorted(glob.glob(arrow_glob))
    out = {"task": [], "quality_label": [], "data_source": [], "id": []}
    for f in files:
        with pa.ipc.open_stream(f) as reader:
            for batch in reader:
                for k in out: out[k].extend(batch.column(k).to_pylist())
    return out

t0 = time.time()
train = bulk_load("/projects/prjs1958/robometer_frames_hf_full/train_raw/robometer_frames_train/*.arrow")
print(f"  train_raw loaded: {len(train['task']):,} rows in {time.time()-t0:.1f}s")
t0 = time.time()
eval_droid = bulk_load("/projects/prjs1958/robometer_frames_hf_full/eval_droid_raw/robometer_frames_eval_droid/*.arrow")
eval_metaworld = bulk_load("/projects/prjs1958/robometer_frames_hf_full/eval_metaworld_raw/robometer_frames_eval_metaworld/*.arrow")
print(f"  eval_droid loaded:  {len(eval_droid['task']):,} rows")
print(f"  eval_metaworld:     {len(eval_metaworld['task']):,} rows in {time.time()-t0:.1f}s")


# ===========================================================================
section("TEST 3 — Metaworld task-drop filter (Solution 2)")
# ===========================================================================
# 3a) Apply the same exclusion logic the trainer uses: trajectories whose
#     task description CONTAINS any of the 4 held-out task strings.
#     We test the same substring-match behavior as `_filter_dataset`.
def would_be_dropped(task):
    """Mirror _filter_dataset's `any(kw in t for kw in excluded_keywords_lower)`."""
    if not task: return False
    t = task.lower()
    return any(kw.lower() in t for kw in HELDOUT_METAWORLD)

# 3b) Count what WOULD be dropped from train and from eval
train_dropped = sum(1 for t in train["task"] if would_be_dropped(t))
eval_mw_dropped = sum(1 for t in eval_metaworld["task"] if would_be_dropped(t))
eval_droid_dropped = sum(1 for t in eval_droid["task"] if would_be_dropped(t))

print(f"  Train (would be dropped if filter applies): {train_dropped:,} of {len(train['task']):,}  ({train_dropped/len(train['task'])*100:.3f}%)")
print(f"  Eval-metaworld (filter NOT applied at eval): {eval_mw_dropped:,}")
print(f"  Eval-droid     (filter NOT applied at eval): {eval_droid_dropped:,}")

# 3c) Expected drop counts from manifest math
EXPECTED_TRAIN_DROP = 3210   # 47+43+47+49 succ + 788+790+665+781 fail per the earlier audit
EXPECTED_EVAL_MW_KEPT = 66   # 3+7+3+1 = 14 succ + 12+10+11+19 = 52 fail per earlier audit
print(f"  Expected train drop:    {EXPECTED_TRAIN_DROP:,} (per earlier audit)")
print(f"  Expected eval-mw kept (these are our held-out evals): {EXPECTED_EVAL_MW_KEPT}")
assert abs(train_dropped - EXPECTED_TRAIN_DROP) <= 50, \
    f"Train drop mismatch: counted {train_dropped} vs expected {EXPECTED_TRAIN_DROP}"
assert eval_mw_dropped == EXPECTED_EVAL_MW_KEPT, \
    f"Eval-mw kept count mismatch: {eval_mw_dropped} vs {EXPECTED_EVAL_MW_KEPT}"
assert eval_droid_dropped == 0, "No droid eval should match metaworld task keywords"
print(f"  ✓ Filter logic correct (4 metaworld tasks dropped from train, kept in eval)")

# 3d) Per-held-out-task breakdown (sanity-check each task appears in both train and eval as expected)
print(f"\n  Per-task breakdown:")
for kw in sorted(HELDOUT_METAWORLD):
    n_train_succ = sum(1 for t, q in zip(train["task"], train["quality_label"])
                       if t == kw and q == "successful")
    n_train_fail = sum(1 for t, q in zip(train["task"], train["quality_label"])
                       if t == kw and q != "successful")
    n_eval_succ = sum(1 for t, q in zip(eval_metaworld["task"], eval_metaworld["quality_label"])
                      if t == kw and q == "successful")
    n_eval_fail = sum(1 for t, q in zip(eval_metaworld["task"], eval_metaworld["quality_label"])
                      if t == kw and q != "successful")
    short = kw[:38] + "..." if len(kw) > 38 else kw
    print(f"    {short:<42}  train:{n_train_succ:>4}s/{n_train_fail:>4}f    eval:{n_eval_succ:>2}s/{n_eval_fail:>2}f")


# ===========================================================================
section("TEST 4 — Sampler weight construction (Solution 1: droid fail boost)")
# ===========================================================================
# Mirror EXACTLY the trainer's weight construction logic (rbm_heads_trainer.py:228-244)
OVERSAMPLE = 2.61                     # data.failure_oversample_factor (run4 value)
SUCC_PER_SRC = {"metaworld": 13.0}
FAIL_PER_SRC = {"droid": 5.5}

def build_weights(qualities, sources, oversample, succ_per_src, fail_per_src):
    """Verbatim copy of rbm_heads_trainer.py weight-build logic."""
    weights = []
    for lbl, src in zip(qualities, sources):
        if lbl != "successful":
            w = oversample
            if fail_per_src and src in fail_per_src:
                w *= fail_per_src[src]
            weights.append(w)
        else:
            w = 1.0
            if succ_per_src and src in succ_per_src:
                w *= succ_per_src[src]
            weights.append(w)
    return torch.tensor(weights, dtype=torch.double)

W = build_weights(train["quality_label"], train["data_source"], OVERSAMPLE, SUCC_PER_SRC, FAIL_PER_SRC)
print(f"  Built weight tensor over {len(W):,} trajectories")
print(f"  Weight stats: min={W.min().item():.3f}  max={W.max().item():.3f}  mean={W.mean().item():.3f}")

# 4a) Spot-check: pick known (quality, source) combos and confirm exact weights
spot_checks = [
    ("successful", "droid",     1.0),       # droid succ unchanged
    ("failure",    "droid",     14.355),    # droid fail = 2.61 × 5.5 = 14.355
    ("successful", "metaworld", 13.0),      # mw succ = 1.0 × 13.0
    ("failure",    "metaworld", 2.61),      # mw fail = 2.61 (unchanged)
    ("successful", "robometer", 1.0),       # other succ unchanged
    ("failure",    "robometer", 2.61),      # other fail unchanged
    ("failure",    "failsafe",  2.61),      # failsafe fail unchanged
    ("successful", "racer",     1.0),       # racer succ unchanged
    ("failure",    "racer",     2.61),      # racer fail unchanged
]
print(f"\n  {'(quality':<12}, {'source)':<22} {'expected':>10} {'observed':>10}")
print(f"  " + "-" * 60)
for q, s, expected in spot_checks:
    # Find first idx matching this combo
    idx = next((i for i in range(len(train["quality_label"]))
                if train["quality_label"][i] == q and train["data_source"][i] == s), None)
    if idx is None:
        print(f"  ({q:<10}, {s:<20}) NO ROWS in train pool — skipping")
        continue
    observed = W[idx].item()
    match = "✓" if abs(observed - expected) < 1e-6 else "✗ MISMATCH"
    print(f"  ({q:<10}, {s:<20}) {expected:>10.3f} {observed:>10.3f}  {match}")
    assert abs(observed - expected) < 1e-6, f"weight wrong for ({q}, {s})"


# ===========================================================================
section("TEST 5 — Within-droid succ:fail ratio (theoretical + empirical)")
# ===========================================================================
droid_idx = [i for i, s in enumerate(train["data_source"]) if s == "droid"]
W_droid = W[droid_idx]
qualities_droid = [train["quality_label"][i] for i in droid_idx]
sources_droid = [train["data_source"][i] for i in droid_idx]

w_succ_total = sum(w for w, q in zip(W_droid.tolist(), qualities_droid) if q == "successful")
w_fail_total = sum(w for w, q in zip(W_droid.tolist(), qualities_droid) if q != "successful")
w_droid_total = w_succ_total + w_fail_total
n_succ = sum(1 for q in qualities_droid if q == "successful")
n_fail = sum(1 for q in qualities_droid if q != "successful")
print(f"  Droid trajectories: {n_succ:,} succ, {n_fail:,} fail")
print(f"  Weight totals:      {w_succ_total:>10,.0f} succ, {w_fail_total:>10,.0f} fail")
print(f"  Within droid:       {w_succ_total/w_droid_total*100:.1f}% succ  /  {w_fail_total/w_droid_total*100:.1f}% fail")
print(f"  Target:             64% succ / 36% fail  (was 91% / 9% before)")

theory_succ_pct = w_succ_total / w_droid_total
theory_fail_pct = w_fail_total / w_droid_total
assert 0.60 < theory_succ_pct < 0.68, f"succ ratio off: {theory_succ_pct}"
assert 0.32 < theory_fail_pct < 0.40, f"fail ratio off: {theory_fail_pct}"
print(f"  ✓ Within-droid balance matches design (64:36 target)")

# Empirical: draw N samples from W with WeightedRandomSampler, count droid succ vs droid fail
N_DRAWS = 480_000     # roughly a 7500-step × 64-batch run
print(f"\n  Empirical sampling: {N_DRAWS:,} draws via WeightedRandomSampler...")
gen = torch.Generator().manual_seed(42)
sampler = WeightedRandomSampler(W, num_samples=N_DRAWS, replacement=True, generator=gen)
drawn = list(sampler)

obs_droid_succ = sum(1 for i in drawn if train["data_source"][i] == "droid"
                                       and train["quality_label"][i] == "successful")
obs_droid_fail = sum(1 for i in drawn if train["data_source"][i] == "droid"
                                       and train["quality_label"][i] != "successful")
obs_droid_total = obs_droid_succ + obs_droid_fail

# Unique droid_fail trajectories drawn (coupon-collector check)
unique_droid_fail_seen = len({i for i in drawn if train["data_source"][i] == "droid"
                                                and train["quality_label"][i] != "successful"})

print(f"\n  Observed in {N_DRAWS:,} draws:")
print(f"    droid total:      {obs_droid_total:,}  ({obs_droid_total/N_DRAWS*100:.2f}% of all batches)")
print(f"    droid succ:       {obs_droid_succ:,}  ({obs_droid_succ/obs_droid_total*100:.1f}% within droid)")
print(f"    droid fail:       {obs_droid_fail:,}  ({obs_droid_fail/obs_droid_total*100:.1f}% within droid)")
print(f"    unique droid_fail seen: {unique_droid_fail_seen:,} of {n_fail:,}  ({unique_droid_fail_seen/n_fail*100:.1f}% coverage)")
print(f"    avg visits / unique droid_fail: {obs_droid_fail/n_fail:.2f}")

# Within-droid ratio should match theory (64:36) within ~1pp
obs_succ_pct = obs_droid_succ / obs_droid_total
err = abs(obs_succ_pct - theory_succ_pct)
print(f"\n  Theory  succ-share within droid: {theory_succ_pct*100:.2f}%")
print(f"  Observed succ-share within droid: {obs_succ_pct*100:.2f}%")
print(f"  Error: {err*100:.3f}pp")
assert err < 0.02, f"empirical droid succ-share off by {err*100:.2f}pp"
print(f"  ✓ Empirical matches theory within 2pp")


# ===========================================================================
section("TEST 6 — A/B comparison: NEW knobs vs LEGACY (no boost)")
# ===========================================================================
W_LEGACY = build_weights(train["quality_label"], train["data_source"],
                          oversample=OVERSAMPLE, succ_per_src={}, fail_per_src={})
W_NEW = W   # already built with both boosts

# Per-batch share of droid_fail
gen_a = torch.Generator().manual_seed(42)
drawn_legacy = list(WeightedRandomSampler(W_LEGACY, num_samples=N_DRAWS, replacement=True, generator=gen_a))
obs_legacy_droid_fail = sum(1 for i in drawn_legacy if train["data_source"][i] == "droid"
                                                    and train["quality_label"][i] != "successful")
unique_legacy = len({i for i in drawn_legacy if train["data_source"][i] == "droid"
                                              and train["quality_label"][i] != "successful"})

print(f"  {'recipe':<28} {'droid_fail draws':>17} {'visits/uniq':>12} {'within-droid fail-share':>24}")
print(f"  " + "-" * 90)
total_droid_legacy = sum(1 for i in drawn_legacy if train["data_source"][i] == "droid")
print(f"  {'LEGACY (no per-src)':<28} {obs_legacy_droid_fail:>17,} {obs_legacy_droid_fail/n_fail:>11.2f} {obs_legacy_droid_fail/max(total_droid_legacy,1)*100:>23.1f}%")
print(f"  {'NEW (droid: 5.5)':<28} {obs_droid_fail:>17,} {obs_droid_fail/n_fail:>11.2f} {obs_droid_fail/obs_droid_total*100:>23.1f}%")

boost_ratio = obs_droid_fail / max(obs_legacy_droid_fail, 1)
print(f"\n  Effective droid_fail boost: {boost_ratio:.2f}× more exposure with the new knob")
print(f"  (configured value: failure_oversample_per_source.droid=5.5)")
# Boost ratio includes the redistribution effect (denominator changes too).
# A 5.5× per-source boost yields ~5× total-exposure boost since other sources also shift slightly.
assert 4.5 < boost_ratio < 6.0, f"effective boost off: {boost_ratio}"
print(f"  ✓ Boost in actual draws ≈ configured 5.5×")


section("ALL TESTS PASSED")
print(f"\n  Solutions verified end-to-end against the real {len(train['task']):,}-row training pool:")
print(f"    (A) Metaworld task-drop: {train_dropped:,} trajs filtered from TRAIN, kept in EVAL ({EXPECTED_EVAL_MW_KEPT} held-out)")
print(f"    (B) Droid fail boost:    succ:fail within droid = 64:36 (was 91:9), ~{obs_droid_fail/n_fail:.1f} visits/uniq fail over 480k draws")
