#!/usr/bin/env python3
"""End-to-end test of the data.keep_orphan_task_failures override.

Tests:
  1. DataConfig has the new field, defaults to False, settable to True.
  2. base.py imports cleanly.
  3. The job script's yaml→hydra-override sed pipeline emits the flag for
     every one of the 6 run yamls.
  4. BaseDataset built with the flag OFF has 846,220 trajectories
     (the filter dropped 13,610 orphan-task failures, matching the live run4 log).
  5. BaseDataset built with the flag ON has 859,830 trajectories
     (the filter is bypassed, delta = exactly +13,610).
  6. At least one known orphan-task source (`roboarena`) gains trajectories
     when the flag is ON.

Run from any directory:
    /home/pkarageorgis1/.conda/envs/robometer_qwen35_gpu/bin/python3 \
        /gpfs/home3/pkarageorgis1/Master-Thesis/test_keep_orphan_task_failures.py
"""
from __future__ import annotations

import os
import sys
import subprocess
from collections import Counter
from pathlib import Path

# Resolve the Robometer package on PYTHONPATH the same way the production job does.
ROBOMETER_ROOT = "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer"
if ROBOMETER_ROOT not in sys.path:
    sys.path.insert(0, ROBOMETER_ROOT)

# Datasets live on the shared project filesystem (set by the job scripts).
os.environ.setdefault(
    "ROBOMETER_PROCESSED_DATASETS_PATH",
    "/scratch-shared/pkarageorgis1/robometer_frames_hf_full_step2",
)


def banner(s: str) -> None:
    print()
    print("=" * 80)
    print(s)
    print("=" * 80)


# ---------- 1. DataConfig field ----------
banner("TEST 1: DataConfig.keep_orphan_task_failures field exists, default False")
from robometer.configs.experiment_configs import DataConfig

c_default = DataConfig()
assert hasattr(c_default, "keep_orphan_task_failures"), "field missing on DataConfig"
assert c_default.keep_orphan_task_failures is False, (
    f"default should be False, got {c_default.keep_orphan_task_failures!r}"
)
print(f"  DataConfig().keep_orphan_task_failures = {c_default.keep_orphan_task_failures}  OK")

c_on = DataConfig(keep_orphan_task_failures=True)
assert c_on.keep_orphan_task_failures is True
print(f"  DataConfig(keep_orphan_task_failures=True).keep_orphan_task_failures = {c_on.keep_orphan_task_failures}  OK")


# ---------- 2. base.py imports cleanly ----------
banner("TEST 2: BaseDataset imports cleanly")
from robometer.data.datasets.base import BaseDataset  # noqa: F401
print("  import robometer.data.datasets.base.BaseDataset  OK")


# ---------- 3. job script yaml→override pipeline emits the flag ----------
banner("TEST 3: job-script sed pipeline emits ++data.keep_orphan_task_failures=true")
yamls = [
    "/gpfs/home3/pkarageorgis1/Master-Thesis/Qwen35-FT/configs/run4_icl_ours.yaml",
    "/gpfs/home3/pkarageorgis1/Master-Thesis/Qwen35-FT/configs/run5_noicl_ours.yaml",
    "/gpfs/home3/pkarageorgis1/Master-Thesis/Qwen35-FT/configs/run6_noicl_standard.yaml",
    "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer-FT/configs/run1_icl_ours.yaml",
    "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer-FT/configs/run2_noicl_ours.yaml",
    "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer-FT/configs/run3_noicl_standard.yaml",
]
flag_token = "++data.keep_orphan_task_failures=true"
for yp in yamls:
    pipeline = (
        f"grep -hv '^[[:space:]]*#' {yp} "
        f"| sed -e 's/[[:space:]]*#.*//' "
        f"| grep -v '^[[:space:]]*$' "
        f"| sed -e 's/: */=/' -e 's/^/++/'"
    )
    out = subprocess.check_output(pipeline, shell=True, executable="/bin/bash").decode()
    assert flag_token in out, f"missing {flag_token!r} in pipeline output for {yp}:\n{out}"
    print(f"  {Path(yp).parents[1].name}/{Path(yp).name}  -> {flag_token}  OK")


# ---------- 4 / 5. Real BaseDataset build with flag OFF then ON ----------
banner("TEST 4 + 5: BaseDataset trajectory count, flag OFF vs ON")
print("  loading dataset (this takes ~1 min)...")

# Build a DataConfig that matches what the job assembles from
# train_base.yaml + run4_icl_ours.yaml on the data axes that matter for filtering.
# We don't need ICL (skip pairs_index_path); we just need the loader to know
# which split to pull and that we're in progress-only training mode (which
# triggers the success-only escape-hatch logic at base.py:87-128).
base_cfg = dict(
    train_datasets=["robometer_frames_train"],
    eval_datasets=[],
    sample_type_ratio=[0, 1, 0],
    use_icl=False,
    icl_prob=0.0,
    pairs_index_path=None,
    icl_min_coverage=0.0,
    stratified_batch_balance=False,
    failure_oversample_factor=2.61,
    warmup_steps=0,
    failure_label_smoothing="linear",
)

cfg_off = DataConfig(keep_orphan_task_failures=False, **base_cfg)
print(f"  building BaseDataset with keep_orphan_task_failures=False ...")
ds_off = BaseDataset(cfg_off, is_evaluation=False)
n_off = len(ds_off)
print(f"  flag OFF: {n_off:,} trajectories")

cfg_on = DataConfig(keep_orphan_task_failures=True, **base_cfg)
print(f"  building BaseDataset with keep_orphan_task_failures=True ...")
ds_on = BaseDataset(cfg_on, is_evaluation=False)
n_on = len(ds_on)
print(f"  flag ON : {n_on:,} trajectories")

delta = n_on - n_off
print(f"  delta   : {delta:+,}")

EXPECTED_OFF = 846_220
EXPECTED_ON = 859_830
EXPECTED_DELTA = 13_610
assert n_off == EXPECTED_OFF, f"flag OFF expected {EXPECTED_OFF:,}, got {n_off:,}"
assert n_on == EXPECTED_ON, f"flag ON expected {EXPECTED_ON:,}, got {n_on:,}"
assert delta == EXPECTED_DELTA, f"delta expected +{EXPECTED_DELTA:,}, got {delta:+,}"
print(f"  counts match expected ({EXPECTED_OFF:,} / {EXPECTED_ON:,} / +{EXPECTED_DELTA:,})  OK")


# ---------- 6. Orphan-task source (roboarena) recovers under flag=ON ----------
banner("TEST 6: roboarena traj counts: flag OFF vs flag ON (expect +11,232)")
src_off = Counter(ds_off.dataset["data_source"])
src_on = Counter(ds_on.dataset["data_source"])
ra_off = src_off.get("roboarena", 0)
ra_on = src_on.get("roboarena", 0)
ra_delta = ra_on - ra_off
print(f"  roboarena OFF: {ra_off:,}")
print(f"  roboarena ON : {ra_on:,}")
print(f"  delta        : {ra_delta:+,}")
EXPECTED_ROBOARENA_DELTA = 11_232
assert ra_delta == EXPECTED_ROBOARENA_DELTA, (
    f"roboarena delta expected +{EXPECTED_ROBOARENA_DELTA:,}, got {ra_delta:+,}"
)

# And spot-check failsafe + metaworld too.
for source, expected_delta in [("failsafe", 1_232), ("metaworld", 600), ("droid", 353)]:
    d = src_on.get(source, 0) - src_off.get(source, 0)
    assert d == expected_delta, f"{source}: expected +{expected_delta:,}, got {d:+,}"
    print(f"  {source:12s} delta: {d:+,}  OK")


banner("ALL TESTS PASSED")
