#!/usr/bin/env python3
"""Drop frame_labels on metaworld success trajectories across step 1 and step 2
HF datasets. Vectorized PyArrow column rewrite — no per-row Python iteration,
no frame re-decoding, no NPZ regeneration.

Background
----------
Robometer/robometer/data/samplers/base.py:1019-1084 routes trajectories with
non-None `frame_labels` through the failure-rubric path (target_progress is
capped at rubric[4] = 0.75). The success t/T path (target → 1.0 at terminal
frames) only fires when frame_labels is None.

MetaWorld/generate_dataset.py erroneously wrote frame_labels for successes
too. Result: metaworld successes' target_progress maxes at 0.75 instead of
1.0 → indistinguishable from failures at terminal frames → eval AUC
collapses.

Fix: set frame_labels = None for metaworld + successful rows. Failures
keep their (correct) 1-4 rubric labels.

Patch surface
-------------
- Step 1 (shared, at /projects/prjs1958/robometer_frames_hf_full/):
    train_raw/robometer_frames_train
    train_no_extras_raw/robometer_frames_train_no_extras
    eval_metaworld_raw/robometer_frames_eval_metaworld
- Step 2 (per-user, at /scratch-shared/$USER/robometer_frames_hf_full_step2/):
    robometer_frames_train/processed_dataset
    robometer_frames_train_no_extras/processed_dataset
    robometer_frames_eval_metaworld/processed_dataset

Training reads from step 2 (samplers/base.py:_load_preprocessed_cache), so
step 2 is the must-patch. Step 1 is patched too so step 2 rebuilds (if
anyone does that) start from correct labels.

For Leonardo (separate user): same script, his scratch path. He runs on his
side; we can't reach his /scratch-shared/lbarcellona/ from this account.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
from datasets import Dataset, load_from_disk


def patch_one_dataset(path: str, *, dry_run: bool) -> dict:
    """Vectorized column-replace: set frame_labels=None where data_source==metaworld AND quality_label==successful."""
    t0 = time.time()
    print(f"\n=== {path}")
    ds = load_from_disk(path)
    n = len(ds)
    print(f"  rows: {n:,}  (loaded in {time.time() - t0:.1f}s)")

    # Single Arrow table view — no Python row iteration
    table = ds.data.table

    if "data_source" not in table.column_names or "quality_label" not in table.column_names:
        print(f"  SKIP — required columns missing")
        return {"path": path, "patched": 0, "skipped": True}
    if "frame_labels" not in table.column_names:
        print(f"  SKIP — no frame_labels column")
        return {"path": path, "patched": 0, "skipped": True}

    mask = pc.and_(
        pc.equal(table.column("data_source"), "metaworld"),
        pc.equal(table.column("quality_label"), "successful"),
    )

    # Build new frame_labels column: keep existing where mask=False, null where mask=True.
    # Use a null array of the SAME logical type (List<int>) so the schema stays intact.
    fl_field = table.schema.field("frame_labels")
    null_col = pa.nulls(n, type=fl_field.type)
    new_fl = pc.if_else(mask, null_col, table.column("frame_labels"))

    n_to_patch = int(pc.sum(mask.cast(pa.int64())).as_py())
    print(f"  metaworld + successful rows to clear: {n_to_patch:,}")

    if dry_run:
        # Spot-check by sampling
        ds_mask = mask.to_numpy(zero_copy_only=False)
        hits = [i for i, m in enumerate(ds_mask) if m][:3]
        for i in hits:
            print(f"    sample row {i}: data_source={ds[i]['data_source']} quality={ds[i]['quality_label']} frame_labels={ds[i]['frame_labels']} → None")
        return {"path": path, "patched": n_to_patch, "dry_run": True}

    if n_to_patch == 0:
        print(f"  nothing to patch")
        return {"path": path, "patched": 0, "dry_run": False}

    # Replace column in the underlying table, build a fresh Dataset, save
    col_idx = table.column_names.index("frame_labels")
    new_table = table.set_column(col_idx, "frame_labels", new_fl)
    new_ds = Dataset(new_table)

    tmp_path = path + ".patched_tmp"
    bak_path = path + ".bak_pre_drop_metaworld_success_labels"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)

    t_save = time.time()
    new_ds.save_to_disk(tmp_path)
    print(f"  saved patched dataset to tmp in {time.time() - t_save:.1f}s")

    if os.path.exists(bak_path):
        print(f"  WARN: backup {bak_path} already exists — leaving it; original goes to {bak_path}_2")
        bak_path = bak_path + "_2"
    shutil.move(path, bak_path)
    shutil.move(tmp_path, path)
    print(f"  backup: {bak_path}")
    print(f"  in-place patched: {path}")
    print(f"  total wallclock: {time.time() - t0:.1f}s")
    return {"path": path, "patched": n_to_patch, "dry_run": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--step1-only", action="store_true", help="patch only the shared step1 datasets, skip step2")
    ap.add_argument("--step2-only", action="store_true", help="patch only the per-user step2 cache, skip step1")
    args = ap.parse_args()

    user = os.environ.get("USER", "UNKNOWN")
    print(f"Running as user: {user}")

    step1_paths = [
        "/projects/prjs1958/robometer_frames_hf_full/train_raw/robometer_frames_train",
        "/projects/prjs1958/robometer_frames_hf_full/train_no_extras_raw/robometer_frames_train_no_extras",
        "/projects/prjs1958/robometer_frames_hf_full/eval_metaworld_raw/robometer_frames_eval_metaworld",
    ]
    step2_root = f"/scratch-shared/{user}/robometer_frames_hf_full_step2"
    step2_paths = [
        f"{step2_root}/robometer_frames_train/processed_dataset",
        f"{step2_root}/robometer_frames_train_no_extras/processed_dataset",
        f"{step2_root}/robometer_frames_eval_metaworld/processed_dataset",
    ]

    paths: list[str] = []
    if not args.step2_only:
        paths.extend(step1_paths)
    if not args.step1_only:
        paths.extend(step2_paths)

    summary: list[dict] = []
    for p in paths:
        if not os.path.isdir(p):
            print(f"\n=== SKIP (missing): {p}")
            continue
        summary.append(patch_one_dataset(p, dry_run=args.dry_run))

    print("\n=== Summary ===")
    for s in summary:
        if s.get("skipped"):
            verb = "skipped (schema)"
        elif s.get("dry_run"):
            verb = "would clear"
        else:
            verb = "cleared"
        print(f"  {verb} {s.get('patched', 0):>5,} rows  in {s['path']}")


if __name__ == "__main__":
    main()
