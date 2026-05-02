#!/usr/bin/env python3
"""
Integrate Leonardo's Group B output into the canonical
robometer_frame_dataset format.

Differs from Group A in two ways:
  1. Score files are sharded — Leonardo ran each archive across 8 SLURM jobs,
     producing <archive>_scored_shard{1..8}.jsonl. We merge them first.
  2. Raw archives live in raw_archives/split/<archive>/<archive>.tar.part-*.
     We concatenate the parts into a single .tar in scratch before passing
     to the Group A integration logic.

Apart from these two pre-steps, processing reuses
integrate_group_a_to_frame_dataset.process_archive() unchanged.

Usage:
  python integrate_group_b_to_frame_dataset.py --archive jesbu1_soar_rfm_soar_rfm
  python integrate_group_b_to_frame_dataset.py --archive all
  python integrate_group_b_to_frame_dataset.py --archive all --keep-raw-tar
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import integrate_group_a_to_frame_dataset as iga

USER = os.environ.get("USER", "pkarageorgis1")

GROUP_B_FAMILIES = {
    "jesbu1_soar_rfm_soar_rfm": "soar",
    "jesbu1_roboarena_0825_rfm_roboarena": "roboarena",
    "jesbu1_roboarena_eval_debug_nowrist_roboarena_eval_debug_nowrist": "roboarena",
}

LEONARDO_B_OUT = Path(f"/scratch-shared/{USER}/leonardo_groupB/robometer_out")
RAW_SPLIT      = Path("/projects/prjs1958/robometer_full_dataset/raw_archives/split")
RAW_TMP        = Path(f"/scratch-shared/{USER}/leonardo_groupB/raw")


def merge_sharded_scores(archive: str) -> Path:
    """cat <archive>_scored_shard*.jsonl -> <archive>_scored.jsonl"""
    scores_dir = LEONARDO_B_OUT / "scores"
    out = scores_dir / f"{archive}_scored.jsonl"
    if out.exists():
        n = sum(1 for _ in open(out))
        print(f"  scores already merged: {out.name} ({n} rows)", flush=True)
        return out
    shards = sorted(scores_dir.glob(f"{archive}_scored_shard*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"No score shards for {archive}")
    print(f"  merging {len(shards)} score shards -> {out.name}", flush=True)
    with open(out, "wb") as outf:
        for s in shards:
            with open(s, "rb") as inf:
                shutil.copyfileobj(inf, outf, length=4 * 1024 * 1024)
    n = sum(1 for _ in open(out))
    print(f"    merged: {n} episodes", flush=True)
    return out


def concat_split_parts(archive: str) -> Path:
    """cat <archive>.tar.part-* -> RAW_TMP/<archive>.tar"""
    out = RAW_TMP / f"{archive}.tar"
    out.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = RAW_SPLIT / archive
    parts = sorted(parts_dir.glob(f"{archive}.tar.part-*"))
    if not parts:
        raise FileNotFoundError(f"No split parts for {archive}")
    expected_size = sum(p.stat().st_size for p in parts)
    if out.exists() and out.stat().st_size == expected_size:
        print(f"  raw tar already concatenated: {out} ({expected_size/1e9:.1f} GB)", flush=True)
        return out
    print(f"  concatenating {len(parts)} parts ({expected_size/1e9:.1f} GB) -> {out}", flush=True)
    t0 = time.time()
    # subprocess.cat is faster than Python copyfileobj for many GB
    with open(out, "wb") as outf:
        subprocess.run(["cat"] + [str(p) for p in parts], stdout=outf, check=True)
    elapsed = time.time() - t0
    print(f"    concatenated in {elapsed:.0f}s ({expected_size/elapsed/1e6:.0f} MB/s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="all",
                    choices=list(GROUP_B_FAMILIES.keys()) + ["all"])
    ap.add_argument("--keep-raw-tar", action="store_true",
                    help="Don't delete concatenated raw tars after integration")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    archives = list(GROUP_B_FAMILIES.keys()) if args.archive == "all" else [args.archive]

    # Patch the iga module to point at Group B inputs
    iga.LEONARDO_OUT = LEONARDO_B_OUT
    iga.RAW_SINGLE   = RAW_TMP
    iga.FAMILY.update(GROUP_B_FAMILIES)

    all_stats = []
    t_overall = time.time()
    for arch in archives:
        print(f"\n=== {arch} ===", flush=True)
        merge_sharded_scores(arch)
        concat_split_parts(arch)
        t = time.time()
        stats = iga.process_archive(arch, dry_run=args.dry_run)
        stats["elapsed_s"] = round(time.time() - t, 1)
        all_stats.append(stats)
        if not args.keep_raw_tar and not args.dry_run:
            tar = RAW_TMP / f"{arch}.tar"
            if tar.exists():
                tar.unlink()
                print(f"  deleted raw tar {tar}", flush=True)

    print("\n=== SUMMARY ===")
    print(f"{'archive':70s}  {'exp':>6s}  {'wrote':>6s}  {'viol':>5s}  {'miss':>5s}  {'sec':>7s}")
    for s in all_stats:
        print(f"{s['archive']:70s}  {s['expected']:6d}  {s['written']:6d}  "
              f"{s['violations']:5d}  {s['missing_npz']:5d}  {s['elapsed_s']:7.1f}")
    print(f"total wall: {round(time.time() - t_overall, 1)}s")


if __name__ == "__main__":
    main()
