#!/usr/bin/env python3
"""Build a 3.5x-success-skewed train split for the next LoRA bake-off.

Derives from the existing _unified splits dir without rebuilding eval:
  * keeps the 4 eval_*.jsonl + pairs_index_eval_*.jsonl byte-identical (copied)
  * subsamples 4,500 of the existing 9k failure pool (preserves source mix)
  * keeps the existing 9k successes, adds 6,750 new orphan successes from
    non-oxe_droid archives (cap on oxe_droid kept at 4,500)
  * resulting train pool: 4,500 fail + 15,750 succ = 20,250 rows (3.5x ratio)

Produces:
  /scratch-shared/pkarageorgis1/robometer_frames_splits_3p5x/
    train.jsonl                       (20,250 episode_ids)
    pairs_index_train.jsonl           (matching pair rows)
    eval_*.jsonl + pairs_index_eval_*.jsonl  (copied verbatim)
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

SRC_DIR = Path("/scratch-shared/pkarageorgis1/robometer_frames_splits_unified")
DST_DIR = Path("/scratch-shared/pkarageorgis1/robometer_frames_splits_3p5x")
PAIRS_UNIFIED = Path("/projects/prjs1958/robometer_frame_dataset/pairs_unified.jsonl")
SEED = 42

ORPHAN_SOURCE = "robometer_orphan_success"
OXE_DROID_ARCHIVE = "jesbu1_oxe_rfm_oxe_droid"

N_FAIL_TARGET = 4000
N_NEW_SUCC = 5000  # added on top of the existing 9k → 14k succ total → 3.5x ratio at 18k total


def stream_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
            n += 1
    return n


def main() -> int:
    rng = random.Random(SEED)
    DST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[init] dst: {DST_DIR}")

    # Eval files: copy verbatim. These define the eval scope and the excluded set.
    eval_eids: set[str] = set()
    for name in ("eval_droid", "eval_robometer", "eval_metaworld", "eval_failsafe"):
        for fname in (f"{name}.jsonl", f"pairs_index_{name}.jsonl"):
            src = SRC_DIR / fname
            dst = DST_DIR / fname
            shutil.copyfile(src, dst)
            print(f"  copied {fname}")
        for r in stream_jsonl(SRC_DIR / f"{name}.jsonl"):
            eval_eids.add(r["episode_id"])
    print(f"[eval] copied 4 splits + 4 pair indices, {len(eval_eids):,} unique eval eids")

    # Existing train: split into successes/failures via pairs_index_train.jsonl labels.
    src_train_pairs = SRC_DIR / "pairs_index_train.jsonl"
    existing_fail_rows: list[dict] = []
    existing_succ_rows: list[dict] = []
    for r in stream_jsonl(src_train_pairs):
        (existing_fail_rows if r.get("label") == "failure" else existing_succ_rows).append(r)
    print(f"[existing-train] {len(existing_fail_rows):,} fail, {len(existing_succ_rows):,} succ")

    # Sample 4,500 from existing failures uniformly (preserves source mix in expectation).
    train_fail = rng.sample(existing_fail_rows, N_FAIL_TARGET)
    src_count = Counter(r.get("source") for r in train_fail)
    print(f"[train-fail] sampled {len(train_fail):,}; source mix: {dict(src_count)}")

    # Successes: keep the existing 9k as-is, then add N_NEW_SUCC from orphan
    # archives that are NOT oxe_droid AND NOT already used.
    used_eids: set[str] = set(eval_eids)
    used_eids.update(r["episode_id"] for r in existing_succ_rows)
    used_eids.update(r["episode_id"] for r in existing_fail_rows)  # exclude fail-pool entirely

    # Build the augmentation candidate pool: orphan_success rows from non-oxe_droid
    # archives, with non-null partner_episode_id (so ICL pair lookup works), excluding
    # anything already in train/eval.
    new_succ_pool: list[dict] = []
    n_total = 0
    n_dropped_no_partner = 0
    for r in stream_jsonl(PAIRS_UNIFIED):
        n_total += 1
        if r.get("source") != ORPHAN_SOURCE:
            continue
        if r.get("archive") == OXE_DROID_ARCHIVE:
            continue
        if r.get("episode_id") in used_eids:
            continue
        if not r.get("partner_episode_id"):
            n_dropped_no_partner += 1
            continue
        new_succ_pool.append(r)
    print(f"[scan] pairs_unified scanned ({n_total:,}); "
          f"non-oxe-droid orphan augmentation pool size = {len(new_succ_pool):,} "
          f"({n_dropped_no_partner:,} dropped for null partner)")

    if len(new_succ_pool) < N_NEW_SUCC:
        print(f"FATAL: pool ({len(new_succ_pool):,}) < target ({N_NEW_SUCC:,})", file=sys.stderr)
        return 2

    new_succ = rng.sample(new_succ_pool, N_NEW_SUCC)
    print(f"[train-succ-new] sampled {len(new_succ):,} new successes")

    train_succ = existing_succ_rows + new_succ
    print(f"[train-succ] total succ = {len(train_succ):,} "
          f"(existing {len(existing_succ_rows):,} + new {len(new_succ):,})")

    # Sanity: archive distribution on the success side.
    arch_counts = Counter(r.get("archive") for r in train_succ)
    oxe_count = arch_counts.get(OXE_DROID_ARCHIVE, 0)
    other_count = sum(v for k, v in arch_counts.items() if k != OXE_DROID_ARCHIVE)
    print(f"[train-succ] oxe_droid={oxe_count:,}, other_archives={other_count:,} "
          f"({len(arch_counts)-1} distinct other archives)")

    # Stitch + shuffle for SGD-friendliness.
    train_rows = train_fail + train_succ
    rng.shuffle(train_rows)
    print(f"[train] total = {len(train_rows):,} "
          f"(target 18,000; ratio = {len(train_succ)/len(train_fail):.2f}x)")

    # Write train.jsonl (episode-ids only) and pairs_index_train.jsonl (full rows).
    n1 = write_jsonl(DST_DIR / "train.jsonl",
                     ({"episode_id": r["episode_id"]} for r in train_rows))
    n2 = write_jsonl(DST_DIR / "pairs_index_train.jsonl", train_rows)
    print(f"[write] train.jsonl={n1:,}  pairs_index_train.jsonl={n2:,}")

    # Quick post-hoc sanity: zero overlap with eval.
    train_eids = {r["episode_id"] for r in train_rows}
    overlap = train_eids & eval_eids
    if overlap:
        print(f"FATAL: {len(overlap)} train ids overlap eval", file=sys.stderr)
        return 3
    print(f"[sanity] train ∩ eval = {len(overlap)} (must be 0)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
