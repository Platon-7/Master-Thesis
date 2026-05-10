"""Concatenate per-shard HF Arrow datasets produced by the preprocess array
job into a single _raw dataset that downstream tools (preprocess_datasets,
samplers) consume as one dataset.

After this:
    <hf_out_dir>/<split>_raw/
        robometer_frames_<split>/    ← merged HF Arrow Dataset
        frames/                      ← collected NPZ files (symlinks)
        ...

Usage:
    python concat_split_shards.py \
        --hf-out-dir /projects/prjs1958/robometer_frames_hf_full \
        --split train \
        --n-shards 16
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from datasets import concatenate_datasets, load_from_disk


def find_part_arrow_dirs(hf_out_dir: Path, split: str, n_shards: int) -> list[Path]:
    """Locate the Arrow Dataset directory inside each <split>_part<NN>_raw."""
    parts = []
    for i in range(n_shards):
        raw_root = hf_out_dir / f"{split}_part{i:02d}_raw"
        if not raw_root.is_dir():
            raise FileNotFoundError(f"missing shard output: {raw_root}")
        candidates = [
            d for d in raw_root.iterdir()
            if d.is_dir() and d.name.startswith("robometer_frames_")
        ]
        if not candidates:
            raise FileNotFoundError(f"no robometer_frames_* subdir under {raw_root}")
        parts.append(candidates[0])
    return parts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-out-dir", type=Path, required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    args = ap.parse_args()

    parts = find_part_arrow_dirs(args.hf_out_dir, args.split, args.n_shards)
    print(f"merging {len(parts)} shards:")
    for p in parts:
        print(f"  {p}")

    datasets = [load_from_disk(str(p)) for p in parts]
    total = sum(len(d) for d in datasets)
    print(f"  total rows across shards: {total:,}")

    merged = concatenate_datasets(datasets)
    print(f"  merged length: {len(merged):,}")

    # Rewrite the `frames` column from relative ("robometer_frames_train_part00/
    # batch_0000/trajectory_0007.mp4") to absolute ("<hf_out_dir>/train_part00_raw/
    # robometer_frames_train_part00/batch_0000/trajectory_0007.mp4"). Necessary
    # because step 2 (preprocess_datasets) resolves relative paths against the
    # merged Arrow's location (<hf_out_dir>/train_raw/) but the actual MP4 files
    # live under <hf_out_dir>/train_part<NN>_raw/. Absolute paths sidestep the
    # filesystem-topology assumption and work for any user reading from this
    # shared cache (e.g. Leonardo running step 2 with his own cache_dir on
    # /scratch-shared).
    import re
    _PART_RE = re.compile(r"^(robometer_frames_train_part\d+)/")

    def _absolutize_frames(row):
        rel = row["frames"]
        m = _PART_RE.match(rel or "")
        if not m:
            return row  # eval/other splits don't use this concat path
        part_dirname = m.group(1)            # e.g. "robometer_frames_train_part00"
        part_idx = part_dirname.rsplit("part", 1)[-1]  # "00"
        absolute = str(args.hf_out_dir / f"train_part{part_idx}_raw" / rel)
        row["frames"] = absolute
        return row

    print("  rewriting `frames` column to absolute paths …")
    merged = merged.map(_absolutize_frames, desc="absolutize frames")
    sample = merged[0]["frames"]
    print(f"  sample frames[0]: {sample[:120]}{'...' if len(sample) > 120 else ''}")

    # Write the merged dataset where downstream tools expect it
    merged_root = args.hf_out_dir / f"{args.split}_raw"
    merged_root.mkdir(parents=True, exist_ok=True)
    merged_target = merged_root / f"robometer_frames_{args.split}"
    if merged_target.exists():
        print(f"  removing existing {merged_target}")
        shutil.rmtree(merged_target)
    merged.save_to_disk(str(merged_target))
    print(f"  saved → {merged_target}")

    # The `frames/` subdir under each <split>_part<NN>_raw holds the actual
    # NPZ files referenced by the rows. The HF rows store absolute paths to
    # those NPZs, so we don't need to relocate them — they stay where they
    # are. The merged dataset just points at them.
    print()
    print("note: per-shard frames/ directories remain in place; merged Arrow rows")
    print("reference them by absolute path. Don't move or delete the part<NN>_raw")
    print("subdirs unless you also move the frames they contain.")


if __name__ == "__main__":
    main()
