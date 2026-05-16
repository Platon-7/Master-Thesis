"""Dump one droid + one robometer training example as image grids.

Loads the v2 HF Arrow cache (the one the bake-off trained on), picks one
row per source, and writes a side-by-side PNG grid of all 16 keyframes
exactly as the model receives them after preprocessing.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from datasets import Dataset, load_from_disk

CACHE = "/projects/prjs1958/robometer_frames_hf_v2/_projects_prjs1958_robometer_frames_hf_v2_train_raw_robometer_frames_train/processed_dataset"
OUT_DIR = Path("/gpfs/home3/pkarageorgis1/Master-Thesis/rbm_examples")
OUT_DIR.mkdir(exist_ok=True)


def grid(frames: np.ndarray, cols: int = 4) -> Image.Image:
    """frames: (T, H, W, 3) uint8 → PIL grid."""
    T, H, W, _ = frames.shape
    rows = (T + cols - 1) // cols
    canvas = Image.new("RGB", (cols * W, rows * H), (0, 0, 0))
    for i in range(T):
        r, c = divmod(i, cols)
        canvas.paste(Image.fromarray(frames[i]), (c * W, r * H))
    return canvas


def find_first(ds: Dataset, source: str, label: str | None = None) -> int:
    for i in range(len(ds)):
        row = ds[i]
        if row.get("data_source") != source:
            continue
        if label is not None and row.get("quality_label") != label:
            continue
        return i
    raise LookupError(f"No row with source={source} label={label}")


def dump_one(ds: Dataset, idx: int, tag: str):
    row = ds[idx]
    npz_path = row["frames"]
    npz = np.load(npz_path)
    # NPZ archives use a single key like 'frames' or 'arr_0'
    key = npz.files[0]
    frames = npz[key]
    print(f"\n[{tag}] idx={idx}")
    print(f"  id            : {row.get('id')}")
    print(f"  source        : {row.get('data_source')}")
    print(f"  task          : {(row.get('task') or '').strip()[:80]}")
    print(f"  label         : {row.get('quality_label')}")
    print(f"  frames_path   : {row.get('frames_path')}")
    print(f"  frames shape  : {frames.shape}  dtype={frames.dtype}")
    print(f"  frames range  : [{frames.min()}, {frames.max()}]  mean={frames.mean():.1f}")
    img = grid(frames, cols=4)
    out = OUT_DIR / f"{tag}_grid.png"
    img.save(out)
    print(f"  → {out}  ({img.size})")
    Image.fromarray(frames[0]).save(OUT_DIR / f"{tag}_frame00.png")


print(f"loading {CACHE}")
ds = load_from_disk(CACHE)
print(f"  rows: {len(ds)}  cols: {ds.column_names}")
print(f"  source counts:")
src_counter = {}
for s in ds["data_source"]:
    src_counter[s] = src_counter.get(s, 0) + 1
for s, c in sorted(src_counter.items()):
    print(f"    {s}: {c}")

# HF cache renames sources: pairs_unified `droid` → HF `robometer_frames_droid`,
# `robometer` → HF `robometer_frames_<family>` (racer/soar/roboarena/etc.).
dump_one(ds, find_first(ds, "robometer_frames_droid", "failure"),    "droid_failure")
dump_one(ds, find_first(ds, "robometer_frames_droid", "successful"), "droid_success")
dump_one(ds, find_first(ds, "robometer_frames_racer", "failure"),    "robometer_racer_failure")
dump_one(ds, find_first(ds, "robometer_frames_oxe_bridge_v2", "successful"), "robometer_orphan_bridge_success")
