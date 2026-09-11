"""Fix the droid shape leak in the 3p5x train cache.

Other-agent finding: every droid failure in train is 240×400, every droid
success is 240×426 (different upstream archives, different native aspects).
The model learns "shape → label" as a free shortcut (16×24 vs 16×28 patch
grids) — explains why droid AUC plateaus at ~0.5 every run.

Fix: for every droid success NPZ that's (16, 240, 426, 3), center-crop the
width to 400 → (16, 240, 400, 3). Update frames_shape in the Arrow dataset
in lockstep so downstream code sees consistent shape.

Eval set is already shape-consistent (both classes 240×400) so no eval
changes needed.
"""
import os
import sys
import shutil
import numpy as np
from datasets import load_from_disk, Sequence, Value

TRAIN_ROOT = "/projects/prjs1958/robometer_frames_hf_3p5x/_projects_prjs1958_robometer_frames_hf_3p5x_train_raw_robometer_frames_train"
TARGET_W = 400


def center_crop_w(arr: np.ndarray, target_w: int) -> np.ndarray:
    """Center-crop width of (T, H, W, 3) to target_w."""
    if arr.ndim != 4:
        raise ValueError(f"unexpected ndim {arr.ndim}")
    w = arr.shape[2]
    if w == target_w:
        return arr
    if w < target_w:
        raise ValueError(f"can't widen {w} → {target_w}")
    start = (w - target_w) // 2
    return arr[:, :, start:start + target_w, :]


def main():
    ds_path = os.path.join(TRAIN_ROOT, "processed_dataset")
    print(f"loading {ds_path}")
    ds = load_from_disk(ds_path)
    print(f"  {len(ds)} rows")

    droid_426_idxs = []
    for i, (src, sh) in enumerate(zip(ds["data_source"], ds["frames_shape"])):
        if src and "droid" in src and tuple(sh) == (16, 240, 426, 3):
            droid_426_idxs.append(i)
    print(f"  {len(droid_426_idxs)} droid samples at (16, 240, 426, 3) — these will be center-cropped to 400 wide")

    # Step 1: rewrite each NPZ in place
    n_done = 0
    for i in droid_426_idxs:
        r = ds[i]
        npz_path = r["frames"]
        d = np.load(npz_path)
        arr = d["frames"]
        if arr.shape != (16, 240, 426, 3):
            print(f"  skip {npz_path}: shape={arr.shape} (already changed?)")
            continue
        cropped = center_crop_w(arr, TARGET_W)
        np.savez_compressed(
            npz_path,
            frames=cropped,
            shape=cropped.shape,
            num_frames=cropped.shape[0],
        )
        n_done += 1
        if n_done % 200 == 0 or n_done == len(droid_426_idxs):
            print(f"    rewrote {n_done}/{len(droid_426_idxs)} NPZs")

    # Step 2: update frames_shape in Arrow dataset (so collator's filter passes)
    print("  updating frames_shape in Arrow dataset...")
    target_set = set(droid_426_idxs)

    def fix_shape(row, idx):
        if idx in target_set:
            return {"frames_shape": [16, 240, 400, 3]}
        return {}

    ds_fixed = ds.map(fix_shape, with_indices=True, num_proc=1, desc="patching frames_shape")

    # save back (atomic via tmp dir + rename)
    tmp_dir = ds_path + "_tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    ds_fixed.save_to_disk(tmp_dir)
    shutil.rmtree(ds_path)
    shutil.move(tmp_dir, ds_path)

    print(f"  done. rewrote {n_done} NPZs and patched Arrow shape column.")


if __name__ == "__main__":
    main()
