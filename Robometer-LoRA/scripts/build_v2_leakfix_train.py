"""Build a leak-fixed clone of the v2 train cache without touching v2 itself.

For each row in v2 train Arrow:
  - if it's a droid sample at (16, 240, 426, 3): write a new center-cropped NPZ
    at v2_leakfix path (16, 240, 400, 3) and update Arrow `frames` to that path
  - otherwise: leave Arrow `frames` pointing at the original v2 NPZ (no copy,
    no symlink — just an absolute path reference)

Eval splits don't change so we just symlink them at the new cache root, so
the trainer's `_get_available_datasets` still finds them under the new
ROBOMETER_PROCESSED_DATASETS_PATH.
"""
import os
import shutil
import json
import numpy as np
from datasets import load_from_disk

V2_ROOT = "/projects/prjs1958/robometer_frames_hf_v2"
V2_TRAIN = f"{V2_ROOT}/_projects_prjs1958_robometer_frames_hf_v2_train_raw_robometer_frames_train"
NEW_ROOT = "/projects/prjs1958/robometer_frames_hf_v2_leakfix"
NEW_TRAIN = f"{NEW_ROOT}/_projects_prjs1958_robometer_frames_hf_v2_leakfix_train_raw_robometer_frames_train"
TARGET_W = 400


def main():
    os.makedirs(NEW_ROOT, exist_ok=True)

    # --- 1. Symlink eval splits + path-encoded eval dirs from v2 to new root ---
    print(f"[clone] symlinking eval splits + shortname symlinks from v2")
    for entry in os.listdir(V2_ROOT):
        if entry.startswith("eval_") or entry.startswith("test_") or entry.startswith("warmup_") or entry.startswith("_projects_prjs1958_robometer_frames_hf_v2_eval_") or entry.startswith("_projects_prjs1958_robometer_frames_hf_v2_test_") or entry.startswith("_projects_prjs1958_robometer_frames_hf_v2_warmup_"):
            src = f"{V2_ROOT}/{entry}"
            dst = f"{NEW_ROOT}/{entry}"
            if os.path.exists(dst) or os.path.islink(dst):
                continue
            os.symlink(src, dst)
    # shortname symlinks (robometer_frames_<split>) for eval/test/warmup → derive new ones
    # The trainer's discovery uses os.path.join(cache_dir, dataset_name) so we need
    # NEW_ROOT/robometer_frames_eval_droid → some valid dir.
    for ds_name, target in [
        ("robometer_frames_eval_droid", "_projects_prjs1958_robometer_frames_hf_v2_eval_droid_raw_robometer_frames_eval_droid"),
        ("robometer_frames_eval_robometer", "_projects_prjs1958_robometer_frames_hf_v2_eval_robometer_raw_robometer_frames_eval_robometer"),
        ("robometer_frames_eval_metaworld", "_projects_prjs1958_robometer_frames_hf_v2_eval_metaworld_raw_robometer_frames_eval_metaworld"),
        ("robometer_frames_eval_failsafe", "_projects_prjs1958_robometer_frames_hf_v2_eval_failsafe_raw_robometer_frames_eval_failsafe"),
        ("robometer_frames_test", "_projects_prjs1958_robometer_frames_hf_v2_test_raw_robometer_frames_test"),
        ("robometer_frames_warmup", "_projects_prjs1958_robometer_frames_hf_v2_warmup_raw_robometer_frames_warmup"),
    ]:
        link = f"{NEW_ROOT}/{ds_name}"
        if os.path.exists(link) or os.path.islink(link):
            continue
        os.symlink(target, link)

    # --- 2. Build new train cache: clone Arrow + write modified droid_success NPZs ---
    print(f"[clone] loading v2 train Arrow")
    ds = load_from_disk(f"{V2_TRAIN}/processed_dataset")
    print(f"  v2 train: {len(ds)} rows")

    new_frames_dir = f"{NEW_TRAIN}/frames"
    os.makedirs(new_frames_dir, exist_ok=True)

    # Build a mapping idx -> new frames path for the droid 240×426 rows
    droid_426_idxs = []
    for i, (src, sh) in enumerate(zip(ds["data_source"], ds["frames_shape"])):
        if src and "droid" in src and tuple(sh) == (16, 240, 426, 3):
            droid_426_idxs.append(i)
    print(f"  found {len(droid_426_idxs)} droid samples at (16,240,426,3) — will rewrite")

    # Step 2a: write new NPZs for those rows
    droid_426_set = set(droid_426_idxs)
    new_paths = {}  # idx -> new_path
    for i in droid_426_idxs:
        r = ds[i]
        old_npz = r["frames"]
        fname = os.path.basename(old_npz)
        new_npz = f"{new_frames_dir}/{fname}"
        new_paths[i] = new_npz
        d = np.load(old_npz)
        arr = d["frames"]
        if arr.shape != (16, 240, 426, 3):
            print(f"    skip {old_npz}: shape={arr.shape}")
            continue
        start = (arr.shape[2] - TARGET_W) // 2
        cropped = arr[:, :, start:start + TARGET_W, :]
        np.savez_compressed(
            new_npz,
            frames=cropped,
            shape=cropped.shape,
            num_frames=cropped.shape[0],
        )
        if (len(new_paths) % 500) == 0:
            print(f"    wrote {len(new_paths)}/{len(droid_426_idxs)} new NPZs")

    # Step 2b: build a new Arrow dataset with updated `frames` paths and `frames_shape` for those rows
    print(f"  patching Arrow with new frames paths and shapes...")
    def patch(row, idx):
        if idx in droid_426_set:
            return {"frames": new_paths[idx], "frames_shape": [16, 240, 400, 3]}
        return {}

    ds_new = ds.map(patch, with_indices=True, num_proc=1, desc="patching")

    out_processed = f"{NEW_TRAIN}/processed_dataset"
    if os.path.exists(out_processed):
        shutil.rmtree(out_processed)
    ds_new.save_to_disk(out_processed)
    print(f"  saved patched Arrow to {out_processed}")

    # Step 2c: dataset_info.json that the trainer's discovery reads
    info = {
        "dataset_path": f"{NEW_ROOT}/train_raw",
        "subset": "robometer_frames_train",
        "total_trajectories": len(ds_new),
        "cache_timestamp": "v2_leakfix",
        "config_hash": "v2_leakfix_droid_400",
    }
    with open(f"{NEW_TRAIN}/dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Step 2d: index_mappings.json — copy from v2 (same indices, same row order)
    src_im = f"{V2_TRAIN}/index_mappings.json"
    dst_im = f"{NEW_TRAIN}/index_mappings.json"
    if os.path.exists(src_im):
        shutil.copy(src_im, dst_im)
        print(f"  copied index_mappings.json")

    # Step 2e: shortname symlink for train
    train_link = f"{NEW_ROOT}/robometer_frames_train"
    if os.path.exists(train_link) or os.path.islink(train_link):
        os.remove(train_link)
    os.symlink(os.path.basename(NEW_TRAIN), train_link)
    print(f"  symlinked {train_link}")

    print(f"\n[clone] DONE. New cache root: {NEW_ROOT}")
    print(f"  - eval/test/warmup: symlinked from v2 (untouched)")
    print(f"  - train: cloned Arrow + {len(new_paths)} rewritten droid_success NPZs (240×400)")
    print(f"  - non-droid train rows: Arrow `frames` still points at v2 NPZs (no copy)")


if __name__ == "__main__":
    main()
