"""Build a parallel `robometer_frames_eval_droid_squared` HF dataset by
center-cropping the existing eval_droid NPZs from (16, 240, 400, 3) to
(16, 240, 240, 3). Pure data transform — no retraining.

Use case: test whether droid's persistent ~0.5 AUC across run5–9 was driven by
the 16×24 vs 16×16 patch grid mismatch (droid wider than other sources). Eval
of run9-step3000 against this squared cache is OOD-aware: a HIGHER AUC than
0.55 baseline would be a strong "shape matters" signal; lower or equal is
inconclusive.
"""
import os
import sys
import shutil
import numpy as np
from datasets import load_from_disk

SRC_HF_ROOT = "/projects/prjs1958/robometer_frames_hf_3p5x/_projects_prjs1958_robometer_frames_hf_3p5x_eval_droid_raw_robometer_frames_eval_droid"
DST_HF_ROOT = "/projects/prjs1958/robometer_frames_hf_3p5x/_projects_prjs1958_robometer_frames_hf_3p5x_eval_droid_squared_raw_robometer_frames_eval_droid_squared"
SHORTNAME_LINK = "/projects/prjs1958/robometer_frames_hf_3p5x/robometer_frames_eval_droid_squared"


def center_crop_horizontal(arr: np.ndarray, target_w: int = 240) -> np.ndarray:
    """For (T, H, W, 3) with H==240 and W>=240, center-crop horizontally to W=240."""
    if arr.ndim != 4 or arr.shape[1] != 240:
        raise ValueError(f"unexpected shape {arr.shape}")
    w = arr.shape[2]
    if w == target_w:
        return arr
    if w < target_w:
        raise ValueError(f"width {w} < target {target_w}, cannot center-crop")
    start = (w - target_w) // 2
    return arr[:, :, start:start + target_w, :]


def main():
    if os.path.exists(DST_HF_ROOT):
        print(f"WARN: {DST_HF_ROOT} exists; removing")
        shutil.rmtree(DST_HF_ROOT)

    os.makedirs(DST_HF_ROOT, exist_ok=True)
    os.makedirs(os.path.join(DST_HF_ROOT, "frames"), exist_ok=True)

    print(f"loading source: {SRC_HF_ROOT}/processed_dataset")
    ds = load_from_disk(os.path.join(SRC_HF_ROOT, "processed_dataset"))
    print(f"  rows: {len(ds)}")

    # Build new rows: copy NPZs but with squared frames; update frames_path/shape
    new_rows = []
    for i, row in enumerate(ds):
        src_npz_path = row["frames"]
        new_filename = os.path.basename(src_npz_path)
        dst_npz_path = os.path.join(DST_HF_ROOT, "frames", new_filename)

        # Load, crop, save
        d = np.load(src_npz_path)
        arr = d["frames"]
        squared = center_crop_horizontal(arr, 240)
        np.savez_compressed(
            dst_npz_path,
            frames=squared,
            shape=squared.shape,
            num_frames=squared.shape[0],
        )

        new_row = dict(row)
        new_row["frames"] = dst_npz_path
        new_row["frames_shape"] = list(squared.shape)
        new_row["data_source"] = "robometer_frames_droid_squared"
        new_rows.append(new_row)

        if (i + 1) % 100 == 0 or i == len(ds) - 1:
            print(f"  {i + 1}/{len(ds)} processed")

    # Write new dataset (Arrow)
    from datasets import Dataset
    new_ds = Dataset.from_list(new_rows)
    new_ds.save_to_disk(os.path.join(DST_HF_ROOT, "processed_dataset"))

    # Stub dataset_info.json at parent so trainer's _get_available_datasets finds it
    import json
    with open(os.path.join(DST_HF_ROOT, "dataset_info.json"), "w") as f:
        json.dump({"num_rows": len(new_rows)}, f)

    # Symlink at short name
    if os.path.exists(SHORTNAME_LINK) or os.path.islink(SHORTNAME_LINK):
        os.remove(SHORTNAME_LINK)
    os.symlink(os.path.basename(DST_HF_ROOT), SHORTNAME_LINK)

    print(f"\nDONE — squared cache at {DST_HF_ROOT}")
    print(f"      symlinked as {SHORTNAME_LINK}")
    print(f"      first NPZ shape: {squared.shape}")


if __name__ == "__main__":
    main()
