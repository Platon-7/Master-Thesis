#!/usr/bin/env python3
"""
Download missing PlayWorld latent videos from HuggingFace.
Run this from a login node (requires internet access).

Usage:
    python download_latents.py
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi, login
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

HF_TOKEN = os.environ.get("HF_TOKEN", "hf_YOUR_TOKEN_HERE")
login(token=HF_TOKEN, add_to_git_credential=False)

REPO_ID = "tennyyyin/playworld_dataset_preview"
LOCAL_DIR = Path("/scratch-shared/pkarageorgis/playworld_data/v0_2025_12_28_2000")
SPLITS = ["train", "val"]
N_WORKERS = 3  # parallel downloads (keep low to avoid HF rate limits)

print_lock = threading.Lock()

def safe_print(*args):
    with print_lock:
        print(*args, flush=True)


def download_trajectory(tid, split, existing_ids):
    """Download 3 latent files (0.pt, 1.pt, 2.pt) for one trajectory."""
    if tid in existing_ids:
        return tid, "skipped"

    out_dir = LOCAL_DIR / "latent_videos" / split / tid
    out_dir.mkdir(parents=True, exist_ok=True)

    for view_idx in range(3):
        remote_path = f"v0_2025_12_28_2000/latent_videos/{split}/{tid}/{view_idx}.pt"
        local_path = out_dir / f"{view_idx}.pt"

        if local_path.exists():
            continue

        try:
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=remote_path,
                local_dir=str(LOCAL_DIR.parent),
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            return tid, f"error: {e}"

    return tid, "downloaded"


def main():
    api = HfApi()

    for split in SPLITS:
        print(f"\n{'='*60}")
        print(f"Processing split: {split}")
        print(f"{'='*60}")

        # Get all trajectory IDs on HuggingFace
        remote_dirs = list(api.list_repo_tree(
            REPO_ID, repo_type="dataset",
            path_in_repo=f"v0_2025_12_28_2000/latent_videos/{split}",
            recursive=False,
        ))
        all_ids = [d.path.split("/")[-1] for d in remote_dirs]
        print(f"  Remote: {len(all_ids)} trajectories")

        # Check what we already have
        local_lat_dir = LOCAL_DIR / "latent_videos" / split
        existing_ids = set()
        if local_lat_dir.exists():
            for d in local_lat_dir.iterdir():
                if d.is_dir() and all((d / f"{v}.pt").exists() for v in range(3)):
                    existing_ids.add(d.name)
        print(f"  Local (complete): {len(existing_ids)}")
        print(f"  To download: {len(all_ids) - len(existing_ids)}")

        if len(all_ids) == len(existing_ids):
            print(f"  All done, skipping.")
            continue

        # Download in parallel
        downloaded = 0
        errors = 0
        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {
                executor.submit(download_trajectory, tid, split, existing_ids): tid
                for tid in all_ids
            }
            for i, future in enumerate(as_completed(futures)):
                tid, status = future.result()
                if status == "downloaded":
                    downloaded += 1
                    if downloaded % 50 == 0:
                        safe_print(f"  Progress: {downloaded} downloaded, {errors} errors, "
                                   f"{i+1}/{len(all_ids)} total")
                elif status.startswith("error"):
                    errors += 1
                    safe_print(f"  ERROR {tid}: {status}")

        print(f"\n  Done: {downloaded} downloaded, {errors} errors")

    # Also download metainfo
    print(f"\nDownloading metainfo...")
    for fname in ["stat.json", "train_sample.json", "val_sample.json"]:
        local_path = LOCAL_DIR / "metainfo" / fname
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=f"v0_2025_12_28_2000/metainfo/{fname}",
                    local_dir=str(LOCAL_DIR.parent),
                    local_dir_use_symlinks=False,
                )
                print(f"  Downloaded: {fname}")
            except Exception as e:
                print(f"  Error {fname}: {e}")

    # Final count
    total = sum(
        1 for split in SPLITS
        for d in (LOCAL_DIR / "latent_videos" / split).iterdir()
        if d.is_dir() and all((d / f"{v}.pt").exists() for v in range(3))
    )
    print(f"\nFinal: {total} complete trajectories on disk")


if __name__ == "__main__":
    main()
