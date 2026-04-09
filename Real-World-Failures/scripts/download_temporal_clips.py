#!/usr/bin/env python3
"""
RoboReward Temporal Clip Downloader
====================================

Downloads all temporally clipped videos from the RoboReward dataset.
These are truncated versions of successful episodes (the success ending is removed),
identified by the `_attempt_N_score_N` suffix in their filenames.

Organizes downloads by source dataset for future video completion experiments.

Usage:
    python download_temporal_clips.py [--output-dir DIR] [--workers N] [--dry-run]

Author: Platon Karageorgis
"""

import os
import sys
import json
import re
import argparse
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

HF_REPO = "teetone/RoboReward"
HF_SHA = "469b9af76e8539e8d2ac553b081307738fd92ca5"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_SHA}/train"

DEFAULT_OUTPUT_DIR = "/var/scratch/pkarageo/roboreward_video_completion"
DEFAULT_METADATA_DIR = "/var/scratch/pkarageo/roboreward_dataset"
DEFAULT_WORKERS = 4
CHUNK_SIZE = 8192
REQUEST_TIMEOUT = 60
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0  # seconds, doubles each retry

TEMPORAL_CLIP_PATTERN = re.compile(r'_attempt_\d+_score_\d+\.mp4$')


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_temporal_clip(file_name: str) -> bool:
    """Check if a filename corresponds to a temporal clip."""
    return bool(TEMPORAL_CLIP_PATTERN.search(file_name))


def extract_dataset(file_name: str) -> str:
    """Extract dataset name from file path (first directory component)."""
    return file_name.split('/')[0]


def build_download_url(file_name: str) -> str:
    """Construct HuggingFace HTTPS download URL from relative file path."""
    return f"{HF_BASE_URL}/{file_name}"


def download_file(url: str, output_path: str, timeout: int = REQUEST_TIMEOUT) -> Tuple[bool, int, str]:
    """
    Download a file from URL to output_path with exponential backoff on 429s.

    Returns:
        Tuple of (success, file_size_bytes, error_message)
    """
    last_error = ""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, stream=True, timeout=timeout)
            if response.status_code == 429:
                backoff = INITIAL_BACKOFF * (2 ** attempt)
                time.sleep(backoff)
                last_error = "429 Too Many Requests"
                continue
            response.raise_for_status()

            # Ensure parent directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Write to temp file first, then rename (atomic-ish)
            tmp_path = output_path + '.tmp'
            file_size = 0
            with open(tmp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    file_size += len(chunk)

            os.rename(tmp_path, output_path)
            return True, file_size, ""

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                backoff = INITIAL_BACKOFF * (2 ** attempt)
                time.sleep(backoff)
                last_error = str(e)
                continue
            # Clean up partial download
            tmp_path = output_path + '.tmp'
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False, 0, str(e)

        except Exception as e:
            # Clean up partial download
            tmp_path = output_path + '.tmp'
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False, 0, str(e)

    # All retries exhausted
    tmp_path = output_path + '.tmp'
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return False, 0, f"Failed after {MAX_RETRIES} retries: {last_error}"


# ============================================================================
# MAIN LOGIC
# ============================================================================

def get_original_base(file_name: str) -> str:
    """Get the original (non-augmented) filename from a clip filename."""
    return re.sub(r'_attempt_\d+_score_\d+\.mp4$', '.mp4', file_name)


def load_temporal_clips(metadata_dir: str) -> List[Dict]:
    """
    Load TRUE temporal clip entries from metadata.

    The _attempt_N_score_N suffix is used for BOTH augmentation types:
      - Temporal clipping: truncated video, same task as original
      - Counterfactual relabeling: same video, different task from original

    This function filters to only temporal clips by comparing each clip's
    task against its original video's task. Counterfactual entries (where
    the task differs from the original) are excluded.
    """
    metadata_file = os.path.join(metadata_dir, "metadata.jsonl")
    if not os.path.exists(metadata_file):
        print(f"ERROR: {metadata_file} not found!")
        sys.exit(1)

    print(f"Scanning {metadata_file} for temporal clips...")
    print("  (Filtering out counterfactual relabeling entries)")

    # First pass: build map of original video tasks and collect candidate clips
    originals = {}  # original filename -> task
    candidates = []  # all _attempt_ entries
    with open(metadata_file) as f:
        for line in f:
            entry = json.loads(line)
            fn = entry['file_name']
            if is_temporal_clip(fn):
                candidates.append(entry)
            else:
                originals[fn] = entry['task']

    print(f"  Found {len(candidates):,} _attempt_ entries total")
    print(f"  Found {len(originals):,} original videos")

    # Second pass: keep only clips whose task matches the original (= temporal)
    clips = []
    counterfactual = 0
    unmatched = 0
    for entry in candidates:
        fn = entry['file_name']
        original_fn = get_original_base(fn)
        orig_task = originals.get(original_fn)

        if orig_task is None:
            unmatched += 1
            continue
        if entry['task'] != orig_task:
            counterfactual += 1
            continue

        clips.append({
            'video': fn,
            'task': entry['task'],
            'reward': entry['reward'],
            'dataset': extract_dataset(fn),
            'score': int(re.search(r'_score_(\d+)', fn).group(1)),
        })

    print(f"  Temporal clips (same task as original): {len(clips):,}")
    print(f"  Counterfactual relabeling (different task, excluded): {counterfactual:,}")
    if unmatched:
        print(f"  No matching original found (excluded): {unmatched:,}")

    return clips


def load_manifest(manifest_path: str) -> Dict:
    """Load existing download manifest or return empty one."""
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)
    return {
        'total_clips': 0,
        'downloaded': 0,
        'failed': 0,
        'skipped': 0,
        'total_bytes': 0,
        'downloaded_files': {},
        'failed_files': {},
    }


def save_manifest(manifest: Dict, manifest_path: str):
    """Save download manifest."""
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Download all temporal clips from RoboReward dataset"
    )
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--metadata-dir', type=str, default=DEFAULT_METADATA_DIR,
                        help=f'Directory containing metadata files (default: {DEFAULT_METADATA_DIR})')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help=f'Number of concurrent download threads (default: {DEFAULT_WORKERS})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only count clips and show plan, do not download')
    parser.add_argument('--max-clips', type=int, default=None,
                        help='Download at most N clips (for testing)')
    args = parser.parse_args()

    # ---- Load clips ----
    clips = load_temporal_clips(args.metadata_dir)
    if not clips:
        print("No temporal clips found. Exiting.")
        sys.exit(1)

    if args.max_clips:
        clips = clips[:args.max_clips]
        print(f"  Limited to first {args.max_clips} clips (--max-clips)")

    # ---- Dataset breakdown ----
    datasets = {}
    for clip in clips:
        ds = clip['dataset']
        datasets[ds] = datasets.get(ds, 0) + 1

    print(f"\nDataset breakdown ({len(datasets)} datasets):")
    for ds in sorted(datasets, key=datasets.get, reverse=True):
        print(f"  {ds}: {datasets[ds]:,} clips")

    if args.dry_run:
        print(f"\nDry run complete. Would download {len(clips):,} clips.")
        return

    # ---- Setup directories ----
    clips_dir = os.path.join(args.output_dir, "clips")
    completions_dir = os.path.join(args.output_dir, "completions")
    metadata_dir = os.path.join(args.output_dir, "metadata")

    # Create dataset subdirs
    for ds in datasets:
        os.makedirs(os.path.join(clips_dir, ds), exist_ok=True)
        os.makedirs(os.path.join(completions_dir, ds), exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    # ---- Load manifest for resume ----
    manifest_path = os.path.join(metadata_dir, "download_manifest.json")
    manifest = load_manifest(manifest_path)
    manifest['total_clips'] = len(clips)

    # ---- Filter out already downloaded ----
    to_download = []
    skipped = 0
    for clip in clips:
        filename = os.path.basename(clip['video'])
        ds = clip['dataset']
        output_path = os.path.join(clips_dir, ds, filename)

        if os.path.exists(output_path):
            skipped += 1
            # Track in manifest if not already there
            if clip['video'] not in manifest['downloaded_files']:
                manifest['downloaded_files'][clip['video']] = {
                    'size': os.path.getsize(output_path),
                    'dataset': ds,
                }
            continue

        to_download.append((clip, output_path))

    manifest['skipped'] = skipped
    # Clear previous failures so they get retried
    manifest['failed_files'] = {}
    print(f"\nResume check: {skipped:,} already downloaded, {len(to_download):,} remaining")

    if not to_download:
        print("All clips already downloaded!")
        manifest['downloaded'] = len(manifest['downloaded_files'])
        save_manifest(manifest, manifest_path)
        return

    # ---- Download with thread pool ----
    print(f"\nStarting download with {args.workers} workers...")
    start_time = time.time()
    downloaded = 0
    failed = 0
    total_bytes = sum(v.get('size', 0) for v in manifest['downloaded_files'].values())

    def download_clip(clip_and_path):
        clip, output_path = clip_and_path
        url = build_download_url(clip['video'])
        return clip, output_path, download_file(url, output_path)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_clip, item): item for item in to_download}

        for i, future in enumerate(as_completed(futures), 1):
            clip, output_path, (success, size, error) = future.result()

            if success:
                downloaded += 1
                total_bytes += size
                manifest['downloaded_files'][clip['video']] = {
                    'size': size,
                    'dataset': clip['dataset'],
                }
            else:
                failed += 1
                manifest['failed_files'][clip['video']] = {
                    'error': error,
                    'dataset': clip['dataset'],
                }

            # Progress update every 100 files or on failure
            if i % 100 == 0 or not success:
                elapsed = time.time() - start_time
                rate = downloaded / elapsed if elapsed > 0 else 0
                mb = total_bytes / (1024 * 1024)
                status = "OK" if success else f"FAIL: {error[:60]}"
                print(f"  [{i:,}/{len(to_download):,}] {downloaded:,} OK, {failed:,} failed | "
                      f"{mb:.1f} MB | {rate:.1f} files/s | {status}")

            # Save manifest periodically
            if i % 500 == 0:
                manifest['downloaded'] = len(manifest['downloaded_files'])
                manifest['failed'] = len(manifest['failed_files'])
                manifest['total_bytes'] = total_bytes
                save_manifest(manifest, manifest_path)

    # ---- Final summary ----
    elapsed = time.time() - start_time
    manifest['downloaded'] = len(manifest['downloaded_files'])
    manifest['failed'] = len(manifest['failed_files'])
    manifest['total_bytes'] = total_bytes
    manifest['elapsed_seconds'] = elapsed
    save_manifest(manifest, manifest_path)

    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"  Total clips:    {manifest['total_clips']:,}")
    print(f"  Downloaded:     {manifest['downloaded']:,}")
    print(f"  Failed:         {manifest['failed']:,}")
    print(f"  Skipped (existing): {manifest['skipped']:,}")
    print(f"  Total size:     {total_bytes / (1024**3):.2f} GB")
    print(f"  Time:           {elapsed/60:.1f} minutes")
    print(f"  Manifest:       {manifest_path}")
    print(f"{'='*60}")

    if manifest['failed'] > 0:
        print(f"\nWARNING: {manifest['failed']} downloads failed. "
              f"Re-run the script to retry (resume-capable).")


if __name__ == '__main__':
    main()
