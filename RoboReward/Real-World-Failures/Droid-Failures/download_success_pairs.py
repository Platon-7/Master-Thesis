#!/usr/bin/env python3
"""
Download DROID success episodes matched to our failures for in-context learning.

Matching strategy:
  1. Task-level match (scene_id + generic_task) — preferred
  2. Scene-level fallback (scene_id only) — for the ~544 failures without task match

For each (scene, task) pair, downloads up to N success episodes where
N = number of failures for that pair, enabling unique pairing.

Each episode in the manifest is flagged with match_type="task" or "scene_fallback"
so scene-only matches can be excluded later if desired.

Requires: gsutil (available in roboreward conda env)
Run from login node (needs internet).

Usage:
    python download_success_pairs.py --step manifest    # Build manifest only
    python download_success_pairs.py --step download     # Download from manifest
    python download_success_pairs.py                     # Both
"""

import os
import json
import argparse
import subprocess
import random
from collections import defaultdict, Counter
from datetime import datetime


# ============================================================
# Paths
# ============================================================

RELABELED_PATH = "/var/scratch/pkarageo/droid_failures_labeled/qwen3_relabeled.jsonl"
FAILURE_MANIFEST = "/var/scratch/pkarageo/droid_failures_raw/download_manifest.json"
METADATA_DIR = "/var/scratch/pkarageo/droid_success_pairs/_metadata"
OUTPUT_DIR = "/var/scratch/pkarageo/droid_success_pairs"
VIDEO_DIR = os.path.join(OUTPUT_DIR, "task_matched")

GCS_BUCKET = "gs://gresearch/robotics/droid_raw/1.0.1"


def load_failure_data():
    """Load failure episodes, return (failure_generic, ep_scene, failure_counts)."""
    failure_generic = {}  # episode -> generic_task (lowered)
    with open(RELABELED_PATH) as f:
        for line in f:
            d = json.loads(line.strip())
            failure_generic[d["episode"]] = d["generic_task"].strip().lower()

    with open(FAILURE_MANIFEST) as f:
        manifest = json.load(f)

    ep_scene = {}
    for ep, info in manifest["completed_episodes"].items():
        parts = info.get("uuid", "").split("+")
        if len(parts) >= 2:
            ep_scene[ep] = parts[1]

    failure_counts = Counter()
    for ep, task in failure_generic.items():
        sid = ep_scene.get(ep)
        if sid:
            failure_counts[(sid, task)] += 1

    return failure_generic, ep_scene, failure_counts


def scan_success_metadata(failure_counts):
    """Scan all success metadata. Returns (task_pool, scene_pool).

    task_pool: (scene_id, task) -> list of episode dicts  (exact task match)
    scene_pool: scene_id -> list of episode dicts           (any task in scene)
    """
    # Which scenes do we need?
    needed_scenes = set(sid for sid, _ in failure_counts)

    task_pool = defaultdict(list)
    scene_pool = defaultdict(list)

    for lab_dir in sorted(os.listdir(METADATA_DIR)):
        lab_path = os.path.join(METADATA_DIR, lab_dir)
        if not os.path.isdir(lab_path):
            continue

        lab_task = 0
        lab_scene = 0
        for fn in os.listdir(lab_path):
            if not fn.endswith(".json"):
                continue
            fpath = os.path.join(lab_path, fn)
            try:
                with open(fpath) as f:
                    d = json.load(f)
                uuid = d.get("uuid", "")
                parts = uuid.split("+")
                if len(parts) < 3:
                    continue
                sid = parts[1]
                if sid not in needed_scenes:
                    continue
                task = d.get("current_task", d.get("task", "")).strip().lower()
                if not task:
                    continue

                # Reconstruct GCS episode dir from hdf5_path
                # hdf5_path: "success/DATE/EPISODE_DIR/trajectory.h5"
                hdf5_path = d.get("hdf5_path", "")
                if hdf5_path:
                    ep_rel = "/".join(hdf5_path.split("/")[:-1])  # strip trajectory.h5
                    gcs_dir = f"{GCS_BUCKET}/{parts[0]}/{ep_rel}/"
                else:
                    gcs_dir = None

                ep_info = {
                    "uuid": uuid,
                    "lab": parts[0],
                    "task_original": d.get("current_task", d.get("task", "")),
                    "metadata_file": fpath,
                    "gcs_dir": gcs_dir,
                }

                pair = (sid, task)
                if pair in failure_counts:
                    task_pool[pair].append(ep_info)
                    lab_task += 1

                scene_pool[sid].append(ep_info)
                lab_scene += 1
            except Exception:
                continue

        if lab_task > 0 or lab_scene > 0:
            print(f"  {lab_dir}: {lab_task} task matches, {lab_scene} scene-level")

    return task_pool, scene_pool


def build_manifest():
    """Build download manifest with task-level matching + scene-level fallback."""

    print("=" * 60)
    print("Building task-level matching manifest (with scene fallback)")
    print("=" * 60)

    # 1. Load failures
    print("\n1. Loading failure episodes...")
    failure_generic, ep_scene, failure_counts = load_failure_data()
    print(f"  Failure episodes: {len(failure_generic)}")
    print(f"  Unique (scene, task) pairs: {len(failure_counts)}")

    # 2. Scan success metadata
    print("\n2. Scanning success metadata...")
    task_pool, scene_pool = scan_success_metadata(failure_counts)
    print(f"\n  Task-level matched pairs: {len(task_pool)} / {len(failure_counts)}")
    print(f"  Task-level matched episodes: {sum(len(v) for v in task_pool.values())}")
    print(f"  Scenes with any success: {len(scene_pool)}")

    # 3. Select episodes
    print("\n3. Selecting success episodes...")
    random.seed(42)

    selected = []
    task_matched_pairs = 0
    task_matched_full = 0
    task_matched_partial = 0
    scene_fallback_pairs = 0
    scene_fallback_episodes = 0

    # Track which uuids are already selected (avoid duplicates across pairs)
    selected_uuids = set()

    for pair, fcount in failure_counts.most_common():
        sid, task = pair
        pool = task_pool.get(pair, [])

        if pool:
            # --- Task-level match ---
            n_needed = fcount
            available = [ep for ep in pool if ep["uuid"] not in selected_uuids]

            if len(available) >= n_needed:
                chosen = random.sample(available, n_needed)
                task_matched_full += 1
            else:
                chosen = available
                task_matched_partial += 1

            task_matched_pairs += 1
            for ep in chosen:
                selected_uuids.add(ep["uuid"])
                selected.append({
                    "scene_id": sid,
                    "failure_task": task,
                    "task_original": ep["task_original"],
                    "uuid": ep["uuid"],
                    "lab": ep["lab"],
                    "metadata_file": ep["metadata_file"],
                    "gcs_dir": ep.get("gcs_dir"),
                    "match_type": "task",
                })
        else:
            # --- Scene-level fallback ---
            scene_eps = scene_pool.get(sid, [])
            if not scene_eps:
                continue

            n_needed = fcount
            available = [ep for ep in scene_eps if ep["uuid"] not in selected_uuids]

            if len(available) >= n_needed:
                chosen = random.sample(available, n_needed)
            else:
                chosen = available

            scene_fallback_pairs += 1
            scene_fallback_episodes += len(chosen)
            for ep in chosen:
                selected_uuids.add(ep["uuid"])
                selected.append({
                    "scene_id": sid,
                    "failure_task": task,
                    "task_original": ep["task_original"],
                    "uuid": ep["uuid"],
                    "lab": ep["lab"],
                    "metadata_file": ep["metadata_file"],
                    "gcs_dir": ep.get("gcs_dir"),
                    "match_type": "scene_fallback",
                })

    task_selected = sum(1 for ep in selected if ep["match_type"] == "task")
    scene_selected = sum(1 for ep in selected if ep["match_type"] == "scene_fallback")

    print(f"  Task-level matches: {task_selected} episodes ({task_matched_pairs} pairs, "
          f"{task_matched_full} full / {task_matched_partial} partial)")
    print(f"  Scene-level fallback: {scene_selected} episodes ({scene_fallback_pairs} pairs)")
    print(f"  Total selected: {len(selected)}")

    # 4. Check GCS path resolution (paths come from metadata hdf5_path)
    print("\n4. Checking GCS paths...")
    resolved = sum(1 for ep in selected if ep.get("gcs_dir"))
    unresolved = len(selected) - resolved
    print(f"  Resolved: {resolved}")
    print(f"  Unresolved: {unresolved}")

    # Filter to resolved only
    selected = [ep for ep in selected if ep.get("gcs_dir")]

    task_final = sum(1 for ep in selected if ep["match_type"] == "task")
    scene_final = sum(1 for ep in selected if ep["match_type"] == "scene_fallback")

    # 5. Save manifest
    manifest_path = os.path.join(OUTPUT_DIR, "task_match_manifest.json")
    manifest_data = {
        "created": datetime.now().isoformat(),
        "total_selected": len(selected),
        "task_matched": task_final,
        "scene_fallback": scene_final,
        "task_matched_pairs": task_matched_pairs,
        "task_matched_full": task_matched_full,
        "task_matched_partial": task_matched_partial,
        "scene_fallback_pairs": scene_fallback_pairs,
        "failure_episodes_total": len(failure_generic),
        "failure_pairs_total": len(failure_counts),
        "episodes": selected,
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Manifest saved: {manifest_path}")
    print(f"Episodes to download: {len(selected)}")
    print(f"  task-level: {task_final}")
    print(f"  scene-fallback: {scene_final}")
    print(f"Estimated size: ~{len(selected) * 0.032:.0f} GB")
    print(f"{'=' * 60}")

    return selected


def download_videos(selected=None):
    """Download MP4 videos for selected success episodes."""

    print("\n" + "=" * 60)
    print("Downloading success videos")
    print("=" * 60)

    if selected is None:
        manifest_path = os.path.join(OUTPUT_DIR, "task_match_manifest.json")
        with open(manifest_path) as f:
            data = json.load(f)
        selected = data["episodes"]

    os.makedirs(VIDEO_DIR, exist_ok=True)

    downloaded = 0
    skipped = 0
    errors = 0

    for i, ep in enumerate(selected):
        gcs_dir = ep.get("gcs_dir")
        if not gcs_dir:
            errors += 1
            continue

        # Output dir: task_matched/LAB_SCENEID_TIMESTAMP/
        parts = ep["uuid"].split("+")
        flat_name = "_".join(parts)
        ep_out = os.path.join(VIDEO_DIR, flat_name)

        # Check if already downloaded
        if os.path.exists(ep_out) and len(os.listdir(ep_out)) > 0:
            skipped += 1
            continue

        os.makedirs(ep_out, exist_ok=True)

        # Download mono MP4s only
        mp4_gcs = gcs_dir.rstrip("/") + "/recordings/MP4/"

        try:
            result = subprocess.run(
                ["gsutil", "ls", mp4_gcs],
                capture_output=True, text=True, timeout=30
            )
        except subprocess.TimeoutExpired:
            errors += 1
            try:
                os.rmdir(ep_out)
            except OSError:
                pass
            continue

        if result.returncode != 0:
            errors += 1
            try:
                os.rmdir(ep_out)
            except OSError:
                pass
            continue

        # Filter for mono mp4s
        mono_files = [
            f.strip() for f in result.stdout.strip().split("\n")
            if f.strip().endswith(".mp4") and "-stereo" not in f
        ]

        if not mono_files:
            errors += 1
            try:
                os.rmdir(ep_out)
            except OSError:
                pass
            continue

        # Download
        ok = True
        for mf in mono_files:
            try:
                r = subprocess.run(
                    ["gsutil", "cp", "-n", mf, ep_out],
                    capture_output=True, text=True, timeout=120
                )
                if r.returncode != 0:
                    ok = False
            except subprocess.TimeoutExpired:
                ok = False

        if ok:
            downloaded += 1
        else:
            errors += 1

        # Progress
        total_done = downloaded + skipped
        if total_done <= 10 or total_done % 100 == 0:
            print(f"  [{total_done}/{len(selected)}] downloaded={downloaded} "
                  f"skipped={skipped} errors={errors}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Download complete")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Total: {downloaded + skipped + errors}")
    print(f"  Output: {VIDEO_DIR}")
    print(f"{'=' * 60}")

    # Update manifest with download status
    manifest_path = os.path.join(OUTPUT_DIR, "task_match_manifest.json")
    with open(manifest_path) as f:
        data = json.load(f)
    data["download_status"] = {
        "downloaded": downloaded,
        "skipped": skipped,
        "errors": errors,
        "timestamp": datetime.now().isoformat(),
    }
    with open(manifest_path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Download task-matched DROID success pairs")
    parser.add_argument("--step", choices=["manifest", "download", "all"], default="all")
    args = parser.parse_args()

    selected = None

    if args.step in ("manifest", "all"):
        selected = build_manifest()

    if args.step in ("download", "all"):
        download_videos(selected)


if __name__ == "__main__":
    main()
