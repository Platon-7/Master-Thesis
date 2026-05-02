#!/usr/bin/env python3
"""
extract_successes.py — Extract 16 uniformly-spaced keyframes from every
success episode across all Robometer archives.

Reads archives from /projects/prjs1958/robometer_full_dataset/raw_archives/.
Outputs to /projects/prjs1958/robometer_full_dataset/keyframes_success/.

Output layout:
    <output_dir>/
      <archive_name>/
        ep_<idx>_<uuid>/
          frame_0_0.00s.jpg   # 0% progress
          frame_1_...jpg
          ...
          frame_15_...jpg     # 100% progress
      manifests/
        <archive_name>_successes.jsonl

Each episode gets 16 frames uniformly spaced so that frame i represents
i/15 * 100% task progress (frame 0 = start, frame 3 ≈ 25%, frame 7 ≈ 50%,
frame 11 ≈ 75%, frame 15 = end).

Resume-safe: episodes already in the manifest are skipped on re-run.

Usage:
    python extract_successes.py --archive jesbu1_racer_rfm_racer_train
    python extract_successes.py --archive all
"""

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

from robometer_families import family_of

# ────────────────────────────────────────────────────────────────────────────
# Archive registry (from pair_robometer.py)
# ────────────────────────────────────────────────────────────────────────────

ARCH_ROOT = Path("/projects/prjs1958/robometer_full_dataset/raw_archives")
SINGLE_DIR = ARCH_ROOT / "single"
SPLIT_DIR  = ARCH_ROOT / "split"

# Unified frame dataset: DROID + Robometer successes + (later) Group A/B failures
# Layout:
#   robometer_frame_dataset/
#     droid/                         (moved from /projects/prjs1958/Droid_Failures)
#     robometer/
#       keyframes_success/<archive>/<episode_id>/frame_*.jpg
#       manifests/<archive>_successes.jsonl
DEFAULT_OUTPUT = Path("/projects/prjs1958/robometer_frame_dataset/robometer/keyframes_success")
DEFAULT_MANIFEST_DIR = Path("/projects/prjs1958/robometer_frame_dataset/robometer/manifests")

# Every archive that contributes successes to Robometer.
# Maps archive_name -> robot family.
SUCCESS_ARCHIVES = {
    # --- From failure archives (Group A single-file) ---
    "jesbu1_racer_rfm_racer_train":                                             "racer",
    "jesbu1_racer_rfm_racer_val":                                               "racer",
    "jesbu1_auto_eval_rfm_auto_eval_rfm":                                       "auto_eval",
    "jesbu1_rfm_new_mit_franka_rfm_rfm_new_mit_franka_rfm":                     "mit_franka",
    "jesbu1_rfm_new_mit_franka_rfm_nowrist_rfm_new_mit_franka_rfm_nowrist":     "mit_franka",
    "jesbu1_mit_franka_p-rank_rfm_mit_franka_p-rank_rfm":                       "mit_franka",
    "jesbu1_usc_koch_p_ranking_rfm_usc_koch_p_ranking_all":                     "usc_koch",
    "aliangdw_utd_so101_policy_ranking_utd_so101_policy_ranking":               "utd_so101",
    "jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top":     "utd_so101",
    "jesbu1_utd_so101_clean_policy_ranking_wrist_utd_so101_clean_policy_ranking_wrist": "utd_so101",
    "aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking":                 "usc_xarm",
    "aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking":             "usc_franka",
    "ykorkmaz_usc_trossen_rfm_usc_trossen":                                     "usc_trossen",
    # --- From failure archives (Group B split) ---
    "jesbu1_soar_rfm_soar_rfm":                                                 "soar",
    "jesbu1_roboarena_0825_rfm_roboarena":                                      "roboarena",
    "jesbu1_roboarena_eval_debug_nowrist_roboarena_eval_debug_nowrist":         "roboarena",
    # --- Success-pool extras (success-only archives) ---
    "abraranwar_libero_rfm_libero256_10":                                       "libero",
    "abraranwar_libero_rfm_libero256_90":                                       "libero",
    "abraranwar_libero_rfm_libero256_goal":                                     "libero",
    "abraranwar_libero_rfm_libero256_object":                                   "libero",
    "abraranwar_libero_rfm_libero256_spatial":                                  "libero",
    "abraranwar_usc_koch_rewind_rfm_usc_koch_rewind":                          "usc_koch",
    "jesbu1_usc_koch_human_robot_paired_usc_koch_human_robot_paired_robot":     "usc_koch",
    "aliangdw_utd_so101_human_utd_so101_human":                                "utd_so101",
}

N_KEYFRAMES = 16
FPS_DEFAULT = 8.0  # assumed source fps for timestamp labels


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def log(msg):
    print(msg, flush=True)


def open_tar(archive_name: str):
    """Return an open tarfile.TarFile for the archive (single or split)."""
    single_tar = SINGLE_DIR / f"{archive_name}.tar"
    split_dir  = SPLIT_DIR / archive_name

    if single_tar.exists():
        return tarfile.open(single_tar, "r")

    if split_dir.exists():
        parts = sorted(split_dir.glob(f"{archive_name}.tar.part-*"))
        if not parts:
            raise FileNotFoundError(f"No parts for {archive_name}")
        # cat all parts into a pipe, open as streaming tar
        cat = subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE)
        return tarfile.open(fileobj=cat.stdout, mode="r|"), cat

    raise FileNotFoundError(f"Archive not found: {archive_name}")


def build_index_to_uuid(members) -> dict[int, str]:
    """Build {trajectory_index: uuid} from embedding filenames.

    Most archives use hex UUIDs; a few (e.g. ykorkmaz_usc_trossen) use
    task-name identifiers like `ykorkmaz__aloha_stir_pot_ep_000002`. Accept any
    non-empty identifier between the numeric index and `_embeddings.pt`.
    """
    pattern = re.compile(r"trajectory_(\d+)_(.+)_embeddings\.pt$")
    mapping = {}
    for m in members:
        match = pattern.search(m.name)
        if match:
            mapping[int(match.group(1))] = match.group(2)
    return mapping


def build_index_to_task(task_indices: dict) -> dict[int, str]:
    result = {}
    for task, indices in task_indices.items():
        for idx in indices:
            result[idx] = task
    return result


def extract_keyframes(frames_array: np.ndarray, n: int = 16):
    """Sample n evenly spaced frames from (T, H, W, 3) uint8 array."""
    T = frames_array.shape[0]
    if T <= n:
        indices = list(range(T))
        while len(indices) < n:
            indices.append(T - 1)
    else:
        indices = [int(round(i * (T - 1) / (n - 1))) for i in range(n)]
    return [frames_array[i] for i in indices], indices


def save_keyframes(frames, indices, fps, out_dir: Path):
    """Save keyframes as JPEG. Returns list of filenames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fnames = []
    for kf_idx, (frame, src_idx) in enumerate(zip(frames, indices)):
        timestamp = src_idx / fps
        fname = f"frame_{kf_idx}_{timestamp:.2f}s.jpg"
        Image.fromarray(frame).save(out_dir / fname, quality=90)
        fnames.append(fname)
    return fnames


# ────────────────────────────────────────────────────────────────────────────
# Core: extract successes from a single-file (seekable) archive
# ────────────────────────────────────────────────────────────────────────────

def extract_successes_seekable(archive_name, tf, output_dir, manifest_dir, fps, resume_set):
    """Extract success keyframes from a seekable tar (single-file archives)."""
    members = tf.getmembers()

    # Read index_mappings.json
    mapping_member = next(
        (m for m in members if m.name.endswith("index_mappings.json")), None
    )
    if mapping_member is None:
        log(f"  WARNING: index_mappings.json not found — skipping")
        return 0

    index_mappings = json.loads(tf.extractfile(mapping_member).read())
    quality_indices = index_mappings.get("quality_indices", {})
    task_indices = index_mappings.get("task_indices", {})

    success_indices = set(quality_indices.get("successful", []))
    if not success_indices:
        log(f"  No successes in this archive")
        return 0

    log(f"  Found {len(success_indices)} success trajectories")

    idx_to_uuid = build_index_to_uuid(members)
    idx_to_task = build_index_to_task(task_indices)

    # Build uuid → npz member
    uuid_to_member = {}
    for m in members:
        if m.name.endswith(".npz") and "/frames/" in m.name:
            stem = Path(m.name).stem
            uuid = stem.replace("trajectory_", "")
            uuid_to_member[uuid] = m

    manifest_path = manifest_dir / f"{archive_name}_successes.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    with open(manifest_path, "a") as mf:
        for traj_idx in sorted(success_indices):
            uuid = idx_to_uuid.get(traj_idx)
            if uuid is None:
                continue

            episode_id = f"ep_{traj_idx:06d}_{uuid}"
            if episode_id in resume_set:
                skipped += 1
                continue

            task = idx_to_task.get(traj_idx, "unknown task")

            npz_member = uuid_to_member.get(uuid)
            if npz_member is None:
                log(f"  WARNING: NPZ not found for idx {traj_idx} / uuid {uuid}")
                continue

            raw = tf.extractfile(npz_member).read()
            npz = np.load(io.BytesIO(raw), allow_pickle=True)
            frames_array = npz["frames"]

            keyframes, src_indices = extract_keyframes(frames_array, N_KEYFRAMES)

            ep_dir = output_dir / archive_name / episode_id
            save_keyframes(keyframes, src_indices, fps, ep_dir)

            entry = {
                "episode_id": episode_id,
                "archive": archive_name,
                "family": family_of(archive_name),
                "task": task,
                "label": "successful",
                "n_source_frames": int(frames_array.shape[0]),
                "n_keyframes": N_KEYFRAMES,
                "keyframes_dir": f"{archive_name}/{episode_id}",
            }
            mf.write(json.dumps(entry) + "\n")
            mf.flush()

            processed += 1
            if processed % 100 == 0:
                log(f"  {processed} episodes extracted ({skipped} resumed-skipped)")

    return processed


# ────────────────────────────────────────────────────────────────────────────
# Core: extract successes from a streaming (split) archive
# ────────────────────────────────────────────────────────────────────────────

def extract_successes_streaming(archive_name, output_dir, manifest_dir, fps, resume_set):
    """Extract success keyframes from a split-part (streaming) archive.

    Streaming tars cannot seek, so we read index_mappings.json and NPZ files
    in a single pass.
    """
    inner_prefix = f"{archive_name}/"
    split_dir = SPLIT_DIR / archive_name
    parts = sorted(split_dir.glob(f"{archive_name}.tar.part-*"))
    if not parts:
        raise FileNotFoundError(f"No parts for {archive_name}")

    # First pass: read index_mappings.json (small, near start)
    log(f"  Reading index_mappings.json from split archive...")
    cat = subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE)
    inner_path = f"{archive_name}/index_mappings.json"
    tar_proc = subprocess.Popen(
        ["tar", "-xOf", "-", inner_path],
        stdin=cat.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    cat.stdout.close()
    data, _ = tar_proc.communicate()
    cat.wait()
    if not data:
        raise RuntimeError(f"Failed to extract index_mappings.json from {archive_name}")

    index_mappings = json.loads(data)
    quality_indices = index_mappings.get("quality_indices", {})
    task_indices = index_mappings.get("task_indices", {})

    success_indices = set(quality_indices.get("successful", []))
    if not success_indices:
        log(f"  No successes in this archive")
        return 0

    log(f"  Found {len(success_indices)} success trajectories")
    idx_to_task = build_index_to_task(task_indices)

    # Loose pattern to also match task-name identifiers (e.g. usc_trossen).
    traj_pattern = re.compile(r"trajectory_(\d+)_(.+)_embeddings\.pt$")
    npz_pattern = re.compile(r"/frames/trajectory_(.+)\.npz$")

    manifest_path = manifest_dir / f"{archive_name}_successes.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: scan embedding filenames only to build uuid->idx map (no data read)
    log(f"  Pass 1: scanning embedding filenames for uuid->idx map...")
    cat = subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE)
    tf = tarfile.open(fileobj=cat.stdout, mode="r|")
    uuid_to_idx = {}
    for member in tf:
        m = traj_pattern.search(member.name)
        if m:
            uuid_to_idx[m.group(2)] = int(m.group(1))
    tf.close()
    cat.wait()
    log(f"  Pass 1 done: {len(uuid_to_idx)} trajectories mapped")

    # Pass 2: stream NPZs; extract only for success UUIDs. No buffering.
    log(f"  Pass 2: streaming NPZs and extracting success keyframes...")
    cat = subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE)
    tf = tarfile.open(fileobj=cat.stdout, mode="r|")
    processed = 0
    with open(manifest_path, "a") as mf:
        for member in tf:
            npz_match = npz_pattern.search(member.name)
            if not (npz_match and member.isfile()):
                continue
            tuuid = npz_match.group(1)
            tidx = uuid_to_idx.get(tuuid)
            if tidx is None or tidx not in success_indices:
                continue
            episode_id = f"ep_{tidx:06d}_{tuuid}"
            if episode_id in resume_set:
                continue
            raw = tf.extractfile(member).read()
            npz = np.load(io.BytesIO(raw), allow_pickle=True)
            frames_array = npz["frames"]
            task = idx_to_task.get(tidx, "unknown task")
            keyframes, src_indices = extract_keyframes(frames_array, N_KEYFRAMES)
            ep_dir = output_dir / archive_name / episode_id
            save_keyframes(keyframes, src_indices, fps, ep_dir)
            entry = {
                "episode_id": episode_id,
                "archive": archive_name,
                "family": family_of(archive_name),
                "task": task,
                "label": "successful",
                "n_source_frames": int(frames_array.shape[0]),
                "n_keyframes": N_KEYFRAMES,
                "keyframes_dir": f"{archive_name}/{episode_id}",
            }
            mf.write(json.dumps(entry) + "\n")
            mf.flush()
            del raw, npz, frames_array, keyframes
            processed += 1
            if processed % 100 == 0:
                log(f"  {processed} episodes extracted")
    tf.close()
    cat.wait()

    return processed


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def process_archive(archive_name, output_dir, manifest_dir, fps):
    """Extract successes from one archive. Returns count of new episodes."""
    log(f"\n{'='*60}")
    log(f"Archive: {archive_name} [{SUCCESS_ARCHIVES.get(archive_name, '?')}]")
    log(f"{'='*60}")

    # Load resume set
    manifest_path = manifest_dir / f"{archive_name}_successes.jsonl"
    resume_set = set()
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                entry = json.loads(line)
                resume_set.add(entry["episode_id"])
        log(f"  Resuming: {len(resume_set)} episodes already done")

    single_tar = SINGLE_DIR / f"{archive_name}.tar"
    split_dir  = SPLIT_DIR / archive_name

    if single_tar.exists():
        log(f"  Opening single tar: {single_tar}")
        with tarfile.open(single_tar, "r") as tf:
            return extract_successes_seekable(archive_name, tf, output_dir, manifest_dir, fps, resume_set)
    elif split_dir.exists():
        log(f"  Processing split archive: {split_dir}")
        return extract_successes_streaming(archive_name, output_dir, manifest_dir, fps, resume_set)
    else:
        log(f"  ERROR: Archive not found at {single_tar} or {split_dir}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Extract 16 uniform keyframes from Robometer success episodes"
    )
    parser.add_argument(
        "--archive", required=True,
        help="Archive name, or 'all' to process every success archive"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Keyframes output directory (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--manifest-dir", type=str,
        default=str(DEFAULT_MANIFEST_DIR),
        help=f"Manifest directory (default: {DEFAULT_MANIFEST_DIR})"
    )
    parser.add_argument(
        "--fps", type=float, default=FPS_DEFAULT,
        help=f"Assumed source fps for timestamp labels (default: {FPS_DEFAULT})"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifest_dir = Path(args.manifest_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    if args.archive == "all":
        archives = list(SUCCESS_ARCHIVES.keys())
    else:
        if args.archive not in SUCCESS_ARCHIVES:
            log(f"WARNING: {args.archive} not in SUCCESS_ARCHIVES registry, proceeding anyway")
        archives = [args.archive]

    total = 0
    for arch in archives:
        try:
            n = process_archive(arch, output_dir, manifest_dir, args.fps)
            total += n
            log(f"  -> {n} new success episodes extracted")
        except Exception as e:
            import traceback
            log(f"  ERROR processing {arch}: {e}")
            traceback.print_exc()

    log(f"\n{'='*60}")
    log(f"Done. Total new success episodes: {total}")
    log(f"Output: {output_dir}/")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
