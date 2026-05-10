#!/usr/bin/env python3
"""
extract_orphan_successes.py — Extract 16 uniformly-spaced keyframes from every
ORPHAN success episode in the Robometer dataset, and write them as
WebDataset-style tar shards (NOT loose JPEGs).

Why shards: loose JPEGs would add ~5.5M inodes to the prjs1958 project
allocation, which has a 1.9M-inode hard cap. Packing ~256 episodes per shard
collapses that to ~1,400 inodes total while preserving the same bytes.

An "orphan success" is a success trajectory whose archive contains zero
failures AND whose robot family is not represented in any failure archive.

This script targets the 1,255,963 orphan successes identified by
audit_all_archives.py, MINUS:
  • ~6,186 already extracted by extract_successes.py (libero256, usc_koch_rewind,
    usc_koch_paired_robot, utd_so101_human — their family has failures elsewhere)
  • ~366,618 human-only / human-hand archives (egodex, epic, rh20t_human, h2r)
  • ~541,939 humanoid-robot archives (agibotworld ×2, galaxea r1_lite ×5)

Net target: ~341,220 episodes across 28 archives (mostly OXE robot arms).

Output layout:
    /projects/prjs1958/robometer_frame_dataset/robometer/
      keyframes_orphan_success/
        <archive_name>/
          shard-00000.tar        ← contains 256 episodes × (16 jpgs + 1 json)
          shard-00001.tar
          ...
      manifests/
        <archive_name>_orphan_successes.jsonl

Inside each shard (WebDataset convention — one key per episode):
    ep_000123_<uuid>/frame_00_0.00s.jpg
    ep_000123_<uuid>/frame_01_0.12s.jpg
    ...
    ep_000123_<uuid>/frame_15_1.88s.jpg
    ep_000123_<uuid>/meta.json        ← {task, family, n_source_frames, fps, ...}

Downstream loading example (PyTorch):
    import webdataset as wds
    ds = wds.WebDataset("shard-{00000..00050}.tar").decode("pil")
    # each sample is a dict {"frame_00_0.00s.jpg": PIL.Image, ..., "meta.json": dict}

Resume-safe: episodes already in the manifest are skipped. Extraction resumes
into a NEW shard (partial shards are not re-opened; small-overhead safety).

Usage:
    # count only (no extraction) — verifies expected numbers
    python extract_orphan_successes.py --count-only

    # extract a single archive
    python extract_orphan_successes.py --archive jesbu1_oxe_rfm_oxe_bridge_v2

    # extract every orphan archive in registry
    python extract_orphan_successes.py --archive all
"""

import argparse
import io
import json
import re
import subprocess
import tarfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from robometer_families import family_of

# ────────────────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────────────────

ARCH_ROOT = Path("/projects/prjs1958/robometer_full_dataset/raw_archives")
SINGLE_DIR = ARCH_ROOT / "single"
SPLIT_DIR  = ARCH_ROOT / "split"
AUDIT_REPORT = Path("/projects/prjs1958/robometer_full_dataset/audit_report.json")

DEFAULT_OUTPUT = Path("/projects/prjs1958/robometer_frame_dataset/robometer/keyframes_orphan_success")
DEFAULT_MANIFEST_DIR = Path("/projects/prjs1958/robometer_frame_dataset/robometer/manifests")

# Group-specific output roots (mirror robometer/'s subfolder layout).
GROUP_ROOTS = {
    "robot":      Path("/projects/prjs1958/robometer_frame_dataset/robometer"),
    "humanoid":   Path("/projects/prjs1958/robometer_frame_dataset/humanoid"),
    "human_hand": Path("/projects/prjs1958/robometer_frame_dataset/human_hand"),
}

N_KEYFRAMES = 16
FPS_DEFAULT = 8.0
EPISODES_PER_SHARD = 256  # ≈ 4096 JPEGs ≈ 50 MB at quality=90

# ────────────────────────────────────────────────────────────────────────────
# Orphan archive registry — maps archive_name -> robot family.
# ────────────────────────────────────────────────────────────────────────────

# Humanoid + human-hand registries — extracted under separate top-level
# subfolders (humanoid/ and human_hand/) but reuse the same orphan-success
# format as robometer/.
HUMANOID_ARCHIVES = {
    "jesbu1_galaxea_rfm_galaxea_part1_r1_lite":                                   "galaxea_r1_lite",
    "jesbu1_galaxea_rfm_galaxea_part2_r1_lite":                                   "galaxea_r1_lite",
    "jesbu1_galaxea_rfm_galaxea_part3_r1_lite":                                   "galaxea_r1_lite",
    "jesbu1_galaxea_rfm_galaxea_part4_r1_lite":                                   "galaxea_r1_lite",
    "jesbu1_galaxea_rfm_galaxea_part5_r1_lite":                                   "galaxea_r1_lite",
    "jesbu1_humanoid_everyday_rfm_humanoid_everyday_rfm":                         "humanoid_everyday",
}

HUMAN_HAND_ARCHIVES = {
    "jesbu1_egodex_rfm_egodex_part1":                                             "egodex",
    "jesbu1_egodex_rfm_egodex_test":                                              "egodex",
    "jesbu1_epic_rfm_epic":                                                       "epic",
    "anqil_rh20t_subset_rfm_rh20t_human":                                         "rh20t_human",
    "jesbu1_h2r_rfm_h2r":                                                         "h2r",
    "jesbu1_usc_koch_human_robot_paired_usc_koch_human_robot_paired_human":       "usc_koch_human",
    "jesbu1_hand_paired_rfm_hand_paired_human":                                   "hand_paired_human",
}

ORPHAN_ARCHIVES = {
    # --- OXE robot-arm subset ---
    "jesbu1_oxe_rfm_oxe_aloha_mobile":                                            "oxe_aloha_mobile",
    "jesbu1_oxe_rfm_oxe_austin_buds_dataset_converted_externally_to_rlds":        "oxe_austin_buds",
    "jesbu1_oxe_rfm_oxe_bc_z":                                                    "oxe_bc_z",
    "jesbu1_oxe_rfm_oxe_berkeley_cable_routing":                                  "oxe_berkeley_cable_routing",
    "jesbu1_oxe_rfm_oxe_berkeley_fanuc_manipulation":                             "oxe_berkeley_fanuc",
    "jesbu1_oxe_rfm_oxe_berkeley_mvp_converted_externally_to_rlds":               "oxe_berkeley_mvp",
    "jesbu1_oxe_rfm_oxe_berkeley_rpt_converted_externally_to_rlds":               "oxe_berkeley_rpt",
    "jesbu1_oxe_rfm_oxe_bridge_v2":                                               "oxe_bridge_v2",
    "jesbu1_oxe_rfm_eval_oxe_bridge_v2_eval":                                     "oxe_bridge_v2",
    "jesbu1_oxe_rfm_oxe_dlr_edan_shared_control_converted_externally_to_rlds":    "oxe_dlr_edan",
    "jesbu1_oxe_rfm_oxe_fractal20220817_data":                                    "oxe_fractal",
    "jesbu1_oxe_rfm_oxe_furniture_bench_dataset_converted_externally_to_rlds":    "oxe_furniture_bench",
    "jesbu1_oxe_rfm_oxe_iamlab_cmu_pickup_insert_converted_externally_to_rlds":   "oxe_iamlab_cmu",
    "jesbu1_oxe_rfm_oxe_imperialcollege_sawyer_wrist_cam":                        "oxe_imperial_sawyer",
    "jesbu1_oxe_rfm_oxe_jaco_play":                                               "oxe_jaco_play",
    "jesbu1_oxe_rfm_oxe_language_table":                                          "oxe_language_table",
    "jesbu1_oxe_rfm_oxe_nyu_rot_dataset_converted_externally_to_rlds":            "oxe_nyu_rot",
    "jesbu1_oxe_rfm_oxe_robo_set":                                                "oxe_robo_set",
    "jesbu1_oxe_rfm_oxe_stanford_hydra_dataset_converted_externally_to_rlds":     "oxe_stanford_hydra",
    "jesbu1_oxe_rfm_oxe_tokyo_u_lsmo_converted_externally_to_rlds":               "oxe_tokyo_lsmo",
    "jesbu1_oxe_rfm_oxe_toto":                                                    "oxe_toto",
    "jesbu1_oxe_rfm_oxe_ucsd_kitchen_dataset_converted_externally_to_rlds":       "oxe_ucsd_kitchen",
    "jesbu1_oxe_rfm_oxe_utaustin_mutex":                                          "oxe_utaustin_mutex",

    # --- DROID orphan successes (149,804) — distinct from user's droid/
    #     subtree which holds their own downloaded failures+matched-successes.
    "jesbu1_oxe_rfm_oxe_droid":                                                   "droid",

    # --- RH20T robot half (paired dataset; human half excluded) ---
    "anqil_rh20t_subset_rfm_rh20t_robot":                                         "rh20t",

    # --- MolmoACT (Franka Panda) ---
    "jesbu1_molmoact_rfm_molmoact_dataset_household":                             "molmoact",
    "jesbu1_molmoact_rfm_molmoact_dataset_tabletop":                              "molmoact",

    # --- Misc small paired/single datasets ---
    "jesbu1_motif_rfm_motif_rfm":                                                 "motif",
    "jesbu1_fino_net_rfm_fino_net":                                               "fino_net",
}

EXCLUDED_HUMAN = {
    "jesbu1_egodex_rfm_egodex_part1", "jesbu1_egodex_rfm_egodex_part2",
    "jesbu1_egodex_rfm_egodex_part3", "jesbu1_egodex_rfm_egodex_part4",
    "jesbu1_egodex_rfm_egodex_part5", "jesbu1_egodex_rfm_egodex_test",
    "jesbu1_epic_rfm_epic",
    "anqil_rh20t_subset_rfm_rh20t_human",
    "jesbu1_h2r_rfm_h2r",
}

EXCLUDED_HUMANOID = {
    "abraranwar_agibotworld_alpha_rfm_agibotworld",
    "abraranwar_agibotworld_alpha_headcam_rfm_agibotworld",
    "jesbu1_galaxea_rfm_galaxea_part1_r1_lite",
    "jesbu1_galaxea_rfm_galaxea_part2_r1_lite",
    "jesbu1_galaxea_rfm_galaxea_part3_r1_lite",
    "jesbu1_galaxea_rfm_galaxea_part4_r1_lite",
    "jesbu1_galaxea_rfm_galaxea_part5_r1_lite",
}

EXCLUDED_ALREADY_EXTRACTED = {
    "abraranwar_libero_rfm_libero256_10",
    "abraranwar_libero_rfm_libero256_90",
    "abraranwar_libero_rfm_libero256_goal",
    "abraranwar_libero_rfm_libero256_object",
    "abraranwar_libero_rfm_libero256_spatial",
    "abraranwar_usc_koch_rewind_rfm_usc_koch_rewind",
    "jesbu1_usc_koch_human_robot_paired_usc_koch_human_robot_paired_robot",
    "aliangdw_utd_so101_human_utd_so101_human",
}


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def log(msg):
    print(msg, flush=True)


def build_index_to_uuid(members):
    pattern = re.compile(r"trajectory_(\d+)_(.+)_embeddings\.pt$")
    mapping = {}
    for m in members:
        match = pattern.search(m.name)
        if match:
            mapping[int(match.group(1))] = match.group(2)
    return mapping


def build_index_to_task(task_indices):
    result = {}
    for task, indices in task_indices.items():
        for idx in indices:
            result[idx] = task
    return result


def extract_keyframes(frames_array, n=N_KEYFRAMES):
    T = frames_array.shape[0]
    if T <= n:
        indices = list(range(T))
        while len(indices) < n:
            indices.append(T - 1)
    else:
        indices = [int(round(i * (T - 1) / (n - 1))) for i in range(n)]
    return [frames_array[i] for i in indices], indices


# ────────────────────────────────────────────────────────────────────────────
# ShardWriter — packs episodes into tar shards
# ────────────────────────────────────────────────────────────────────────────

class ShardWriter:
    """Write episodes (each = 16 JPEGs + 1 meta.json) into tar shards.

    Sharding is determined by the episode count already in the shard directory
    plus the number of episodes we are about to write. Partial shards are
    rotated at EPISODES_PER_SHARD.
    """

    def __init__(self, shard_dir: Path, episodes_per_shard: int = EPISODES_PER_SHARD):
        self.shard_dir = shard_dir
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_per_shard = episodes_per_shard
        # Start a fresh shard for this run (any partial shard from a previous
        # run stays intact; resume logic skips its episodes via the manifest).
        existing = sorted(self.shard_dir.glob("shard-*.tar"))
        self.shard_idx = len(existing)
        self.current_tar = None
        self.current_count = 0
        self._open_new_shard()

    def _open_new_shard(self):
        if self.current_tar is not None:
            self.current_tar.close()
        path = self.shard_dir / f"shard-{self.shard_idx:05d}.tar"
        self.current_tar = tarfile.open(path, "w")
        self.current_count = 0
        log(f"  Opened {path.name}")

    def write_episode(self, episode_id: str, keyframes, src_indices, fps: float, meta: dict):
        if self.current_count >= self.episodes_per_shard:
            self.shard_idx += 1
            self._open_new_shard()

        for kf_idx, (frame, src_idx) in enumerate(zip(keyframes, src_indices)):
            timestamp = src_idx / fps
            fname = f"{episode_id}/frame_{kf_idx:02d}_{timestamp:.2f}s.jpg"
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, format="JPEG", quality=90)
            data = buf.getvalue()
            info = tarfile.TarInfo(name=fname)
            info.size = len(data)
            info.mtime = int(time.time())
            self.current_tar.addfile(info, io.BytesIO(data))

        meta_bytes = json.dumps(meta).encode("utf-8")
        info = tarfile.TarInfo(name=f"{episode_id}/meta.json")
        info.size = len(meta_bytes)
        info.mtime = int(time.time())
        self.current_tar.addfile(info, io.BytesIO(meta_bytes))

        self.current_count += 1

    def close(self):
        if self.current_tar is not None:
            self.current_tar.close()
            self.current_tar = None


# ────────────────────────────────────────────────────────────────────────────
# Extraction (seekable single-file archives)
# ────────────────────────────────────────────────────────────────────────────

def extract_seekable(archive_name, tf, output_dir, manifest_path, fps, resume_set,
                     max_episodes=None):
    members = tf.getmembers()
    mapping_member = next((m for m in members if m.name.endswith("index_mappings.json")), None)
    if mapping_member is None:
        log("  WARNING: index_mappings.json not found — skipping")
        return 0

    index_mappings = json.loads(tf.extractfile(mapping_member).read())
    quality_indices = index_mappings.get("quality_indices", {})
    task_indices = index_mappings.get("task_indices", {})

    success_indices = set(quality_indices.get("successful", []))
    if not success_indices:
        log("  No successes in this archive")
        return 0

    log(f"  Found {len(success_indices)} success trajectories")
    idx_to_uuid = build_index_to_uuid(members)
    idx_to_task = build_index_to_task(task_indices)

    uuid_to_member = {}
    for m in members:
        if m.name.endswith(".npz") and "/frames/" in m.name:
            uuid = Path(m.name).stem.replace("trajectory_", "")
            uuid_to_member[uuid] = m

    shard_dir = output_dir / archive_name
    writer = ShardWriter(shard_dir)
    processed = skipped = 0

    try:
        with open(manifest_path, "a") as mf:
            for traj_idx in sorted(success_indices):
                uuid = idx_to_uuid.get(traj_idx)
                if uuid is None:
                    continue
                episode_id = f"ep_{traj_idx:06d}_{uuid}"
                if episode_id in resume_set:
                    skipped += 1
                    continue

                npz_member = uuid_to_member.get(uuid)
                if npz_member is None:
                    log(f"  WARNING: NPZ not found for idx {traj_idx} / uuid {uuid}")
                    continue

                raw = tf.extractfile(npz_member).read()
                npz = np.load(io.BytesIO(raw), allow_pickle=True)
                frames_array = npz["frames"]
                keyframes, src_indices = extract_keyframes(frames_array, N_KEYFRAMES)

                meta = {
                    "episode_id": episode_id,
                    "archive": archive_name,
                    "family": family_of(archive_name),
                    "task": idx_to_task.get(traj_idx, "unknown task"),
                    "label": "successful",
                    "orphan": True,
                    "n_source_frames": int(frames_array.shape[0]),
                    "n_keyframes": N_KEYFRAMES,
                    "fps": fps,
                }
                writer.write_episode(episode_id, keyframes, src_indices, fps, meta)

                manifest_entry = {
                    **meta,
                    "shard": f"shard-{writer.shard_idx:05d}.tar",
                }
                mf.write(json.dumps(manifest_entry) + "\n")
                mf.flush()

                processed += 1
                if processed % 100 == 0:
                    log(f"  {processed} extracted ({skipped} resumed-skipped)")
                if max_episodes is not None and processed >= max_episodes:
                    log(f"  reached --max-episodes={max_episodes}, stopping")
                    break
    finally:
        writer.close()

    return processed


# ────────────────────────────────────────────────────────────────────────────
# Extraction (streaming split-part archives)
# ────────────────────────────────────────────────────────────────────────────

def extract_streaming(archive_name, output_dir, manifest_path, fps, resume_set,
                      max_episodes=None):
    split_dir = SPLIT_DIR / archive_name
    parts = sorted(split_dir.glob(f"{archive_name}.tar.part-*"))
    if not parts:
        raise FileNotFoundError(f"No parts for {archive_name}")

    log("  Reading index_mappings.json from split archive...")
    inner_path = f"{archive_name}/index_mappings.json"
    cat = subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE)
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
        log("  No successes in this archive")
        return 0

    log(f"  Found {len(success_indices)} success trajectories")
    idx_to_task = build_index_to_task(task_indices)

    traj_pattern = re.compile(r"trajectory_(\d+)_(.+)_embeddings\.pt$")
    npz_pattern  = re.compile(r"/frames/trajectory_(.+)\.npz$")

    log("  Pass 1: scanning embedding filenames for uuid->idx map...")
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

    log("  Pass 2: streaming NPZs and extracting orphan-success keyframes...")
    cat = subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE)
    tf = tarfile.open(fileobj=cat.stdout, mode="r|")

    shard_dir = output_dir / archive_name
    writer = ShardWriter(shard_dir)
    processed = 0

    try:
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
                keyframes, src_indices = extract_keyframes(frames_array, N_KEYFRAMES)

                meta = {
                    "episode_id": episode_id,
                    "archive": archive_name,
                    "family": family_of(archive_name),
                    "task": idx_to_task.get(tidx, "unknown task"),
                    "label": "successful",
                    "orphan": True,
                    "n_source_frames": int(frames_array.shape[0]),
                    "n_keyframes": N_KEYFRAMES,
                    "fps": fps,
                }
                writer.write_episode(episode_id, keyframes, src_indices, fps, meta)

                manifest_entry = {
                    **meta,
                    "shard": f"shard-{writer.shard_idx:05d}.tar",
                }
                mf.write(json.dumps(manifest_entry) + "\n")
                mf.flush()
                del raw, npz, frames_array, keyframes
                processed += 1
                if processed % 100 == 0:
                    log(f"  {processed} extracted")
                if max_episodes is not None and processed >= max_episodes:
                    log(f"  reached --max-episodes={max_episodes}, stopping early")
                    break
    finally:
        writer.close()
        tf.close()
        cat.terminate()
        try:
            cat.wait(timeout=5)
        except Exception:
            cat.kill()

    return processed


# ────────────────────────────────────────────────────────────────────────────
# Per-archive driver
# ────────────────────────────────────────────────────────────────────────────

def process_archive(archive_name, output_dir, manifest_dir, fps, max_episodes=None):
    log(f"\n{'='*60}")
    log(f"Archive: {archive_name} [{family_of(archive_name)}]"
        + (f"  (cap {max_episodes})" if max_episodes else ""))
    log(f"{'='*60}")

    manifest_path = manifest_dir / f"{archive_name}_orphan_successes.jsonl"
    resume_set = set()
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                resume_set.add(json.loads(line)["episode_id"])
        log(f"  Resuming: {len(resume_set)} episodes already in manifest")

    # Adjust the cap by what's already been written so we extract exactly
    # max_episodes total (resume + new), not max_episodes more.
    remaining = None
    if max_episodes is not None:
        remaining = max(0, max_episodes - len(resume_set))
        if remaining == 0:
            log(f"  cap already met by resume_set ({len(resume_set)} >= {max_episodes})")
            return 0

    single_tar = SINGLE_DIR / f"{archive_name}.tar"
    split_dir  = SPLIT_DIR / archive_name

    if single_tar.exists():
        log(f"  Opening single tar: {single_tar}")
        with tarfile.open(single_tar, "r") as tf:
            return extract_seekable(archive_name, tf, output_dir, manifest_path,
                                     fps, resume_set, max_episodes=remaining)
    if split_dir.exists():
        log(f"  Processing split archive: {split_dir}")
        return extract_streaming(archive_name, output_dir, manifest_path,
                                  fps, resume_set, max_episodes=remaining)

    log(f"  ERROR: Archive not found at {single_tar} or {split_dir}")
    return 0


# ────────────────────────────────────────────────────────────────────────────
# Counting (uses audit_report.json — no archive scan needed)
# ────────────────────────────────────────────────────────────────────────────

def count_orphans():
    """Print orphan-success breakdown using audit_report.json."""
    if not AUDIT_REPORT.exists():
        log(f"ERROR: {AUDIT_REPORT} not found. Run audit_all_archives.py first.")
        return

    audit = json.loads(AUDIT_REPORT.read_text())
    orphans = {row["archive"]: row["successes"] for row in audit["orphan_successes"]}
    total_orphan = audit.get("orphan_success_total", sum(orphans.values()))

    excluded_human    = sum(orphans.get(a, 0) for a in EXCLUDED_HUMAN)
    excluded_humanoid = sum(orphans.get(a, 0) for a in EXCLUDED_HUMANOID)
    excluded_already  = sum(orphans.get(a, 0) for a in EXCLUDED_ALREADY_EXTRACTED)
    target_total      = sum(orphans.get(a, 0) for a in ORPHAN_ARCHIVES)

    sanity = excluded_human + excluded_humanoid + excluded_already + target_total

    log(f"\n{'='*64}")
    log("ORPHAN SUCCESS BREAKDOWN  (from audit_report.json)")
    log(f"{'='*64}")
    log(f"  Total orphan successes (audit):                {total_orphan:>10,}")
    log(f"  Excluded — human-only / human-hand archives:   {excluded_human:>10,}  ({len(EXCLUDED_HUMAN)} archives)")
    log(f"  Excluded — humanoid-robot archives:            {excluded_humanoid:>10,}  ({len(EXCLUDED_HUMANOID)} archives)")
    log(f"  Excluded — already extracted (family-orphans): {excluded_already:>10,}  ({len(EXCLUDED_ALREADY_EXTRACTED)} archives)")
    log(f"  → Target for extraction:                       {target_total:>10,}  ({len(ORPHAN_ARCHIVES)} archives)")
    log(f"  Sanity (sum of the four above):                {sanity:>10,}")
    if sanity != total_orphan:
        log(f"  ⚠  Mismatch of {total_orphan - sanity}: registry may be out of sync with audit.")
    log("")
    log("Per-archive (target only, sorted desc):")
    for arch, fam in sorted(ORPHAN_ARCHIVES.items(), key=lambda kv: -orphans.get(kv[0], 0)):
        n = orphans.get(arch, 0)
        log(f"  {n:>8,}  [{fam:<28}]  {arch}")
    log(f"{'='*64}\n")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract 16 uniform keyframes from Robometer ORPHAN-success episodes into tar shards"
    )
    parser.add_argument(
        "--archive",
        help="Archive name from the selected --group registry, or 'all'. Required unless --count-only.",
    )
    parser.add_argument(
        "--group", choices=sorted(GROUP_ROOTS.keys()), default="robot",
        help="Archive group: 'robot' (default, OXE arms etc.), 'humanoid', or 'human_hand'. "
             "Switches the archive registry AND the output/manifest roots."
    )
    parser.add_argument(
        "--count-only", action="store_true",
        help="Print orphan-success breakdown from audit_report.json and exit."
    )
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output dir (default: <group_root>/keyframes_orphan_success).")
    parser.add_argument("--manifest-dir", type=str, default=None,
                        help="Override manifest dir (default: <group_root>/manifests).")
    parser.add_argument("--fps", type=float, default=FPS_DEFAULT)
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Per-archive cap (resume-aware). For split tars this enables "
                             "early-exit during pass 2, which can shave significant wall-time "
                             "on huge archives like epic (206 GB).")
    args = parser.parse_args()

    if args.count_only:
        count_orphans()
        return

    if not args.archive:
        parser.error("--archive is required (or use --count-only)")

    # Pick registry + default paths from --group.
    registry = {
        "robot":      ORPHAN_ARCHIVES,
        "humanoid":   HUMANOID_ARCHIVES,
        "human_hand": HUMAN_HAND_ARCHIVES,
    }[args.group]
    group_root = GROUP_ROOTS[args.group]
    output_dir = Path(args.output_dir) if args.output_dir else group_root / "keyframes_orphan_success"
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else group_root / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    if args.archive == "all":
        archives = list(registry.keys())
    else:
        if args.archive not in registry:
            log(f"ERROR: {args.archive} not in {args.group} registry. "
                f"Valid archives for group={args.group}: {sorted(registry)}")
            return
        archives = [args.archive]

    total = 0
    for arch in archives:
        try:
            n = process_archive(arch, output_dir, manifest_dir, args.fps,
                                max_episodes=args.max_episodes)
            total += n
            log(f"  -> {n} new orphan-success episodes extracted")
        except Exception as e:
            import traceback
            log(f"  ERROR processing {arch}: {e}")
            traceback.print_exc()

    log(f"\n{'='*60}")
    log(f"Done. Total new orphan-success episodes: {total}")
    log(f"Output: {output_dir}/")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
