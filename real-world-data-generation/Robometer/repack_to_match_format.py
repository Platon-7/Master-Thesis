#!/usr/bin/env python3
"""
repack_to_match_format.py — Repack already-migrated shard tars so they match
the format produced by extract_orphan_successes.py:

  • frame names go from `frame_N_T.TTs.jpg`  →  `frame_NN_T.TTs.jpg`
    (zero-padded to 2 digits so alphabetical = numerical order)
  • each episode gets a `<episode_id>/meta.json` synthesized from the
    archive's source metadata (index_mappings.json for Robometer; the
    droid success/failure indexes for the droid keyframe roots)

Usage:
    python repack_to_match_format.py --shard-dir <dir> --kind {robometer,droid} \
        [--label successful|failure] [--root-name keyframes_success]

The script processes every `shard-*.tar` under --shard-dir, writes each one
to `<name>.tar.new`, then atomically renames over the original. Resume-safe:
re-running on already-converted shards is a no-op (detected by the presence
of meta.json in the first episode).
"""
from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

from robometer_families import family_of

# ────────────────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────────────────

ARCH_ROOT = Path("/projects/prjs1958/robometer_full_dataset/raw_archives")
SINGLE_DIR = ARCH_ROOT / "single"
SPLIT_DIR  = ARCH_ROOT / "split"

DROID_META = Path("/projects/prjs1958/robometer_frame_dataset/droid/metadata")
DROID_SUCCESS_INDEX = DROID_META / "success_index.jsonl"
DROID_FAILURE_INDEX = DROID_META / "failure_index.jsonl"

FPS_DEFAULT = 8.0  # consistent with extract_orphan_successes.py / extract_successes.py

# Family lookup comes from robometer_families.FAMILY_REGISTRY (single source
# of truth shared with extract_successes.py / extract_orphan_successes.py).

# ────────────────────────────────────────────────────────────────────────────
# Source-metadata loaders
# ────────────────────────────────────────────────────────────────────────────

def log(msg):
    print(msg, flush=True)


def read_index_mappings(archive_name: str) -> Optional[dict]:
    candidates = [
        f"{archive_name}/index_mappings.json",
        f"./{archive_name}/index_mappings.json",
        f"processed_datasets/{archive_name}/index_mappings.json",
        f"./processed_datasets/{archive_name}/index_mappings.json",
    ]
    single_tar = SINGLE_DIR / f"{archive_name}.tar"
    split_dir  = SPLIT_DIR / archive_name

    if single_tar.exists():
        with tarfile.open(single_tar, "r") as t:
            for p in candidates:
                try:
                    f = t.extractfile(p)
                    if f is not None:
                        return json.loads(f.read())
                except KeyError:
                    continue
        return None

    if split_dir.exists():
        parts = sorted(split_dir.glob(f"{archive_name}.tar.part-*"))
        if not parts:
            return None
        for p in candidates:
            cat = subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE)
            tar = subprocess.Popen(
                ["tar", "--occurrence=1", "-xOf", "-", p],
                stdin=cat.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            cat.stdout.close()
            data = tar.stdout.read()
            tar.wait()
            cat.terminate()
            cat.wait()
            if data:
                return json.loads(data)
        return None
    return None


def build_robometer_idx_to_task(archive_name: str) -> Dict[int, str]:
    idx = read_index_mappings(archive_name)
    if idx is None:
        log(f"  WARNING: index_mappings.json not found for {archive_name} — task will be 'unknown'")
        return {}
    out = {}
    for task, indices in idx.get("task_indices", {}).items():
        for i in indices:
            out[i] = task
    return out


def load_droid_metadata() -> Dict[str, dict]:
    """episode_id -> {task, lab, gcs_prefix, scene_id, ...}"""
    out: Dict[str, dict] = {}
    for path, label in [(DROID_SUCCESS_INDEX, "successful"), (DROID_FAILURE_INDEX, "failure")]:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                ep = r.get("episode_id")
                if not ep:
                    continue
                out[ep] = {
                    "task": r.get("current_task", "unknown task"),
                    "lab": r.get("lab"),
                    "gcs_prefix": r.get("gcs_prefix"),
                    "scene_id": r.get("scene_id"),
                    "uuid": r.get("uuid"),
                    "label_inferred": label,
                }
    return out


# ────────────────────────────────────────────────────────────────────────────
# Frame-name conversion
# ────────────────────────────────────────────────────────────────────────────

FRAME_RE = re.compile(r"^(?P<dir>[^/]+)/frame_(?P<idx>\d+)_(?P<ts>[\d.]+)s\.jpg$")


def repad_frame_name(name: str) -> Optional[tuple]:
    """Returns (new_name, episode_dir, idx) or None if not a frame entry."""
    m = FRAME_RE.match(name)
    if not m:
        return None
    ep = m.group("dir")
    idx = int(m.group("idx"))
    ts  = m.group("ts")
    new_name = f"{ep}/frame_{idx:02d}_{ts}s.jpg"
    return new_name, ep, idx, float(ts)


# ────────────────────────────────────────────────────────────────────────────
# Episode metadata builders
# ────────────────────────────────────────────────────────────────────────────

EP_ROBOMETER_RE = re.compile(r"^ep_(\d{6})_(.+)$")  # ep_NNNNNN_<UUID>


def split_droid_ep_dir(ep_dir: str, droid_lookup: Dict[str, dict]) -> tuple:
    """Robust split of `<episode_id>__<task_slug>` — episode_id itself can
    contain `__` (e.g. `Fri_Jul__7_...`), so we try every `__` split point
    and pick the first whose LHS is a known droid episode_id."""
    parts = ep_dir.split("__")
    for cut in range(1, len(parts)):
        candidate = "__".join(parts[:cut])
        if candidate in droid_lookup:
            return candidate, "__".join(parts[cut:])
    # Fallback: split on LAST `__` (assume task slug has no `__`)
    if "__" in ep_dir:
        i = ep_dir.rfind("__")
        return ep_dir[:i], ep_dir[i + 2:]
    return ep_dir, None


def build_episode_meta_robometer(ep_dir: str, archive: str, label: str,
                                  idx_to_task: Dict[int, str], n_keyframes: int,
                                  last_ts: float, fps: float) -> dict:
    m = EP_ROBOMETER_RE.match(ep_dir)
    traj_idx = int(m.group(1)) if m else None
    uuid     = m.group(2) if m else None
    task = idx_to_task.get(traj_idx, "unknown task") if traj_idx is not None else "unknown task"
    n_source_est = int(round(last_ts * fps)) + 1 if last_ts is not None else None
    return {
        "episode_id": ep_dir,
        "archive": archive,
        "family": family_of(archive),
        "task": task,
        "label": label,
        "orphan": False,
        "n_source_frames": n_source_est,
        "n_keyframes": n_keyframes,
        "fps": fps,
        "trajectory_index": traj_idx,
        "uuid": uuid,
    }


def build_episode_meta_droid(ep_dir: str, archive: str, root_name: str, label: str,
                              droid_lookup: Dict[str, dict], n_keyframes: int,
                              last_ts: float, fps: float) -> dict:
    episode_id, task_slug = split_droid_ep_dir(ep_dir, droid_lookup)
    info = droid_lookup.get(episode_id, {})
    n_source_est = int(round(last_ts * fps)) + 1 if last_ts is not None else None
    inferred_label = info.get("label_inferred", label)
    return {
        "episode_id": episode_id,
        "ep_dir_key": ep_dir,
        "archive": archive,
        "family": "droid",
        "view": root_name,           # keyframes / keyframes_ext2 / keyframes_wrist / *_success*
        "task": info.get("task") or task_slug or "unknown task",
        "task_slug": task_slug,
        "label": inferred_label,
        "lab": info.get("lab"),
        "gcs_prefix": info.get("gcs_prefix"),
        "scene_id": info.get("scene_id"),
        "uuid": info.get("uuid"),
        "n_source_frames": n_source_est,
        "n_keyframes": n_keyframes,
        "fps": fps,
    }


# ────────────────────────────────────────────────────────────────────────────
# Per-shard repack
# ────────────────────────────────────────────────────────────────────────────

def shard_already_converted(shard_path: Path) -> bool:
    """A shard is in the new format iff the first non-frame entry is a meta.json."""
    try:
        with tarfile.open(shard_path, "r") as t:
            for m in t:
                if m.name.endswith("/meta.json"):
                    return True
                if m.name.endswith(".jpg"):
                    # First JPEG before any meta -> need conversion (also detect padding)
                    base = m.name.split("/")[-1]
                    fm = re.match(r"frame_(\d+)_", base)
                    if fm and len(fm.group(1)) >= 2:
                        # zero-padded already; still need meta.json check beyond first entry
                        # walk a few more to be sure
                        found_meta = False
                        for i, mm in enumerate(t):
                            if i > 30:
                                break
                            if mm.name.endswith("/meta.json"):
                                found_meta = True
                                break
                        return found_meta
                    return False
    except Exception:
        return False
    return False


def repack_one_shard(shard_path: Path, kind: str, archive: str, root_name: str,
                     label: str, idx_to_task: Dict[int, str],
                     droid_lookup: Dict[str, dict], fps: float) -> tuple:
    """Returns (n_episodes, n_frames)."""
    new_path = shard_path.with_suffix(".tar.new")

    # Pass 1 — read everything into memory grouped by episode dir
    ep_data: Dict[str, Dict[int, tuple]] = defaultdict(dict)  # ep -> {idx: (new_name, ts, bytes)}
    with tarfile.open(shard_path, "r") as t:
        for m in t:
            if not m.isfile():
                continue
            res = repad_frame_name(m.name)
            if res is None:
                continue
            new_name, ep, idx, ts = res
            data = t.extractfile(m).read()
            ep_data[ep][idx] = (new_name, ts, data)

    # Pass 2 — write new shard with meta.json per episode
    n_eps = n_frames = 0
    now = int(time.time())
    with tarfile.open(new_path, "w") as out:
        for ep in sorted(ep_data.keys()):
            frames = ep_data[ep]
            for idx in sorted(frames.keys()):
                new_name, ts, data = frames[idx]
                info = tarfile.TarInfo(name=new_name)
                info.size = len(data)
                info.mtime = now
                out.addfile(info, io.BytesIO(data))
                n_frames += 1
            last_ts = max(ts for _, ts, _ in frames.values())
            n_kf = len(frames)
            if kind == "robometer":
                meta = build_episode_meta_robometer(ep, archive, label, idx_to_task,
                                                    n_kf, last_ts, fps)
            else:
                meta = build_episode_meta_droid(ep, archive, root_name, label,
                                                 droid_lookup, n_kf, last_ts, fps)
            mb = json.dumps(meta).encode("utf-8")
            mi = tarfile.TarInfo(name=f"{ep}/meta.json")
            mi.size = len(mb)
            mi.mtime = now
            out.addfile(mi, io.BytesIO(mb))
            n_eps += 1

    new_path.replace(shard_path)  # atomic on POSIX
    return n_eps, n_frames


# ────────────────────────────────────────────────────────────────────────────
# Driver
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", required=True,
                        help="Directory containing shard-*.tar files (one archive's worth)")
    parser.add_argument("--kind", required=True, choices=["robometer", "droid"])
    parser.add_argument("--archive", default=None,
                        help="Archive name (defaults to basename of --shard-dir)")
    parser.add_argument("--root-name", default=None,
                        help="Migrated-root name (e.g. keyframes_success / keyframes_ext2). "
                             "Defaults to the parent dir name of --shard-dir.")
    parser.add_argument("--label", default="successful", choices=["successful", "failure"],
                        help="Default label assigned in meta.json (droid auto-inferred from index)")
    parser.add_argument("--fps", type=float, default=FPS_DEFAULT)
    parser.add_argument("--force", action="store_true",
                        help="Re-pack even if shards look already-converted")
    args = parser.parse_args()

    shard_dir = Path(args.shard_dir)
    if not shard_dir.is_dir():
        log(f"ERROR: {shard_dir} is not a directory"); return

    archive  = args.archive or shard_dir.name
    root_name = args.root_name or shard_dir.parent.name

    log(f"Repacking shards in: {shard_dir}")
    log(f"  kind={args.kind}  archive={archive}  root={root_name}  label={args.label}  fps={args.fps}")

    idx_to_task: Dict[int, str] = {}
    droid_lookup: Dict[str, dict] = {}
    if args.kind == "robometer":
        log("  loading source index_mappings.json...")
        idx_to_task = build_robometer_idx_to_task(archive)
        log(f"  -> {len(idx_to_task)} idx->task entries")
    else:
        log("  loading droid success/failure indexes...")
        droid_lookup = load_droid_metadata()
        log(f"  -> {len(droid_lookup)} droid episode entries")

    shards = sorted(shard_dir.glob("shard-*.tar"))
    log(f"  found {len(shards)} shards")

    total_eps = total_frames = total_skipped = 0
    for sp in shards:
        if not args.force and shard_already_converted(sp):
            log(f"  [SKIP already-converted] {sp.name}")
            total_skipped += 1
            continue
        try:
            n_eps, n_frames = repack_one_shard(
                sp, args.kind, archive, root_name, args.label,
                idx_to_task, droid_lookup, args.fps,
            )
            total_eps += n_eps
            total_frames += n_frames
            log(f"  [OK] {sp.name}: {n_eps} episodes, {n_frames} frames")
        except Exception as e:
            import traceback
            log(f"  [ERROR] {sp.name}: {e}")
            traceback.print_exc()

    log(f"\nDone. {total_eps} episodes, {total_frames} frames, {total_skipped} shards skipped (already-converted)")


if __name__ == "__main__":
    main()
