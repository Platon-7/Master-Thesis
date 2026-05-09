"""Normalize droid layout to match failsafe/metaworld/robometer/roboreward.

Operates IN-PLACE on a scratch copy of the dataset. Original /projects/ data
is untouched. The scratch copy's `droid/` subtree is rewritten to:
  - keyframes*/<archive>/shard-NNNNN.tar (nested per-archive instead of flat shards/)
  - JPG dirs inside tars renamed from <eid>__<task_words> → <eid>
  - manifests split into <fam>_<archive>_<failures|successes>.jsonl per archive
  - manifest rows have `archive` set to the eid prefix (not the useless "droid"),
    `keyframes_dir` populated as "<archive>/<eid>", and successes manifest built
    by walking the success-side tars (didn't exist before).

Idempotent: detects already-normalized state by checking for `shards.bak_pre_norm/`
in the keyframe dirs and skips if found.
"""
import argparse
import glob
import io
import json
import os
import re
import shutil
import sys
import tarfile
import time
from collections import defaultdict

DEFAULT_ROOT = "/scratch-shared/pkarageorgis1/robometer_frame_dataset_20260505_164035"
KEYFRAME_DIRS = [
    "keyframes",
    "keyframes_ext2",
    "keyframes_wrist",
    "keyframes_success",
    "keyframes_success_ext2",
    "keyframes_success_wrist",
]

# Episode IDs in droid look like "<archive>_<datetime_words>" — first segment up
# to the first underscore is the archive.
EID_ARCHIVE_RE = re.compile(r"^([^_]+)_")

# Inside a tar, JPG paths look like "<eid>__<task_words>/frame_NN_<seconds>s.jpg".
# IMPORTANT: split on the LAST "__" — droid eids themselves can contain "__" (e.g.
# `REAL_2023-04-06_Thu_Apr__6_...` for single-digit dates where `Apr  6` gets its
# two-space separator translated to `__`). Naive split-on-first-`__` corrupts these
# eids and merges multiple trajectories into one bogus directory.


def archive_from_eid(eid: str) -> str:
    m = EID_ARCHIVE_RE.match(eid)
    return m.group(1) if m else "unknown"


def parse_dir_in_tar(dir_name: str):
    """Return (eid, task_words_or_none) given a tar-internal directory name.

    For droid: 'REAL_2023-07-13_..._..._Use_object_to_pick_up_something' →
      ('REAL_2023-07-13_..._...', 'Use_object_to_pick_up_something')

    Uses rsplit so eids with `__` inside them (single-digit-date timestamps)
    parse correctly.
    """
    if "__" not in dir_name:
        return dir_name, None
    eid, task = dir_name.rsplit("__", 1)
    return eid, task


def repack_one_keyframe_dir(droid_root: str, view_name: str, dry_run: bool = False) -> dict:
    """Repack droid/<view>/shards/*.tar → droid/<view>/<archive>/shard-NNNNN.tar.

    Returns a dict {archive: [eid, ...]} of episodes seen (used for building
    successes manifests later).
    """
    view_path = os.path.join(droid_root, view_name)
    shards_dir = os.path.join(view_path, "shards")
    bak_dir = os.path.join(view_path, "shards.bak_pre_norm")

    if not os.path.isdir(shards_dir):
        if os.path.isdir(bak_dir):
            print(f"  [{view_name}] already normalized (bak_pre_norm exists)")
            return {}
        print(f"  [{view_name}] no shards/ dir — skipping")
        return {}

    print(f"  [{view_name}] repacking from {shards_dir}")
    old_tars = sorted(glob.glob(os.path.join(shards_dir, "*.tar")))
    print(f"    {len(old_tars)} input tars")

    # Output: per-archive new tars. We create one tar per archive (don't worry
    # about shard size — droid is small enough that a single tar per archive
    # is fine; matches the simpler convention).
    out_writers: dict[str, tarfile.TarFile] = {}
    out_paths: dict[str, str] = {}
    archives_seen: dict[str, list] = defaultdict(list)

    def get_writer(archive):
        if archive in out_writers:
            return out_writers[archive]
        out_dir = os.path.join(view_path, archive)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "shard-00000.tar")
        out_paths[archive] = out_path
        if dry_run:
            return None
        tw = tarfile.open(out_path, "w")
        out_writers[archive] = tw
        return tw

    members_processed = 0
    for old_tar_path in old_tars:
        with tarfile.open(old_tar_path, "r|") as tin:
            for member in tin:
                if not member.isfile():
                    continue
                # member.name like "TRI_2023-10-23__Hang_or_.../frame_00_0.00s.jpg"
                parts = member.name.split("/")
                if len(parts) < 2:
                    continue
                old_dir = parts[-2]
                jpg_name = parts[-1]
                eid, task = parse_dir_in_tar(old_dir)
                arch = archive_from_eid(eid)
                # New path inside the output tar: <eid>/<jpg_name>
                new_name = f"{eid}/{jpg_name}"

                if eid not in archives_seen[arch]:
                    archives_seen[arch].append(eid)

                if dry_run:
                    members_processed += 1
                    if members_processed <= 3:
                        print(f"    sample: {member.name} → {new_name}")
                    continue

                # Read bytes from old tar, write to new tar
                buf = tin.extractfile(member).read()
                # Build a new TarInfo with the new path (preserves size, copies mode)
                ti = tarfile.TarInfo(name=new_name)
                ti.size = len(buf)
                ti.mode = member.mode
                ti.mtime = member.mtime
                writer = get_writer(arch)
                writer.addfile(ti, io.BufferedReader(io.BytesIO(buf)))
                members_processed += 1

        if members_processed % 5000 == 0 and members_processed > 0:
            print(f"    {members_processed} members processed...", flush=True)

    print(f"    {members_processed} JPG members repacked into {len(out_paths)} per-archive tars")

    # Close output tars
    for w in out_writers.values():
        w.close()

    # Move old shards/ → shards.bak_pre_norm/
    if not dry_run:
        os.rename(shards_dir, bak_dir)
        print(f"    {shards_dir} → {bak_dir}")

    return dict(archives_seen)


def split_failures_manifest(droid_root: str, archive_to_eids_default_view: dict, dry_run: bool):
    """Split droid_failures.jsonl into per-archive files with `archive` and
    `keyframes_dir` populated correctly. Backs up the old single-file manifest."""
    manifest_dir = os.path.join(droid_root, "manifests")
    src_path = os.path.join(manifest_dir, "droid_failures.jsonl")
    bak_path = src_path + ".bak_pre_norm"
    if not os.path.exists(src_path):
        if os.path.exists(bak_path):
            print(f"  [manifest] failures already split (bak exists)")
            return
        print(f"  [manifest] no droid_failures.jsonl found")
        return

    print(f"  [manifest] splitting {src_path}")
    by_archive: dict[str, list[dict]] = defaultdict(list)
    n_total = 0
    with open(src_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            eid = row["episode_id"]
            arch = archive_from_eid(eid)
            # Rewrite fields to match other families' convention
            new_row = dict(row)
            new_row["archive"] = arch                    # was useless "droid"
            new_row["keyframes_dir"] = f"{arch}/{eid}"   # was null
            # paired_success_id stays null for now
            by_archive[arch].append(new_row)
            n_total += 1

    if dry_run:
        print(f"    would write {len(by_archive)} per-archive failures manifests:")
        for arch, rows in sorted(by_archive.items()):
            print(f"      droid_{arch}_failures.jsonl: {len(rows)} rows")
        return

    for arch, rows in sorted(by_archive.items()):
        out_path = os.path.join(manifest_dir, f"droid_{arch}_failures.jsonl")
        with open(out_path, "w") as out:
            for r in rows:
                out.write(json.dumps(r) + "\n")
        print(f"    wrote droid_{arch}_failures.jsonl ({len(rows)} rows)")

    os.rename(src_path, bak_path)
    print(f"    {src_path} → {bak_path}  (total {n_total} failure rows split into {len(by_archive)} archives)")


def build_successes_manifests(droid_root: str, archive_to_eids: dict, dry_run: bool):
    """Walk the success-side tars (default view) to build per-archive successes
    manifests. Pulls the task string from the original __<task_words> suffix
    (which was stripped during repack but we recover it from the .bak shards
    OR from the new tars if .bak was already deleted).

    Successes have no frame_labels (correct per user clarification — droid
    successes were never authored with labels).
    """
    print(f"  [manifest] building per-archive successes manifests")
    manifest_dir = os.path.join(droid_root, "manifests")

    # The success-side default view is keyframes_success/. After repack, its
    # archives are populated with new tars (no __task suffix). To recover the
    # task string for each episode, we need to read from the .bak_pre_norm
    # shards/ directory, which still has the original dir names with __task.
    bak_shards = os.path.join(droid_root, "keyframes_success", "shards.bak_pre_norm")
    if not os.path.isdir(bak_shards):
        # If user already cleaned up, fall back to "task=null" entries
        print(f"    [warn] no bak_pre_norm available, will write task=null for successes")
        bak_shards = None

    # Collect (archive, eid) → (task, n_keyframes) by iterating the bak shards
    success_episodes: dict[str, dict[str, dict]] = defaultdict(dict)  # archive → eid → {task, n_keyframes}
    if bak_shards is not None:
        for bak_tar in sorted(glob.glob(os.path.join(bak_shards, "*.tar"))):
            with tarfile.open(bak_tar, "r|") as tin:
                for m in tin:
                    if not m.isfile() or not m.name.endswith(".jpg"):
                        continue
                    parts = m.name.split("/")
                    if len(parts) < 2:
                        continue
                    old_dir = parts[-2]
                    eid, task = parse_dir_in_tar(old_dir)
                    arch = archive_from_eid(eid)
                    if eid not in success_episodes[arch]:
                        success_episodes[arch][eid] = {"task": task, "n_keyframes": 0}
                    success_episodes[arch][eid]["n_keyframes"] += 1

    if not success_episodes:
        # Either bak was missing OR the success tars were empty
        if archive_to_eids:
            print(f"    [warn] no episodes found in bak_pre_norm; cannot build successes manifests")
            return
        print(f"    no successes data; skipping")
        return

    n_total = sum(len(d) for d in success_episodes.values())
    print(f"    found {n_total} success episodes across {len(success_episodes)} archives")

    if dry_run:
        for arch, eids in sorted(success_episodes.items()):
            print(f"      droid_{arch}_successes.jsonl: {len(eids)} rows")
        return

    for arch, eids in sorted(success_episodes.items()):
        out_path = os.path.join(manifest_dir, f"droid_{arch}_successes.jsonl")
        with open(out_path, "w") as out:
            for eid, meta in eids.items():
                row = {
                    "episode_id": eid,
                    "archive": arch,
                    "family": "droid",
                    "task": (meta["task"] or "").replace("_", " "),
                    "label": "success",
                    "terminal_reward": None,
                    "n_source_frames": None,
                    "n_keyframes": meta["n_keyframes"],
                    "keyframes_dir": f"{arch}/{eid}",
                    "paired_success_id": None,
                    "frame_labels": None,   # droid successes have no labels (per user)
                    "fps": None,
                }
                out.write(json.dumps(row) + "\n")
        print(f"    wrote droid_{arch}_successes.jsonl ({len(eids)} rows)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    droid_root = os.path.join(args.root, "droid")
    if not os.path.isdir(droid_root):
        print(f"ERROR: droid not found at {droid_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Normalizing: {droid_root}")
    print(f"Dry-run: {args.dry_run}")

    t0 = time.time()
    print("\n=== Phase 1: repack keyframe tars (6 dirs) ===")
    archive_to_eids = {}
    for view in KEYFRAME_DIRS:
        ae = repack_one_keyframe_dir(droid_root, view, dry_run=args.dry_run)
        if view == "keyframes":
            archive_to_eids = ae

    print("\n=== Phase 2: split failures manifest ===")
    split_failures_manifest(droid_root, archive_to_eids, dry_run=args.dry_run)

    print("\n=== Phase 3: build successes manifests ===")
    build_successes_manifests(droid_root, archive_to_eids, dry_run=args.dry_run)

    print(f"\nDONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
