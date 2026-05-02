#!/usr/bin/env python3
"""
pack_loose_to_shards.py — Pack loose-JPEG keyframe directories into
WebDataset-style tar shards.

Why: prjs1958 has a 1.9M-inode project quota. Loose JPEGs (16 per episode)
blow past it. Packing ~256 episodes per shard collapses the inode footprint
~4000×.

Supported layouts (auto-detected):
  • Layout A (multi-archive):    <root>/<archive>/<episode>/frame_*.jpg
  • Layout B (single-archive):   <root>/<episode>/frame_*.jpg

Output (same root, parallel to the loose data):
  • Layout A:  <root>/<archive>/shard-00000.tar, shard-00001.tar, ...
  • Layout B:  <root>/shards/shard-00000.tar, shard-00001.tar, ...

Each tar contains entries of the form:
    <episode_id>/frame_00_0.00s.jpg
    <episode_id>/frame_01_...jpg
    ...

Safety:
  • --dry-run:   print the plan, do nothing
  • --verify:    re-open each shard and check all expected files are present
                 BEFORE deleting the loose JPEGs
  • --keep-loose: write shards but do not delete loose files (useful for a
                  first pass, before you trust the migration)
  • Resume-safe: shards already in the output dir are not overwritten;
                 episodes that are already packed are skipped.

Usage:
    # dry-run on a small layout-A root
    python pack_loose_to_shards.py /projects/prjs1958/robometer_frame_dataset/robometer/keyframes_orphan_success --dry-run

    # pack one archive, keep loose files for now
    python pack_loose_to_shards.py \
        /projects/prjs1958/robometer_frame_dataset/robometer/keyframes_orphan_success \
        --archive jesbu1_oxe_rfm_oxe_bc_z --keep-loose

    # full migration with verify + delete
    python pack_loose_to_shards.py \
        /projects/prjs1958/robometer_frame_dataset/droid/keyframes_success \
        --verify
"""

import argparse
import io
import json
import shutil
import tarfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

EPISODES_PER_SHARD = 256


def log(msg):
    print(msg, flush=True)


# ────────────────────────────────────────────────────────────────────────────
# Layout detection
# ────────────────────────────────────────────────────────────────────────────

def detect_layout(root: Path) -> str:
    """Return 'A' for multi-archive (root/arch/ep/frame) or 'B' for single-archive (root/ep/frame)."""
    first_level = [d for d in root.iterdir() if d.is_dir()]
    if not first_level:
        raise ValueError(f"{root} has no subdirectories")

    sample = first_level[0]
    has_jpgs = any(c.suffix == ".jpg" for c in sample.iterdir() if c.is_file())
    has_subdirs = any(c.is_dir() for c in sample.iterdir())

    if has_jpgs and not has_subdirs:
        return "B"
    if has_subdirs and not has_jpgs:
        return "A"
    # Mixed — prefer layout A if at least one grandchild has jpgs
    for child in sample.iterdir():
        if child.is_dir():
            if any(c.suffix == ".jpg" for c in child.iterdir() if c.is_file()):
                return "A"
    raise ValueError(f"Could not determine layout of {root}")


def find_episode_dirs(group_root: Path):
    """Yield episode dirs (each contains frame_*.jpg) under a group root."""
    for ep in sorted(group_root.iterdir()):
        if not ep.is_dir():
            continue
        # Skip output artifacts
        if ep.name.startswith("shard-") or ep.name == "shards":
            continue
        yield ep


# ────────────────────────────────────────────────────────────────────────────
# Packing
# ────────────────────────────────────────────────────────────────────────────

class ShardWriter:
    def __init__(self, shard_dir: Path, episodes_per_shard: int = EPISODES_PER_SHARD):
        self.shard_dir = shard_dir
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_per_shard = episodes_per_shard
        existing = sorted(self.shard_dir.glob("shard-*.tar"))
        self.shard_idx = len(existing)
        self.current_tar = None
        self.current_path = None
        self.current_count = 0

    def _open(self):
        self.current_path = self.shard_dir / f"shard-{self.shard_idx:05d}.tar"
        self.current_tar = tarfile.open(self.current_path, "w")
        self.current_count = 0
        log(f"    -> opened {self.current_path.name}")

    def write_episode(self, episode_id: str, episode_dir: Path):
        if self.current_tar is None:
            self._open()
        if self.current_count >= self.episodes_per_shard:
            self.current_tar.close()
            self.shard_idx += 1
            self._open()

        files = sorted(f for f in episode_dir.iterdir() if f.is_file())
        for f in files:
            arcname = f"{episode_id}/{f.name}"
            self.current_tar.add(f, arcname=arcname, recursive=False)
        self.current_count += 1
        return self.current_path.name

    def close(self):
        if self.current_tar is not None:
            self.current_tar.close()
            self.current_tar = None


def already_packed_episodes(shard_dir: Path):
    """Return set of episode_ids already present in existing shards."""
    packed = set()
    if not shard_dir.exists():
        return packed
    for tar_path in sorted(shard_dir.glob("shard-*.tar")):
        try:
            with tarfile.open(tar_path, "r") as tf:
                for m in tf.getmembers():
                    # members look like "<episode_id>/frame_..jpg"
                    head = m.name.split("/", 1)[0]
                    if head:
                        packed.add(head)
        except Exception as e:
            log(f"  WARNING: could not read {tar_path}: {e}")
    return packed


def pack_group(group_root: Path, shard_dir: Path, dry_run: bool = False) -> Tuple[int, List[Path]]:
    """Pack all episodes under group_root into shards under shard_dir.

    Returns (num_new_episodes_packed, list_of_episode_dirs_packed).
    """
    episodes = list(find_episode_dirs(group_root))
    if not episodes:
        log(f"  No episode dirs in {group_root}")
        return 0, []

    packed_already = already_packed_episodes(shard_dir)
    if packed_already:
        log(f"  Resume: {len(packed_already)} episodes already in existing shards")

    to_pack = [ep for ep in episodes if ep.name not in packed_already]
    log(f"  {len(episodes)} episodes total, {len(to_pack)} to pack")

    if dry_run:
        for ep in to_pack[:3]:
            log(f"    would pack: {ep.name}")
        if len(to_pack) > 3:
            log(f"    ... and {len(to_pack) - 3} more")
        return 0, []

    writer = ShardWriter(shard_dir)
    packed_dirs: List[Path] = []
    try:
        for i, ep in enumerate(to_pack):
            writer.write_episode(ep.name, ep)
            packed_dirs.append(ep)
            if (i + 1) % 500 == 0:
                log(f"    packed {i + 1}/{len(to_pack)}")
    finally:
        writer.close()

    return len(packed_dirs), packed_dirs


def verify_shards(shard_dir: Path, packed_dirs: List[Path]) -> bool:
    """Verify every packed episode's JPEGs are readable from some shard."""
    if not packed_dirs:
        return True
    log(f"  Verifying {len(packed_dirs)} packed episodes...")

    expected = {}  # episode_id -> set(filenames)
    for ep in packed_dirs:
        expected[ep.name] = {f.name for f in ep.iterdir() if f.is_file()}

    found = {k: set() for k in expected}
    for tar_path in sorted(shard_dir.glob("shard-*.tar")):
        with tarfile.open(tar_path, "r") as tf:
            for m in tf.getmembers():
                if "/" not in m.name:
                    continue
                ep_id, fname = m.name.split("/", 1)
                if ep_id in found:
                    found[ep_id].add(fname)

    ok = True
    for ep_id, exp_set in expected.items():
        if found[ep_id] != exp_set:
            missing = exp_set - found[ep_id]
            extra = found[ep_id] - exp_set
            log(f"  MISMATCH {ep_id}: missing={missing} extra={extra}")
            ok = False
    if ok:
        log("  ✓ All packed episodes verified.")
    return ok


def delete_loose(packed_dirs: List[Path]):
    for ep in packed_dirs:
        shutil.rmtree(ep)
    log(f"  Deleted {len(packed_dirs)} loose episode dirs.")


# ────────────────────────────────────────────────────────────────────────────
# Drivers
# ────────────────────────────────────────────────────────────────────────────

def drive_layout_a(root: Path, archive_filter: Optional[str], dry_run: bool, verify: bool, keep_loose: bool):
    """For root/<archive>/<episode>/frame_*.jpg — pack per archive."""
    archives = sorted(d for d in root.iterdir() if d.is_dir() and d.name not in {"shards", "manifests"})
    if archive_filter and archive_filter != "all":
        archives = [d for d in archives if d.name == archive_filter]
        if not archives:
            log(f"ERROR: archive {archive_filter} not found under {root}")
            return

    total_new = 0
    for arch_dir in archives:
        log(f"\n--- {arch_dir.name} ---")
        shard_dir = arch_dir  # shards live alongside loose episodes, then we remove loose
        n_new, packed_dirs = pack_group(arch_dir, shard_dir, dry_run=dry_run)
        if dry_run:
            continue
        if verify and not verify_shards(shard_dir, packed_dirs):
            log(f"  verification failed — NOT deleting loose for {arch_dir.name}")
            continue
        if not keep_loose:
            delete_loose(packed_dirs)
        total_new += n_new
    log(f"\nLayout A total: {total_new} episodes packed across {len(archives)} archive(s)")


def drive_layout_b(root: Path, dry_run: bool, verify: bool, keep_loose: bool):
    """For root/<episode>/frame_*.jpg — pack into root/shards/."""
    shard_dir = root / "shards"
    log(f"\n--- packing {root.name} into {shard_dir}/ ---")
    n_new, packed_dirs = pack_group(root, shard_dir, dry_run=dry_run)
    if dry_run:
        return
    if verify and not verify_shards(shard_dir, packed_dirs):
        log("  verification failed — NOT deleting loose")
        return
    if not keep_loose:
        delete_loose(packed_dirs)
    log(f"\nLayout B total: {n_new} episodes packed")


def main():
    parser = argparse.ArgumentParser(
        description="Pack loose-JPEG keyframe directories into tar shards."
    )
    parser.add_argument("root", type=str,
                        help="Root dir (layout auto-detected). e.g., keyframes_orphan_success or droid/keyframes_success")
    parser.add_argument("--archive", type=str, default=None,
                        help="(layout A only) restrict to one archive; default = all")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan; do not write or delete anything.")
    parser.add_argument("--verify", action="store_true",
                        help="Re-open shards and verify every packed episode before deleting loose.")
    parser.add_argument("--keep-loose", action="store_true",
                        help="Write shards but keep the loose JPEGs (no deletion).")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        log(f"ERROR: {root} does not exist")
        return

    layout = detect_layout(root)
    log(f"Detected layout: {layout}  (root: {root})")
    log(f"Episodes per shard: {EPISODES_PER_SHARD}")
    log(f"Mode: dry_run={args.dry_run}  verify={args.verify}  keep_loose={args.keep_loose}")
    t0 = time.time()

    if layout == "A":
        drive_layout_a(root, args.archive, args.dry_run, args.verify, args.keep_loose)
    else:
        drive_layout_b(root, args.dry_run, args.verify, args.keep_loose)

    log(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
