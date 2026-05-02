#!/usr/bin/env python3
"""
patch_meta_family.py — Fix the `family` field in every `<ep>/meta.json` inside
already-repacked shards.

Context: `repack_to_match_format.py` was run before `robometer_families.py`
became the single source of truth. Its local `ROBOMETER_FAMILY` dict only
covered 13 archives, so every other archive got `family = <archive_name>` in
its in-shard meta.json. This script walks all repacked shard trees, looks up
the correct family via the unified `FAMILY_REGISTRY`, and rewrites ONLY the
affected meta.json entries — all JPEG frames are copied through byte-for-byte.

Atomicity: we write to `<shard>.tar.new` and POSIX-rename into place.
Idempotency: we skip shards whose meta.json entries all already carry the
correct family.

Scopes covered:
  * /projects/prjs1958/robometer_frame_dataset/droid/keyframes*/shards/
  * /projects/prjs1958/robometer_frame_dataset/robometer/keyframes_success/<archive>/
  * /projects/prjs1958/robometer_frame_dataset/robometer/keyframes_orphan_success/<archive>/

The orphan-success tree was extracted by the already-patched
extract_orphan_successes.py and should be a no-op; scanning it is cheap and
catches any stragglers.

Usage:
    # dry-run — report what would change, write nothing
    python patch_meta_family.py --dry-run

    # patch everything in-place
    python patch_meta_family.py

    # restrict to a single subtree (for debugging)
    python patch_meta_family.py --scope robometer_successes
    python patch_meta_family.py --scope droid
    python patch_meta_family.py --scope orphan_successes
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

from robometer_families import family_of

DATA_ROOT = Path("/projects/prjs1958/robometer_frame_dataset")

SCOPES = {
    "droid": [
        DATA_ROOT / "droid" / "keyframes" / "shards",
        DATA_ROOT / "droid" / "keyframes_ext2" / "shards",
        DATA_ROOT / "droid" / "keyframes_wrist" / "shards",
        DATA_ROOT / "droid" / "keyframes_success" / "shards",
        DATA_ROOT / "droid" / "keyframes_success_ext2" / "shards",
        DATA_ROOT / "droid" / "keyframes_success_wrist" / "shards",
    ],
    "robometer_successes": [DATA_ROOT / "robometer" / "keyframes_success"],
    "orphan_successes":    [DATA_ROOT / "robometer" / "keyframes_orphan_success"],
}


def log(msg: str) -> None:
    print(msg, flush=True)


def find_shards(roots: list[Path]) -> list[Path]:
    """Return all shard-*.tar files under the given roots (recursive)."""
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            log(f"  (skip: {root} does not exist)")
            continue
        out.extend(sorted(root.rglob("shard-*.tar")))
    return out


def shard_needs_patch(shard: Path) -> tuple[bool, int, int]:
    """Cheap pre-pass: open shard read-only, read every meta.json, count how
    many have the wrong family. Returns (needs_patch, wrong, total).
    """
    wrong = total = 0
    with tarfile.open(shard, "r") as tf:
        for m in tf:
            if not m.name.endswith("/meta.json"):
                continue
            total += 1
            data = tf.extractfile(m).read()
            try:
                meta = json.loads(data)
            except json.JSONDecodeError:
                wrong += 1  # unreadable meta — force rewrite
                continue
            archive = meta.get("archive")
            if archive is None:
                continue
            correct = family_of(archive)
            if meta.get("family") != correct:
                wrong += 1
    return (wrong > 0, wrong, total)


def rewrite_shard(shard: Path, dry_run: bool = False) -> tuple[int, int]:
    """Stream-copy shard, rewriting every meta.json with the correct family.
    Returns (fixed, total_meta).
    """
    tmp = shard.with_suffix(".tar.new")
    fixed = total = 0

    with tarfile.open(shard, "r") as src:
        if dry_run:
            sink = None
        else:
            sink = tarfile.open(tmp, "w")
        try:
            for m in src:
                if m.name.endswith("/meta.json") and m.isfile():
                    total += 1
                    data = src.extractfile(m).read()
                    try:
                        meta = json.loads(data)
                    except json.JSONDecodeError:
                        log(f"    ! unparseable meta: {m.name} — passing through")
                        if sink is not None:
                            info = tarfile.TarInfo(name=m.name)
                            info.size = len(data)
                            info.mtime = int(time.time())
                            sink.addfile(info, io.BytesIO(data))
                        continue

                    archive = meta.get("archive")
                    if archive is not None:
                        correct = family_of(archive)
                        if meta.get("family") != correct:
                            meta["family"] = correct
                            fixed += 1
                            data = json.dumps(meta).encode("utf-8")

                    if sink is not None:
                        info = tarfile.TarInfo(name=m.name)
                        info.size = len(data)
                        info.mtime = int(time.time())
                        sink.addfile(info, io.BytesIO(data))
                else:
                    if sink is not None:
                        f = src.extractfile(m) if m.isfile() else None
                        sink.addfile(m, f)
        finally:
            if sink is not None:
                sink.close()

    if dry_run:
        return fixed, total

    if fixed == 0:
        tmp.unlink(missing_ok=True)
        return 0, total

    os.replace(tmp, shard)
    return fixed, total


def process(roots: list[Path], dry_run: bool) -> None:
    shards = find_shards(roots)
    log(f"Found {len(shards)} shards")

    grand_fixed = grand_total = grand_shards_changed = 0
    for shard in shards:
        try:
            needs, wrong, total = shard_needs_patch(shard)
        except Exception as e:
            log(f"  ! skip (scan error) {shard}: {e}")
            continue

        if not needs:
            continue

        rel = shard.relative_to(DATA_ROOT) if shard.is_relative_to(DATA_ROOT) else shard
        log(f"  [{wrong}/{total} bad family]  {rel}")

        try:
            fixed, _tot = rewrite_shard(shard, dry_run=dry_run)
        except Exception as e:
            log(f"    ! rewrite failed: {e}")
            continue

        grand_fixed += fixed
        grand_total += total
        if fixed:
            grand_shards_changed += 1

    verb = "would fix" if dry_run else "fixed"
    log("─" * 60)
    log(f"  {verb} {grand_fixed} meta.json entries across "
        f"{grand_shards_changed} shards ({grand_total} meta.json scanned)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="Report changes without rewriting.")
    p.add_argument(
        "--scope",
        choices=[*SCOPES.keys(), "all"],
        default="all",
        help="Subtree to patch (default: all).",
    )
    args = p.parse_args()

    if args.scope == "all":
        roots = [r for rs in SCOPES.values() for r in rs]
    else:
        roots = SCOPES[args.scope]

    log("=" * 60)
    log(f"patch_meta_family.py  scope={args.scope}  dry_run={args.dry_run}")
    log("=" * 60)
    process(roots, args.dry_run)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
