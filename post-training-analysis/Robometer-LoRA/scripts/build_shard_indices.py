#!/usr/bin/env python3
"""
build_shard_indices.py — Build `shard_index.json` files that map
episode_id -> shard filename for every shard-packed keyframe dir in
/projects/prjs1958/robometer_frame_dataset/.

For metaworld/failsafe/robometer, the top-level dir inside a shard IS the
episode_id. For DROID the top-level dir is `{episode_id}__{task_slug}`, so we
resolve it by longest-known-prefix match against failure_index+success_index.

Output per archive dir:
  <keyframes_subdir>/<archive>/shard_index.json    # { "<episode_id>": "shard-NNNNN.tar", ... }

Droid is single-archive-per-view, so the output is:
  droid/<keyframes_subdir>/shards/shard_index.json
"""

import json
import tarfile
from pathlib import Path

ROOT = Path("/projects/prjs1958/robometer_frame_dataset")

# (parent_dir, is_droid)
SIMPLE_SOURCES = [
    (ROOT / "metaworld" / "keyframes",            False),
    (ROOT / "metaworld" / "keyframes_success",    False),
    (ROOT / "failsafe"  / "keyframes",            False),
    (ROOT / "failsafe"  / "keyframes_success",    False),
    (ROOT / "robometer" / "keyframes",            False),
    (ROOT / "robometer" / "keyframes_success",    False),
    (ROOT / "robometer" / "keyframes_orphan_success", False),
    (ROOT / "roboreward" / "keyframes",           False),
    (ROOT / "roboreward" / "keyframes_success",   False),
]

DROID_VIEW_DIRS = [
    ROOT / "droid" / "keyframes"         / "shards",    # ext1 failure
    ROOT / "droid" / "keyframes_ext2"    / "shards",
    ROOT / "droid" / "keyframes_wrist"   / "shards",
    ROOT / "droid" / "keyframes_success" / "shards",
    ROOT / "droid" / "keyframes_success_ext2" / "shards",
    ROOT / "droid" / "keyframes_success_wrist" / "shards",
]


def build_simple(parent):
    """parent contains <archive>/shard-*.tar. One index per archive."""
    if not parent.exists():
        print(f"  skip (missing): {parent}")
        return
    for archive_dir in sorted(parent.iterdir()):
        if not archive_dir.is_dir():
            continue
        shards = sorted(archive_dir.glob("shard-*.tar"))
        if not shards:
            continue
        index = {}
        collisions = 0
        for shard in shards:
            try:
                with tarfile.open(shard, "r") as t:
                    names = t.getnames()
            except Exception as e:
                print(f"  ERROR reading {shard}: {e}")
                continue
            top_dirs = set()
            for n in names:
                p = n.split("/", 1)[0]
                if p:
                    top_dirs.add(p)
            for d in top_dirs:
                if d in index and index[d] != shard.name:
                    collisions += 1
                else:
                    index[d] = shard.name
        out_path = archive_dir / "shard_index.json"
        with open(out_path, "w") as f:
            json.dump(index, f)
        tag = f"({collisions} dup)" if collisions else ""
        print(f"  {archive_dir.name}: {len(index)} episodes across {len(shards)} shards {tag}")


def load_droid_known_ids():
    meta = ROOT / "droid" / "metadata"
    ids = set()
    for p in [meta / "failure_index.jsonl", meta / "success_index.jsonl"]:
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                ids.add(json.loads(line)["episode_id"])
    return ids


def resolve_droid_dir(dirname, known_ids):
    """Return the episode_id such that dirname == f'{ep_id}__{task_slug}'.
    Some episode_ids contain '__' (e.g., 'Jul__7'), so we can't just rsplit.
    Strategy: iterate candidate split points (every '__' occurrence) and
    check if the left half is a known episode_id.
    """
    i = 0
    candidates = []
    while True:
        j = dirname.find("__", i)
        if j < 0:
            break
        candidates.append(j)
        i = j + 2
    # Try splits from the RIGHTMOST first — task slug is trailing.
    for pos in reversed(candidates):
        left = dirname[:pos]
        if left in known_ids:
            return left
    return None


def build_droid():
    known = load_droid_known_ids()
    print(f"  DROID known episode_ids: {len(known)}")
    for view_dir in DROID_VIEW_DIRS:
        if not view_dir.exists():
            print(f"  skip (missing): {view_dir}")
            continue
        shards = sorted(view_dir.glob("shard-*.tar"))
        if not shards:
            continue
        index = {}
        unresolved = 0
        for shard in shards:
            try:
                with tarfile.open(shard, "r") as t:
                    names = t.getnames()
            except Exception as e:
                print(f"  ERROR reading {shard}: {e}")
                continue
            top_dirs = set()
            for n in names:
                p = n.split("/", 1)[0]
                if p:
                    top_dirs.add(p)
            for d in top_dirs:
                ep_id = resolve_droid_dir(d, known)
                if ep_id is None:
                    unresolved += 1
                    continue
                # Store mapping ep_id -> (shard, inner_dir)
                # Different views may have different inner_dir for the same ep_id.
                index[ep_id] = {"shard": shard.name, "inner_dir": d}
        out_path = view_dir / "shard_index.json"
        with open(out_path, "w") as f:
            json.dump(index, f)
        tag = f"({unresolved} unresolved)" if unresolved else ""
        print(f"  {view_dir.parent.name}/{view_dir.name}: {len(index)} episodes across {len(shards)} shards {tag}")


def main():
    print("=== Simple sources ===")
    for parent, _ in SIMPLE_SOURCES:
        print(f"[{parent.relative_to(ROOT)}]")
        build_simple(parent)
    print("\n=== DROID ===")
    build_droid()
    print("\nDone.")


if __name__ == "__main__":
    main()
