#!/usr/bin/env python3
"""Check overlap between the 5,503 droid successes we downloaded (matched
to droid failures) and the droid successes inside the Robometer raw
archive jesbu1_oxe_rfm_oxe_droid.

Robometer's droid archive identifies trajectories ONLY by full UUID
(stored in the file path `frames/trajectory_<UUID>.npz` and in
`embeddings/trajectory_<INDEX>_<UUID>_embeddings.pt`).

The user's metadata uses the droid raw `episode_id` (lab + timestamp) as
its primary key, but `success_index.jsonl` also carries an 8-character
UUID prefix in the `uuid` field — e.g.
`AUTOLab+84bd5053+2023-07-21-13h-07m-06s`. We use that 8-char prefix as
the bridge.

Steps:
  1. Build episode_id -> uuid8 map from success_index.jsonl.
  2. For each entry in success_pairs.jsonl (5,503), pull its uuid8.
  3. Stream the droid tar TOC, regex out every UUID, take its 8-char prefix.
  4. Intersect.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ARCH_NAME = "jesbu1_oxe_rfm_oxe_droid"
SPLIT_DIR = Path("/projects/prjs1958/robometer_full_dataset/raw_archives/split") / ARCH_NAME
PAIRS = Path("/projects/prjs1958/robometer_frame_dataset/droid/metadata/success_pairs.jsonl")
INDEX = Path("/projects/prjs1958/robometer_frame_dataset/droid/metadata/success_index.jsonl")
UUID_DUMP = Path("/projects/prjs1958/robometer_full_dataset/droid_uuid_list.txt")

UUID_RE = re.compile(r"trajectory_[0-9_]*([0-9a-f]{8})-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def stream_uuid_prefixes():
    """Stream tar TOC across all parts and yield 8-char UUID prefixes."""
    parts = sorted(SPLIT_DIR.glob(f"{ARCH_NAME}.tar.part-*"))
    print(f"  streaming TOC across {len(parts)} parts...", flush=True)
    cat = subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE)
    tar = subprocess.Popen(
        ["tar", "-tf", "-"],
        stdin=cat.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1,
    )
    cat.stdout.close()
    seen = set()
    n_lines = 0
    for line in tar.stdout:
        n_lines += 1
        m = UUID_RE.search(line)
        if m:
            seen.add(m.group(1))
        if n_lines % 100000 == 0:
            print(f"    read {n_lines:,} entries, {len(seen):,} unique UUIDs", flush=True)
    tar.wait()
    cat.wait()
    print(f"  done: {n_lines:,} entries, {len(seen):,} unique UUIDs", flush=True)
    return seen


def load_robometer_uuids():
    if UUID_DUMP.exists():
        print(f"Reusing cached UUID list: {UUID_DUMP}", flush=True)
        with open(UUID_DUMP) as f:
            return {line.strip() for line in f if line.strip()}
    print("Streaming droid tar TOC to harvest UUIDs...", flush=True)
    uuids = stream_uuid_prefixes()
    UUID_DUMP.write_text("\n".join(sorted(uuids)) + "\n")
    print(f"Cached {len(uuids):,} UUIDs to {UUID_DUMP}", flush=True)
    return uuids


def main():
    # 1. episode_id -> uuid8 from success_index.jsonl (covers all 59,475)
    ep2uuid = {}
    with open(INDEX) as f:
        for line in f:
            r = json.loads(line)
            uuid_field = r.get("uuid")  # "AUTOLab+84bd5053+2023-..."
            if uuid_field and "+" in uuid_field:
                parts = uuid_field.split("+")
                if len(parts) >= 2 and len(parts[1]) == 8:
                    ep2uuid[r["episode_id"]] = parts[1]
    print(f"Built episode_id -> uuid8 map: {len(ep2uuid):,} entries", flush=True)

    # 2. Pull uuid8 for each of the 5,503 pairs
    pair_uuids = set()
    pair_total = 0
    pair_missing = 0
    with open(PAIRS) as f:
        for line in f:
            r = json.loads(line)
            pair_total += 1
            ep = r["success_ep"]
            u = ep2uuid.get(ep)
            if u:
                pair_uuids.add(u)
            else:
                pair_missing += 1
    print(f"Pairs: {pair_total} total, {len(pair_uuids):,} with uuid8, {pair_missing} missing", flush=True)

    # 3. Robometer UUIDs
    robometer_uuids = load_robometer_uuids()

    # 4. Overlap
    overlap = pair_uuids & robometer_uuids
    print(f"\n{'='*60}")
    print(f"OVERLAP: {len(overlap):,} of your {len(pair_uuids):,} matched-success UUIDs")
    print(f"         exist inside the Robometer droid archive")
    print(f"         ({len(overlap)/max(1,len(pair_uuids))*100:.1f}%)")
    print(f"{'='*60}")
    if overlap:
        print("Sample overlap UUID prefixes:", list(overlap)[:5])


if __name__ == "__main__":
    main()
