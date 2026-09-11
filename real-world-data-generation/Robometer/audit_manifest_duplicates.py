#!/usr/bin/env python3
"""
Audit RoboMeter manifests for duplicate and near-duplicate episodes.

Checks performed:
1) Exact duplicate episode_id across all manifests
2) Duplicate UUID suffix in episode_id (ep_<idx>_<uuid>)
3) Duplicate keyframes_dir values
4) Heuristic candidates for "same datapoint, different view" by metadata
5) Optional image-hash candidates from first keyframe (more expensive)

Usage examples:
  python audit_manifest_duplicates.py \
      --manifests /scratch-shared/pkarageorgis/robometer_groupC/manifests

  python audit_manifest_duplicates.py \
      --manifests /scratch-shared/pkarageorgis/robometer_groupC/manifests \
      --manifests /scratch-shared/pkarageorgis/robometer_failures/manifests \
      --base-dirs /scratch-shared/pkarageorgis/robometer_groupC \
      --with-image-hash
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


EP_RE = re.compile(r"^ep_(\d+)_([0-9a-f\-]+)$")


@dataclass(frozen=True)
class Episode:
    manifest_file: str
    episode_id: str
    archive: str
    source_dataset: str
    task: str
    n_source_frames: int
    keyframes_dir: str
    traj_idx: int | None
    uuid: str | None


def normalize_task(task: str) -> str:
    return " ".join((task or "").strip().lower().split())


def parse_episode_id(episode_id: str) -> tuple[int | None, str | None]:
    m = EP_RE.match(episode_id)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)


def iter_manifest_files(manifest_dirs: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for d in manifest_dirs:
        if not d.exists() or not d.is_dir():
            continue
        files.extend(sorted(d.glob("*.jsonl")))
    return files


def load_episodes(manifest_files: list[Path]) -> list[Episode]:
    episodes: list[Episode] = []
    for mf in manifest_files:
        with mf.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                ep_id = row.get("episode_id", "")
                traj_idx, uuid = parse_episode_id(ep_id)
                episodes.append(
                    Episode(
                        manifest_file=mf.name,
                        episode_id=ep_id,
                        archive=row.get("archive", ""),
                        source_dataset=row.get("source_dataset", ""),
                        task=row.get("task", ""),
                        n_source_frames=int(row.get("n_source_frames", -1)),
                        keyframes_dir=row.get("keyframes_dir", ""),
                        traj_idx=traj_idx,
                        uuid=uuid,
                    )
                )
    return episodes


def average_hash_64(img_path: Path) -> str | None:
    if not img_path.exists():
        return None
    try:
        img = Image.open(img_path).convert("L").resize((8, 8), Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.float32)
        m = arr.mean()
        bits = (arr >= m).astype(np.uint8).flatten()
        as_int = 0
        for b in bits:
            as_int = (as_int << 1) | int(b)
        return f"{as_int:016x}"
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifests",
        action="append",
        required=True,
        help="Manifest directory path (can be passed multiple times)",
    )
    parser.add_argument(
        "--base-dirs",
        action="append",
        default=[],
        help="Base data dir(s) used to resolve keyframes_dir for hashing",
    )
    parser.add_argument(
        "--with-image-hash",
        action="store_true",
        help="Compute first-keyframe 64-bit average hash candidates",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="How many duplicate groups to print per section",
    )
    parser.add_argument(
        "--hash-limit",
        type=int,
        default=0,
        help="If >0, hash only the first N episodes (for fast sanity checks)",
    )
    args = parser.parse_args()

    manifest_dirs = [Path(p) for p in args.manifests]
    manifest_files = iter_manifest_files(manifest_dirs)
    if not manifest_files:
        print("No manifest files found.")
        return

    episodes = load_episodes(manifest_files)
    print(f"Loaded {len(episodes):,} episodes from {len(manifest_files)} manifest files")

    by_epid: dict[str, list[Episode]] = defaultdict(list)
    by_uuid: dict[str, list[Episode]] = defaultdict(list)
    by_kdir: dict[str, list[Episode]] = defaultdict(list)
    by_meta_candidate: dict[tuple[str, str, int], list[Episode]] = defaultdict(list)

    for ep in episodes:
        by_epid[ep.episode_id].append(ep)
        if ep.uuid:
            by_uuid[ep.uuid].append(ep)
        if ep.keyframes_dir:
            by_kdir[ep.keyframes_dir].append(ep)

        # Heuristic for cross-view duplicates: same source/task/framecount, different UUIDs.
        key = (ep.source_dataset or "", normalize_task(ep.task), ep.n_source_frames)
        by_meta_candidate[key].append(ep)

    dup_epid = [v for v in by_epid.values() if len(v) > 1]
    dup_uuid = [v for v in by_uuid.values() if len(v) > 1]
    dup_kdir = [v for v in by_kdir.values() if len(v) > 1]

    print(f"Exact duplicate episode_id groups: {len(dup_epid):,}")
    print(f"Duplicate UUID groups:            {len(dup_uuid):,}")
    print(f"Duplicate keyframes_dir groups:   {len(dup_kdir):,}")

    def print_groups(title: str, groups: list[list[Episode]], top_n: int) -> None:
        if not groups:
            return
        print(f"\n{title}")
        groups = sorted(groups, key=len, reverse=True)
        for g in groups[:top_n]:
            print(f"  size={len(g)}")
            for ep in g[:6]:
                print(
                    "   "
                    f"{ep.manifest_file} | {ep.archive} | {ep.episode_id} | "
                    f"frames={ep.n_source_frames}"
                )
            if len(g) > 6:
                print("    ...")

    print_groups("Largest duplicate episode_id groups", dup_epid, args.top)
    print_groups("Largest duplicate UUID groups", dup_uuid, args.top)
    print_groups("Largest duplicate keyframes_dir groups", dup_kdir, args.top)

    # Metadata-based candidate groups; require at least 2 different UUIDs.
    meta_candidates: list[list[Episode]] = []
    for key, grp in by_meta_candidate.items():
        uuids = {e.uuid for e in grp if e.uuid}
        if len(grp) > 1 and len(uuids) > 1:
            meta_candidates.append(grp)

    print(f"\nHeuristic multi-view candidate groups (metadata): {len(meta_candidates):,}")
    if meta_candidates:
        meta_candidates = sorted(meta_candidates, key=len, reverse=True)
        print("Top metadata candidate groups:")
        for grp in meta_candidates[: args.top]:
            ex = grp[0]
            print(
                f"  size={len(grp)} | source={ex.source_dataset or 'unknown'} | "
                f"task='{normalize_task(ex.task)[:80]}' | frames={ex.n_source_frames}"
            )

    if args.with_image_hash:
        base_dirs = [Path(p) for p in args.base_dirs]
        if not base_dirs:
            print("\n--with-image-hash was set but no --base-dirs provided; skipping hash step.")
            return

        print("\nComputing first-keyframe hashes (this can take a while) ...")
        by_hash: dict[str, list[Episode]] = defaultdict(list)
        missing = 0
        for idx, ep in enumerate(episodes):
            if args.hash_limit > 0 and idx >= args.hash_limit:
                break
            if not ep.keyframes_dir:
                continue
            found_img: Path | None = None
            for bd in base_dirs:
                d = bd / ep.keyframes_dir
                p = d / "frame_0_0.00s.jpg"
                if p.exists():
                    found_img = p
                    break
                # Fallback if timestamp formatting differs.
                cands = sorted(d.glob("frame_0_*.jpg")) if d.exists() else []
                if cands:
                    found_img = cands[0]
                    break
            if found_img is None:
                missing += 1
                continue

            h = average_hash_64(found_img)
            if h is not None:
                by_hash[h].append(ep)

        hash_dups = []
        for grp in by_hash.values():
            if len(grp) < 2:
                continue
            uuids = {e.uuid for e in grp if e.uuid}
            # Potentially same event from different views if UUID differs.
            if len(uuids) > 1:
                hash_dups.append(grp)

        print(f"Missing first keyframe for hashing: {missing:,}")
        print(f"Image-hash candidate groups: {len(hash_dups):,}")
        if hash_dups:
            hash_dups = sorted(hash_dups, key=len, reverse=True)
            print("Top image-hash candidate groups:")
            for grp in hash_dups[: args.top]:
                ex = grp[0]
                print(
                    f"  size={len(grp)} | archive={ex.archive} | "
                    f"task='{normalize_task(ex.task)[:80]}' | frames={ex.n_source_frames}"
                )


if __name__ == "__main__":
    main()
