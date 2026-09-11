#!/usr/bin/env python3
"""
pack_scored_output.py — Convert the scoring pipeline's loose-JPEG output
(FOR_LEONARDO.md layout) into the 256-episodes-per-tar sharded format used
by robometer_frame_dataset/robometer/keyframes/.

Why: FOR_LEONARDO.md writes each episode as a directory of loose JPEGs —
cheap for VLM input but inode-expensive (~8 inodes per episode). Packing
collapses that to ~N/256 tar files while preserving bytes, and attaches a
per-episode meta.json (with family + score + vlm_description) in the same
format as every other subtree of /projects/prjs1958/robometer_frame_dataset/.

Idempotent: skip shards that already exist (by count) in the output dir.
Atomic: write to <shard>.tar.new then POSIX-rename.

Input layout (FOR_LEONARDO.md $DATA_DIR):
    keyframes/<archive>/<episode_id>/frame_NN_X.Xs.jpg
    manifests/<archive>.jsonl
    scores/<archive>_scored.jsonl            (optional but expected)
    vlm_descriptions/<archive>_descriptions.jsonl  (optional)

Output layout (merge into robometer_frame_dataset/):
    keyframes/<archive>/shard-NNNNN.tar
    manifests/<archive>_failures.jsonl       (one line per episode, with score)

Each shard tar entry:
    <episode_id>/frame_NN_X.Xs.jpg        (copied verbatim, bytes untouched)
    <episode_id>/meta.json                (freshly generated)

Usage:
    python3 pack_scored_output.py \
        --data-dir /scratch-shared/$USER/robometer_out \
        --output-dir /scratch-shared/$USER/robometer_out_packed

    # pack a single archive only
    python3 pack_scored_output.py --data-dir ... --output-dir ... \
        --archive jesbu1_racer_rfm_racer_train

    # delete the source loose-JPEG tree for an archive after a successful pack
    python3 pack_scored_output.py --data-dir ... --output-dir ... \
        --archive jesbu1_racer_rfm_racer_train --cleanup-source
"""
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
import tarfile
import time
from pathlib import Path

from robometer_families import family_of

EPISODES_PER_SHARD = 256
FRAME_RE = re.compile(r"frame_(\d+)_.*\.jpg$")


def log(msg: str) -> None:
    print(msg, flush=True)


def load_jsonl(path: Path, key: str = "episode_id") -> dict[str, dict]:
    """Load a JSONL into a dict keyed by `episode_id` (last-write-wins)."""
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = obj.get(key)
            if k:
                out[k] = obj
    return out


def build_meta(episode_id: str, archive: str,
               manifest_row: dict | None,
               score_row: dict | None,
               vlm_row: dict | None,
               n_keyframes: int) -> dict:
    """Compose the per-episode meta.json from whichever sources are present."""
    m = manifest_row or {}
    s = score_row or {}
    v = vlm_row or {}

    meta = {
        "episode_id": episode_id,
        "archive": archive,
        "family": family_of(archive),
        "task": m.get("task") or s.get("task") or "unknown task",
        "label": m.get("label") or s.get("robometer_label") or "failure",
        "n_source_frames": m.get("n_source_frames"),
        "n_keyframes": n_keyframes,
        "fps": m.get("fps"),
    }
    # Drop Nones so downstream code can .get() cleanly.
    meta = {k: v for k, v in meta.items() if v is not None}

    # Attach score + vlm description if we have them.
    if "score" in s:
        meta["score"] = s["score"]
    if "description" in v:
        meta["vlm_description"] = v["description"]
    elif "vlm_description" in v:
        meta["vlm_description"] = v["vlm_description"]

    return meta


def sorted_frames(ep_dir: Path) -> list[Path]:
    frames: list[tuple[int, Path]] = []
    for p in ep_dir.iterdir():
        match = FRAME_RE.search(p.name)
        if match:
            frames.append((int(match.group(1)), p))
    frames.sort(key=lambda pair: pair[0])
    return [p for _, p in frames]


def pack_archive(archive: str, data_dir: Path, out_dir: Path,
                 cleanup_source: bool = False) -> int:
    src_root = data_dir / "keyframes" / archive
    if not src_root.is_dir():
        log(f"  [skip] {archive}: no keyframes dir at {src_root}")
        return 0

    manifest_rows = load_jsonl(data_dir / "manifests" / f"{archive}.jsonl")
    score_rows    = load_jsonl(data_dir / "scores"    / f"{archive}_scored.jsonl")
    vlm_rows      = load_jsonl(data_dir / "vlm_descriptions" / f"{archive}_descriptions.jsonl")

    episode_dirs = sorted([p for p in src_root.iterdir() if p.is_dir()])
    log(f"  {archive}: {len(episode_dirs)} episodes found under {src_root}")

    shard_dir = out_dir / "keyframes" / archive
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = out_dir / "manifests" / f"{archive}_failures.jsonl"
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    existing_shards = sorted(shard_dir.glob("shard-*.tar"))
    # Resume semantics: skip episode dirs whose episode_id is already in the
    # output manifest (written per-episode below with flush()).
    already: set[str] = set()
    if manifest_out.exists():
        with manifest_out.open() as mf:
            for line in mf:
                try:
                    already.add(json.loads(line)["episode_id"])
                except Exception:
                    pass
    shard_idx = len(existing_shards)

    current_tar: tarfile.TarFile | None = None
    current_tar_path: Path | None = None
    current_count = 0
    total_packed = 0

    def open_new_shard() -> None:
        nonlocal current_tar, current_tar_path, current_count, shard_idx
        if current_tar is not None:
            current_tar.close()
            # Atomic swap in: .tar.new -> .tar.
            assert current_tar_path is not None
            current_tar_path.replace(current_tar_path.with_suffix(".tar"))
        path = shard_dir / f"shard-{shard_idx:05d}.tar.new"
        current_tar = tarfile.open(path, "w")
        current_tar_path = path
        current_count = 0
        shard_idx += 1
        log(f"    opened {path.with_suffix('.tar').name}")

    open_new_shard()

    with manifest_out.open("a") as mout:
        for ep_dir in episode_dirs:
            episode_id = ep_dir.name
            if episode_id in already:
                continue
            frames = sorted_frames(ep_dir)
            if not frames:
                log(f"    ! {episode_id}: no frames, skipping")
                continue

            if current_count >= EPISODES_PER_SHARD:
                open_new_shard()

            meta = build_meta(
                episode_id=episode_id,
                archive=archive,
                manifest_row=manifest_rows.get(episode_id),
                score_row=score_rows.get(episode_id),
                vlm_row=vlm_rows.get(episode_id),
                n_keyframes=len(frames),
            )

            for fp in frames:
                data = fp.read_bytes()
                info = tarfile.TarInfo(name=f"{episode_id}/{fp.name}")
                info.size = len(data)
                info.mtime = int(time.time())
                current_tar.addfile(info, io.BytesIO(data))

            meta_bytes = json.dumps(meta).encode("utf-8")
            info = tarfile.TarInfo(name=f"{episode_id}/meta.json")
            info.size = len(meta_bytes)
            info.mtime = int(time.time())
            current_tar.addfile(info, io.BytesIO(meta_bytes))

            current_count += 1
            total_packed += 1

            manifest_entry = {**meta, "shard": f"shard-{shard_idx - 1:05d}.tar"}
            mout.write(json.dumps(manifest_entry) + "\n")
            mout.flush()

            if total_packed % 500 == 0:
                log(f"    packed {total_packed} episodes")

    if current_tar is not None:
        current_tar.close()
        assert current_tar_path is not None
        current_tar_path.replace(current_tar_path.with_suffix(".tar"))

    log(f"  {archive}: packed {total_packed} new episodes "
        f"({len(already)} already present)")

    if cleanup_source and total_packed > 0:
        log(f"  [cleanup] removing {src_root}")
        shutil.rmtree(src_root)

    return total_packed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-dir", required=True, type=Path,
                   help="FOR_LEONARDO.md $DATA_DIR (root with keyframes/ manifests/ scores/ ...).")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Destination root (will get keyframes/<archive>/shard-*.tar + manifests/).")
    p.add_argument("--archive", default="all",
                   help="Archive name to pack, or 'all' (default).")
    p.add_argument("--cleanup-source", action="store_true",
                   help="rm -rf <data-dir>/keyframes/<archive>/ after successful pack (frees inodes).")
    args = p.parse_args()

    if not args.data_dir.exists():
        p.error(f"--data-dir not found: {args.data_dir}")

    kf_root = args.data_dir / "keyframes"
    if args.archive == "all":
        archives = sorted(x.name for x in kf_root.iterdir() if x.is_dir())
    else:
        archives = [args.archive]

    log("=" * 60)
    log(f"pack_scored_output.py  data_dir={args.data_dir}")
    log(f"                        output_dir={args.output_dir}")
    log(f"                        archives={len(archives)}  cleanup={args.cleanup_source}")
    log("=" * 60)

    grand = 0
    for arch in archives:
        log(f"\n--- {arch} ---")
        try:
            grand += pack_archive(arch, args.data_dir, args.output_dir,
                                  cleanup_source=args.cleanup_source)
        except Exception as e:
            import traceback
            log(f"  ERROR packing {arch}: {e}")
            traceback.print_exc()

    log("\n" + "=" * 60)
    log(f"Total new episodes packed: {grand}")
    log("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
