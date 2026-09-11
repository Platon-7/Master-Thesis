#!/usr/bin/env python3
"""
pack_rbm1m_ood.py — pack the staged keyframes + scores into the THREE output
buckets used by /projects/prjs1958/robometer_frame_dataset/robometer/:

  output/keyframes/<archive>/shard-NNNNN.tar
        ↳ FAILURES only. meta.json has frame_labels, terminal_reward,
          paired_success_id, keyframes_dir.

  output/keyframes_success/<archive>/shard-NNNNN.tar
        ↳ SUCCESSES from archives that ALSO have failures. meta.json has
          orphan: false (these are paired-success companions).

  output/keyframes_orphan_success/<archive>/shard-NNNNN.tar
        ↳ SUCCESSES from archives that have NO failures. meta.json has
          orphan: true.

Plus the existing manifest convention:
  output/manifests/<archive>_failures.jsonl    (one line per failure episode)
  output/manifests/<archive>_successes.jsonl   (one line per success episode)

Episode dirs in the input come from:
  data_dir/keyframes/<archive>/<ep_id>/frame_NN_TT.TTs.jpg

Episodes are classified into failure vs success via the existing manifest
files at data_dir/manifests/<archive>_{failures,successes}.jsonl.
For failures, frame_labels + terminal_reward come from
data_dir/scores/<archive>_scored.jsonl (post-processed by extract_frame_labels.py).

Usage:
  python pack_rbm1m_ood.py \
      --data-dir /projects/prjs1958/robometer_frame_eval_dataset/staging \
      --output-dir /projects/prjs1958/robometer_frame_eval_dataset
"""
from __future__ import annotations
import argparse
import io
import json
import re
import sys
import tarfile
import time
from pathlib import Path
from typing import Iterable

EPISODES_PER_SHARD = 256
N_KEYFRAMES = 16


def load_jsonl(p: Path) -> dict[str, dict]:
    """Load a JSONL into a dict keyed by episode_id. Empty if file missing."""
    out: dict[str, dict] = {}
    if not p.exists():
        return out
    with p.open() as f:
        for line in f:
            try:
                d = json.loads(line)
                eid = d.get("episode_id")
                if eid:
                    out[eid] = d
            except Exception:
                pass
    return out


def sorted_frames(ep_dir: Path) -> list[Path]:
    pattern = re.compile(r"frame_(\d+)_(\d+\.\d+)s\.jpg")
    pairs: list[tuple[int, Path]] = []
    for p in ep_dir.glob("frame_*.jpg"):
        m = pattern.match(p.name)
        if m:
            pairs.append((int(m.group(1)), p))
    pairs.sort(key=lambda x: x[0])
    return [p for _, p in pairs]


def write_shard_atomic(shard_dir: Path, shard_idx: int, episodes: list[tuple[str, dict, list[Path]]]) -> Path:
    """Write one shard to <shard_dir>/shard-NNNNN.tar. Atomic via .tmp rename.

    episodes: list of (episode_id, meta_dict, [frame_paths])
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    final = shard_dir / f"shard-{shard_idx:05d}.tar"
    tmp = shard_dir / f"shard-{shard_idx:05d}.tar.tmp"
    if tmp.exists():
        tmp.unlink()
    with tarfile.open(tmp, "w") as tar:
        for ep_id, meta, frames in episodes:
            for fp in frames:
                data = fp.read_bytes()
                info = tarfile.TarInfo(name=f"{ep_id}/{fp.name}")
                info.size = len(data)
                info.mtime = int(time.time())
                tar.addfile(info, io.BytesIO(data))
            mb = json.dumps(meta).encode("utf-8")
            mi = tarfile.TarInfo(name=f"{ep_id}/meta.json")
            mi.size = len(mb)
            mi.mtime = int(time.time())
            tar.addfile(mi, io.BytesIO(mb))
    tmp.replace(final)
    return final


def shard_episodes(episodes: list[tuple[str, dict, list[Path]]], shard_dir: Path) -> int:
    """Pack episodes into N shards of EPISODES_PER_SHARD each. Returns # shards written."""
    n_shards = 0
    for i in range(0, len(episodes), EPISODES_PER_SHARD):
        batch = episodes[i: i + EPISODES_PER_SHARD]
        path = write_shard_atomic(shard_dir, n_shards, batch)
        print(f"    wrote {path.name} ({len(batch)} episodes)")
        n_shards += 1
    return n_shards


def build_failure_meta(
    episode_id: str, archive: str, family: str,
    manifest_row: dict, score_row: dict, n_keyframes: int,
) -> dict:
    return {
        "episode_id": episode_id,
        "archive": archive,
        "family": family,
        "task": manifest_row.get("task") or score_row.get("task") or "unknown",
        "label": "failure",
        "terminal_reward": int(score_row.get("terminal_reward", 1)),
        "n_source_frames": manifest_row.get("n_source_frames"),
        "n_keyframes": n_keyframes,
        "keyframes_dir": f"{archive}/{episode_id}",
        "paired_success_id": score_row.get("paired_success_id"),
        "frame_labels": list(score_row.get("frame_labels", [])) or [1] * n_keyframes,
    }


def build_success_meta(
    episode_id: str, archive: str, family: str,
    manifest_row: dict, n_keyframes: int, *, orphan: bool,
) -> dict:
    return {
        "episode_id": episode_id,
        "archive": archive,
        "family": family,
        "task": manifest_row.get("task") or "unknown",
        "label": "successful",
        "orphan": orphan,
        "n_source_frames": manifest_row.get("n_source_frames"),
        "n_keyframes": n_keyframes,
    }


def write_manifest(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    src_keyframes = args.data_dir / "keyframes"
    src_manifests = args.data_dir / "manifests"
    src_scores = args.data_dir / "scores"

    # Discover archives
    archives = sorted(p.name for p in src_keyframes.iterdir() if p.is_dir())
    print(f"Discovered {len(archives)} archives: {archives}\n")

    # Determine paired-vs-orphan for successes: an archive is "paired" if it has BOTH
    # a failures manifest AND a successes manifest.
    pair_status: dict[str, str] = {}  # 'paired' | 'orphan' | 'failures_only'
    for arch in archives:
        has_fail = (src_manifests / f"{arch}_failures.jsonl").exists()
        has_succ = (src_manifests / f"{arch}_successes.jsonl").exists()
        if has_fail and has_succ:
            pair_status[arch] = "paired"
        elif has_succ:
            pair_status[arch] = "orphan"
        elif has_fail:
            pair_status[arch] = "failures_only"
        else:
            pair_status[arch] = "unknown"

    grand_fail = grand_succ = grand_orphan = 0
    for arch in archives:
        family = arch  # we use archive == family for rbm-1m-ood (single-family archives)
        print(f"--- {arch}  (status: {pair_status[arch]}) ---")

        # Load row sources
        fail_rows = load_jsonl(src_manifests / f"{arch}_failures.jsonl")
        succ_rows = load_jsonl(src_manifests / f"{arch}_successes.jsonl")
        score_rows = load_jsonl(src_scores / f"{arch}_scored.jsonl")

        # Walk every episode dir under keyframes/<arch>/, classify, build meta
        ep_dirs = sorted([p for p in (src_keyframes / arch).iterdir() if p.is_dir()])
        fail_eps: list[tuple[str, dict, list[Path]]] = []
        succ_eps: list[tuple[str, dict, list[Path]]] = []
        fail_manifest_rows: list[dict] = []
        succ_manifest_rows: list[dict] = []

        for ep_dir in ep_dirs:
            ep_id = ep_dir.name
            frames = sorted_frames(ep_dir)
            if not frames:
                print(f"    ! {ep_id}: no frames; skipping", file=sys.stderr)
                continue
            if ep_id in fail_rows:
                manifest_row = fail_rows[ep_id]
                score_row = score_rows.get(ep_id, {})
                meta = build_failure_meta(ep_id, arch, family, manifest_row, score_row, len(frames))
                fail_eps.append((ep_id, meta, frames))
                fail_manifest_rows.append(meta)
            elif ep_id in succ_rows:
                manifest_row = succ_rows[ep_id]
                orphan_flag = (pair_status[arch] == "orphan")
                meta = build_success_meta(ep_id, arch, family, manifest_row, len(frames), orphan=orphan_flag)
                succ_eps.append((ep_id, meta, frames))
                succ_manifest_rows.append(meta)
            else:
                print(f"    ! {ep_id}: not in any manifest; skipping", file=sys.stderr)

        # Pack failures → output/keyframes/<arch>/
        if fail_eps:
            shard_dir = args.output_dir / "keyframes" / arch
            shard_episodes(fail_eps, shard_dir)
            write_manifest(fail_manifest_rows, args.output_dir / "manifests" / f"{arch}_failures.jsonl")
            grand_fail += len(fail_eps)
            print(f"  packed {len(fail_eps)} failures → keyframes/{arch}/")

        # Pack successes → keyframes_success or keyframes_orphan_success
        if succ_eps:
            if pair_status[arch] == "orphan":
                bucket = "keyframes_orphan_success"
                grand_orphan += len(succ_eps)
            else:
                bucket = "keyframes_success"
                grand_succ += len(succ_eps)
            shard_dir = args.output_dir / bucket / arch
            shard_episodes(succ_eps, shard_dir)
            # Match the existing convention for success manifests
            mfn = f"{arch}_orphan_successes.jsonl" if pair_status[arch] == "orphan" else f"{arch}_successes.jsonl"
            write_manifest(succ_manifest_rows, args.output_dir / "manifests" / mfn)
            print(f"  packed {len(succ_eps)} successes → {bucket}/{arch}/")

        print()

    print("=" * 60)
    print(f"DONE. failures={grand_fail}  paired_successes={grand_succ}  orphan_successes={grand_orphan}")
    print(f"      total = {grand_fail + grand_succ + grand_orphan}")


if __name__ == "__main__":
    main()
