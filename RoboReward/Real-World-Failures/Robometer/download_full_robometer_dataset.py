#!/usr/bin/env python3
"""
Download all RoboMeter processed_datasets archives to a scratch folder.

Key properties:
- Downloads every .tar and .tar.part-* file from
  robometer/processed_datasets on HuggingFace.
- Organizes files into a clean layout:
    <dest>/raw_archives/single/<archive>.tar
    <dest>/raw_archives/split/<archive>/<archive>.tar.part-xx
- Skips already complete files (size match).
- Resumes partial files via HTTP Range requests.
- Optionally adopts existing local downloads (copy or hardlink) before downloading.

Examples:
  python download_full_robometer_dataset.py \
      --dest /scratch-shared/pkarageorgis/robometer_full_dataset

  python download_full_robometer_dataset.py \
      --dest /scratch-shared/pkarageorgis/robometer_full_dataset \
      --adopt-from /scratch-shared/pkarageorgis \
      --adopt-mode hardlink

  HF_TOKEN=... python download_full_robometer_dataset.py --dest ... --workers 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from huggingface_hub import HfApi


REPO_ID = "robometer/processed_datasets"
BASE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"

TAR_RE = re.compile(r"^(?P<archive>.+)\.tar$")
PART_RE = re.compile(r"^(?P<archive>.+)\.tar\.(?P<part>part-[a-z0-9]+)$")


@dataclass(frozen=True)
class RemoteFile:
    remote_path: str
    size: int
    archive: str
    part: str | None


def _session(token: str) -> requests.Session:
    s = requests.Session()
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    return s


def _discover_remote_files(token: str) -> list[RemoteFile]:
    api = HfApi(token=token or None)
    info = api.dataset_info(repo_id=REPO_ID, files_metadata=True)

    files: list[RemoteFile] = []
    for sib in info.siblings:
        rp = sib.rfilename
        m1 = TAR_RE.match(rp)
        if m1:
            files.append(
                RemoteFile(
                    remote_path=rp,
                    size=int(sib.size or 0),
                    archive=m1.group("archive"),
                    part=None,
                )
            )
            continue

        m2 = PART_RE.match(rp)
        if m2:
            files.append(
                RemoteFile(
                    remote_path=rp,
                    size=int(sib.size or 0),
                    archive=m2.group("archive"),
                    part=m2.group("part"),
                )
            )

    files.sort(key=lambda x: x.remote_path)
    return files


def _target_path(dest_root: Path, rf: RemoteFile) -> Path:
    if rf.part is None:
        return dest_root / "raw_archives" / "single" / rf.remote_path
    return dest_root / "raw_archives" / "split" / rf.archive / rf.remote_path


def _shard_owner(key: str, num_shards: int) -> int:
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % num_shards


def _scan_candidates(paths: Iterable[Path]) -> dict[str, Path]:
    """Map basename -> first discovered path for tar / tar.part files."""
    out: dict[str, Path] = {}
    for p in paths:
        if not p.exists() or not p.is_dir():
            continue
        for f in p.rglob("*"):
            if not f.is_file():
                continue
            name = f.name
            if TAR_RE.match(name) or PART_RE.match(name):
                out.setdefault(name, f)
    return out


def _adopt_if_available(target: Path, expected_size: int, existing_by_name: dict[str, Path], mode: str) -> bool:
    """Adopt existing file into target location using hardlink/copy if valid."""
    if target.exists():
        return target.stat().st_size == expected_size

    src = existing_by_name.get(target.name)
    if src is None or not src.exists():
        return False

    if expected_size > 0 and src.stat().st_size != expected_size:
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        os.link(src, target)
    else:
        shutil.copy2(src, target)
    return True


def _download_one(
    rf: RemoteFile,
    dest_root: Path,
    token: str,
    retries: int,
    timeout_s: int,
    chunk_mb: int,
) -> tuple[str, Path, int]:
    """
    Return status tuple: (status, target_path, bytes_written)
    status in {"skipped", "downloaded", "resumed", "failed"}
    """
    target = _target_path(dest_root, rf)
    expected_size = rf.size
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        cur = target.stat().st_size
        if expected_size > 0 and cur == expected_size:
            return "skipped", target, 0
        if expected_size > 0 and cur > expected_size:
            target.unlink()
            cur = 0
    else:
        cur = 0

    url = f"{BASE_URL}/{rf.remote_path}"
    wrote = 0
    chunk_size = max(1, chunk_mb) * 1024 * 1024

    for attempt in range(1, retries + 1):
        try:
            sess = _session(token)
            headers = {}
            mode = "wb"
            start = 0
            status = "downloaded"

            if target.exists():
                cur = target.stat().st_size
                if cur > 0:
                    headers["Range"] = f"bytes={cur}-"
                    mode = "ab"
                    start = cur
                    status = "resumed"

            with sess.get(url, stream=True, timeout=timeout_s, headers=headers) as r:
                if r.status_code in (200, 206):
                    # If range requested but server responded 200, restart full file.
                    if "Range" in headers and r.status_code == 200:
                        target.unlink(missing_ok=True)
                        mode = "wb"
                        start = 0
                        status = "downloaded"

                    with open(target, mode) as f:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if not chunk:
                                continue
                            f.write(chunk)
                            wrote += len(chunk)
                else:
                    raise RuntimeError(f"HTTP {r.status_code}")

            final_size = target.stat().st_size if target.exists() else 0
            if expected_size > 0 and final_size != expected_size:
                raise RuntimeError(
                    f"size mismatch after download for {rf.remote_path}: got {final_size}, expected {expected_size}"
                )
            return status, target, wrote

        except Exception:
            if attempt >= retries:
                return "failed", target, wrote
            time.sleep(min(30, 2 * attempt))

    return "failed", target, wrote


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True, help="Destination root directory on scratch")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--chunk-mb", type=int, default=8)
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of concurrent shards/jobs",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Current shard index in [0, num-shards)",
    )
    parser.add_argument(
        "--adopt-from",
        action="append",
        default=[],
        help="Existing directory to scan for already downloaded .tar / .tar.part-* files",
    )
    parser.add_argument(
        "--adopt-mode",
        choices=["hardlink", "copy"],
        default="hardlink",
        help="How to place adopted files into destination",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1:
        print("ERROR: --num-shards must be >= 1")
        return 1
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        print("ERROR: --shard-index must satisfy 0 <= shard-index < num-shards")
        return 1

    report_name = (
        f"download_report_shard{args.shard_index:02d}_of_{args.num_shards:02d}.json"
        if args.num_shards > 1
        else "download_report.json"
    )

    dest_root = Path(args.dest).expanduser().resolve()
    (dest_root / "raw_archives" / "single").mkdir(parents=True, exist_ok=True)
    (dest_root / "raw_archives" / "split").mkdir(parents=True, exist_ok=True)

    print(f"Discovering files in {REPO_ID} ...")
    remote_files = _discover_remote_files(args.hf_token)
    if not remote_files:
        print("No .tar/.tar.part files found in dataset repo.")
        return 1

    n_single = sum(1 for f in remote_files if f.part is None)
    n_parts = len(remote_files) - n_single
    total_bytes = sum(f.size for f in remote_files if f.size > 0)
    print(f"Found {len(remote_files)} archive objects ({n_single} single + {n_parts} split parts)")
    print(f"Remote size total: {total_bytes / (1024**4):.2f} TiB")

    if args.num_shards > 1:
        all_count = len(remote_files)
        remote_files = [
            rf for rf in remote_files
            if _shard_owner(rf.remote_path, args.num_shards) == args.shard_index
        ]
        shard_bytes = sum(f.size for f in remote_files if f.size > 0)
        print(
            f"Shard mode: {args.shard_index + 1}/{args.num_shards} -> "
            f"{len(remote_files)}/{all_count} files, {shard_bytes / (1024**4):.2f} TiB"
        )

    existing_by_name: dict[str, Path] = {}
    if args.adopt_from:
        print("Scanning adopt-from paths for existing files ...")
        existing_by_name = _scan_candidates(Path(p).expanduser().resolve() for p in args.adopt_from)
        print(f"Adopt scan found {len(existing_by_name)} candidate files")

    summary = {
        "timestamp": int(time.time()),
        "repo": REPO_ID,
        "dest": str(dest_root),
        "counts": {
            "total_remote_files": len(remote_files),
            "single_files": n_single,
            "split_parts": n_parts,
            "skipped": 0,
            "adopted": 0,
            "downloaded": 0,
            "resumed": 0,
            "failed": 0,
        },
        "files": [],
        "shard": {
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
        },
    }

    # First pass: skip/adopt classification.
    to_download: list[RemoteFile] = []
    for rf in remote_files:
        tgt = _target_path(dest_root, rf)
        done = False
        if tgt.exists() and rf.size > 0 and tgt.stat().st_size == rf.size:
            summary["counts"]["skipped"] += 1
            summary["files"].append({"remote": rf.remote_path, "target": str(tgt), "status": "skipped"})
            done = True
        elif existing_by_name:
            try:
                adopted = _adopt_if_available(tgt, rf.size, existing_by_name, args.adopt_mode)
            except Exception:
                adopted = False
            if adopted:
                summary["counts"]["adopted"] += 1
                summary["files"].append({"remote": rf.remote_path, "target": str(tgt), "status": "adopted"})
                done = True

        if not done:
            to_download.append(rf)

    print(
        "Plan: "
        f"skip={summary['counts']['skipped']}, "
        f"adopt={summary['counts']['adopted']}, "
        f"download={len(to_download)}"
    )

    if args.dry_run:
        out = dest_root / report_name
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Dry run complete. Report: {out}")
        return 0

    if args.workers < 1:
        args.workers = 1

    lock = threading.Lock()

    def _record(status: str, rf: RemoteFile, target: Path, bytes_written: int):
        with lock:
            summary["counts"][status] += 1
            summary["files"].append(
                {
                    "remote": rf.remote_path,
                    "target": str(target),
                    "status": status,
                    "bytes_written": bytes_written,
                }
            )

    print("Starting downloads ...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                _download_one,
                rf,
                dest_root,
                args.hf_token,
                args.retries,
                args.timeout,
                args.chunk_mb,
            ): rf
            for rf in to_download
        }

        done_n = 0
        for fut in as_completed(futures):
            rf = futures[fut]
            try:
                status, target, bw = fut.result()
            except Exception:
                status, target, bw = "failed", _target_path(dest_root, rf), 0
            _record(status, rf, target, bw)
            done_n += 1
            if done_n % 10 == 0 or status == "failed":
                print(
                    f"Progress {done_n}/{len(to_download)} | "
                    f"downloaded={summary['counts']['downloaded']} "
                    f"resumed={summary['counts']['resumed']} "
                    f"failed={summary['counts']['failed']}"
                )

    out = dest_root / report_name
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"Report: {out}")
    print(
        "Counts: "
        f"skipped={summary['counts']['skipped']} "
        f"adopted={summary['counts']['adopted']} "
        f"downloaded={summary['counts']['downloaded']} "
        f"resumed={summary['counts']['resumed']} "
        f"failed={summary['counts']['failed']}"
    )

    return 0 if summary["counts"]["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
