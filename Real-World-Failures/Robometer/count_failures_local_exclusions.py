#!/usr/bin/env python3
"""
Count RoboMeter failure episodes from locally downloaded archives and report:
1) Failures excluding MetaWorld, human-hand, and humanoid datasets
2) Total failures for human-hand datasets
3) Total failures for humanoid datasets

This script reads from a local full-dataset mirror created by
`download_full_robometer_dataset.py`:
  <root>/raw_archives/single/*.tar
  <root>/raw_archives/split/<archive>/*.tar.part-*

Design goals:
- Be robust for very large archives.
- Prefer fast tail scanning of tar files.
- Fall back to full archive parsing when needed.
- Persist machine-readable results to JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tarfile
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent))
from datasets_catalog import DATASETS


TAR_RE = re.compile(r"^(?P<archive>.+)\.tar$")
PART_RE = re.compile(r"^(?P<archive>.+)\.tar\.(?P<part>part-[a-z0-9]+)$")
RANGE_SIZES_MB = (5, 50, 200)


def _parse_tar_header(block: bytes):
    if len(block) < 512:
        return None
    if block[:512] == b"\x00" * 512:
        return None

    magic = block[257:263]
    if not (magic.startswith(b"ustar") or magic.startswith(b"\x00")):
        return None

    name = block[:100].split(b"\x00")[0].decode("utf-8", errors="replace").strip()
    prefix = block[345:500].split(b"\x00")[0].decode("utf-8", errors="replace").strip()
    if prefix:
        name = prefix + "/" + name

    size_field = block[124:136].split(b"\x00")[0].decode("ascii", errors="replace").strip()
    try:
        size = int(size_field, 8) if size_field else 0
    except ValueError:
        size = 0

    return name, size


def _scan_tail_for_mapping(data: bytes, range_start: int) -> dict | None:
    align = 512 - (range_start % 512) if range_start % 512 else 0
    data = data[align:]

    for i in range(0, len(data) - 512, 512):
        parsed = _parse_tar_header(data[i : i + 512])
        if parsed is None:
            continue

        name, entry_size = parsed
        if "index_mappings.json" not in name:
            continue

        start = i + 512
        end = start + entry_size
        if end > len(data):
            return None

        try:
            return json.loads(data[start:end])
        except json.JSONDecodeError:
            return None

    return None


def _read_tail_single(path: Path, nbytes: int) -> tuple[bytes, int]:
    size = path.stat().st_size
    start = max(0, size - nbytes)
    with open(path, "rb") as f:
        f.seek(start)
        return f.read(), start


def _read_tail_split(parts: list[Path], nbytes: int) -> tuple[bytes, int]:
    sizes = [p.stat().st_size for p in parts]
    total = sum(sizes)
    start = max(0, total - nbytes)

    out = bytearray()
    current = 0
    for part, size in zip(parts, sizes):
        part_end = current + size
        if part_end <= start:
            current = part_end
            continue

        local_start = max(0, start - current)
        with open(part, "rb") as f:
            f.seek(local_start)
            out.extend(f.read())

        current = part_end

    return bytes(out), start


def _fallback_single(path: Path) -> dict | None:
    with tarfile.open(path, mode="r:") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith("index_mappings.json"):
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            return json.loads(extracted.read())
    return None


def _fallback_split(parts: list[Path]) -> dict | None:
    parts_quoted = " ".join(str(p) for p in parts)
    cmd = (
        f"cat {parts_quoted} | "
        "tar -xOf - --wildcards --no-anchored 'index_mappings.json'"
    )
    proc = subprocess.run(cmd, shell=True, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return None

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def _count_from_mapping(mapping: dict) -> tuple[int, int]:
    quality = mapping.get("quality_indices", {})
    failures = len(quality.get("failure", []))
    total = sum(len(v) for v in quality.values())
    return failures, total


def _build_archive_index(raw_root: Path):
    single_root = raw_root / "single"
    split_root = raw_root / "split"

    singles: dict[str, Path] = {}
    splits: dict[str, list[Path]] = {}

    if single_root.exists():
        for f in single_root.glob("*.tar"):
            m = TAR_RE.match(f.name)
            if m:
                singles[m.group("archive")] = f

    if split_root.exists():
        for d in split_root.iterdir():
            if not d.is_dir():
                continue
            parts = sorted([p for p in d.glob("*.tar.part-*") if PART_RE.match(p.name)])
            if parts:
                splits[d.name] = parts

    return singles, splits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="/projects/prjs1958/robometer_full_dataset",
        help="Root directory containing raw_archives/single and raw_archives/split",
    )
    parser.add_argument(
        "--output",
        default="/projects/prjs1958/robometer_full_dataset/failure_summary_exclusions.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()

    raw_root = Path(args.dataset_root).expanduser().resolve() / "raw_archives"
    singles, splits = _build_archive_index(raw_root)

    results: dict[str, dict] = {}

    for ds in DATASETS:
        archive = ds["archive"]
        category = ds["category"]
        description = ds.get("description", "")

        mapping = None
        method = ""

        if archive in singles:
            src = ("single", singles[archive])
        elif archive in splits:
            src = ("split", splits[archive])
        else:
            results[archive] = {
                "category": category,
                "description": description,
                "failures": None,
                "total": None,
                "method": "missing-local",
            }
            continue

        for mb in RANGE_SIZES_MB:
            nbytes = mb * 1024 * 1024
            if src[0] == "single":
                tail, start = _read_tail_single(src[1], nbytes)
            else:
                tail, start = _read_tail_split(src[1], nbytes)

            mapping = _scan_tail_for_mapping(tail, start)
            if mapping is not None:
                method = f"tail-{mb}MB"
                break

        if mapping is None:
            if src[0] == "single":
                mapping = _fallback_single(src[1])
                method = "fallback-single"
            else:
                mapping = _fallback_split(src[1])
                method = "fallback-split"

        if mapping is None:
            results[archive] = {
                "category": category,
                "description": description,
                "failures": None,
                "total": None,
                "method": "unresolved",
            }
            continue

        failures, total = _count_from_mapping(mapping)
        results[archive] = {
            "category": category,
            "description": description,
            "failures": failures,
            "total": total,
            "method": method,
        }

    metaworld = {d["archive"] for d in DATASETS if "metaworld" in d["archive"].lower()}
    human_hand = {d["archive"] for d in DATASETS if "hand_paired" in d["archive"].lower()}
    humanoid = {
        d["archive"]
        for d in DATASETS
        if "humanoid" in d["archive"].lower() or "humanoid" in d.get("description", "").lower()
    }

    included = set(results) - metaworld - human_hand - humanoid

    def aggregate(group: set[str]):
        total_fail = 0
        unresolved = []
        for archive in sorted(group):
            entry = results[archive]
            if entry["failures"] is None:
                unresolved.append(archive)
            else:
                total_fail += int(entry["failures"])
        return total_fail, unresolved

    included_fail, included_unresolved = aggregate(included)
    human_hand_fail, human_hand_unresolved = aggregate(human_hand)
    humanoid_fail, humanoid_unresolved = aggregate(humanoid)

    report = {
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "archives_in_catalog": len(DATASETS),
        "archives_resolved": sum(1 for r in results.values() if r["failures"] is not None),
        "archives_unresolved": [a for a, r in results.items() if r["failures"] is None],
        "requested_counts": {
            "failures_excluding_metaworld_human_hand_humanoid": included_fail,
            "human_hand_failures": human_hand_fail,
            "humanoid_failures": humanoid_fail,
        },
        "requested_unresolved": {
            "excluded_group": included_unresolved,
            "human_hand_group": human_hand_unresolved,
            "humanoid_group": humanoid_unresolved,
        },
        "groups": {
            "metaworld_archives": sorted(metaworld),
            "human_hand_archives": sorted(human_hand),
            "humanoid_archives": sorted(humanoid),
        },
        "per_archive": results,
    }

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Wrote report:", out)
    print("failures_excluding_metaworld_human_hand_humanoid:", included_fail)
    print("human_hand_failures:", human_hand_fail)
    print("humanoid_failures:", humanoid_fail)

    if report["requested_unresolved"]["excluded_group"] or report["requested_unresolved"]["human_hand_group"] or report["requested_unresolved"]["humanoid_group"]:
        print("WARNING: Some archives were unresolved; see requested_unresolved in report")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
