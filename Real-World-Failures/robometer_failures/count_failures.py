"""
count_failures.py

Counts failure episodes in each RoboMeter archive by fetching just the
last 5 MB of each tar via HTTP range request, parsing out index_mappings.json.

Includes: all archives except MetaWorld and human_egocentric.
Includes roboreward archives (unlike the main pipeline).
Skips: split-part archives (too complex to range-request reliably).

Usage:
    python count_failures.py [--timeout 30]

Output:
    Prints a table and a timing estimate based on DROID throughput.
"""

import os
import sys
import json
import time
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from datasets_catalog import DATASETS

HF_TOKEN = os.environ.get("HF_TOKEN", "")
BASE_URL = "https://huggingface.co/datasets/robometer/processed_datasets/resolve/main"
RANGE_SIZES = [5, 50, 200]  # MB — tried in order until index_mappings.json is found

# Human/egocentric archives to skip (our VLM prompts are robot-centric)
HUMAN_EGOCENTRIC_ARCHIVES = {
    d["archive"] for d in DATASETS if d["category"] == "human_egocentric"
}
# MetaWorld: user has own failures
METAWORLD_ARCHIVES = {d["archive"] for d in DATASETS if "metaworld" in d["archive"]}

SKIP_ARCHIVES = HUMAN_EGOCENTRIC_ARCHIVES | METAWORLD_ARCHIVES

# DROID throughput baseline (from actual run, 2-GPU A100 job)
EPISODES_PER_HOUR = 118.6   # episodes/hour on 1 job (2 GPUs)
MAX_HOURS_PER_JOB = 22.0    # conservative (24h limit minus overhead)
EPISODES_PER_JOB = int(EPISODES_PER_HOUR * MAX_HOURS_PER_JOB)  # ~2607


def _parse_tar_header(block: bytes):
    """Parse a 512-byte POSIX tar header block. Returns (name, size) or None."""
    if len(block) < 512:
        return None
    if block[:512] == b"\x00" * 512:
        return None  # end-of-archive block
    # Magic at offset 257
    magic = block[257:263]
    if not (magic.startswith(b"ustar") or magic.startswith(b"\x00")):
        return None
    name = block[:100].split(b"\x00")[0].decode("utf-8", errors="replace").strip()
    # Prefix (POSIX) at offset 345
    prefix = block[345:500].split(b"\x00")[0].decode("utf-8", errors="replace").strip()
    if prefix:
        name = prefix + "/" + name
    # Size in octal at offset 124
    size_field = block[124:136].split(b"\x00")[0].decode("ascii", errors="replace").strip()
    try:
        size = int(size_field, 8) if size_field else 0
    except ValueError:
        size = 0
    return name, size


def _scan_for_mapping(data: bytes, range_start: int) -> dict | None:
    """Scan a byte buffer (fetched from range_start in the file) for index_mappings.json."""
    align = 512 - (range_start % 512) if range_start % 512 else 0
    data = data[align:]
    for i in range(0, len(data) - 512, 512):
        block = data[i : i + 512]
        parsed = _parse_tar_header(block)
        if parsed is None:
            continue
        name, entry_size = parsed
        if "index_mappings.json" not in name:
            continue
        file_start = i + 512
        file_end = file_start + entry_size
        if file_end > len(data):
            return {"status": "cut-off", "need": entry_size}
        try:
            mapping = json.loads(data[file_start:file_end])
        except json.JSONDecodeError as e:
            return {"status": "json-error", "note": str(e)[:80]}
        q = mapping.get("quality_indices", {})
        fail_list = q.get("failure", [])
        total = sum(len(v) for v in q.values())
        return {"status": "ok", "failures": len(fail_list), "total": total}
    return None


def get_failure_count(archive_name: str, session: requests.Session, timeout: int = 60) -> dict:
    """
    Fetch the tail of the archive tar via HTTP range requests (adaptive size)
    and parse index_mappings.json to count failures.
    Returns dict with keys: failures, total, status, note.
    """
    url = f"{BASE_URL}/{archive_name}.tar"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    # 1. HEAD to get file size
    try:
        r = session.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        if r.status_code == 404:
            return {"failures": None, "total": None, "status": "404", "note": "not found"}
        content_length = r.headers.get("Content-Length") or r.headers.get("content-length")
        if not content_length:
            return {"failures": None, "total": None, "status": "no-size", "note": "no Content-Length"}
        file_size = int(content_length)
    except Exception as e:
        return {"failures": None, "total": None, "status": "error", "note": str(e)[:80]}

    # 2. Try increasing range sizes until we find the JSON
    for mb in RANGE_SIZES:
        fetch_bytes = mb * 1024 * 1024
        range_start = max(0, file_size - fetch_bytes)
        try:
            r = session.get(
                url,
                headers={**headers, "Range": f"bytes={range_start}-"},
                timeout=timeout,
            )
            if r.status_code not in (200, 206):
                return {"failures": None, "total": None, "status": str(r.status_code), "note": "range req failed"}
            data = r.content
        except Exception as e:
            return {"failures": None, "total": None, "status": "error", "note": str(e)[:80]}

        result = _scan_for_mapping(data, range_start)
        if result is None:
            continue  # not found yet, try larger window
        if result["status"] == "ok":
            return {"failures": result["failures"], "total": result["total"],
                    "status": f"ok@{mb}MB", "note": ""}
        if result["status"] == "cut-off":
            continue  # JSON header found but data cut off, need larger window
        # json-error or other
        return {"failures": None, "total": None, "status": result["status"],
                "note": result.get("note", "")}

    return {
        "failures": None, "total": None,
        "status": "not-found-in-range",
        "note": f"index_mappings.json not in last {RANGE_SIZES[-1]}MB — archive may be compressed",
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--include-splits", action="store_true",
                        help="Also attempt split-part archives (unreliable)")
    args = parser.parse_args()

    # Build list of archives to scan
    to_scan = []
    for d in DATASETS:
        name = d["archive"]
        cat = d["category"]
        is_split = d.get("split_parts", False)

        if name in SKIP_ARCHIVES:
            continue
        if cat == "human_egocentric":
            continue
        if is_split and not args.include_splits:
            continue
        to_scan.append(d)

    print(f"Archives to scan: {len(to_scan)}")
    print(f"Skipping: {len(SKIP_ARCHIVES)} (MetaWorld + human egocentric)")
    print(f"Skipping split-part archives: {sum(1 for d in DATASETS if d.get('split_parts') and d['archive'] not in SKIP_ARCHIVES and d['category'] != 'human_egocentric')}")
    print()

    session = requests.Session()
    results = []
    total_failures = 0
    total_episodes = 0
    unknown_count = 0

    print(f"{'Archive':<65} {'Failures':>9} {'Total':>9} {'Rate':>6}  Status")
    print("-" * 100)

    for d in to_scan:
        name = d["archive"]
        t0 = time.time()
        res = get_failure_count(name, session, timeout=args.timeout)
        elapsed = time.time() - t0

        f = res["failures"]
        tot = res["total"]
        status = res["status"]
        note = res["note"]

        if f is not None:
            total_failures += f
            if tot:
                total_episodes += tot
            rate = f"{f/tot*100:.0f}%" if tot else "100%"
            print(f"{name:<65} {f:>9,} {tot:>9,} {rate:>6}  {status} ({elapsed:.1f}s)")
        else:
            unknown_count += 1
            print(f"{name:<65} {'?':>9} {'?':>9} {'?':>6}  {status}: {note} ({elapsed:.1f}s)")

        results.append({"archive": name, **res})
        sys.stdout.flush()

    print("-" * 100)
    print(f"\n{'TOTAL (known archives)':<65} {total_failures:>9,} {total_episodes:>9,}")
    if unknown_count:
        print(f"  + {unknown_count} archives with unknown failure count (see status above)")
    print()

    # Save results
    out_path = Path(__file__).parent / "failure_counts.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "total_failures_known": total_failures}, f, indent=2)
    print(f"Saved to {out_path}")

    # ── Timing estimate ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("SCORING TIME ESTIMATE (based on DROID run)")
    print("=" * 60)
    print(f"Throughput: {EPISODES_PER_HOUR:.0f} episodes/hour per 2-GPU A100 job")
    print(f"  (VLM Qwen3.5-35B-A3B: 3.9s/ep, LLM Qwen3-32B: 26.4s/ep)")
    print()

    for n_episodes in ([total_failures] if total_failures else []):
        hours_1job = n_episodes / EPISODES_PER_HOUR
        n_jobs = -(-n_episodes // EPISODES_PER_JOB)  # ceil division
        wall_hours = hours_1job / n_jobs
        print(f"Failures to score: {n_episodes:,}")
        print(f"  Single job (sequential): {hours_1job:.0f}h = {hours_1job/24:.1f} days")
        print(f"  With {n_jobs} parallel shards ({EPISODES_PER_JOB} ep/job): ~{wall_hours:.1f}h wall time")
        print(f"  Recommended: {n_jobs} SLURM jobs × 24h each")


if __name__ == "__main__":
    main()
