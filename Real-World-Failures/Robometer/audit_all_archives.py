#!/usr/bin/env python3
"""
audit_all_archives.py — Scan every archive in the Robometer dataset and report
success/failure counts per archive. Identifies orphan successes (archives with
successes but zero failures anywhere in their robot family).

Output: prints full table + writes JSON report to --output.
"""

import json
import os
import subprocess
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

ARCH_ROOT = Path("/projects/prjs1958/robometer_full_dataset/raw_archives")
SINGLE_DIR = ARCH_ROOT / "single"
SPLIT_DIR  = ARCH_ROOT / "split"

# Excluded per user request: humanoid + hand + metaworld + failsafe + droid
EXCLUDED = {
    "jesbu1_oxe_rfm_oxe_droid",
    "aliangdw_metaworld_metaworld_eval",
    "aliangdw_metaworld_metaworld_train",
    "jesbu1_hand_paired_rfm_hand_paired_human",
    "jesbu1_hand_paired_rfm_hand_paired_robot",
    "jesbu1_humanoid_everyday_rfm_humanoid_everyday_rfm",
    "jesbu1_failsafe_rfm_failsafe",
    "jesbu1_usc_koch_human_robot_paired_usc_koch_human_robot_paired_human",
}


def log(msg):
    print(msg, flush=True)


def read_index_mappings(archive_name):
    """Extract index_mappings.json from a single or split archive.

    Tries multiple internal layouts — different archives wrap the JSON
    under different prefixes (some at archive root, some under
    processed_datasets/).
    """
    candidates = [
        f"{archive_name}/index_mappings.json",
        f"./{archive_name}/index_mappings.json",
        f"processed_datasets/{archive_name}/index_mappings.json",
        f"./processed_datasets/{archive_name}/index_mappings.json",
    ]
    single_tar = SINGLE_DIR / f"{archive_name}.tar"
    split_dir  = SPLIT_DIR / archive_name

    if single_tar.exists():
        with tarfile.open(single_tar, "r") as t:
            for p in candidates:
                try:
                    f = t.extractfile(p)
                    if f is not None:
                        return json.loads(f.read())
                except KeyError:
                    continue
        return None

    if split_dir.exists():
        parts = sorted(split_dir.glob(f"{archive_name}.tar.part-*"))
        if not parts:
            return None
        for p in candidates:
            cat = subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE)
            tar = subprocess.Popen(
                ["tar", "-xOf", "-", p],
                stdin=cat.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            cat.stdout.close()
            data, _ = tar.communicate(timeout=600)
            cat.wait()
            if data:
                return json.loads(data)
        return None
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="/projects/prjs1958/robometer_full_dataset/audit_report.json",
        help="Path to write JSON report",
    )
    parser.add_argument(
        "--no-exclusions", action="store_true",
        help="Scan EVERY archive — ignore the EXCLUDED set. Use this for a full, "
             "no-blind-spots audit (droid, metaworld, failsafe, hand_paired, etc.).",
    )
    args = parser.parse_args()

    global EXCLUDED
    if args.no_exclusions:
        log("--no-exclusions: scanning every archive (EXCLUDED set ignored)")
        EXCLUDED = set()

    # Collect all archive names
    all_archives = set()
    for f in os.listdir(SINGLE_DIR):
        if f.endswith(".tar"):
            all_archives.add(f.replace(".tar", ""))
    for d in os.listdir(SPLIT_DIR):
        if os.path.isdir(SPLIT_DIR / d):
            all_archives.add(d)

    log(f"Found {len(all_archives)} total archives ({len(EXCLUDED)} excluded)")
    log(f"Scanning {len(all_archives) - len(all_archives & EXCLUDED)} archives...\n")

    results = []
    errors = []

    for i, arch in enumerate(sorted(all_archives)):
        if arch in EXCLUDED:
            log(f"  [{i+1}/{len(all_archives)}] SKIP (excluded): {arch}")
            continue
        log(f"  [{i+1}/{len(all_archives)}] Scanning: {arch}")
        try:
            idx = read_index_mappings(arch)
            if idx is None:
                errors.append({"archive": arch, "error": "could not read index_mappings.json"})
                log(f"    -> ERROR: could not read index_mappings.json")
                continue
            qi = idx.get("quality_indices", {})
            ns = len(qi.get("successful", []))
            nf = len(qi.get("failure", []))
            np_ = len(qi.get("partial_success", []))
            results.append({
                "archive": arch,
                "successes": ns,
                "failures": nf,
                "partial_success": np_,
            })
            log(f"    -> succ={ns}  fail={nf}  partial={np_}")
        except Exception as e:
            errors.append({"archive": arch, "error": str(e)})
            log(f"    -> ERROR: {e}")

    # Compute totals
    total_succ = sum(r["successes"] for r in results)
    total_fail = sum(r["failures"] for r in results)
    total_partial = sum(r["partial_success"] for r in results)

    # Identify orphan successes: archives with successes but zero failures
    orphans = [r for r in results if r["successes"] > 0 and r["failures"] == 0]
    orphan_succ = sum(r["successes"] for r in orphans)

    # Archives with both
    both = [r for r in results if r["successes"] > 0 and r["failures"] > 0]

    # Archives with only failures
    fail_only = [r for r in results if r["failures"] > 0 and r["successes"] == 0]

    log(f"\n{'='*70}")
    log(f"RESULTS (excluding {len(EXCLUDED)} excluded archives)")
    log(f"{'='*70}")
    log(f"Total archives scanned: {len(results)}")
    log(f"Total successes:        {total_succ}")
    log(f"Total failures:         {total_fail}")
    log(f"Total partial_success:  {total_partial}")
    log(f"Grand total episodes:   {total_succ + total_fail + total_partial}")

    log(f"\n--- Archives with BOTH successes and failures: {len(both)} ---")
    for r in sorted(both, key=lambda x: -x["successes"]):
        log(f"  {r['archive']}: {r['successes']} succ, {r['failures']} fail")

    log(f"\n--- ORPHAN successes (succ > 0, fail = 0): {len(orphans)} archives, {orphan_succ} episodes ---")
    for r in sorted(orphans, key=lambda x: -x["successes"]):
        log(f"  {r['archive']}: {r['successes']} successes")

    log(f"\n--- Failure-only archives (fail > 0, succ = 0): {len(fail_only)} ---")
    for r in sorted(fail_only, key=lambda x: -x["failures"]):
        log(f"  {r['archive']}: {r['failures']} failures")

    if errors:
        log(f"\n--- ERRORS ({len(errors)}) ---")
        for e in errors:
            log(f"  {e['archive']}: {e['error']}")

    # Write JSON report
    report = {
        "total_successes": total_succ,
        "total_failures": total_fail,
        "total_partial_success": total_partial,
        "total_episodes": total_succ + total_fail + total_partial,
        "archives_scanned": len(results),
        "archives_excluded": sorted(EXCLUDED),
        "per_archive": sorted(results, key=lambda x: x["archive"]),
        "orphan_successes": sorted(orphans, key=lambda x: -x["successes"]),
        "orphan_success_total": orphan_succ,
        "errors": errors,
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    log(f"\nJSON report written to: {args.output}")


if __name__ == "__main__":
    main()
