#!/usr/bin/env python3
"""Re-scan only the archives that errored in audit_report.json and merge
the fresh results back in. Avoids a full 1.5h re-audit when the only
issue was a path-lookup bug.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audit_all_archives import read_index_mappings

REPORT = Path("/projects/prjs1958/robometer_full_dataset/audit_report.json")


def main():
    with open(REPORT) as f:
        rep = json.load(f)

    errored = list(rep.get("errors", []))
    if not errored:
        print("No errored archives — nothing to do.")
        return

    print(f"Re-scanning {len(errored)} previously errored archives...\n")

    fixed_results = []
    still_errored = []
    for e in errored:
        arch = e["archive"]
        print(f"  Scanning: {arch}")
        try:
            idx = read_index_mappings(arch)
            if idx is None:
                still_errored.append({"archive": arch, "error": "still not found after patch"})
                print(f"    -> STILL FAILS")
                continue
            qi = idx.get("quality_indices", {})
            ns = len(qi.get("successful", []))
            nf = len(qi.get("failure", []))
            np_ = len(qi.get("partial_success", []))
            fixed_results.append({
                "archive": arch,
                "successes": ns,
                "failures": nf,
                "partial_success": np_,
            })
            print(f"    -> succ={ns}  fail={nf}  partial={np_}")
        except Exception as ex:
            still_errored.append({"archive": arch, "error": str(ex)})
            print(f"    -> ERROR: {ex}")

    # Merge: drop the fixed ones from errors, append to per_archive
    fixed_names = {r["archive"] for r in fixed_results}
    rep["errors"] = [e for e in rep["errors"] if e["archive"] not in fixed_names]
    rep["errors"].extend(still_errored)

    per_archive = list(rep["per_archive"])
    # Replace if present, else append
    by_name = {r["archive"]: r for r in per_archive}
    for r in fixed_results:
        by_name[r["archive"]] = r
    per_archive = sorted(by_name.values(), key=lambda x: x["archive"])
    rep["per_archive"] = per_archive

    # Recompute totals
    rep["total_successes"] = sum(r["successes"] for r in per_archive)
    rep["total_failures"] = sum(r["failures"] for r in per_archive)
    rep["total_partial_success"] = sum(r["partial_success"] for r in per_archive)
    rep["total_episodes"] = (
        rep["total_successes"] + rep["total_failures"] + rep["total_partial_success"]
    )
    rep["archives_scanned"] = len(per_archive)

    # Recompute orphan_successes
    orphans = [r for r in per_archive if r["successes"] > 0 and r["failures"] == 0]
    rep["orphan_successes"] = sorted(orphans, key=lambda x: -x["successes"])
    rep["orphan_success_total"] = sum(r["successes"] for r in orphans)

    with open(REPORT, "w") as f:
        json.dump(rep, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Fixed: {len(fixed_results)} archives")
    print(f"Still errored: {len(still_errored)}")
    print(f"Total scanned: {rep['archives_scanned']}")
    print(f"Total successes: {rep['total_successes']}")
    print(f"Total failures:  {rep['total_failures']}")
    print(f"Total episodes:  {rep['total_episodes']}")
    print(f"Orphan successes: {rep['orphan_success_total']} across {len(orphans)} archives")
    print(f"\nUpdated: {REPORT}")


if __name__ == "__main__":
    main()
