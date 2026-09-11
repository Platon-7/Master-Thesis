"""Deep-dive audit of /projects/prjs1958/robometer_frame_dataset/.

For every episode in every family's manifest, count ACTUAL JPG files in the
corresponding `keyframes/` tar shard (the default view that the loader uses)
and compare against:
  - manifest's `n_keyframes` field
  - len(manifest's `frame_labels`)

Flags any row where actual_jpg_count != n_keyframes, or actual_jpg_count != len(labels),
or actual_jpg_count < 16. Writes a TSV of all faulty episodes.

Outputs:
  /scratch-shared/$USER/dataset_audit_report.tsv  — every flagged row, full detail
  /scratch-shared/$USER/dataset_audit_summary.txt — per-family summary stats
"""
import json
import os
import re
import sys
import tarfile
import time
import glob
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.environ.get("AUDIT_ROOT", "/scratch-shared/pkarageorgis1/robometer_frame_dataset_20260505_164035")
FAMILIES = ["droid", "failsafe", "metaworld", "robometer", "roboreward"]
TARGET = 16

USER = os.environ.get("USER", "pkarageorgis1")
OUT_TSV = f"/scratch-shared/{USER}/dataset_audit_report.tsv"
OUT_SUMMARY = f"/scratch-shared/{USER}/dataset_audit_summary.txt"


def episode_id_from_jpg_path(member_name: str) -> str | None:
    """Extract episode_id from JPG path inside a tar.

    Examples seen earlier:
      'REAL_2023-07-13_..._something/frame_00_0.00s.jpg'
    -> episode_id is the directory name before the slash.
    """
    if not member_name.endswith(".jpg"):
        return None
    parts = member_name.split("/")
    if len(parts) < 2:
        return None
    dir_name = parts[-2]
    # The dir often appends '__<task_words>' to the episode_id; strip that.
    if "__" in dir_name:
        return dir_name.split("__", 1)[0]
    return dir_name


def count_jpgs_in_tar(tar_path: str) -> dict[str, int]:
    """Returns {episode_id: jpg_count} for one tar shard. Reads headers only."""
    counts: dict[str, int] = defaultdict(int)
    try:
        with tarfile.open(tar_path, "r|") as tf:  # streaming mode, no random access
            for member in tf:
                if not member.isfile():
                    continue
                eid = episode_id_from_jpg_path(member.name)
                if eid:
                    counts[eid] += 1
    except Exception as e:
        print(f"  [WARN] failed to read {tar_path}: {e}", flush=True)
    return dict(counts)


def main():
    t_start = time.time()

    # --- Phase 1: enumerate JPG counts per episode in BOTH keyframes/ AND
    # keyframes_success/ views. Failures live in keyframes/, successes in
    # keyframes_success/ — manifest's `label` field tells us which to query.
    # Skip *.bak_pre_norm/ subdirs (rollback safety nets, not live data).
    # Recursive glob catches both old and normalized layouts.
    print("=== Phase 1: counting JPGs per episode in keyframes/ + keyframes_success/ ===", flush=True)
    keyframe_tars = []  # list of (family, view_label, tar_path) where view_label is "failure" or "success"
    for fam in FAMILIES:
        for view_subdir, view_label in [("keyframes", "failure"), ("keyframes_success", "success")]:
            view_root = os.path.join(ROOT, fam, view_subdir)
            if not os.path.isdir(view_root):
                continue
            all_tars = glob.glob(os.path.join(view_root, "**", "*.tar"), recursive=True)
            tars = [t for t in all_tars if "bak_pre_norm" not in t]
            keyframe_tars.extend((fam, view_label, t) for t in sorted(tars))
            print(f"  {fam}/{view_subdir}: {len(tars)} tar shards", flush=True)
    print(f"  total: {len(keyframe_tars)} tars to scan", flush=True)

    # Index by (family, view_label) → {eid: count} where view_label is "failure" or "success"
    actual_counts_by_view: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)

    n_workers = max(1, min(16, len(keyframe_tars)))
    print(f"  using {n_workers} workers", flush=True)
    t_p1 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(count_jpgs_in_tar, tar_path): (fam, view_label, tar_path)
                   for fam, view_label, tar_path in keyframe_tars}
        for i, fut in enumerate(as_completed(futures)):
            fam, view_label, tpath = futures[fut]
            res = fut.result()
            actual_counts_by_view[(fam, view_label)].update(res)
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(keyframe_tars)} tars done", flush=True)
    print(f"  Phase 1 done in {time.time() - t_p1:.1f}s", flush=True)
    for fam in FAMILIES:
        n_fail = len(actual_counts_by_view.get((fam, "failure"), {}))
        n_succ = len(actual_counts_by_view.get((fam, "success"), {}))
        print(f"    {fam}: failure-view={n_fail} eps  success-view={n_succ} eps", flush=True)

    # --- Phase 2: walk manifests, cross-reference, write flagged rows ---
    print("\n=== Phase 2: cross-referencing manifests against actual JPG counts ===", flush=True)
    per_family_stats: dict[str, dict[str, int]] = {fam: defaultdict(int) for fam in FAMILIES}
    flagged: list[dict] = []

    for fam in FAMILIES:
        manifest_dir = os.path.join(ROOT, fam, "manifests")
        manifests = sorted(glob.glob(os.path.join(manifest_dir, "*.jsonl")))
        for mpath in manifests:
            with open(mpath) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue

                    eid = row.get("episode_id", "<no_id>")
                    n_kf_claim = row.get("n_keyframes")
                    labels = row.get("frame_labels")
                    n_lab = len(labels) if labels else 0
                    # Look up actual count in the right physical view based on label
                    label = row.get("label", "")
                    view_label = "failure" if label == "failure" else "success"
                    n_actual = actual_counts_by_view.get((fam, view_label), {}).get(eid)

                    s = per_family_stats[fam]
                    s["total"] += 1
                    if n_actual is None:
                        s["no_jpgs_found"] += 1
                    if n_kf_claim is not None and n_actual is not None and n_kf_claim != n_actual:
                        s["claim_vs_actual_mismatch"] += 1
                    if n_lab > 0 and n_actual is not None and n_lab != n_actual:
                        s["labels_vs_actual_mismatch"] += 1
                    if n_actual is not None and n_actual < TARGET:
                        s["actual_short"] += 1
                    if n_lab > 0 and n_lab < TARGET:
                        s["labels_short"] += 1
                    if n_kf_claim is not None and n_kf_claim < TARGET:
                        s["claim_short"] += 1

                    is_flagged = (
                        (n_actual is None)
                        or (n_kf_claim is not None and n_kf_claim != n_actual)
                        or (n_lab > 0 and n_lab != n_actual)
                        or (n_actual is not None and n_actual < TARGET)
                    )
                    if is_flagged:
                        flagged.append({
                            "episode_id": eid,
                            "family": fam,
                            "manifest": os.path.basename(mpath),
                            "n_keyframes_claim": n_kf_claim if n_kf_claim is not None else "",
                            "n_frame_labels": n_lab,
                            "n_jpgs_actual": n_actual if n_actual is not None else "",
                            "label": row.get("label", ""),
                            "claim_eq_actual": int(n_kf_claim is not None and n_actual is not None and n_kf_claim == n_actual),
                            "labels_eq_actual": int(n_lab > 0 and n_actual is not None and n_lab == n_actual),
                            "actual_short_lt16": int(n_actual is not None and n_actual < TARGET),
                            "no_jpgs_found": int(n_actual is None),
                        })

    # --- Phase 3: write outputs ---
    print(f"\n=== Phase 3: writing report ===", flush=True)
    os.makedirs(os.path.dirname(OUT_TSV), exist_ok=True)
    with open(OUT_TSV, "w") as f:
        cols = ["episode_id", "family", "manifest", "n_keyframes_claim", "n_frame_labels",
                "n_jpgs_actual", "label", "claim_eq_actual", "labels_eq_actual",
                "actual_short_lt16", "no_jpgs_found"]
        f.write("\t".join(cols) + "\n")
        for r in flagged:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")

    with open(OUT_SUMMARY, "w") as f:
        def emit(line=""):
            print(line, flush=True)
            f.write(line + "\n")

        emit(f"Dataset audit run: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        emit(f"Total wall time: {time.time() - t_start:.1f}s")
        emit("=" * 100)
        emit(f"{'family':<12} {'total':>10} {'no_jpgs':>10} {'claim<>act':>11} {'lbl<>act':>10} {'actual<16':>10} {'lbl<16':>8} {'claim<16':>9}")
        emit("=" * 100)
        grand = defaultdict(int)
        for fam in FAMILIES:
            s = per_family_stats[fam]
            for k, v in s.items():
                grand[k] += v
            emit(f"{fam:<12} {s['total']:>10} {s['no_jpgs_found']:>10} {s['claim_vs_actual_mismatch']:>11} {s['labels_vs_actual_mismatch']:>10} {s['actual_short']:>10} {s['labels_short']:>8} {s['claim_short']:>9}")
        emit("-" * 100)
        emit(f"{'TOTAL':<12} {grand['total']:>10} {grand['no_jpgs_found']:>10} {grand['claim_vs_actual_mismatch']:>11} {grand['labels_vs_actual_mismatch']:>10} {grand['actual_short']:>10} {grand['labels_short']:>8} {grand['claim_short']:>9}")
        emit("")
        emit(f"Flagged rows written to: {OUT_TSV} ({len(flagged)} total)")
        emit("")
        emit("KEY for columns:")
        emit("  no_jpgs:    manifest has the episode but tar has no JPGs (extraction never ran or failed)")
        emit("  claim<>act: manifest's n_keyframes != actual JPG count in tar (this is the bug class we hit)")
        emit("  lbl<>act:   len(frame_labels) != actual JPG count (training-time mismatch)")
        emit("  actual<16:  actual JPG count < 16 (short trajectory; legitimate if loader supports min_frames<16)")
        emit("  lbl<16:     len(frame_labels) < 16")
        emit("  claim<16:   manifest's n_keyframes < 16")


if __name__ == "__main__":
    main()
