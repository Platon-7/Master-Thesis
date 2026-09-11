"""Deep episode-level audit of the robometer_frame_dataset scratch copy.

Per episode (not per tar), checks:
  M01 — not in pairs_unified.jsonl as a query
  T01 — tar file not found at expected path (from manifest's keyframes_dir)
  T02 — episode dir not found inside the tar
  T03 — JPG count < 16
  T04 — JPG count > 16
  T05 — at least one sampled JPG fails to decode as a valid image (PIL)
  L01 — failure trajectory but manifest has no frame_labels
  L02 — frame_labels length != 16
  L03 — frame_labels contains values outside [1, 5]

Outputs a TSV with one row per (episode, issue) plus a per-family summary table.
This is INTENTIONALLY slower than the tar-header audit — it actually opens tars
per episode, reads bytes, and decodes a sample JPG to detect file corruption.

Default: scans the scratch copy. Override with AUDIT_ROOT env var.
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import tarfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

from PIL import Image

ROOT = os.environ.get(
    "AUDIT_ROOT",
    "/scratch-shared/pkarageorgis1/robometer_frame_dataset_20260505_164035",
)
PAIRS_UNIFIED = os.path.join(ROOT, "pairs_unified.jsonl")
USER = os.environ.get("USER", "pkarageorgis1")
OUT_TSV = f"/scratch-shared/{USER}/dataset_deep_audit.tsv"
OUT_SUMMARY = f"/scratch-shared/{USER}/dataset_deep_audit_summary.txt"

FAMILIES = ["droid", "failsafe", "metaworld", "robometer", "roboreward"]
TARGET_FRAMES = 16

# Map (family, label) → physical view directory.
#
# Quirks across families:
#   - robometer's orphan-success rows live in keyframes_orphan_success/ and use
#     filenames ending in '_orphan_successes.jsonl'.
#   - The `label` field uses *different strings per family*:
#       droid / failsafe / metaworld → 'success' or 'failure'
#       robometer / roboreward       → 'successful' or 'failure'
#     We accept both 'success' and 'successful' as the success indicator. Earlier
#     versions only matched 'success' and silently mis-routed every robometer/
#     roboreward success row to keyframes/, producing thousands of false-positive
#     T01/T02 flags. Don't repeat that.
def view_for(family: str, label: str, archive: Optional[str], manifest_basename: str) -> str:
    if family == "robometer" and "orphan" in manifest_basename:
        return "keyframes_orphan_success"
    if label in ("success", "successful"):
        return "keyframes_success"
    return "keyframes"


def resolve_tar_path(family: str, view: str, archive: str, episode_id: str, root: str) -> Optional[str]:
    """Find the tar shard file containing this episode. We don't know which shard
    a priori, so caller groups by (family, view, archive) and scans all matching
    shards once."""
    arch_dir = os.path.join(root, family, view, archive)
    if os.path.isdir(arch_dir):
        return arch_dir  # caller iterates *.tar in this dir
    # Droid shards/ legacy fallback (if not normalized)
    legacy = os.path.join(root, family, view, "shards")
    if os.path.isdir(legacy):
        return legacy
    return None


def index_tar_episodes(tar_dir: str) -> dict[str, list[tuple[str, list[tarfile.TarInfo]]]]:
    """Open every *.tar in tar_dir, return {episode_id: [(tar_path, [members])]}.
    Each tarinfo we keep is for files (JPGs) belonging to that episode.

    Assumes POST-NORMALIZATION layout where the JPG dir name inside the tar IS
    the episode_id (no __<task_words> suffix). For all 5 families on the
    normalized scratch copy, the directory name and manifest's episode_id match
    exactly. Don't rsplit on '__' here — droid eids legitimately contain '__'
    when timestamps land on single-digit dates (e.g.,
    'PennPAL_2023-10-06_Fri_Oct__6_19-24-55_2023'), and a naive rsplit
    truncates them, producing 100% false-positive T02 flags for those rows.
    """
    out: dict[str, list[tuple[str, list[tarfile.TarInfo]]]] = defaultdict(list)
    if not os.path.isdir(tar_dir):
        return out
    for fname in sorted(os.listdir(tar_dir)):
        if not fname.endswith(".tar"):
            continue
        tar_path = os.path.join(tar_dir, fname)
        try:
            ep_members: dict[str, list[tarfile.TarInfo]] = defaultdict(list)
            with tarfile.open(tar_path, "r|") as tf:
                for m in tf:
                    if not m.isfile() or not m.name.endswith(".jpg"):
                        continue
                    parts = m.name.split("/")
                    if len(parts) < 2:
                        continue
                    eid = parts[-2]   # post-normalization: dir_name == episode_id
                    ep_members[eid].append(m)
            for eid, members in ep_members.items():
                out[eid].append((tar_path, members))
        except Exception as e:
            print(f"  [tar-read-fail] {tar_path}: {e}", flush=True)
    return out


def decode_sample_jpg(tar_path: str, member: tarfile.TarInfo) -> tuple[bool, str]:
    """Open the tar at tar_path, extract this member's bytes, decode as image.
    Returns (ok, detail)."""
    try:
        with tarfile.open(tar_path, "r:") as tf:
            f = tf.extractfile(member)
            if f is None:
                return False, "extractfile-returned-None"
            data = f.read()
        img = Image.open(io.BytesIO(data))
        img.verify()  # PIL integrity check
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def audit_one_archive(args: tuple) -> tuple[str, str, str, list[dict]]:
    """Worker: audit all episodes for one (family, view, archive) tar group.
    Returns (family, archive, view, list-of-issues)."""
    family, view, archive, root, episodes_for_check = args
    issues: list[dict] = []

    tar_dir = resolve_tar_path(family, view, archive, "_unused_", root)
    if tar_dir is None:
        for ep_info in episodes_for_check:
            issues.append({**ep_info, "issue_code": "T01", "issue_detail": f"tar dir not found: {family}/{view}/{archive}"})
        return family, archive, view, issues

    tar_index = index_tar_episodes(tar_dir)

    for ep_info in episodes_for_check:
        eid = ep_info["episode_id"]
        if eid not in tar_index:
            issues.append({**ep_info, "issue_code": "T02",
                          "issue_detail": f"episode dir not in any tar in {tar_dir}"})
            continue

        # Collect all members across any tars for this eid (usually exactly 1 tar)
        all_members = []
        primary_tar = None
        for tar_path, members in tar_index[eid]:
            all_members.extend(members)
            if primary_tar is None:
                primary_tar = tar_path

        n_jpgs = len(all_members)
        if n_jpgs < TARGET_FRAMES:
            issues.append({**ep_info, "issue_code": "T03",
                          "issue_detail": f"JPG count {n_jpgs} < {TARGET_FRAMES}"})
        elif n_jpgs > TARGET_FRAMES:
            issues.append({**ep_info, "issue_code": "T04",
                          "issue_detail": f"JPG count {n_jpgs} > {TARGET_FRAMES}"})

        # Sample first JPG, attempt decode
        if all_members:
            ok, detail = decode_sample_jpg(primary_tar, all_members[0])
            if not ok:
                issues.append({**ep_info, "issue_code": "T05",
                              "issue_detail": detail[:200]})

    return family, archive, view, issues


def main():
    t_start = time.time()

    # === Phase 1: Build pairs_unified query index ===
    print("=== Phase 1: indexing pairs_unified.jsonl ===", flush=True)
    pairs_query_eids: set[str] = set()
    n_unified_rows = 0
    with open(PAIRS_UNIFIED) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            n_unified_rows += 1
            eid = d.get("episode_id")
            if eid:
                pairs_query_eids.add(eid)
    print(f"  {n_unified_rows} rows in pairs_unified, {len(pairs_query_eids)} unique query eids", flush=True)

    # === Phase 2: collect all manifest rows + group by (family, view, archive) ===
    print("\n=== Phase 2: walking manifests ===", flush=True)
    archive_groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    n_manifest_rows = 0
    pre_issues: list[dict] = []  # M01 (pairs absence) + L01-L03 (label issues)
    for fam in FAMILIES:
        manifest_dir = os.path.join(ROOT, fam, "manifests")
        if not os.path.isdir(manifest_dir):
            continue
        for fname in sorted(os.listdir(manifest_dir)):
            if not fname.endswith(".jsonl"):
                continue
            mpath = os.path.join(manifest_dir, fname)
            for line in open(mpath):
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                n_manifest_rows += 1
                eid = row.get("episode_id")
                if not eid:
                    continue
                archive = row.get("archive", "unknown")
                label = row.get("label", "")
                view = view_for(fam, label, archive, fname)

                ep_info = {
                    "episode_id": eid,
                    "family": fam,
                    "archive": archive,
                    "label": label,
                    "manifest": fname,
                }

                # M01: not in pairs_unified as query
                if eid not in pairs_query_eids:
                    pre_issues.append({**ep_info, "issue_code": "M01",
                                      "issue_detail": "absent from pairs_unified.jsonl as a query"})

                # L01-L03: failure-side label checks
                if label == "failure":
                    fl = row.get("frame_labels")
                    if fl is None:
                        pre_issues.append({**ep_info, "issue_code": "L01",
                                          "issue_detail": "failure but frame_labels is None"})
                    elif not isinstance(fl, list):
                        pre_issues.append({**ep_info, "issue_code": "L02",
                                          "issue_detail": f"frame_labels not a list (type={type(fl).__name__})"})
                    elif len(fl) != TARGET_FRAMES:
                        pre_issues.append({**ep_info, "issue_code": "L02",
                                          "issue_detail": f"frame_labels length {len(fl)} != {TARGET_FRAMES}"})
                    else:
                        bad = [v for v in fl if not isinstance(v, int) or v < 1 or v > 5]
                        if bad:
                            pre_issues.append({**ep_info, "issue_code": "L03",
                                              "issue_detail": f"frame_labels has out-of-range values (sample: {bad[:3]})"})

                # Stash for tar-side checks
                archive_groups[(fam, view, archive)].append(ep_info)

    print(f"  scanned {n_manifest_rows} manifest rows across {len(archive_groups)} (family,view,archive) groups", flush=True)
    print(f"  pre-tar issues found so far: {len(pre_issues)}", flush=True)

    # === Phase 3: parallel tar audits ===
    print("\n=== Phase 3: per-archive tar audits (decoding sampled JPGs) ===", flush=True)
    n_workers = max(1, min(16, len(archive_groups)))
    print(f"  using {n_workers} workers across {len(archive_groups)} archive groups", flush=True)
    args_list = [(fam, view, arch, ROOT, eps) for (fam, view, arch), eps in archive_groups.items()]
    tar_issues: list[dict] = []
    t_p3 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(audit_one_archive, a): a for a in args_list}
        for i, fut in enumerate(as_completed(futures)):
            fam, archive, view, issues = fut.result()
            tar_issues.extend(issues)
            done = i + 1
            if done % 20 == 0 or done == len(args_list):
                print(f"    {done}/{len(args_list)} groups done, "
                      f"running tar issues={len(tar_issues)}, "
                      f"elapsed={time.time()-t_p3:.0f}s", flush=True)
    print(f"  Phase 3 done in {time.time()-t_p3:.1f}s", flush=True)

    # === Phase 4: write outputs ===
    all_issues = pre_issues + tar_issues
    print(f"\n=== Phase 4: writing report ({len(all_issues)} total issues) ===", flush=True)
    os.makedirs(os.path.dirname(OUT_TSV), exist_ok=True)
    cols = ["episode_id", "family", "archive", "label", "manifest", "issue_code", "issue_detail"]
    with open(OUT_TSV, "w") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in all_issues:
            w.writerow({c: r.get(c, "") for c in cols})

    # Per-family x issue_code summary
    summary_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in all_issues:
        summary_counts[r["family"]][r["issue_code"]] += 1
    family_totals: dict[str, int] = {fam: 0 for fam in FAMILIES}
    for r in all_issues:
        family_totals[r.get("family", "unknown")] = family_totals.get(r.get("family", "unknown"), 0) + 1

    with open(OUT_SUMMARY, "w") as f:
        def emit(line=""):
            print(line, flush=True)
            f.write(line + "\n")
        emit(f"Deep episode audit run: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        emit(f"AUDIT_ROOT: {ROOT}")
        emit(f"Total wall time: {time.time()-t_start:.1f}s")
        emit(f"Total manifest episodes scanned: {n_manifest_rows}")
        emit(f"Total issues logged: {len(all_issues)}")
        emit("")
        emit("Issue code legend:")
        emit("  M01 — episode not in pairs_unified.jsonl as a query")
        emit("  T01 — tar dir not found")
        emit("  T02 — episode dir not in tar")
        emit("  T03 — JPG count < 16")
        emit("  T04 — JPG count > 16")
        emit("  T05 — JPG fails to decode")
        emit("  L01 — failure trajectory has no frame_labels")
        emit("  L02 — frame_labels length != 16")
        emit("  L03 — frame_labels contains values outside [1,5]")
        emit("")
        codes = ["M01", "T01", "T02", "T03", "T04", "T05", "L01", "L02", "L03"]
        emit(f"{'family':<14}" + "".join(f"{c:>8}" for c in codes) + f"{'total':>9}")
        emit("-" * 100)
        grand = defaultdict(int)
        for fam in FAMILIES:
            sc = summary_counts.get(fam, {})
            line = f"{fam:<14}"
            row_total = 0
            for c in codes:
                v = sc.get(c, 0)
                grand[c] += v
                row_total += v
                line += f"{v:>8}"
            line += f"{row_total:>9}"
            emit(line)
        emit("-" * 100)
        line = f"{'TOTAL':<14}" + "".join(f"{grand[c]:>8}" for c in codes) + f"{sum(grand.values()):>9}"
        emit(line)
        emit("")
        emit(f"Full per-issue TSV: {OUT_TSV}")


if __name__ == "__main__":
    main()
