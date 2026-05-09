"""Read-only inconsistency detector for robometer_frame_dataset.

Catalogs structural differences between families across:
  A. Top-level subdirectory layout (which keyframes_* variants exist)
  B. Manifest filename patterns + counts
  C. Manifest field-presence per family
  D. Keyframes tar shard layout (flat vs nested per-archive)
  E. JPG directory naming convention inside tars (<eid>__<task> vs <eid>)
  F. Manifest's `keyframes_dir` field convention

Default target: the scratch copy. Override with --root.
"""
import argparse
import glob
import json
import os
import sys
import tarfile
from collections import defaultdict, Counter

DEFAULT_ROOT = "/scratch-shared/pkarageorgis1/robometer_frame_dataset_20260505_164035"
FAMILIES = ["droid", "failsafe", "metaworld", "robometer", "roboreward"]


def section(title):
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)


def cat_a_subdirs(root):
    section("A. Top-level subdirectory layout per family")
    subdir_sets = {}
    for fam in FAMILIES:
        fam_path = os.path.join(root, fam)
        if not os.path.isdir(fam_path):
            print(f"  {fam}: (missing)")
            continue
        entries = sorted(d for d in os.listdir(fam_path)
                        if os.path.isdir(os.path.join(fam_path, d)))
        subdir_sets[fam] = set(entries)
        print(f"  {fam}: {entries}")

    # Find common vs unique
    if subdir_sets:
        all_dirs = set().union(*subdir_sets.values())
        common = set.intersection(*subdir_sets.values())
        print(f"\n  common to all: {sorted(common)}")
        for fam, dirs in subdir_sets.items():
            unique = dirs - common
            if unique:
                print(f"  unique to {fam}: {sorted(unique)}")


def cat_b_manifest_naming(root):
    section("B. Manifest filename patterns")
    for fam in FAMILIES:
        manifest_dir = os.path.join(root, fam, "manifests")
        if not os.path.isdir(manifest_dir):
            print(f"  {fam}: no manifests/ dir")
            continue
        files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(manifest_dir, "*.jsonl")))
        # Extract pattern: count by suffix
        suffixes = Counter()
        for f in files:
            for s in ["_failures.jsonl", "_successes.jsonl", "_orphan_successes.jsonl",
                     "_failures_paired.jsonl", "_failures_orphan.jsonl"]:
                if f.endswith(s):
                    suffixes[s] += 1
                    break
            else:
                suffixes["other"] += 1
        print(f"  {fam}: {len(files)} files; suffixes={dict(suffixes)}")
        # Print first 3 + last 1 file names for shape inference
        if len(files) <= 4:
            for f in files: print(f"    {f}")
        else:
            for f in files[:3]: print(f"    {f}")
            print(f"    ... ({len(files)-4} more)")
            print(f"    {files[-1]}")


def cat_c_manifest_fields(root, sample_n=20):
    section("C. Manifest field-presence per family (sampled rows)")
    for fam in FAMILIES:
        manifest_dir = os.path.join(root, fam, "manifests")
        manifests = sorted(glob.glob(os.path.join(manifest_dir, "*.jsonl")))
        if not manifests:
            continue

        # Count field presence (non-null) across sample of rows from up to 3 manifests
        sampled = manifests[:3]
        field_present = Counter()
        field_null = Counter()
        n_rows = 0
        for mpath in sampled:
            with open(mpath) as fh:
                for i, line in enumerate(fh):
                    if i >= sample_n: break
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    n_rows += 1
                    for k, v in row.items():
                        if v is None or v == "":
                            field_null[k] += 1
                        else:
                            field_present[k] += 1

        all_fields = sorted(set(field_present) | set(field_null))
        print(f"\n  {fam} (sample of {n_rows} rows from {len(sampled)} manifests):")
        for k in all_fields:
            p = field_present[k]
            n = field_null[k]
            total = p + n
            if n_rows == 0: continue
            pct_null = 100 * n / total
            marker = "(always null)" if pct_null == 100 else ("(always populated)" if pct_null == 0 else f"({pct_null:.0f}% null)")
            print(f"    {k:<30}  populated={p:>3}  null={n:>3}  {marker}")


def cat_d_keyframes_layout(root):
    section("D. Keyframes/ tar shard layout per family")
    for fam in FAMILIES:
        kf = os.path.join(root, fam, "keyframes")
        if not os.path.isdir(kf):
            print(f"  {fam}: no keyframes/")
            continue
        entries = os.listdir(kf)
        # Three patterns:
        # - 'shards/' subdir (flat)
        # - <archive>/shard-*.tar files inside (nested per-archive)
        # - direct shard-*.tar files (no subdir)
        if "shards" in entries and os.path.isdir(os.path.join(kf, "shards")):
            n_tars = len(glob.glob(os.path.join(kf, "shards", "*.tar")))
            print(f"  {fam}: FLAT (keyframes/shards/) — {n_tars} tar files")
        else:
            # Could be nested archives or direct tars
            archives = [e for e in entries if os.path.isdir(os.path.join(kf, e))]
            tars = [e for e in entries if e.endswith(".tar")]
            if archives:
                tar_count = sum(len(glob.glob(os.path.join(kf, a, "*.tar"))) for a in archives)
                print(f"  {fam}: NESTED (keyframes/<archive>/) — {len(archives)} archives, {tar_count} tar files total")
                if len(archives) <= 5:
                    print(f"    archives: {archives}")
                else:
                    print(f"    archives (first 5 of {len(archives)}): {archives[:5]}")
            elif tars:
                print(f"  {fam}: DIRECT (keyframes/*.tar) — {len(tars)} tar files")
            else:
                print(f"  {fam}: empty or unknown layout")


def cat_e_tar_dir_naming(root):
    section("E. JPG directory naming convention inside tars (sampled)")
    for fam in FAMILIES:
        kf = os.path.join(root, fam, "keyframes")
        # Find first tar
        tars = glob.glob(os.path.join(kf, "**", "*.tar"), recursive=True)
        if not tars:
            print(f"  {fam}: no tars found")
            continue
        sample_tar = tars[0]
        try:
            with tarfile.open(sample_tar, "r|") as tf:
                seen_dirs = set()
                for i, m in enumerate(tf):
                    if i > 30: break
                    if m.name.endswith(".jpg"):
                        d = m.name.split("/")[-2] if "/" in m.name else None
                        if d:
                            seen_dirs.add(d)
        except Exception as e:
            print(f"  {fam}: failed to open {sample_tar}: {e}")
            continue
        if not seen_dirs:
            print(f"  {fam}: no JPGs found in sample tar")
            continue
        # Check naming convention
        sample = sorted(seen_dirs)[0]
        has_dunder = "__" in sample
        print(f"  {fam}: tar={os.path.basename(sample_tar)}")
        print(f"    sample dir: {sample!r}")
        print(f"    convention: {'<eid>__<task_words>' if has_dunder else 'just <eid> (no __task suffix)'}")


def cat_f_keyframes_dir_field(root):
    section("F. Manifest 'keyframes_dir' field convention per family")
    for fam in FAMILIES:
        manifest_dir = os.path.join(root, fam, "manifests")
        manifests = sorted(glob.glob(os.path.join(manifest_dir, "*.jsonl")))
        if not manifests:
            continue
        kf_dir_examples = []
        with open(manifests[0]) as fh:
            for i, line in enumerate(fh):
                if i >= 5: break
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                kf_dir_examples.append(row.get("keyframes_dir"))
        all_null = all(x is None for x in kf_dir_examples)
        all_populated = all(x is not None for x in kf_dir_examples)
        marker = "ALL NULL" if all_null else ("ALL POPULATED" if all_populated else "MIXED")
        print(f"  {fam}: {marker}")
        for ex in kf_dir_examples[:3]:
            print(f"    sample: {ex!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=DEFAULT_ROOT)
    args = p.parse_args()

    if not os.path.isdir(args.root):
        print(f"ERROR: --root not found: {args.root}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning: {args.root}")
    cat_a_subdirs(args.root)
    cat_b_manifest_naming(args.root)
    cat_c_manifest_fields(args.root)
    cat_d_keyframes_layout(args.root)
    cat_e_tar_dir_naming(args.root)
    cat_f_keyframes_dir_field(args.root)

    print("\n" + "=" * 100)
    print("  DONE — review above to identify normalization targets")
    print("=" * 100)


if __name__ == "__main__":
    main()
