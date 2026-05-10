"""Produce a `train_no_extras` cache by filtering humanoid + human_hand rows
out of the merged train _raw HF Arrow dataset.

The Robometer FT runs use this cache (no humanoid/hand exposure); the Qwen3.5
pretraining runs use the full `train` cache.

Output:
    <hf_out_dir>/train_no_extras_raw/robometer_frames_train_no_extras/
"""
import argparse
import json
import shutil
from pathlib import Path

from datasets import load_from_disk

# Families introduced by humanoid_orphan_success + human_hand_orphan_success.
# Full list inferred from pairs_unified.jsonl. The `data_source` field on each
# HF row (post-fix-A) is the bare family name.
EXTRAS_FAMILIES = {
    # humanoid_orphan_success
    "galaxea_r1_lite",
    "humanoid_everyday",
    # human_hand_orphan_success
    "rh20t_human",
    "egodex",
    "epic",
    "h2r",
    "hand_paired_human",
    "usc_koch_human",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-out-dir", type=Path, required=True)
    args = ap.parse_args()

    src = args.hf_out_dir / "train_raw" / "robometer_frames_train"
    dst = args.hf_out_dir / "train_no_extras_raw" / "robometer_frames_train_no_extras"
    if not src.is_dir():
        raise FileNotFoundError(f"merged train_raw not found at {src}")

    print(f"[filter] loading {src}")
    ds = load_from_disk(str(src))
    print(f"  total rows: {len(ds):,}")

    n_before = len(ds)
    keep_mask = [s not in EXTRAS_FAMILIES for s in ds["data_source"]]
    filtered = ds.select([i for i, k in enumerate(keep_mask) if k])
    n_after = len(filtered)
    n_dropped = n_before - n_after

    # Sanity-print per-source counts to confirm humanoid+hand really gone
    import collections
    src_counts = collections.Counter(filtered["data_source"])
    print(f"[filter] dropped {n_dropped:,} rows ({100*n_dropped/n_before:.1f}%)")
    print(f"  kept rows: {n_after:,}")
    print(f"  top sources after filter:")
    for s, n in src_counts.most_common(10):
        print(f"    {s}: {n:,}")

    if dst.exists():
        print(f"[filter] removing existing {dst}")
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    filtered.save_to_disk(str(dst))
    print(f"[filter] saved → {dst}")

    # Sanity report
    report = {
        "src": str(src),
        "dst": str(dst),
        "rows_before": n_before,
        "rows_after": n_after,
        "rows_dropped": n_dropped,
        "excluded_families": sorted(EXTRAS_FAMILIES),
        "sources_after": dict(src_counts),
    }
    report_path = dst.parent / "filter_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[filter] report → {report_path}")


if __name__ == "__main__":
    main()
