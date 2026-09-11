"""Build train + 4 per-source eval splits from pairs_unified.jsonl.

Differences from build_splits.py:
  * 100% ICL-pair-resolvable filter on every eval query — drops rows whose
    `partner_episode_id` is null or absent from pairs_unified.
  * No artificial train caps. Train pool = pairs_unified MINUS (eval queries
    + their partners). All ~860k non-eval rows are kept.
  * Eval splits: per-source (droid, metaworld, failsafe, robometer), sized
    to the user's targets.
  * Writes pairs_index_<split>.jsonl in the same schema as the existing
    build_splits.py outputs so the upstream sampler is happy.

Output (all under --output-dir):
  train.jsonl                       — episode_ids in train pool, one per line
  pairs_index_train.jsonl           — pairs_unified rows for train episodes
  eval_<source>.jsonl               — per-split episode_ids
  pairs_index_eval_<source>.jsonl   — pairs_unified rows for those + partners
  build_val_splits_report.json      — counts, sources, leakage check
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set


PAIRS_DEFAULT = Path("/projects/prjs1958/robometer_frame_dataset/pairs_unified.jsonl")
OUT_DEFAULT = Path("/scratch-shared/pkarageorgis1/robometer_frames_splits_full")

# LIBERO-90 hold-out (matches build_splits.py's _LIBERO90_EXCLUDED_ARCHIVES exactly):
# drop these two archives entirely (successes + failures) so LIBERO-90 never enters
# train, eval, or partner lookup. The other LIBERO suites (10/object/spatial/goal)
# are intentionally left untouched.
_LIBERO90_EXCLUDED_ARCHIVES = {
    "abraranwar_libero_rfm_libero256_90",             # libero_90 successes
    "ykorkmaz_libero_failure_rfm_libero_90_failure",  # libero_90 failures
}

# Per-source eval target counts (number of QUERY episodes; partners are added on top).
EVAL_TARGETS = {
    "droid":     500,
    "metaworld": 500,
    "failsafe":  500,
    "robometer": 2000,
}


def stream_jsonl(path: Path):
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_episode_jsonl(path: Path, eids: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for eid in eids:
            f.write(json.dumps({"episode_id": eid}) + "\n")
    print(f"  wrote {len(eids):>7,} episode_ids → {path.name}")


def write_pair_index(path: Path, rows_by_id: Dict[str, dict], eids: List[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w") as f:
        for eid in eids:
            row = rows_by_id.get(eid)
            if row is None:
                continue
            f.write(json.dumps(row) + "\n")
            written += 1
    print(f"  wrote {written:>7,} pair rows → {path.name}")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs-jsonl", type=Path, default=PAIRS_DEFAULT)
    ap.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.pairs_jsonl}")
    rows_by_id: Dict[str, dict] = {}
    by_source: Dict[str, List[str]] = defaultdict(list)
    n_libero90_dropped = 0
    for r in stream_jsonl(args.pairs_jsonl):
        eid = r.get("episode_id")
        if not eid:
            continue
        # LIBERO-90 hold-out: skip rows outright so the suite never enters rows_by_id,
        # by_source, train, eval, or partner lookup. Drop if either side of the pair
        # is libero_90 (archive catches queries, partner_archive catches rows whose
        # ICL partner is a libero_90 episode).
        if (r.get("archive") in _LIBERO90_EXCLUDED_ARCHIVES
                or r.get("partner_archive") in _LIBERO90_EXCLUDED_ARCHIVES):
            n_libero90_dropped += 1
            continue
        rows_by_id[eid] = r
        by_source[r.get("source") or "<None>"].append(eid)
    print(f"  total rows: {len(rows_by_id):,}")
    print(f"  LIBERO-90 hold-out: dropped {n_libero90_dropped:,} rows from {sorted(_LIBERO90_EXCLUDED_ARCHIVES)}")
    print(f"  rows per source:")
    for src, eids in sorted(by_source.items(), key=lambda kv: -len(kv[1])):
        print(f"    {src:30s} {len(eids):>10,}")

    # Eligible eval pool per source = rows whose partner_episode_id is set AND
    # resolvable. This guarantees 100% ICL-pair coverage on every eval query.
    print(f"\n[eval] selecting per-source held-out queries with 100% ICL-pair filter")
    eval_query_eids: Dict[str, List[str]] = {}
    eval_pair_eids: Dict[str, List[str]] = {}  # queries + their partners
    excluded_eids: Set[str] = set()
    for src, target in EVAL_TARGETS.items():
        pool = by_source.get(src, [])
        eligible = [
            eid for eid in pool
            if rows_by_id[eid].get("partner_episode_id")
            and rows_by_id[eid].get("partner_episode_id") in rows_by_id
        ]
        if len(eligible) < target:
            print(f"  WARN: source={src} only has {len(eligible)} ICL-resolvable rows; "
                  f"target was {target}, taking all eligible.")
            target = len(eligible)
        rng.shuffle(eligible)
        queries = eligible[:target]
        # Add partners to "excluded from train"; keep both query + partner rows
        # in the eval pair-index so the eval sampler can resolve pairs cheaply.
        partner_eids = [rows_by_id[eid]["partner_episode_id"] for eid in queries]
        all_eval_eids = list(dict.fromkeys(queries + partner_eids))  # de-dup, preserve order
        eval_query_eids[src] = queries
        eval_pair_eids[src] = all_eval_eids
        excluded_eids.update(all_eval_eids)
        print(f"  {src}: {len(queries):>5,} queries + {len(set(partner_eids)):>5,} unique partners "
              f"→ {len(all_eval_eids):>5,} held-out total")

    # Train pool = everything else
    print(f"\n[train] pool = pairs_unified minus all eval+partner episode_ids")
    train_eids = [eid for eid in rows_by_id if eid not in excluded_eids]
    print(f"  train rows: {len(train_eids):,}")

    # Sanity: zero overlap
    overlap = excluded_eids & set(train_eids)
    assert not overlap, f"BUG: {len(overlap)} eval episodes leaked into train"

    # Write everything
    print(f"\n[write] outputs in {args.output_dir}")
    write_episode_jsonl(args.output_dir / "train.jsonl", train_eids)
    write_pair_index(args.output_dir / "pairs_index_train.jsonl", rows_by_id, train_eids)
    for src in EVAL_TARGETS:
        write_episode_jsonl(args.output_dir / f"eval_{src}.jsonl", eval_pair_eids[src])
        write_pair_index(args.output_dir / f"pairs_index_eval_{src}.jsonl",
                         rows_by_id, eval_pair_eids[src])

    # Report
    report = {
        "seed": args.seed,
        "total_rows_in_pairs_unified": len(rows_by_id),
        "train_rows": len(train_eids),
        "eval_query_counts": {s: len(v) for s, v in eval_query_eids.items()},
        "eval_pool_counts_with_partners": {s: len(v) for s, v in eval_pair_eids.items()},
        "eval_target_counts": EVAL_TARGETS,
        "sources_in_pairs_unified": {s: len(eids) for s, eids in by_source.items()},
    }
    report_path = args.output_dir / "build_val_splits_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  wrote report → {report_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
