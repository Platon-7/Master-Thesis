"""Build a held-out test set from `pairs_unified.jsonl` rows that were NOT used in
train / warmup / eval splits.

Composition (per user spec):
  * metaworld   — failures + paired successes (within-family)
  * failsafe    — failures + paired successes (within-family)
  * Group-A robometer (source=robometer) — failures + paired successes (within-family).
    Group A spans archives: auto_eval, racer, mit_franka, libero, usc_franka, usc_xarm,
    utd_so101, usc_koch, usc_trossen.  We sample proportionally from the largest pools.
  * droid       — skipped (training exhausted droid failures).

Integrity: only sample failure rows where BOTH the failure-query AND its partner-success
are unused — i.e. the partner was never seen as an ICL demo during training. Drops the
strictness gap our prior eval splits had.

Writes `test.jsonl` (episode_ids) and `pairs_index_test.jsonl` (full pair rows) into the
splits dir, ready for `jobs/preprocess_split.job` to materialise into an HF dataset cache.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set


PAIRS_JSONL = Path("/projects/prjs1958/robometer_frame_dataset/pairs_unified.jsonl")
SPLITS_DIR  = Path("/scratch-shared/pkarageorgis1/robometer_frames_splits")

GROUP_A_FAMILIES = ["auto_eval", "racer", "mit_franka", "libero",
                    "usc_franka", "usc_xarm", "utd_so101", "usc_koch", "usc_trossen"]
SIM_FAMILIES = ["metaworld", "failsafe"]


def _stream(path: Path) -> Iterable[Dict]:
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _ids_from_split(path: Path, used: Set[str]) -> None:
    if not path.is_file():
        return
    for row in _stream(path):
        if eid := row.get("episode_id"):
            used.add(eid)
        if pid := row.get("partner_episode_id"):
            used.add(pid)


def _expand_with_partners(ids: List[str], rows_by_id: Dict[str, Dict],
                          used: Set[str]) -> List[str]:
    """Return ids plus any unused partner_episode_id whose row we have."""
    out, seen = list(ids), set(ids)
    for eid in ids:
        row = rows_by_id.get(eid)
        if row is None:
            continue
        partner = row.get("partner_episode_id")
        if partner and partner not in seen and partner not in used and partner in rows_by_id:
            out.append(partner); seen.add(partner)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--per-family-failures", type=int, default=300,
                    help="Per-family failure-query sample target (capped at pool size).")
    ap.add_argument("--total-group-a-failures", type=int, default=1500,
                    help="Total Group-A failure budget (split across the 9 archive families).")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # ----- collect "used" set from prior splits -----
    used: Set[str] = set()
    for fname in ["train.jsonl", "warmup.jsonl",
                  "eval_droid.jsonl", "eval_robometer.jsonl",
                  "eval_metaworld.jsonl", "eval_failsafe.jsonl",
                  "pairs_index_train.jsonl", "pairs_index_warmup.jsonl",
                  "pairs_index_eval_droid.jsonl", "pairs_index_eval_robometer.jsonl",
                  "pairs_index_eval_metaworld.jsonl", "pairs_index_eval_failsafe.jsonl"]:
        _ids_from_split(SPLITS_DIR / fname, used)
    print(f"[init] {len(used):,} episode_ids in 'used' set")

    # ----- index all rows + bucket eligible failure queries by family -----
    rows_by_id: Dict[str, Dict] = {}
    failures_eligible_by_family: Dict[str, List[str]] = defaultdict(list)

    n_total = 0
    for row in _stream(PAIRS_JSONL):
        n_total += 1
        eid = row["episode_id"]
        rows_by_id[eid] = row

    # Need full index before checking partners' "unused" status.
    print(f"[scan] indexed {n_total:,} pairs_unified rows")

    for eid, row in rows_by_id.items():
        if eid in used:
            continue
        if row.get("label") != "failure":
            continue
        fam = row.get("family") or "unknown"
        # Group A: require both query and partner unused so we can pair-expand into a
        # rich fail+success test set per archive (proper AUC + FPR computation).
        if fam in GROUP_A_FAMILIES:
            partner = row.get("partner_episode_id")
            if not partner or partner in used or partner not in rows_by_id:
                continue
            failures_eligible_by_family[fam].append(eid)
        # Sim families: failure-only test (per user spec). Partners aren't required because
        # we won't pair-expand. FPR (computed on failures alone) is the headline metric for
        # these splits; AUC will be undefined (single-class) — that's expected.
        elif fam in SIM_FAMILIES:
            failures_eligible_by_family[fam].append(eid)

    print(f"[eligibility] strictly-unused failure queries (with also-unused partners):")
    for fam in SIM_FAMILIES + GROUP_A_FAMILIES:
        n = len(failures_eligible_by_family.get(fam, []))
        print(f"  {fam:<14} {n:>7,}")

    # ----- sample -----
    test_ids: List[str] = []

    # Sim families — sample full target
    for fam in SIM_FAMILIES:
        pool = failures_eligible_by_family.get(fam, [])
        n = min(args.per_family_failures, len(pool))
        sampled = rng.sample(pool, n) if n else []
        print(f"[sample] {fam:<14} failures={n:>4} of {len(pool):>5} eligible")
        test_ids.extend(sampled)

    # Group A — distribute total budget across archives proportional to pool size
    total_a = sum(len(failures_eligible_by_family.get(f, [])) for f in GROUP_A_FAMILIES)
    if total_a == 0:
        print("[sample] Group A: no eligible failures!")
    else:
        budget = min(args.total_group_a_failures, total_a)
        for fam in GROUP_A_FAMILIES:
            pool = failures_eligible_by_family.get(fam, [])
            if not pool:
                continue
            share = max(1, round(budget * len(pool) / total_a))
            n = min(share, len(pool))
            sampled = rng.sample(pool, n) if n else []
            print(f"[sample] {fam:<14} failures={n:>4} of {len(pool):>5} eligible (proportional)")
            test_ids.extend(sampled)

    # ----- pull in partners ONLY for Group A failures (sim families stay failure-only) -----
    group_a_query_ids = [
        eid for eid in test_ids
        if (rows_by_id[eid].get("family") in GROUP_A_FAMILIES)
    ]
    sim_query_ids = [
        eid for eid in test_ids
        if (rows_by_id[eid].get("family") in SIM_FAMILIES)
    ]
    expanded_group_a = _expand_with_partners(group_a_query_ids, rows_by_id, used)
    test_ids_with_partners = list(sim_query_ids) + expanded_group_a
    print(f"[partners] Group A: {len(expanded_group_a) - len(group_a_query_ids):,} partner rows added")
    print(f"[partners] Sim families (failure-only): {len(sim_query_ids):,} kept as queries")
    print(f"[partners] → test set: {len(test_ids_with_partners):,} episodes total")

    # Sanity: zero overlap with used
    overlap = used & set(test_ids_with_partners)
    assert not overlap, f"BUG: {len(overlap)} test ids overlap with 'used'"

    # ----- write -----
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    out_eps   = SPLITS_DIR / "test.jsonl"
    out_pairs = SPLITS_DIR / "pairs_index_test.jsonl"
    test_ids_with_partners = sorted(set(test_ids_with_partners))
    with out_eps.open("w") as f:
        for eid in test_ids_with_partners:
            f.write(json.dumps({"episode_id": eid}) + "\n")
    n_written = 0
    with out_pairs.open("w") as f:
        for eid in test_ids_with_partners:
            row = rows_by_id.get(eid)
            if row is None:
                continue
            f.write(json.dumps(row) + "\n")
            n_written += 1
    print(f"[write] {out_eps}        ({len(test_ids_with_partners):,} ids)")
    print(f"[write] {out_pairs}      ({n_written:,} pair rows)")

    # Per-family/label summary of the final test set
    by_fl: Counter = Counter()
    for eid in test_ids_with_partners:
        row = rows_by_id.get(eid)
        if row is None:
            continue
        by_fl[(row.get("family", "?"), row.get("label", "?"))] += 1
    print("\n[summary] (family, label) counts in final test set:")
    for (fam, lab), n in sorted(by_fl.items()):
        print(f"  {fam:<25} {lab:<8} {n:>5}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
