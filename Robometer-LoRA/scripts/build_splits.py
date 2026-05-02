#!/usr/bin/env python3
"""Build train and eval splits for the LoRA bake-off from `pairs_unified.jsonl`.

For each split this script writes two files:
  * `<split>.jsonl`           — one episode_id per line; the working list of episodes.
  * `pairs_index_<split>.jsonl` — the corresponding subset of pairs_unified rows so the
    sampler's ICL pair-index loader (Robometer/robometer/data/samplers/base.py) can stream
    only what's needed.

Composition (defaults capture the user's data-split spec — all flags are tuneable):

  Train   ~9k failures + ~9k successes
            * Failures: 90 % of droid failures, plus enough Group-A robometer failures to
              top up to `--train-failures-total-n` (default 9 000).
            * Successes (orphan, all have partners): half from `jesbu1_oxe_rfm_oxe_droid`
              so the success distribution matches the droid-heavy failure side; other half
              sampled across the remaining orphan archives.
  Eval   trajectory-id-disjoint from train, never overlapping the train pool:
            * droid      — the held-out 10 % of droid failures.
            * robometer  — 10 % of Group-A robometer failures, PLUS a matching count of
                           orphan successes (drawn from non-oxe_droid archives by default
                           so the success contribution to robometer-eval doesn't lean on
                           the same archive used heavily in training). Counts are kept
                           equal so eval-robometer is balanced failure↔success.
            * metaworld  — `--eval-metaworld-n` random rows (any label).
            * failsafe   — `--eval-failsafe-n`  random rows (any label).

Splits are deterministic given `--seed`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

# Default location of the source pair file (HPC).
DEFAULT_PAIRS_JSONL = Path("/projects/prjs1958/robometer_frame_dataset/pairs_unified.jsonl")
DEFAULT_OUTPUT_DIR = Path("/scratch-shared") / "robometer_frames_splits"

# Group-A failure detection: these are the curated robometer failures (not orphan successes).
_ORPHAN_SOURCE = "robometer_orphan_success"
_OXE_DROID_ARCHIVE = "jesbu1_oxe_rfm_oxe_droid"


def _stream_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_episode_jsonl(path: Path, episode_ids: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for eid in episode_ids:
            f.write(json.dumps({"episode_id": eid}) + "\n")
    print(f"  wrote {len(episode_ids):>7,} ids → {path}")


def _write_pair_index(path: Path, rows_by_id: Dict[str, Dict], episode_ids: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w") as f:
        for eid in episode_ids:
            row = rows_by_id.get(eid)
            if row is None:
                continue
            f.write(json.dumps(row) + "\n")
            written += 1
    print(f"  wrote {written:>7,} pair rows → {path}")


def _bucket_by_source_and_label(pairs_path: Path) -> Dict[str, Dict]:
    """Stream pairs_unified.jsonl once and bucket episode ids by (family/source, label).

    Returns:
        {
            "rows_by_id":   episode_id → row (only rows with a non-null partner kept),
            "by_family":    {family: List[episode_id]} all-labels,
            "by_source":    {source: List[episode_id]}, e.g. robometer / robometer_orphan_success,
            "failures_by_family":  {family: List[episode_id]} where label == "failure",
            "successes_by_source": {source: List[episode_id]} where label == "success",
            "orphan_by_archive":   {archive: List[episode_id]} for source=robometer_orphan_success,
        }
    """
    rows_by_id: Dict[str, Dict] = {}
    by_family: Dict[str, List[str]] = defaultdict(list)
    by_source: Dict[str, List[str]] = defaultdict(list)
    failures_by_family: Dict[str, List[str]] = defaultdict(list)
    successes_by_source: Dict[str, List[str]] = defaultdict(list)
    orphan_by_archive: Dict[str, List[str]] = defaultdict(list)

    n_total = 0
    n_with_partner = 0
    for row in _stream_jsonl(pairs_path):
        n_total += 1
        # rows_by_id holds EVERY episode so eval-side `_with_partners` can resolve a query's
        # partner even when that partner row's own `partner_episode_id` is None (e.g. droid
        # success rows are listed in pairs_unified but only the failure side carries the link).
        eid = row["episode_id"]
        rows_by_id[eid] = row
        # The query-pool buckets below only enroll rows that themselves have a partner —
        # those are the rows eligible to be eval/train queries. Partner-only rows stay
        # silently in rows_by_id for partner-lookup purposes.
        if not row.get("partner_episode_id"):
            continue
        n_with_partner += 1
        family = row.get("family") or row.get("source") or "unknown"
        source = row.get("source") or "unknown"
        archive = row.get("archive") or "unknown"
        by_family[family].append(eid)
        by_source[source].append(eid)
        if row.get("label") == "failure":
            failures_by_family[family].append(eid)
        if row.get("label") == "success":
            successes_by_source[source].append(eid)
        if source == _ORPHAN_SOURCE:
            orphan_by_archive[archive].append(eid)

    print(f"\n[scan] {n_total:,} rows scanned, {n_with_partner:,} have partner_episode_id "
          f"(eligible as queries); all {len(rows_by_id):,} kept in rows_by_id for partner lookup")
    print(f"[scan] failures by family: " + ", ".join(f"{k}={len(v):,}" for k, v in failures_by_family.items()))
    print(f"[scan] successes by source: " + ", ".join(f"{k}={len(v):,}" for k, v in successes_by_source.items()))
    if orphan_by_archive:
        print(f"[scan] orphan successes by archive: " + ", ".join(
            f"{k}={len(v):,}" for k, v in sorted(orphan_by_archive.items(), key=lambda kv: -len(kv[1]))[:5]
        ))
    return {
        "rows_by_id": rows_by_id,
        "by_family": by_family,
        "by_source": by_source,
        "failures_by_family": failures_by_family,
        "successes_by_source": successes_by_source,
        "orphan_by_archive": orphan_by_archive,
    }


def _sample_disjoint(rng: random.Random, pool: List[str], n: int, exclude: Set[str]) -> List[str]:
    """Sample up to n episode ids from `pool` while avoiding `exclude`. Returns a sorted-stable list."""
    candidates = [x for x in pool if x not in exclude]
    if n >= len(candidates):
        return list(candidates)
    return rng.sample(candidates, n)


def _with_partners(
    episode_ids: List[str], rows_by_id: Dict[str, Dict]
) -> List[str]:
    """Return episode_ids plus any partner_episode_id that isn't already in the list.

    Used for eval-split construction so each task ends up with both quality labels in the
    cache — a hard requirement for trajectory-level metrics (Success AUC, ranking accuracy,
    FP rate) and for the upstream `policy_ranking` custom-eval sampler. Without this, eval
    splits are query-only (typically failure-only) and trajectory metrics are mathematically
    undefined / silently empty.
    """
    out: List[str] = list(episode_ids)
    seen: Set[str] = set(episode_ids)
    n_added = 0
    for eid in episode_ids:
        row = rows_by_id.get(eid)
        if row is None:
            continue
        partner = row.get("partner_episode_id")
        if partner and partner not in seen and partner in rows_by_id:
            out.append(partner)
            seen.add(partner)
            n_added += 1
    print(f"  + {n_added:,} partners pulled in → {len(out):,} eval episodes total "
          f"(was {len(episode_ids):,})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs-jsonl", type=Path, default=DEFAULT_PAIRS_JSONL)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--seed", type=int, default=42)

    # Eval slice sizes.
    ap.add_argument("--eval-droid-frac", type=float, default=0.10,
                    help="Fraction of droid failures held out for eval (default 10%%).")
    ap.add_argument("--eval-robometer-frac-of-sampled", type=float, default=0.10,
                    help="eval_robometer is 10%% of the sampled robometer training data — "
                         "i.e., 10%% of the robometer-failures count plus 10%% of the orphan-successes "
                         "count. Default 10%% gives ~1.4k eval slice for ~14k sampled training data.")
    ap.add_argument("--eval-metaworld-n", type=int, default=500,
                    help="Number of metaworld episodes (any label) held out for eval.")
    ap.add_argument("--eval-failsafe-n", type=int, default=500,
                    help="Number of failsafe episodes (any label) held out for eval.")

    # Train slice sizes.
    ap.add_argument("--train-droid-frac", type=float, default=0.90,
                    help="Fraction of droid failures used for training (the rest goes to eval).")
    ap.add_argument("--train-failures-total-n", type=int, default=9000,
                    help="Target total number of failure episodes in the training pool. "
                         "Robometer Group-A failures top up after droid contributes its 90%%.")
    ap.add_argument("--train-orphan-success-n", type=int, default=9000,
                    help="Total number of orphan-success episodes for the training pool.")
    ap.add_argument("--train-orphan-oxe-droid-n", type=int, default=4500,
                    help="Of train_orphan_success_n, how many to draw from jesbu1_oxe_rfm_oxe_droid.")

    # Warmup slice — extra failure-only pool used in the first N training steps. Disjoint
    # from train + eval so it doesn't steal failures from the main 9k pool.
    ap.add_argument("--warmup-n", type=int, default=1500,
                    help="Number of robometer Group-A failures sampled for the warmup phase. "
                         "Disjoint from train + eval.")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    if not args.pairs_jsonl.is_file():
        print(f"FATAL: pairs jsonl not found: {args.pairs_jsonl}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[init] output dir: {args.output_dir}")

    buckets = _bucket_by_source_and_label(args.pairs_jsonl)
    rows_by_id = buckets["rows_by_id"]

    excluded: Set[str] = set()  # eval episodes — train pool excludes these.

    # ---------------- Eval slices ----------------
    print("\n[eval] sampling held-out slices …")

    # droid eval: fraction of droid failures, then expand to include each query's partner so
    # the cache holds both quality labels per task (required for AUC/ranking/FP-rate metrics).
    droid_failures = buckets["failures_by_family"].get("droid", [])
    n_eval_droid = int(round(args.eval_droid_frac * len(droid_failures)))
    eval_droid_queries = _sample_disjoint(rng, droid_failures, n_eval_droid, excluded)
    eval_droid = _with_partners(eval_droid_queries, rows_by_id)
    excluded.update(eval_droid)
    _write_episode_jsonl(args.output_dir / "eval_droid.jsonl", eval_droid)
    _write_pair_index(args.output_dir / "pairs_index_eval_droid.jsonl", rows_by_id, eval_droid)

    # robometer (Group A/B) failures: all rows with source==robometer and label==failure.
    robometer_failures_groupA = [eid for eid, row in rows_by_id.items()
                                 if row.get("label") == "failure" and row.get("source") == "robometer"]
    print(f"[eval] Group-A robometer failures available: {len(robometer_failures_groupA):,}")

    # eval_robometer is sized as a fraction of *what we plan to sample for training*, not
    # of the full available pool. Targets:
    #   eval_robometer_failures  = frac * (train_failures_total_n - droid contribution)
    #   eval_robometer_successes = frac * train_orphan_success_n
    n_train_droid_target = int(round(args.train_droid_frac * len(droid_failures)))
    n_train_robometer_target = max(0, args.train_failures_total_n - n_train_droid_target)
    n_eval_robometer_failures = int(round(args.eval_robometer_frac_of_sampled * n_train_robometer_target))
    n_eval_robometer_successes = int(round(args.eval_robometer_frac_of_sampled * args.train_orphan_success_n))

    eval_robometer_failures = _sample_disjoint(
        rng, robometer_failures_groupA, n_eval_robometer_failures, excluded
    )
    excluded.update(eval_robometer_failures)

    # Orphan successes drawn from non-oxe_droid archives so the success contribution to
    # eval_robometer doesn't compete with the heavy oxe_droid use on the training side.
    non_oxe_orphan = [eid for arch, ids in buckets["orphan_by_archive"].items()
                      if arch != _OXE_DROID_ARCHIVE for eid in ids]
    eval_robometer_successes = _sample_disjoint(
        rng, non_oxe_orphan, n_eval_robometer_successes, excluded
    )
    excluded.update(eval_robometer_successes)
    print(f"[eval] eval_robometer = {len(eval_robometer_failures):,} failures "
          f"+ {len(eval_robometer_successes):,} orphan successes")

    # eval_robometer is already built as fail-side + success-side from disjoint pools, so it
    # naturally has both labels. We still pass it through `_with_partners` so each task in
    # the cache has its actual paired counterpart, not just label-balance — needed for
    # policy_ranking which groups by (task, quality_label).
    eval_robometer_queries = eval_robometer_failures + eval_robometer_successes
    eval_robometer = _with_partners(eval_robometer_queries, rows_by_id)
    excluded.update(eval_robometer)
    _write_episode_jsonl(args.output_dir / "eval_robometer.jsonl", eval_robometer)
    _write_pair_index(args.output_dir / "pairs_index_eval_robometer.jsonl", rows_by_id, eval_robometer)

    # metaworld eval: fixed N from any metaworld-family row, then pull in partners.
    metaworld_pool = buckets["by_family"].get("metaworld", [])
    eval_metaworld_queries = _sample_disjoint(rng, metaworld_pool, args.eval_metaworld_n, excluded)
    eval_metaworld = _with_partners(eval_metaworld_queries, rows_by_id)
    excluded.update(eval_metaworld)
    _write_episode_jsonl(args.output_dir / "eval_metaworld.jsonl", eval_metaworld)
    _write_pair_index(args.output_dir / "pairs_index_eval_metaworld.jsonl", rows_by_id, eval_metaworld)

    # failsafe eval: fixed N from any failsafe-family row, then pull in partners.
    # Failsafe is the most-trusted source per losses.md (clean simulator labels) — getting
    # both labels in this cache is the highest-priority correctness fix.
    failsafe_pool = buckets["by_family"].get("failsafe", [])
    eval_failsafe_queries = _sample_disjoint(rng, failsafe_pool, args.eval_failsafe_n, excluded)
    eval_failsafe = _with_partners(eval_failsafe_queries, rows_by_id)
    excluded.update(eval_failsafe)
    _write_episode_jsonl(args.output_dir / "eval_failsafe.jsonl", eval_failsafe)
    _write_pair_index(args.output_dir / "pairs_index_eval_failsafe.jsonl", rows_by_id, eval_failsafe)

    # ---------------- Train slice ----------------
    print("\n[train] sampling training pool …")

    train_ids: List[str] = []

    # droid failures: take 90% of what's left after eval_droid carved off 10%.
    droid_remaining = [eid for eid in droid_failures if eid not in excluded]
    n_train_droid = len(droid_remaining)  # all of the non-eval droid failures
    train_droid = _sample_disjoint(rng, droid_remaining, n_train_droid, excluded)
    train_ids.extend(train_droid)
    excluded.update(train_droid)
    print(f"  droid failures: {len(train_droid):,}  (≈ {args.train_droid_frac*100:.0f}% of {len(droid_failures):,})")

    # robometer Group-A failures: top up the failure side to `train_failures_total_n`.
    n_train_robometer_target = max(0, args.train_failures_total_n - len(train_droid))
    train_robometer = _sample_disjoint(rng, robometer_failures_groupA, n_train_robometer_target, excluded)
    train_ids.extend(train_robometer)
    excluded.update(train_robometer)
    print(f"  robometer (Group A/B) failures: {len(train_robometer):,}  (top-up toward "
          f"{args.train_failures_total_n:,} total failures)")

    # orphan successes: 4.5k from oxe_droid + (n − 4.5k) from other orphan archives.
    n_oxe = min(args.train_orphan_oxe_droid_n, args.train_orphan_success_n)
    n_other = args.train_orphan_success_n - n_oxe
    oxe_pool = buckets["orphan_by_archive"].get(_OXE_DROID_ARCHIVE, [])
    other_orphan_pool = [eid for arch, ids in buckets["orphan_by_archive"].items()
                         if arch != _OXE_DROID_ARCHIVE for eid in ids]

    train_oxe = _sample_disjoint(rng, oxe_pool, n_oxe, excluded)
    excluded.update(train_oxe)
    train_other = _sample_disjoint(rng, other_orphan_pool, n_other, excluded)
    excluded.update(train_other)
    train_ids.extend(train_oxe)
    train_ids.extend(train_other)
    print(f"  orphan successes (oxe_droid): {len(train_oxe):,}")
    print(f"  orphan successes (other archives): {len(train_other):,}")

    rng.shuffle(train_ids)
    _write_episode_jsonl(args.output_dir / "train.jsonl", train_ids)
    _write_pair_index(args.output_dir / "pairs_index_train.jsonl", rows_by_id, train_ids)

    # ---------------- Warmup slice ----------------
    # Extra robometer Group-A failures, disjoint from train + eval, for the failure-only
    # warmup phase (data.warmup_steps in the loss preset).
    print("\n[warmup] sampling extra failure-only pool …")
    warmup_ids = _sample_disjoint(rng, robometer_failures_groupA, args.warmup_n, excluded)
    excluded.update(warmup_ids)
    print(f"  warmup robometer failures: {len(warmup_ids):,}")
    _write_episode_jsonl(args.output_dir / "warmup.jsonl", warmup_ids)
    _write_pair_index(args.output_dir / "pairs_index_warmup.jsonl", rows_by_id, warmup_ids)

    # ---------------- Sanity report ----------------
    print("\n[summary]")
    print(f"  total train episodes:      {len(train_ids):,}")
    print(f"  total warmup episodes:     {len(warmup_ids):,}")
    print(f"  total eval episodes (sum): {sum(map(len, [eval_droid, eval_robometer, eval_metaworld, eval_failsafe])):,}")
    print(f"  output dir: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
