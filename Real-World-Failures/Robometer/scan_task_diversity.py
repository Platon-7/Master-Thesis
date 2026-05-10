#!/usr/bin/env python3
"""
scan_task_diversity.py — For each humanoid + human-hand archive we plan to
extract, pull `index_mappings.json` from the source tar (single or split),
read `task_indices` and `quality_indices.successful`, and project how many
ICL same-task pairs `pair_orphan_successes.py` would assemble:

  Tier 1 (in-archive  same task): for each task with N_succ>=2 episodes, all
                                  N pair (with the rare last-leftover
                                  exception); we report N_succ for tasks
                                  with N_succ>=2.
  Tier 2 (in-family   same task, sister archive fresh): tasks shared across
                                  archives in the same family register, after
                                  Tier 1 exhausts.
  No-pair: tasks with only 1 success episode AND no sister archive carries
                                  that task either.

Reuses the same family map as pair_orphan_successes (FAMILY_REGISTRY in
robometer_families.py + group registries in extract_orphan_successes.py).

Usage:
    python scan_task_diversity.py            # all humanoid+human_hand targets
    python scan_task_diversity.py --group humanoid
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tarfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from robometer_families import family_of  # noqa: E402
from extract_orphan_successes import HUMANOID_ARCHIVES, HUMAN_HAND_ARCHIVES  # noqa: E402

ARCH_ROOT = Path("/projects/prjs1958/robometer_full_dataset/raw_archives")
SINGLE_DIR = ARCH_ROOT / "single"
SPLIT_DIR = ARCH_ROOT / "split"

_WS_RE = re.compile(r"\s+")
_TRAIL_PUNCT = ".!?,;:\"')]}"


def norm_task(s: str) -> str:
    if not s:
        return s
    s = _WS_RE.sub(" ", s.lower().strip())
    while s and s[-1] in _TRAIL_PUNCT:
        s = s[:-1].rstrip()
    return s


def read_index_mapping_single(archive: str) -> dict:
    tar_path = SINGLE_DIR / f"{archive}.tar"
    with tarfile.open(tar_path, "r") as tf:
        for m in tf:
            if m.name.endswith("index_mappings.json"):
                return json.loads(tf.extractfile(m).read())
    raise RuntimeError(f"index_mappings.json not found in {tar_path}")


def read_index_mapping_split(archive: str) -> dict:
    parts = sorted((SPLIT_DIR / archive).glob(f"{archive}.tar.part-*"))
    if not parts:
        raise FileNotFoundError(archive)
    inner = f"{archive}/index_mappings.json"
    cat = subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE)
    # --occurrence=1 makes GNU tar stop after the first matching member.
    tar = subprocess.Popen(
        ["tar", "--occurrence=1", "-xOf", "-", inner],
        stdin=cat.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    cat.stdout.close()
    data, _ = tar.communicate()
    cat.kill()
    if not data:
        raise RuntimeError(f"no index_mappings.json in split {archive}")
    return json.loads(data)


def archive_tasks(archive: str) -> dict:
    if (SINGLE_DIR / f"{archive}.tar").exists():
        idx = read_index_mapping_single(archive)
    elif (SPLIT_DIR / archive).exists():
        idx = read_index_mapping_split(archive)
    else:
        raise FileNotFoundError(archive)

    succ = set(idx.get("quality_indices", {}).get("successful", []))
    task_indices = idx.get("task_indices", {})

    # task -> list of trajectory indices that are SUCCESSES
    task_succ = defaultdict(list)
    for task, indices in task_indices.items():
        key = norm_task(task)
        for i in indices:
            if i in succ:
                task_succ[key].append(i)

    return {
        "archive": archive,
        "family": family_of(archive),
        "n_success_total": len(succ),
        "n_distinct_tasks_with_succ": len(task_succ),
        "task_succ_sizes": {t: len(v) for t, v in task_succ.items()},
    }


def project_pairs(per_archive_results: list[dict]) -> dict:
    """Apply pair_orphan_successes.py's tier semantics to project pair counts.

    Tier 1: each task with k>=2 successes pairs all k episodes (last one may
    miss; conservatively count k for k>=2).  k==1 → falls through.
    Tier 2: across sister archives of the same family, for tasks not yet
    fully paired in Tier 1 (i.e., k==1 within archive), if any sister
    archive has the same task with k>=1, pair as Tier 2 (count of singletons
    that find a sister-archive partner).
    Else no_pair.
    """
    # family -> task -> {archive: count}
    fam_task = defaultdict(lambda: defaultdict(dict))
    for r in per_archive_results:
        fam = r["family"]
        for t, k in r["task_succ_sizes"].items():
            fam_task[fam][t][r["archive"]] = k

    summary = {"per_archive": {}, "totals": {"tier1": 0, "tier2": 0, "no_pair": 0,
                                             "n_episodes": 0}}
    for r in per_archive_results:
        a = r["archive"]
        fam = r["family"]
        t1 = t2 = np_ = 0
        for t, k in r["task_succ_sizes"].items():
            if k >= 2:
                t1 += k  # all k get a Tier-1 fresh partner (modulo last-leftover)
            else:
                # singleton in-archive — try Tier 2 (sister archive in family)
                sister_total = sum(
                    cnt for arch, cnt in fam_task[fam][t].items() if arch != a
                )
                if sister_total >= 1:
                    t2 += 1
                else:
                    np_ += 1
        n = r["n_success_total"]
        summary["per_archive"][a] = {
            "family": fam,
            "n_success_total": n,
            "n_distinct_tasks": r["n_distinct_tasks_with_succ"],
            "tier1_same_task_in_archive": t1,
            "tier2_same_task_family_fresh": t2,
            "no_pair": np_,
            "pair_rate": round((t1 + t2) / max(n, 1), 4),
        }
        summary["totals"]["tier1"] += t1
        summary["totals"]["tier2"] += t2
        summary["totals"]["no_pair"] += np_
        summary["totals"]["n_episodes"] += n

    summary["totals"]["pair_rate"] = round(
        (summary["totals"]["tier1"] + summary["totals"]["tier2"])
        / max(summary["totals"]["n_episodes"], 1), 4
    )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["humanoid", "human_hand", "both"], default="both")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    targets = []
    if args.group in ("humanoid", "both"):
        targets += list(HUMANOID_ARCHIVES.keys())
    if args.group in ("human_hand", "both"):
        targets += list(HUMAN_HAND_ARCHIVES.keys())

    print(f"Scanning {len(targets)} archives in parallel (workers={args.workers}) ...",
          flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(archive_tasks, a): a for a in targets}
        for f in as_completed(futs):
            a = futs[f]
            try:
                r = f.result()
                results.append(r)
                print(f"  ✓ {a:65s} succ={r['n_success_total']:>7,} "
                      f"tasks={r['n_distinct_tasks_with_succ']:>5,}", flush=True)
            except Exception as e:
                print(f"  ✗ {a}: {e}", flush=True)

    sm = project_pairs(results)
    print("\nPER-ARCHIVE PAIR PROJECTION")
    print(f"{'archive':70s}  {'fam':18s}  {'eps':>7s}  {'tasks':>6s}  "
          f"{'tier1':>7s}  {'tier2':>6s}  {'noPair':>6s}  {'rate':>5s}")
    for a, s in sorted(sm["per_archive"].items()):
        print(f"{a:70s}  {s['family']:18s}  {s['n_success_total']:>7,}  "
              f"{s['n_distinct_tasks']:>6,}  {s['tier1_same_task_in_archive']:>7,}  "
              f"{s['tier2_same_task_family_fresh']:>6,}  {s['no_pair']:>6,}  "
              f"{s['pair_rate']*100:>4.1f}%")
    t = sm["totals"]
    print(f"\nGRAND TOTAL: {t['n_episodes']:,} episodes  "
          f"tier1={t['tier1']:,}  tier2={t['tier2']:,}  no_pair={t['no_pair']:,}  "
          f"pair_rate={t['pair_rate']*100:.2f}%")


if __name__ == "__main__":
    main()
