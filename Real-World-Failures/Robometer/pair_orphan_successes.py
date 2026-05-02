#!/usr/bin/env python3
"""
pair_orphan_successes.py — Build success↔success pair manifests for every
orphan-success archive under robometer_frame_dataset/robometer/.

Input  : /projects/prjs1958/robometer_frame_dataset/robometer/manifests/
           *_orphan_successes.jsonl
Output : /projects/prjs1958/robometer_frame_dataset/robometer/pairs_orphan/
           <archive>_orphan_pairs.jsonl   # one record per episode
           report.json                    # per-archive + family stats, task-size histogram

Policy (per episode, same-task only; no mixing with non-orphan successes):
  1_same_task_fresh           — same task, same archive, unused partner
  2_same_task_family_fresh    — same task, sister archive (same family), unused
  3_same_task_reused          — same task, least-used partner (cap configurable)
  no_pair                     — no same-task partner available

Self-pair is always forbidden (partner episode_id != self episode_id).

CLI:
  python pair_orphan_successes.py               # uncapped reuse (max coverage)
  python pair_orphan_successes.py --reuse-cap 5 # cap each partner at 5 uses
  python pair_orphan_successes.py --seed 123
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/projects/prjs1958/robometer_frame_dataset/robometer")
MAN_DIR = ROOT / "manifests"
OUT_DIR = ROOT / "pairs_orphan"
MANIFEST_SUFFIX = "_orphan_successes.jsonl"

_WS_RE = re.compile(r"\s+")
_TRAIL_PUNCT = ".!?,;:\"')]}"


def norm_task(s):
    """Lowercase, strip, collapse whitespace, drop trailing punctuation.
    Conservative: keeps articles so semantically distinct tasks don't collide."""
    if not s:
        return s
    s = _WS_RE.sub(" ", s.lower().strip())
    while s and s[-1] in _TRAIL_PUNCT:
        s = s[:-1].rstrip()
    return s


def load_manifests():
    archives = {}
    for p in sorted(MAN_DIR.glob(f"*{MANIFEST_SUFFIX}")):
        archive = p.name[: -len(MANIFEST_SUFFIX)]
        eps = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                eps.append(json.loads(line))
        archives[archive] = eps
    return archives


def task_size_bucket(n):
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    if n <= 10:
        return "3-10"
    if n <= 100:
        return "11-100"
    return "100+"


def pair_all(reuse_cap, seed):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    archives = load_manifests()

    # task lookups
    by_archive_task = defaultdict(lambda: defaultdict(list))      # archive -> task -> [ep_id]
    by_family_task = defaultdict(lambda: defaultdict(list))       # family  -> task -> [(archive, ep_id)]
    episode_lookup = {}                                           # ep_id -> ep dict

    for archive, eps in archives.items():
        for ep in eps:
            ep_id = ep["episode_id"]
            task_key = norm_task(ep["task"])
            family = ep["family"]
            by_archive_task[archive][task_key].append(ep_id)
            by_family_task[family][task_key].append((archive, ep_id))
            episode_lookup[ep_id] = ep

    # Family-level task-size histogram (recoverability ceiling)
    fam_task_size_hist = defaultdict(Counter)
    for family, tasks in by_family_task.items():
        for members in tasks.values():
            fam_task_size_hist[family][task_size_bucket(len(members))] += 1

    cap = reuse_cap if reuse_cap and reuse_cap > 0 else float("inf")
    use_count = Counter()

    report = {
        "config": {"reuse_cap": reuse_cap, "seed": seed},
        "totals": Counter(),
        "per_archive": {},
        "per_family": {},
        "task_size_histogram_by_family": {f: dict(h) for f, h in fam_task_size_hist.items()},
    }

    for archive, eps in archives.items():
        out_path = OUT_DIR / f"{archive}_orphan_pairs.jsonl"
        tier_hist = Counter()
        with open(out_path, "w") as fh:
            for ep in eps:
                ep_id = ep["episode_id"]
                task = ep["task"]
                task_key = norm_task(task)
                family = ep["family"]

                match = None  # (partner_ep_id, partner_archive, tier)

                # Tier 1: same task, same archive, fresh
                cands = [
                    eid for eid in by_archive_task[archive].get(task_key, [])
                    if eid != ep_id and use_count[eid] == 0
                ]
                if cands:
                    pick = rng.choice(cands)
                    match = (pick, archive, "1_same_task_fresh")

                # Tier 2: same task, same family, sister archive, fresh
                if match is None:
                    cands = [
                        (arc, eid) for (arc, eid) in by_family_task[family].get(task_key, [])
                        if arc != archive and eid != ep_id and use_count[eid] == 0
                    ]
                    if cands:
                        pick = rng.choice(cands)
                        match = (pick[1], pick[0], "2_same_task_family_fresh")

                # Tier 3: same task, reused (in-archive first, then family), respect cap
                if match is None:
                    pool = [
                        (archive, eid) for eid in by_archive_task[archive].get(task_key, [])
                        if eid != ep_id and use_count[eid] < cap
                    ]
                    if not pool:
                        pool = [
                            (arc, eid) for (arc, eid) in by_family_task[family].get(task_key, [])
                            if eid != ep_id and use_count[eid] < cap
                        ]
                    if pool:
                        min_c = min(use_count[eid] for (_, eid) in pool)
                        ties = [(arc, eid) for (arc, eid) in pool if use_count[eid] == min_c]
                        pick = rng.choice(ties)
                        match = (pick[1], pick[0], "3_same_task_reused")

                if match is None:
                    rec = {
                        "episode_id": ep_id,
                        "archive": archive,
                        "family": family,
                        "task": task,
                        "partner_episode_id": None,
                        "partner_archive": None,
                        "partner_task": None,
                        "tier": "no_pair",
                    }
                    fh.write(json.dumps(rec) + "\n")
                    tier_hist["no_pair"] += 1
                    continue

                p_id, p_arc, tier = match
                use_count[p_id] += 1
                rec = {
                    "episode_id": ep_id,
                    "archive": archive,
                    "family": family,
                    "task": task,
                    "partner_episode_id": p_id,
                    "partner_archive": p_arc,
                    "partner_task": episode_lookup[p_id]["task"],
                    "tier": tier,
                }
                fh.write(json.dumps(rec) + "\n")
                tier_hist[tier] += 1

        report["per_archive"][archive] = {
            "family": eps[0]["family"] if eps else None,
            "n_episodes": len(eps),
            "tier_hist": dict(tier_hist),
            "no_pair_rate": round(tier_hist["no_pair"] / max(len(eps), 1), 4),
        }
        for k, v in tier_hist.items():
            report["totals"][k] += v
        report["totals"]["episodes"] += len(eps)
        print(f"[{archive}] eps={len(eps)} tiers={dict(tier_hist)}", flush=True)

    fam_rollup = defaultdict(Counter)
    for stats in report["per_archive"].values():
        fam = stats["family"]
        if fam is None:
            continue
        fam_rollup[fam]["episodes"] += stats["n_episodes"]
        for k, v in stats["tier_hist"].items():
            fam_rollup[fam][k] += v
    report["per_family"] = {f: dict(c) for f, c in fam_rollup.items()}
    report["totals"] = dict(report["totals"])

    # Partner-usage stats
    if use_count:
        uses = sorted(use_count.values())
        report["partner_usage"] = {
            "n_partners_used": len(use_count),
            "max_uses": uses[-1],
            "median_uses": uses[len(uses) // 2],
            "mean_uses": round(sum(uses) / len(uses), 2),
        }

    with open(OUT_DIR / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {OUT_DIR/'report.json'}")
    print(f"Totals: {report['totals']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-cap", type=int, default=0,
                    help="Max times a single partner may be reused in tier 3. 0 = uncapped.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    pair_all(args.reuse_cap, args.seed)


if __name__ == "__main__":
    main()
