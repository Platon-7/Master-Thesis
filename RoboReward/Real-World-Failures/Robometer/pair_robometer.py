#!/usr/bin/env python3
"""
pair_robometer.py — Build failure -> success in-context pairs for every
Group A / Group B archive (see FOR_LEONARDO.md).

Input: /projects/prjs1958/robometer_full_dataset/raw_archives/{single,split}/
       Each archive contains index_mappings.json with:
         - quality_indices: {failure:[...], successful:[...], ...}
         - task_indices:    {task_string: [trajectory_indices]}
         - source_indices:  {dataset_tag: [indices]}

Output: /projects/prjs1958/robometer_full_dataset/pairs/
          <archive>_pairs.jsonl     # one record per failure, with tier + success ref
          report.json               # per-archive counts + tier histogram

Pairing policy (per failure, in priority order):
  1_same_task_fresh                — within-archive, same task, unused success
  2_same_task_family_fresh         — same task in a sister archive (same robot family), unused
  3_same_task_reused               — same task (in-archive or family), reused
  4_same_family_other_task_fresh   — any unused success from the same robot family
  5_same_family_other_task_reused  — any success from the same robot family (reused)
  6_unpaired                       — no family-level success available (marked for review)

Each record carries `origin_archive` so downstream filtering is trivial.

Exclusions (per user): DROID (already paired), metaworld, human_hand, humanoid, failsafe.

This script does NO frame extraction — only pair manifests.
"""

import json
import os
import re
import subprocess
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
import random

ARCH_ROOT = Path("/projects/prjs1958/robometer_full_dataset/raw_archives")
OUT_ROOT  = Path("/projects/prjs1958/robometer_full_dataset/pairs")

# ---- Failure archives (from FOR_LEONARDO.md, Groups A + B, minus exclusions) ----
# These are the only archives we draw FAILURES from.
FAILURE_ARCHIVES = {
    "jesbu1_racer_rfm_racer_train":                                             "racer",
    "jesbu1_racer_rfm_racer_val":                                               "racer",
    "jesbu1_auto_eval_rfm_auto_eval_rfm":                                       "auto_eval",
    "ykorkmaz_libero_failure_rfm_libero_90_failure":                            "libero",
    "ykorkmaz_libero_failure_rfm_libero_10_failure":                            "libero",
    "ykorkmaz_libero_failure_rfm_libero_object_failure":                        "libero",
    "ykorkmaz_libero_failure_rfm_libero_spatial_failure":                       "libero",
    "ykorkmaz_libero_failure_rfm_libero_goal_failure":                          "libero",
    "jesbu1_rfm_new_mit_franka_rfm_rfm_new_mit_franka_rfm":                     "mit_franka",
    "jesbu1_rfm_new_mit_franka_rfm_nowrist_rfm_new_mit_franka_rfm_nowrist":     "mit_franka",
    "jesbu1_mit_franka_p-rank_rfm_mit_franka_p-rank_rfm":                       "mit_franka",
    "jesbu1_usc_koch_p_ranking_rfm_usc_koch_p_ranking_all":                     "usc_koch",
    "aliangdw_utd_so101_policy_ranking_utd_so101_policy_ranking":               "utd_so101",
    "jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top":     "utd_so101",
    "jesbu1_utd_so101_clean_policy_ranking_wrist_utd_so101_clean_policy_ranking_wrist": "utd_so101",
    "aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking":                 "usc_xarm",
    "aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking":             "usc_franka",
    "ykorkmaz_usc_trossen_rfm_usc_trossen":                                     "usc_trossen",
    "jesbu1_soar_rfm_soar_rfm":                                                 "soar",
    "jesbu1_roboarena_0825_rfm_roboarena":                                      "roboarena",
    "jesbu1_roboarena_eval_debug_nowrist_roboarena_eval_debug_nowrist":         "roboarena",
}

# ---- Success-pool extras ----
# Additional archives (success-only or other-family failures we ignore) that
# contribute their `successful` indices to a given failure family.
# Keyed by archive name -> family used for pairing.
SUCCESS_POOL_EXTRAS = {
    # LIBERO demos come from abraranwar's archives
    "abraranwar_libero_rfm_libero256_10":                              "libero",
    "abraranwar_libero_rfm_libero256_90":                              "libero",
    "abraranwar_libero_rfm_libero256_goal":                            "libero",
    "abraranwar_libero_rfm_libero256_object":                          "libero",
    "abraranwar_libero_rfm_libero256_spatial":                         "libero",
    # Same-robot success-only archives detected by name pattern
    "abraranwar_usc_koch_rewind_rfm_usc_koch_rewind":                  "usc_koch",
    "jesbu1_usc_koch_human_robot_paired_usc_koch_human_robot_paired_robot": "usc_koch",
    "aliangdw_utd_so101_human_utd_so101_human":                        "utd_so101",
}

# Archives excluded from BOTH sides (per user): DROID (already paired),
# metaworld, human_hand, humanoid, failsafe.
EXCLUDED_ARCHIVES = {
    "jesbu1_oxe_rfm_oxe_droid",
    "aliangdw_metaworld_metaworld_eval",
    "aliangdw_metaworld_metaworld_train",
    "jesbu1_hand_paired_rfm_hand_paired_human",
    "jesbu1_hand_paired_rfm_hand_paired_robot",
    "jesbu1_humanoid_everyday_rfm_humanoid_everyday_rfm",
    "jesbu1_failsafe_rfm_failsafe",
    # human-side koch-paired is human video, not robot demo
    "jesbu1_usc_koch_human_robot_paired_usc_koch_human_robot_paired_human",
}

SINGLE_DIR = ARCH_ROOT / "single"
SPLIT_DIR  = ARCH_ROOT / "split"


def log(m): print(m, flush=True)


def read_index_mappings(archive_name):
    """Extract just index_mappings.json from the archive without unpacking the rest."""
    inner_path = f"{archive_name}/index_mappings.json"
    single_tar = SINGLE_DIR / f"{archive_name}.tar"
    split_dir  = SPLIT_DIR / archive_name

    if single_tar.exists():
        with tarfile.open(single_tar, "r") as t:
            try:
                f = t.extractfile(inner_path)
            except KeyError:
                # Some tars store with leading ./
                f = t.extractfile("./" + inner_path)
            return json.loads(f.read())

    if split_dir.exists():
        parts = sorted(split_dir.glob(f"{archive_name}.tar.part-*"))
        if not parts:
            raise FileNotFoundError(f"No parts for {archive_name}")
        # Stream `cat parts | tar -xO inner_path`
        cat = subprocess.Popen(["cat", *map(str, parts)],
                               stdout=subprocess.PIPE)
        tar = subprocess.Popen(["tar", "-xOf", "-", inner_path],
                               stdin=cat.stdout, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL)
        cat.stdout.close()
        data, _ = tar.communicate()
        cat.wait()
        if not data:
            raise RuntimeError(f"Failed to extract {inner_path} from split archive")
        return json.loads(data)

    raise FileNotFoundError(f"Archive not found: {archive_name}")


def task_of_index(task_indices, idx):
    # Invert task_indices lazily — we'll precompute an index -> task map per archive.
    raise NotImplementedError  # placeholder; done inline below for speed


def build_archive(archive_name, family):
    """Return a dict summarizing one archive's failures/successes."""
    log(f"[{archive_name}] reading index_mappings.json")
    idx = read_index_mappings(archive_name)
    q = idx.get("quality_indices", {})
    t = idx.get("task_indices", {})
    failures = sorted(set(q.get("failure", [])))
    successes = sorted(set(q.get("successful", [])))

    # Build index -> task map
    idx2task = {}
    for task, members in t.items():
        for m in members:
            idx2task[m] = task

    # Task -> successes (only keep successes that are actually labeled successful)
    succ_set = set(successes)
    task_to_succ = defaultdict(list)
    for s in successes:
        task_to_succ[idx2task.get(s, "<UNKNOWN_TASK>")].append(s)

    # Task -> failures
    task_to_fail = defaultdict(list)
    for f in failures:
        task_to_fail[idx2task.get(f, "<UNKNOWN_TASK>")].append(f)

    return {
        "archive":       archive_name,
        "family":        family,
        "failures":      failures,
        "successes":     successes,
        "idx2task":      idx2task,
        "task_to_succ":  dict(task_to_succ),
        "task_to_fail":  dict(task_to_fail),
        "total_trajectories": idx.get("total_trajectories") or len(idx2task) or (len(failures)+len(successes)),
    }


def pair_all():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    # Load every archive whose family we care about (failure archives + pool extras)
    all_archives = {**FAILURE_ARCHIVES, **SUCCESS_POOL_EXTRAS}
    archives = {}
    for name, fam in all_archives.items():
        if name in EXCLUDED_ARCHIVES:
            continue
        try:
            archives[name] = build_archive(name, fam)
        except Exception as e:
            log(f"[{name}] ERROR: {e}")
            archives[name] = {"error": str(e), "family": fam}

    # Per-family: task -> [(archive, succ_idx), ...] — pool successes across
    # every archive that belongs to the family (failure or not).
    family_task_succ = defaultdict(lambda: defaultdict(list))
    family_all_succ  = defaultdict(list)
    for name, a in archives.items():
        if "error" in a: continue
        fam = a["family"]
        for task, slist in a["task_to_succ"].items():
            for s in slist:
                family_task_succ[fam][task].append((name, s))
                family_all_succ[fam].append((name, s, task))

    # Track usage per (archive, idx)
    used = set()                           # set of (archive, idx)
    use_count = Counter()                  # (archive, idx) -> count

    report = {"per_archive": {}, "totals": Counter()}

    # Only iterate FAILURE archives for writing pairs — success-only archives
    # contribute to the pool but aren't queried for failures.
    for name in FAILURE_ARCHIVES:
        a = archives.get(name)
        if a is None or "error" in (a or {}):
            report["per_archive"][name] = {"error": (a or {}).get("error", "not loaded"),
                                           "family": FAILURE_ARCHIVES[name]}
            continue

        out_path = OUT_ROOT / f"{name}_pairs.jsonl"
        fh = open(out_path, "w")
        tier_hist = Counter()
        unmatched_tasks = Counter()  # task strings for tier 4+
        fam = a["family"]

        # Determine in-archive success pools and family pools
        for f_idx in a["failures"]:
            task = a["idx2task"].get(f_idx, "<UNKNOWN_TASK>")

            match = None  # (archive, idx, tier_name)

            # ---- Tier 1: same task, same archive, fresh ----
            cands = [(name, s) for s in a["task_to_succ"].get(task, []) if (name, s) not in used]
            if cands:
                pick = rng.choice(cands)
                match = (pick[0], pick[1], "1_same_task_fresh")

            # ---- Tier 2: same task in family, fresh ----
            if match is None:
                cands = [(arc, s) for (arc, s) in family_task_succ[fam].get(task, [])
                         if (arc, s) not in used]
                if cands:
                    pick = rng.choice(cands)
                    match = (pick[0], pick[1], "2_same_task_family_fresh")

            # ---- Tier 3: same task, reuse (in-archive preferred, then family) ----
            if match is None:
                pool = [(name, s) for s in a["task_to_succ"].get(task, [])]
                if not pool:
                    pool = family_task_succ[fam].get(task, [])
                if pool:
                    # Pick the least-used
                    min_c = min(use_count[k] for k in pool)
                    ties = [k for k in pool if use_count[k] == min_c]
                    pick = rng.choice(ties)
                    match = (pick[0], pick[1], "3_same_task_reused")

            # ---- Tier 4: same family, different task, fresh ----
            if match is None:
                cands = [(arc, s, tt) for (arc, s, tt) in family_all_succ[fam]
                         if (arc, s) not in used]
                if cands:
                    pick = rng.choice(cands)
                    match = (pick[0], pick[1], "4_same_family_other_task_fresh")
                    unmatched_tasks[task] += 1

            # ---- Tier 5: same family, different task, reused ----
            if match is None and family_all_succ[fam]:
                pool = [(arc, s) for (arc, s, _) in family_all_succ[fam]]
                min_c = min(use_count[k] for k in pool)
                ties = [k for k in pool if use_count[k] == min_c]
                pick = rng.choice(ties)
                match = (pick[0], pick[1], "5_same_family_other_task_reused")
                unmatched_tasks[task] += 1

            # ---- Tier 6: no family success available ----
            if match is None:
                rec = {
                    "archive":       name,
                    "family":        fam,
                    "failure_idx":   f_idx,
                    "failure_task":  task,
                    "tier":          "6_unpaired",
                }
                fh.write(json.dumps(rec) + "\n")
                tier_hist["6_unpaired"] += 1
                unmatched_tasks[task] += 1
                continue

            arc, s_idx, tier = match
            used.add((arc, s_idx))
            use_count[(arc, s_idx)] += 1
            succ_task = archives[arc]["idx2task"].get(s_idx, "<UNKNOWN_TASK>")
            rec = {
                "archive":        name,
                "family":         fam,
                "failure_idx":    f_idx,
                "failure_task":   task,
                "success_archive": arc,
                "success_idx":    s_idx,
                "success_task":   succ_task,
                "tier":           tier,
            }
            fh.write(json.dumps(rec) + "\n")
            tier_hist[tier] += 1

        fh.close()

        # Successes left unpaired in this archive (not used by anyone)
        unused_succ = [s for s in a["successes"] if (name, s) not in used]
        report["per_archive"][name] = {
            "family":            fam,
            "total_failures":    len(a["failures"]),
            "total_successes":   len(a["successes"]),
            "tasks_total":       len(set(a["task_to_fail"]) | set(a["task_to_succ"])),
            "tasks_fail_only":   sorted(set(a["task_to_fail"]) - set(a["task_to_succ"])),
            "tier_hist":         dict(tier_hist),
            "unpaired":          tier_hist["6_unpaired"],
            "successes_unused":  len(unused_succ),
            "failures_no_same_task_match": sum(unmatched_tasks.values()),
            "fail_tasks_missing_success":  sorted(unmatched_tasks.items(),
                                                  key=lambda x: -x[1])[:20],
        }
        for k, v in tier_hist.items():
            report["totals"][k] += v
        report["totals"]["failures"] += len(a["failures"])

        log(f"[{name}] fails={len(a['failures'])} succ={len(a['successes'])} "
            f"tiers={dict(tier_hist)}")

    # Family-level summary
    fam_summary = defaultdict(lambda: Counter())
    for name, stats in report["per_archive"].items():
        if "error" in stats: continue
        fam = stats["family"]
        fam_summary[fam]["failures"] += stats["total_failures"]
        fam_summary[fam]["successes"] += stats["total_successes"]
        for k, v in stats["tier_hist"].items():
            fam_summary[fam][k] += v
    report["per_family"] = {k: dict(v) for k, v in fam_summary.items()}
    report["totals"] = dict(report["totals"])

    with open(OUT_ROOT / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    log(f"Report written: {OUT_ROOT/'report.json'}")


if __name__ == "__main__":
    pair_all()
