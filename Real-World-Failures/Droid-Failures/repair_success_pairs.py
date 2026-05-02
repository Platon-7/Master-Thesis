#!/usr/bin/env python3
"""
repair_success_pairs.py — Re-pair any failure whose current matched success
has missing camera-view frames on disk. Picks a replacement from the pool of
already-downloaded, fully-complete successes, preferring the same match tier
(1_exact > 2_same_scene > 3_same_task > 4_same_lab). Writes the corrected
file back to success_pairs.jsonl in place (after backing up).
"""

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

BASE = Path("/projects/prjs1958/robometer_frame_dataset/droid")
PAIRS = BASE / "metadata" / "success_pairs.jsonl"
SUCC_IDX = BASE / "metadata" / "success_index.jsonl"
CAMS = [("ext1", "keyframes_success"),
        ("ext2", "keyframes_success_ext2"),
        ("wrist", "keyframes_success_wrist")]
N_FRAMES = 16


def task_slug(task):
    return re.sub(r'[^a-zA-Z0-9]+', '_', task or "").strip('_')[:60] or "unknown"


def norm_task(t):
    return (t or "").strip().lower()


def has_all_cams(success_ep, success_task):
    folder = f"{success_ep}__{task_slug(success_task)}"
    for _, sub in CAMS:
        d = BASE / sub / folder
        if not d.exists():
            return False
        if sum(1 for _ in d.glob("frame_*.jpg")) < N_FRAMES:
            return False
    return True


def main():
    pairs = [json.loads(l) for l in open(PAIRS)]
    print(f"Loaded {len(pairs)} pairs")

    # Success index for replacement metadata
    succ_idx = {}
    with open(SUCC_IDX) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "error" in d or not d.get("lab"):
                continue
            succ_idx[d["episode_id"]] = d

    # Identify broken pairs
    complete_pairs = []
    broken_pairs = []
    for p in pairs:
        if has_all_cams(p["success_ep"], p["success_task"]):
            complete_pairs.append(p)
        else:
            broken_pairs.append(p)
    print(f"Complete: {len(complete_pairs)} | Broken: {len(broken_pairs)}")

    if not broken_pairs:
        print("Nothing to repair.")
        return

    # Pool of already-downloaded, complete successes — these are the only
    # replacements we can use without triggering more downloads.
    used = {p["success_ep"] for p in complete_pairs}
    avail = defaultdict(list)   # tier-level buckets
    for p in complete_pairs:
        s = succ_idx.get(p["success_ep"])
        if not s:
            continue
        # Bucket by (lab, task, building, scene) and coarser keys
        lab = s["lab"]; task = norm_task(s["current_task"])
        bld = s.get("building",""); scn = s.get("scene_id","")
        avail[("1", lab, task, bld, scn)].append(p)
        avail[("2", lab, task, scn)].append(p)
        avail[("3", lab, task)].append(p)
        avail[("4", lab)].append(p)
        avail[("5",)].append(p)

    # Re-pair each broken entry
    rng_seed = 42
    import random
    rng = random.Random(rng_seed)
    repaired = []
    still_broken = 0
    tier_shifts = defaultdict(int)
    for p in broken_pairs:
        lab = p["failure_lab"]
        task = norm_task(p["failure_task"])
        scn = p["failure_scene"]
        # Try tiers in priority order; prefer unused candidates
        replacement = None; new_tier = None
        for key, t in [(("1", lab, task, "", scn), "1_exact"),
                       (("2", lab, task, scn), "2_same_scene"),
                       (("3", lab, task), "3_same_task"),
                       (("4", lab), "4_same_lab"),
                       (("5",), "5_any")]:
            # NB: we cannot reliably compute building from failure record here,
            # so tier 1 degenerates. Use tier 2 as effective same-scene match.
            cands = avail.get(key, [])
            fresh = [c for c in cands if c["success_ep"] not in used]
            if fresh:
                replacement = rng.choice(fresh)
                new_tier = t
                break
        if replacement is None:
            # Fall back: reuse a complete one (breaks uniqueness but keeps the datapoint)
            for key, t in [(("3", lab, task), "3_same_task_reused"),
                           (("4", lab), "4_same_lab_reused"),
                           (("5",), "5_any_reused")]:
                cands = avail.get(key, [])
                if cands:
                    replacement = rng.choice(cands)
                    new_tier = t
                    break
        if replacement is None:
            still_broken += 1
            continue

        used.add(replacement["success_ep"])
        new = dict(p)
        new["success_ep"]      = replacement["success_ep"]
        new["success_task"]    = replacement["success_task"]
        new["success_scene"]   = replacement["success_scene"]
        new["success_gcs"]     = replacement["success_gcs"]
        new["success_serials"] = replacement["success_serials"]
        new["success_mp4s"]    = replacement["success_mp4s"]
        new["tier"]            = new_tier
        new["repaired_from"]   = p["success_ep"]
        repaired.append(new)
        tier_shifts[(p["tier"], new_tier)] += 1

    print(f"Repaired: {len(repaired)} | Unrepairable: {still_broken}")
    for (old, new), c in sorted(tier_shifts.items()):
        print(f"  {old} -> {new}: {c}")

    # Write back
    backup = PAIRS.with_suffix(".jsonl.prerepair")
    shutil.copy(PAIRS, backup)
    print(f"Backup: {backup}")
    out = complete_pairs + repaired
    # Preserve original order
    order = {p["failure_ep"]: i for i, p in enumerate(pairs)}
    out.sort(key=lambda x: order.get(x["failure_ep"], 1_000_000))
    with open(PAIRS, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(out)} pairs to {PAIRS}")


if __name__ == "__main__":
    main()
