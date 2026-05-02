#!/usr/bin/env python3
"""
match_successes.py — Pair each DROID failure with a success episode.

Reads:
  • /projects/prjs1958/robometer_frame_dataset/droid/metadata/scored_full_droid_shard*.jsonl
    (gives us the 5,503 failure episode_ids and cleaned tasks)
  • /projects/prjs1958/robometer_frame_dataset/droid/metadata/success_index.jsonl
    (all success episodes with lab/current_task/building/scene_id)

Because the failure scored-JSONL lacks building/scene_id, we also fetch each
failure's GCS metadata to get those fields (one-time cost).

Writes:
  • metadata/failure_index.jsonl   — failure metadata (same schema as success_index)
  • metadata/success_pairs.jsonl   — {failure_ep, success_ep, tier, ...}

Match tiers (higher = closer match):
  1 exact        — same (lab, current_task, building, scene_id)
  2 same_scene   — same (lab, current_task, scene_id)
  3 same_task    — same (lab, current_task)
  4 same_lab     — any success from the same lab
  5 any          — any success

Usage:
  python match_successes.py --workers 32
"""

import json
import re
import argparse
import requests
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_DIR  = Path("/projects/prjs1958/robometer_frame_dataset/droid")
META_DIR  = BASE_DIR / "metadata"
FAIL_IDX  = META_DIR / "failure_index.jsonl"
SUCC_IDX  = META_DIR / "success_index.jsonl"
PAIRS_OUT = META_DIR / "success_pairs.jsonl"

DROID_PFX = "robotics/droid_raw/1.0.1"
GCS_LIST  = "https://storage.googleapis.com/storage/v1/b/gresearch/o"

print_lock = Lock()
write_lock = Lock()


def log(msg):
    with print_lock:
        print(msg, flush=True)


def make_session():
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.0,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=64))
    return s


def failure_ep_to_gcs_prefix(ep_id):
    lab, rest = ep_id.split("_", 1)
    date = rest[:10]
    ep_disk = rest[11:]
    ep_gcs = re.sub(r"(\d{2})-(\d{2})-(\d{2})", r"\1:\2:\3", ep_disk, count=1)
    return f"{DROID_PFX}/{lab}/failure/{date}/{ep_gcs}"


def fetch_failure_meta(sess, ep_id):
    prefix = failure_ep_to_gcs_prefix(ep_id)
    try:
        r = sess.get(GCS_LIST, params={"prefix": prefix + "/", "maxResults": 50},
                     timeout=30)
        r.raise_for_status()
        items = r.json().get("items", [])
        meta_obj = next((o for o in items
                         if o["name"].rsplit("/", 1)[-1].startswith("metadata_")
                         and o["name"].endswith(".json")), None)
        if meta_obj is None:
            return None
        import urllib.parse
        enc = urllib.parse.quote(meta_obj["name"], safe="")
        md = sess.get(f"https://storage.googleapis.com/storage/v1/b/gresearch/o/{enc}?alt=media",
                      timeout=30).json()
        mp4s = {o["name"].rsplit("/", 1)[-1]: o["name"]
                for o in items
                if o["name"].endswith(".mp4") and "stereo" not in o["name"]}
        return {
            "episode_id":   ep_id,
            "gcs_prefix":   prefix,
            "lab":          md.get("lab", ""),
            "current_task": md.get("current_task", ""),
            "building":     md.get("building", ""),
            "scene_id":     md.get("scene_id", ""),
            "date":         md.get("date", ""),
            "ext1_serial":  md.get("ext1_cam_serial", ""),
            "ext2_serial":  md.get("ext2_cam_serial", ""),
            "wrist_serial": md.get("wrist_cam_serial", ""),
            "mp4s":         mp4s,
        }
    except Exception as e:
        return {"episode_id": ep_id, "error": str(e)}


def build_failure_index(workers):
    """Fetch GCS metadata for every failure episode (resume-safe)."""
    # Collect all failure episode_ids
    fail_eps = set()
    for p in sorted(META_DIR.glob("scored_full_droid_shard*.jsonl")):
        with open(p) as f:
            for line in f:
                try:
                    fail_eps.add(json.loads(line)["episode_id"])
                except Exception:
                    pass
    log(f"Failures total: {len(fail_eps)}")

    done = set()
    if FAIL_IDX.exists():
        with open(FAIL_IDX) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["episode_id"])
                except Exception:
                    pass
        log(f"Resume: {len(done)} failure metadata already fetched")
    todo = sorted(fail_eps - done)
    if not todo:
        log("Failure index complete.")
        return

    log(f"Fetching metadata for {len(todo)} failures…")
    fh = open(FAIL_IDX, "a")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fetch_failure_meta, make_session(), e): e for e in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            if rec is None:
                continue
            with write_lock:
                fh.write(json.dumps(rec) + "\n")
                if i % 500 == 0:
                    fh.flush()
            if i % 500 == 0:
                log(f"  {i}/{len(todo)}")
    fh.close()
    log("Failure index built.")


def norm_task(t):
    return (t or "").strip().lower()


def match_all():
    # Load success index, grouped for fast matching
    succ_by_4 = defaultdict(list)   # (lab, task, building, scene) -> [records]
    succ_by_3 = defaultdict(list)   # (lab, task, scene)
    succ_by_2 = defaultdict(list)   # (lab, task)
    succ_by_1 = defaultdict(list)   # (lab,)
    succ_all  = []

    n_succ = 0
    with open(SUCC_IDX) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "error" in d or not d.get("lab"):
                continue
            n_succ += 1
            key4 = (d["lab"], norm_task(d["current_task"]), d.get("building",""), d.get("scene_id",""))
            key3 = (d["lab"], norm_task(d["current_task"]), d.get("scene_id",""))
            key2 = (d["lab"], norm_task(d["current_task"]))
            key1 = (d["lab"],)
            succ_by_4[key4].append(d)
            succ_by_3[key3].append(d)
            succ_by_2[key2].append(d)
            succ_by_1[key1].append(d)
            succ_all.append(d)
    log(f"Loaded {n_succ} successes into match index")

    # Load failures
    failures = []
    with open(FAIL_IDX) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "error" in d or not d.get("lab"):
                continue
            failures.append(d)
    log(f"Loaded {len(failures)} failures")

    # Already-matched set (resume)
    already = set()
    if PAIRS_OUT.exists():
        with open(PAIRS_OUT) as f:
            for line in f:
                try:
                    already.add(json.loads(line)["failure_ep"])
                except Exception:
                    pass
        log(f"Resume: {len(already)} pairs already recorded")

    # Track which successes have already been used to avoid reusing one success
    # for many failures when better matches exist.
    used = set(json.loads(line)["success_ep"]
               for line in open(PAIRS_OUT).read().splitlines() if line) \
           if PAIRS_OUT.exists() else set()

    rng = random.Random(42)
    tier_counts = Counter()
    use_counts = Counter()
    for ep in used:
        use_counts[ep] += 1
    fh = open(PAIRS_OUT, "a")

    # Deterministic order for reproducibility
    failures.sort(key=lambda x: x["episode_id"])

    for f in failures:
        if f["episode_id"] in already:
            tier_counts[ "resumed" ] += 1
            continue

        task = norm_task(f["current_task"])
        lab  = f["lab"]
        bld  = f.get("building","")
        scn  = f.get("scene_id","")

        tiers = [(succ_by_4[(lab, task, bld, scn)], "1_exact"),
                 (succ_by_3[(lab, task, scn)],      "2_same_scene"),
                 (succ_by_2[(lab, task)],           "3_same_task"),
                 (succ_by_1[(lab,)],                "4_same_lab"),
                 (succ_all,                         "5_any")]

        # First pass: prefer an unused success at the highest-priority tier
        # that has any fresh candidate. This avoids picking a reused
        # tier-1 success when tier-2/3 still has untouched options for the
        # same task.
        match = None; tier = None
        for cands, t in tiers:
            fresh = [c for c in cands if c["episode_id"] not in used]
            if fresh:
                match = rng.choice(fresh)
                tier = t
                break

        # Second pass: no fresh candidate anywhere — reuse the least-used
        # success at the highest-priority tier that has any candidate.
        if match is None:
            for cands, t in tiers:
                if cands:
                    min_use = min(use_counts[c["episode_id"]] for c in cands)
                    pool = [c for c in cands if use_counts[c["episode_id"]] == min_use]
                    match = rng.choice(pool)
                    tier = t + "_reused"
                    break

        if match is None:
            log(f"  no match for {f['episode_id']}")
            continue

        used.add(match["episode_id"])
        use_counts[match["episode_id"]] += 1
        rec = {
            "failure_ep":      f["episode_id"],
            "failure_task":    f["current_task"],
            "failure_lab":     lab,
            "failure_scene":   str(scn),
            "success_ep":      match["episode_id"],
            "success_task":    match["current_task"],
            "success_scene":   str(match.get("scene_id","")),
            "success_gcs":     match["gcs_prefix"],
            "success_serials": {"ext1": match.get("ext1_serial",""),
                                "ext2": match.get("ext2_serial",""),
                                "wrist": match.get("wrist_serial","")},
            "success_mp4s":    match.get("mp4s", {}),
            "tier":            tier,
        }
        fh.write(json.dumps(rec) + "\n")
        tier_counts[tier.split("_reused")[0]] += 1

    fh.close()
    total = sum(tier_counts.values())
    log("=" * 60)
    log("Match tier breakdown:")
    for t, c in sorted(tier_counts.items()):
        log(f"  {t:20s} {c:6d}  ({100*c/total:5.1f}%)")
    log("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--skip-failure-index", action="store_true")
    args = ap.parse_args()

    if not args.skip_failure_index:
        build_failure_index(args.workers)
    match_all()


if __name__ == "__main__":
    main()
