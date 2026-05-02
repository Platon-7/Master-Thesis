#!/usr/bin/env python3
"""
build_success_index.py — Enumerate all DROID success episodes on GCS and
extract matching keys (lab, current_task, building, scene_id) per episode.

Output: one JSONL line per success episode, written to
  /projects/prjs1958/robometer_frame_dataset/droid/metadata/success_index.jsonl

Resume-safe: existing episode_ids in the output are skipped on re-run.

Usage:
    python build_success_index.py --workers 32
"""

import json
import argparse
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

GCS_LIST_URL = "https://storage.googleapis.com/storage/v1/b/gresearch/o"
DROID_PFX    = "robotics/droid_raw/1.0.1"

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


def list_prefixes(sess, prefix):
    """List subdirectories (common-prefixes) directly under `prefix`."""
    out = []
    token = None
    while True:
        params = {"prefix": prefix, "delimiter": "/", "maxResults": 1000}
        if token:
            params["pageToken"] = token
        r = sess.get(GCS_LIST_URL, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        out.extend(j.get("prefixes", []))
        token = j.get("nextPageToken")
        if not token:
            break
    return out


def list_objects(sess, prefix):
    params = {"prefix": prefix, "maxResults": 50}
    r = sess.get(GCS_LIST_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("items", [])


def episode_id_from_prefix(ep_prefix):
    """
    'robotics/.../AUTOLab/success/2023-07-07/Fri_Jul__7_09:42:23_2023/'
      → 'AUTOLab_2023-07-07_Fri_Jul__7_09-42-23_2023'
    """
    parts = ep_prefix.rstrip("/").split("/")
    lab, _bucket, date, ep_gcs = parts[-4], parts[-3], parts[-2], parts[-1]
    ep_disk = ep_gcs.replace(":", "-", 2)   # HH:MM:SS → HH-MM-SS (first two colons)
    # re.sub would be cleaner but we want the exact opposite of download_extra_views
    import re
    ep_disk = re.sub(r"(\d{2}):(\d{2}):(\d{2})", r"\1-\2-\3", ep_gcs, count=1)
    return f"{lab}_{date}_{ep_disk}"


def fetch_episode_meta(sess, ep_prefix):
    """Return dict with fields we need, or None on failure."""
    try:
        items = list_objects(sess, ep_prefix)
        meta_obj = next((o for o in items
                         if o["name"].rsplit("/", 1)[-1].startswith("metadata_")
                         and o["name"].endswith(".json")), None)
        if meta_obj is None:
            return None
        name = meta_obj["name"]
        import urllib.parse
        enc = urllib.parse.quote(name, safe="")
        url = f"https://storage.googleapis.com/storage/v1/b/gresearch/o/{enc}?alt=media"
        md = sess.get(url, timeout=30).json()

        mp4s = {o["name"].rsplit("/", 1)[-1]: o["name"]
                for o in items
                if o["name"].endswith(".mp4") and "stereo" not in o["name"]}

        return {
            "episode_id":   episode_id_from_prefix(ep_prefix),
            "gcs_prefix":   ep_prefix.rstrip("/"),
            "lab":          md.get("lab", ""),
            "current_task": md.get("current_task", ""),
            "building":     md.get("building", ""),
            "scene_id":     md.get("scene_id", ""),
            "date":         md.get("date", ""),
            "uuid":         md.get("uuid", ""),
            "ext1_serial":  md.get("ext1_cam_serial", ""),
            "ext2_serial":  md.get("ext2_cam_serial", ""),
            "wrist_serial": md.get("wrist_cam_serial", ""),
            "mp4s":         mp4s,
        }
    except Exception as e:
        return {"error": str(e), "gcs_prefix": ep_prefix}


def enumerate_episode_prefixes(sess):
    """Walk labs → dates → episodes under /success/, return all episode prefixes."""
    log("Listing labs…")
    lab_prefixes = list_prefixes(sess, f"{DROID_PFX}/")
    log(f"  Found {len(lab_prefixes)} labs")

    episode_prefixes = []
    for lab_pfx in lab_prefixes:
        success_pfx = f"{lab_pfx}success/"
        date_pfxs = list_prefixes(sess, success_pfx)
        if not date_pfxs:
            continue
        log(f"  {lab_pfx.rstrip('/').split('/')[-1]}: {len(date_pfxs)} dates")
        # Parallelize date-level listing
        with ThreadPoolExecutor(max_workers=16) as pool:
            futs = {pool.submit(list_prefixes, sess, dp): dp for dp in date_pfxs}
            for fut in as_completed(futs):
                episode_prefixes.extend(fut.result())
    return episode_prefixes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",
                    default="/projects/prjs1958/robometer_frame_dataset/droid/metadata/success_index.jsonl")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume set
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if "gcs_prefix" in d:
                        done.add(d["gcs_prefix"])
                except Exception:
                    pass
        log(f"Resume: {len(done)} episodes already indexed")

    sess = make_session()
    ep_prefixes = enumerate_episode_prefixes(sess)
    log(f"Total success episodes on GCS: {len(ep_prefixes)}")

    todo = [p for p in ep_prefixes if p.rstrip("/") not in done]
    if args.limit:
        todo = todo[:args.limit]
    log(f"To fetch metadata for: {len(todo)}")

    out_fh = open(out_path, "a")
    counter = {"ok": 0, "err": 0}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch_episode_meta, make_session(), p): p for p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            if rec is None or "error" in (rec or {}):
                counter["err"] += 1
            else:
                counter["ok"] += 1
            if rec is not None:
                with write_lock:
                    out_fh.write(json.dumps(rec) + "\n")
                    if i % 500 == 0:
                        out_fh.flush()
            if i % 500 == 0:
                log(f"  {i}/{len(todo)} | ok={counter['ok']} err={counter['err']}")

    out_fh.close()
    log(f"DONE | ok={counter['ok']} err={counter['err']} | wrote {out_path}")


if __name__ == "__main__":
    main()
