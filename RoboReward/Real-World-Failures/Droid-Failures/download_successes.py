#!/usr/bin/env python3
"""
download_successes.py — Download 3 camera MP4s per matched success episode
and extract 16 frames per camera at uniformly-spaced timestamps.

Reads:
  /projects/prjs1958/robometer_frame_dataset/droid/metadata/success_pairs.jsonl

Writes:
  /projects/prjs1958/robometer_frame_dataset/droid/keyframes_success/         (ext1)
  /projects/prjs1958/robometer_frame_dataset/droid/keyframes_success_ext2/
  /projects/prjs1958/robometer_frame_dataset/droid/keyframes_success_wrist/

Frame pipeline (same as failure ext2/wrist):
  raw 60fps  →  fps=10, scale=320:180, pad=320:192:0:6:black  →  16 JPEGs

Folder naming: {success_episode_id}__{task_slug}, matching the failure layout.

Resume-safe: episodes with all 16 frames already present are skipped.

Usage:
    python download_successes.py --workers 24
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

GCS_BASE = "https://storage.googleapis.com"
BUCKET   = "gresearch"
CAMERAS  = [("ext1", "keyframes_success"),
            ("ext2", "keyframes_success_ext2"),
            ("wrist", "keyframes_success_wrist")]

print_lock = Lock()
def log(msg):
    with print_lock:
        print(msg, flush=True)


def make_session():
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.5,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def task_slug(task):
    return re.sub(r'[^a-zA-Z0-9]+', '_', task or "").strip('_')[:60] or "unknown"


def download_video(sess, gcs_path, local_path):
    import urllib.parse
    url = f"{GCS_BASE}/{BUCKET}/{urllib.parse.quote(gcs_path, safe='/')}"
    try:
        with sess.get(url, stream=True, timeout=180) as r:
            r.raise_for_status()
            with open(local_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        return True
    except Exception as e:
        log(f"  [dl err] {gcs_path}: {e}")
        return False


def probe_10fps_frame_count(video_path):
    """Count how many frames the video produces after fps=10 filter."""
    cmd = ["ffprobe", "-v", "error",
           "-select_streams", "v:0",
           "-show_entries", "stream=duration",
           "-of", "default=nw=1:nk=1", video_path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        dur = float(r.stdout.strip())
        return max(1, int(dur * 10))
    except Exception:
        return 40  # sane default for ~4 s DROID clip


def extract_frames(video_path, n_frames, out_dir, folder_name):
    ep_dir = out_dir / folder_name
    ep_dir.mkdir(parents=True, exist_ok=True)
    if sum(1 for _ in ep_dir.glob("frame_*.jpg")) >= n_frames:
        return True

    # Clean stale tmp_*.jpg from a prior interrupted run.
    for f in ep_dir.glob("tmp_*.jpg"):
        f.unlink(missing_ok=True)

    total_10fps = probe_10fps_frame_count(video_path)
    # Uniform indices in the 10fps stream (first & last included)
    if total_10fps < 2:
        return False
    indices = [round(i * (total_10fps - 1) / (n_frames - 1)) for i in range(n_frames)]
    # Deduplicate but keep order (very short videos may produce duplicates)
    seen, uniq = set(), []
    for x in indices:
        if x not in seen:
            seen.add(x); uniq.append(x)

    select_expr = "+".join(f"eq(n,{n})" for n in uniq)
    tmp_pattern = str(ep_dir / "tmp_%02d.jpg")
    cmd = ["ffmpeg", "-y", "-i", video_path,
           "-vf", f"fps=10,scale=320:180,pad=320:192:0:6:black,select='{select_expr}'",
           "-vsync", "0", "-q:v", "2", "-loglevel", "error", tmp_pattern]
    res = subprocess.run(cmd, capture_output=True, timeout=180)
    if res.returncode != 0:
        log(f"  [ffmpeg err] {folder_name}: {res.stderr.decode()[:200]}")
        for f in ep_dir.glob("tmp_*.jpg"):
            f.unlink(missing_ok=True)
        return False

    tmp_files = sorted(ep_dir.glob("tmp_*.jpg"))
    n_got = len(tmp_files)
    if n_got == 0:
        return False
    # Pad by duplicating the last frame if we were short (or dedup happened).
    # Indices are written back by matching extraction order; we relabel by
    # `i`, and approximate timestamp as indices[i]/10.
    while len(tmp_files) < n_frames:
        dup = ep_dir / f"tmp_{len(tmp_files) + 1:02d}.jpg"
        shutil.copy(str(tmp_files[-1]), str(dup))
        tmp_files = sorted(ep_dir.glob("tmp_*.jpg"))

    for i, tmp_f in enumerate(tmp_files[:n_frames]):
        ts = indices[i] / 10.0
        final = ep_dir / f"frame_{i}_{ts:.2f}s.jpg"
        tmp_f.rename(final)
    # Clean any leftover tmp files
    for f in ep_dir.glob("tmp_*.jpg"):
        f.unlink(missing_ok=True)
    return True


def process_pair(pair, base_dir, n_frames):
    sess = make_session()
    succ_ep = pair["success_ep"]
    task = pair["success_task"]
    folder = f"{succ_ep}__{task_slug(task)}"
    mp4s = pair["success_mp4s"]
    serials = pair["success_serials"]

    result = {"success_ep": succ_ep}
    for cam_key, out_subdir in CAMERAS:
        out_dir = base_dir / out_subdir
        ep_dir = out_dir / folder
        if ep_dir.exists() and sum(1 for _ in ep_dir.glob("frame_*.jpg")) >= n_frames:
            result[cam_key] = "skipped"
            continue

        serial = serials.get(cam_key, "")
        fname = f"{serial}.mp4"
        if not serial or fname not in mp4s:
            result[cam_key] = f"mp4_not_found({serial})"
            continue
        gcs_path = mp4s[fname]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as t:
            tmp_path = t.name
        try:
            if not download_video(sess, gcs_path, tmp_path):
                result[cam_key] = "download_failed"
                continue
            if extract_frames(tmp_path, n_frames, out_dir, folder):
                result[cam_key] = "ok"
            else:
                result[cam_key] = "extract_failed"
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-file",
                    default="/projects/prjs1958/robometer_frame_dataset/droid/metadata/success_pairs.jsonl")
    ap.add_argument("--droid-dir",
                    default="/projects/prjs1958/robometer_frame_dataset/droid")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--n-frames", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    base = Path(args.droid_dir)
    pairs = []
    with open(args.pairs_file) as f:
        for line in f:
            try:
                pairs.append(json.loads(line))
            except Exception:
                pass
    # Deduplicate by success_ep: multiple failures can be paired with the same
    # success (tier 4 fallbacks mostly). We only want to download+extract each
    # unique success folder once; the pairs file still maps them back by ID.
    seen, dedup = set(), []
    for p in pairs:
        if p["success_ep"] in seen:
            continue
        seen.add(p["success_ep"])
        dedup.append(p)
    log(f"Pairs loaded: {len(pairs)}  |  unique successes: {len(dedup)}")
    pairs = dedup
    if args.limit:
        pairs = pairs[:args.limit]
    start = time.time()
    counters = {"ok": 0, "skipped": 0, "failed": 0}
    n_done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_pair, p, base, args.n_frames): p for p in pairs}
        for fut in as_completed(futs):
            n_done += 1
            try:
                r = fut.result()
            except Exception as e:
                counters["failed"] += 3
                log(f"  EXC: {e}")
                continue
            for cam, _ in CAMERAS:
                st = r.get(cam, "?")
                if st == "ok":
                    counters["ok"] += 1
                elif st == "skipped":
                    counters["skipped"] += 1
                else:
                    counters["failed"] += 1
                    if st != "skipped":
                        log(f"  [{cam}] {r['success_ep']}: {st}")
            if n_done % 100 == 0:
                el = time.time() - start
                rate = n_done / el
                eta = (len(pairs) - n_done) / rate / 60 if rate else 0
                log(f"Progress: {n_done}/{len(pairs)} | ok={counters['ok']} "
                    f"skipped={counters['skipped']} failed={counters['failed']} | ETA {eta:.0f}m")

    log("=" * 60)
    log(f"DONE | ok={counters['ok']} skipped={counters['skipped']} "
        f"failed={counters['failed']} | wall {(time.time()-start)/60:.1f}m")


if __name__ == "__main__":
    main()
