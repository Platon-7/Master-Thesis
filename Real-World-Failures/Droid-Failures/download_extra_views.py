#!/usr/bin/env python3
"""
download_extra_views.py — Download ext2 + wrist keyframes for DROID failure episodes.

For each episode in the scored JSONLs, downloads the ext2 and wrist camera MP4s
from GCS (anonymous, public bucket) and extracts 16 frames at the same timestamps
as the existing ext1 keyframes. Output mirrors the keyframes/ structure under
keyframes_ext2/ and keyframes_wrist/.

The frame pipeline matches the original ext1 processing:
  raw 60fps 1280x720  →  fps=10, scale=320:180, pad=320:192:0:6:black  →  16 JPEGs

Usage:
    python download_extra_views.py \
        --droid-dir "/projects/prjs1958/robometer_frame_dataset/droid" \
        --workers 24

Resume-safe: episodes with all 16 frames already present are skipped.
"""

import json
import os
import re
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

# ── GCS constants ─────────────────────────────────────────────────────────────
GCS_BASE   = "https://storage.googleapis.com"
BUCKET     = "gresearch"
DROID_PFX  = "robotics/droid_raw/1.0.1"
CAMERAS    = ["ext2", "wrist"]

# ── Shared state ──────────────────────────────────────────────────────────────
print_lock = Lock()


def log(msg: str):
    with print_lock:
        print(msg, flush=True)


# ── HTTP session with retry ───────────────────────────────────────────────────

def make_session() -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    return sess


# ── GCS helpers ───────────────────────────────────────────────────────────────

def episode_id_to_gcs_prefix(episode_id: str) -> str:
    """
    'AUTOLab_2023-07-07_Fri_Jul__7_10-02-43_2023'
    → 'robotics/droid_raw/1.0.1/AUTOLab/failure/2023-07-07/Fri_Jul__7_10:02:43_2023'
    """
    lab, rest = episode_id.split("_", 1)
    date      = rest[:10]                    # YYYY-MM-DD
    ep_disk   = rest[11:]                    # Fri_Jul__7_10-02-43_2023
    # Restore HH:MM:SS colons (the only digit-digit-digit-digit-digit-digit pattern)
    ep_gcs    = re.sub(r"(\d{2})-(\d{2})-(\d{2})", r"\1:\2:\3", ep_disk, count=1)
    return f"{DROID_PFX}/{lab}/failure/{date}/{ep_gcs}"


def list_episode_objects(sess: requests.Session, gcs_prefix: str) -> list[dict]:
    """Return all GCS objects directly under gcs_prefix (non-recursive)."""
    url    = f"{GCS_BASE}/storage/v1/b/{BUCKET}/o"
    params = {"prefix": gcs_prefix + "/", "maxResults": 50}
    resp   = sess.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("items", [])


def get_camera_serials(sess: requests.Session, objects: list[dict]) -> dict[str, str]:
    """
    Download the episode metadata JSON (small, ~2 KB) and return
    {'ext1': serial, 'ext2': serial, 'wrist': serial}.
    """
    meta_obj = next(
        (o for o in objects if re.search(r"metadata_.*\.json$", o["name"])),
        None,
    )
    if meta_obj is None:
        return {}

    url  = f"{GCS_BASE}/{BUCKET}/{meta_obj['name']}"
    resp = sess.get(url, timeout=30)
    resp.raise_for_status()
    md   = resp.json()

    return {
        "ext1":  md.get("ext1_cam_serial",  ""),
        "ext2":  md.get("ext2_cam_serial",  ""),
        "wrist": md.get("wrist_cam_serial", ""),
    }


def download_video(sess: requests.Session, gcs_path: str, local_path: str) -> bool:
    """Stream-download a single MP4 to disk. Returns True on success."""
    url = f"{GCS_BASE}/{BUCKET}/{gcs_path}"
    try:
        with sess.get(url, stream=True, timeout=180) as resp:
            resp.raise_for_status()
            with open(local_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):   # 1 MB
                    fh.write(chunk)
        return True
    except Exception as exc:
        log(f"  [download error] {gcs_path}: {exc}")
        return False


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(video_path: str, timestamps: list[float],
                   out_dir: Path, folder_name: str) -> bool:
    """
    Extract 16 frames from video at the given timestamps.

    Pipeline:  fps=10  →  scale=320:180  →  pad=320:192:0:6:black
    The select filter picks frames at 10fps index n = round(ts * 10).

    Output files: frame_0_0.00s.jpg, frame_1_0.20s.jpg, ...
    """
    ep_dir = out_dir / folder_name
    ep_dir.mkdir(parents=True, exist_ok=True)

    # Already done?
    if sum(1 for _ in ep_dir.glob("frame_*.jpg")) >= len(timestamps):
        return True

    # 10fps frame indices for each timestamp
    n_indices = [round(ts * 10) for ts in timestamps]
    select_expr = "+".join(f"eq(n,{n})" for n in n_indices)

    # One ffmpeg call: downsample → resize → select → 16 JPEGs
    tmp_pattern = str(ep_dir / "tmp_%02d.jpg")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps=10,scale=320:180,pad=320:192:0:6:black,select='{select_expr}'",
        "-vsync", "0",
        "-q:v", "2",
        "-loglevel", "error",
        tmp_pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=180)

    if result.returncode != 0:
        log(f"  [ffmpeg error] {folder_name}: {result.stderr.decode()[:200]}")
        for f in ep_dir.glob("tmp_*.jpg"):
            f.unlink(missing_ok=True)
        return False

    tmp_files = sorted(ep_dir.glob("tmp_*.jpg"))
    n_got = len(tmp_files)
    n_want = len(timestamps)

    if n_got == 0:
        log(f"  [no frames extracted] {folder_name}")
        return False

    if n_got < n_want - 1:
        log(f"  [frame count mismatch] {folder_name}: got {n_got}, want {n_want}")
        for f in tmp_files:
            f.unlink(missing_ok=True)
        return False

    # If one frame short (video slightly too short at 10fps), duplicate the last frame.
    if n_got == n_want - 1:
        import shutil
        shutil.copy(str(tmp_files[-1]), str(ep_dir / f"tmp_{n_got + 1:02d}.jpg"))
        tmp_files = sorted(ep_dir.glob("tmp_*.jpg"))

    # Rename to final names matching ext1 convention
    for i, (tmp_f, ts) in enumerate(zip(tmp_files, timestamps)):
        final = ep_dir / f"frame_{i}_{ts:.2f}s.jpg"
        tmp_f.rename(final)

    return True


# ── Per-episode worker ────────────────────────────────────────────────────────

def process_episode(episode_id: str, folder_name: str, timestamps: list[float],
                    base_dir: Path) -> dict:
    """
    Download ext2 and wrist for one episode and extract frames.
    Returns a status dict.
    """
    sess   = make_session()
    prefix = episode_id_to_gcs_prefix(episode_id)
    result = {"episode_id": episode_id, "ext2": "pending", "wrist": "pending"}

    # Check how much work is already done
    cameras_needed = []
    for cam in CAMERAS:
        out_dir   = base_dir / f"keyframes_{cam}"
        ep_dir    = out_dir / folder_name
        n_done    = sum(1 for _ in ep_dir.glob("frame_*.jpg")) if ep_dir.exists() else 0
        if n_done >= len(timestamps):
            result[cam] = "skipped"
        else:
            cameras_needed.append(cam)

    if not cameras_needed:
        return result

    # Fetch episode object listing (one API call, shared across cameras)
    try:
        objects = list_episode_objects(sess, prefix)
    except Exception as exc:
        for cam in cameras_needed:
            result[cam] = f"list_failed: {exc}"
        return result

    if not objects:
        for cam in cameras_needed:
            result[cam] = "no_objects_found"
        return result

    # Get camera serial mapping
    serials = get_camera_serials(sess, objects)
    if not serials:
        for cam in cameras_needed:
            result[cam] = "no_metadata"
        return result

    # Build set of available mono MP4 serials from listing
    mono_mp4s = {
        obj["name"].rsplit("/", 1)[-1].replace(".mp4", ""): obj["name"]
        for obj in objects
        if obj["name"].endswith(".mp4") and "stereo" not in obj["name"]
    }

    # Download and extract each needed camera
    for cam in cameras_needed:
        serial = serials.get(cam, "")
        if not serial:
            result[cam] = "no_serial"
            continue

        if serial not in mono_mp4s:
            result[cam] = f"mp4_not_found ({serial})"
            continue

        gcs_video_path = mono_mp4s[serial]
        out_dir        = base_dir / f"keyframes_{cam}"

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            if not download_video(sess, gcs_video_path, tmp_path):
                result[cam] = "download_failed"
                continue

            if extract_frames(tmp_path, timestamps, out_dir, folder_name):
                result[cam] = "ok"
            else:
                result[cam] = "extract_failed"

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def load_episodes(metadata_dir: Path) -> list[dict]:
    """
    Load all episodes from scored JSONL shards.
    Returns list of {episode_id, folder_name, timestamps}.
    """
    episodes = {}

    for jsonl_path in sorted(metadata_dir.glob("scored_full_droid_shard*.jsonl")):
        with open(jsonl_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                ep_id = d["episode_id"]
                if ep_id in episodes:
                    continue

                frames      = d.get("frames", [])
                timestamps  = [fr["timestamp"] for fr in frames]
                folder_name = frames[0]["image_path"].split("/")[0] if frames else ep_id

                episodes[ep_id] = {
                    "episode_id":  ep_id,
                    "folder_name": folder_name,
                    "timestamps":  timestamps,
                }

    return list(episodes.values())


def main():
    parser = argparse.ArgumentParser(description="Download ext2+wrist keyframes for DROID failures")
    parser.add_argument(
        "--droid-dir",
        default="/projects/prjs1958/robometer_frame_dataset/droid",
        help="Base directory containing metadata/ and keyframes/ (default: %(default)s)",
    )
    parser.add_argument(
        "--workers", type=int, default=24,
        help="Concurrent download workers (default: %(default)s)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N episodes (for testing)",
    )
    args = parser.parse_args()

    base_dir     = Path(args.droid_dir)
    metadata_dir = base_dir / "metadata"

    print("=" * 60)
    print("DROID Extra Views Downloader")
    print(f"  Base dir:  {base_dir}")
    print(f"  Cameras:   {CAMERAS}")
    print(f"  Workers:   {args.workers}")
    print("=" * 60)

    # Load episodes
    episodes = load_episodes(metadata_dir)
    print(f"Episodes loaded: {len(episodes)}")

    if args.limit:
        episodes = episodes[:args.limit]
        print(f"Limited to {args.limit} episodes (--limit)")

    # Quick resume check
    n_already_done = 0
    for ep in episodes:
        done = all(
            (base_dir / f"keyframes_{cam}" / ep["folder_name"]).exists()
            and sum(1 for _ in (base_dir / f"keyframes_{cam}" / ep["folder_name"]).glob("frame_*.jpg")) >= len(ep["timestamps"])
            for cam in CAMERAS
        )
        if done:
            n_already_done += 1
    print(f"Already complete: {n_already_done} / {len(episodes)}")
    print()

    # Process
    counters = {"ok": 0, "skipped": 0, "failed": 0}
    start = time.time()
    n_done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_episode,
                ep["episode_id"],
                ep["folder_name"],
                ep["timestamps"],
                base_dir,
            ): ep["episode_id"]
            for ep in episodes
        }

        for future in as_completed(futures):
            ep_id = futures[future]
            n_done += 1

            try:
                res = future.result()
            except Exception as exc:
                log(f"[EXCEPTION] {ep_id}: {exc}")
                counters["failed"] += 1
                continue

            for cam in CAMERAS:
                status = res.get(cam, "unknown")
                if status == "ok":
                    counters["ok"] += 1
                elif status == "skipped":
                    counters["skipped"] += 1
                else:
                    counters["failed"] += 1
                    if status not in ("skipped",):
                        log(f"  [{cam}] {ep_id}: {status}")

            if n_done % 100 == 0:
                elapsed   = time.time() - start
                rate      = n_done / elapsed
                remaining = (len(episodes) - n_done) / rate if rate > 0 else 0
                log(
                    f"Progress: {n_done}/{len(episodes)} episodes | "
                    f"ok={counters['ok']} skipped={counters['skipped']} failed={counters['failed']} | "
                    f"ETA {remaining/60:.0f}m"
                )

    elapsed = time.time() - start
    print()
    print("=" * 60)
    print("DONE")
    print(f"  Episodes processed: {n_done}")
    print(f"  Camera views ok:    {counters['ok']}")
    print(f"  Skipped (existing): {counters['skipped']}")
    print(f"  Failed:             {counters['failed']}")
    print(f"  Wall time:          {elapsed/60:.1f} min")
    print(f"  Output dirs:")
    for cam in CAMERAS:
        d = base_dir / f"keyframes_{cam}"
        n = sum(1 for _ in d.iterdir()) if d.exists() else 0
        print(f"    {d}  ({n} episode folders)")
    print("=" * 60)


if __name__ == "__main__":
    main()
