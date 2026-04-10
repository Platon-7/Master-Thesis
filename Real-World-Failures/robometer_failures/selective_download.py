#!/usr/bin/env python3
"""
selective_download.py — Stream RoboMeter archives and extract only failures.

Strategy:
  - Archives have 80-85% failure rate → range-based selective download saves <20% bandwidth
    and the TOC scan overhead (one HTTP request per file) would cost more time than it saves.
  - Instead: stream the full archive over HTTP, extract failure NPZs on the fly,
    never write the raw tar to disk. Saves disk space while keeping simplicity.

  Special case — LIBERO failure archives (*_failure.tar):
    ALL episodes are failures, so they stream with zero filtering overhead.

Usage:
    # One archive
    python selective_download.py --archive jesbu1_roboreward_rfm_roboreward_test

    # All single-file archives (skip MetaWorld + human egocentric)
    python selective_download.py --archive all --skip-humans

    # From a local cached tar (no download)
    python selective_download.py --archive jesbu1_roboreward_rfm_roboreward_test \\
        --local-tar /scratch-shared/pkarageorgis/hf_cache/.../jesbu1_roboreward_rfm_roboreward_test.tar

Environment:
    HF_TOKEN   HuggingFace token
    DATA_DIR   root output directory (default: /data/robometer_failures)
"""

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, ReadTimeout
from urllib3.exceptions import IncompleteRead, ProtocolError
from PIL import Image

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATA_DIR  = Path(os.environ.get("DATA_DIR", "/data/robometer_failures"))
BASE_URL  = "https://huggingface.co/datasets/robometer/processed_datasets/resolve/main"

# These archives contain ONLY failure episodes — no filtering needed
ALL_FAILURES_ARCHIVES = {
    "ykorkmaz_libero_failure_rfm_libero_10_failure",
    "ykorkmaz_libero_failure_rfm_libero_90_failure",
    "ykorkmaz_libero_failure_rfm_libero_goal_failure",
    "ykorkmaz_libero_failure_rfm_libero_object_failure",
    "ykorkmaz_libero_failure_rfm_libero_spatial_failure",
}

# Human/egocentric — robot-centric VLM prompts don't apply
HUMAN_ARCHIVES = {
    "jesbu1_egodex_rfm_egodex_test",
    "jesbu1_epic_rfm_epic",
    "jesbu1_hand_paired_rfm_hand_paired_human",
    "jesbu1_humanoid_everyday_rfm_humanoid_everyday_rfm",
    "jesbu1_usc_koch_human_robot_paired_usc_koch_human_robot_paired_human",
}

# User built own version / different failure type handled separately
SKIP_ARCHIVES = {
    "aliangdw_metaworld_metaworld_train",
    "aliangdw_metaworld_metaworld_eval",
    "jesbu1_roboreward_rfm_roboreward_train",
    "jesbu1_roboreward_rfm_roboreward_val",
    "jesbu1_roboreward_rfm_roboreward_test",
    "jesbu1_roboreward_rfm_high_res_roboreward_train",
    "jesbu1_roboreward_rfm_high_res_roboreward_val",
    "jesbu1_roboreward_rfm_high_res_roboreward_test",
}


# ────────────────────────────────────────────────────────────────────────────
# Keyframe helpers
# ────────────────────────────────────────────────────────────────────────────

def extract_keyframes(frames_array: np.ndarray, n: int = 8):
    T = frames_array.shape[0]
    if T <= n:
        indices = list(range(T)) + [T - 1] * (n - T)
    else:
        indices = [int(round(i * (T - 1) / (n - 1))) for i in range(n)]
    return [frames_array[i] for i in indices], indices


def save_keyframes(frames, src_indices, fps, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for kf_i, (frame, src_i) in enumerate(zip(frames, src_indices)):
        ts = src_i / fps
        Image.fromarray(frame).save(out_dir / f"frame_{kf_i}_{ts:.2f}s.jpg", quality=90)


# ────────────────────────────────────────────────────────────────────────────
# Two-pass streaming extractor
# ────────────────────────────────────────────────────────────────────────────

class _MultiPartStream:
    """Reads split archive parts (part-aa, part-ab, …) as one sequential stream."""

    def __init__(self, archive: str, parts: list, token: str,
                 max_retries: int = 8, retry_backoff_s: float = 2.0):
        self._archive = archive
        self._parts   = parts
        self._token   = token
        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_s
        self._idx     = 0
        self._part_pos = 0
        self._raw     = None
        self._resp    = None
        self._sess    = requests.Session()
        self._sess.headers["Authorization"] = f"Bearer {self._token}"
        self._open_next()

    def _open_next(self):
        if self._resp is not None:
            self._resp.close()
            self._resp = None
        if self._idx >= len(self._parts):
            self._raw = None
            return
        suffix = self._parts[self._idx]
        url = f"{BASE_URL}/{self._archive}.tar.{suffix}"
        print(f"    → streaming part {self._idx + 1}/{len(self._parts)}: {suffix}")
        self._part_pos = 0
        self._resp = self._sess.get(url, stream=True, timeout=60)
        self._resp.raise_for_status()
        self._raw = self._resp.raw
        self._idx += 1

    def _reopen_current_part(self):
        if self._idx == 0:
            raise RuntimeError("Cannot reopen current part before first open")
        suffix = self._parts[self._idx - 1]
        url = f"{BASE_URL}/{self._archive}.tar.{suffix}"
        if self._resp is not None:
            self._resp.close()
            self._resp = None
        headers = {"Range": f"bytes={self._part_pos}-"}
        self._resp = self._sess.get(url, stream=True, timeout=60, headers=headers)
        self._resp.raise_for_status()
        self._raw = self._resp.raw

    def read(self, n=-1):
        if self._raw is None:
            return b""

        retries = 0
        while True:
            try:
                data = self._raw.read(n)
                if data:
                    self._part_pos += len(data)
                    return data
                if self._idx < len(self._parts):
                    self._open_next()
                    continue
                return b""
            except (IncompleteRead, ProtocolError, ChunkedEncodingError, ConnectionError, ReadTimeout) as e:
                retries += 1
                if retries > self._max_retries:
                    raise RuntimeError(
                        f"Failed to stream part {self._parts[self._idx - 1]} after "
                        f"{self._max_retries} retries"
                    ) from e
                wait_s = min(30.0, self._retry_backoff_s * retries)
                print(
                    f"    ! stream interrupted on part {self._parts[self._idx - 1]} "
                    f"at byte {self._part_pos:,}; retry {retries}/{self._max_retries} in {wait_s:.1f}s"
                )
                time.sleep(wait_s)
                self._reopen_current_part()

    def readable(self):
        return True


class _ResumableHTTPStream:
    """HTTP stream with automatic reconnect using Range requests."""

    def __init__(self, url: str, token: str,
                 max_retries: int = 8, retry_backoff_s: float = 2.0):
        self._url = url
        self._token = token
        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_s
        self._pos = 0
        self._resp = None
        self._raw = None
        self._sess = requests.Session()
        self._sess.headers["Authorization"] = f"Bearer {self._token}"
        self._open(initial=True)

    def _open(self, initial: bool = False):
        if self._resp is not None:
            self._resp.close()
            self._resp = None
        headers = None if initial else {"Range": f"bytes={self._pos}-"}
        self._resp = self._sess.get(self._url, stream=True, timeout=60, headers=headers)
        self._resp.raise_for_status()
        self._raw = self._resp.raw

    def read(self, n=-1):
        if self._raw is None:
            return b""

        retries = 0
        while True:
            try:
                data = self._raw.read(n)
                if data:
                    self._pos += len(data)
                return data
            except (IncompleteRead, ProtocolError, ChunkedEncodingError, ConnectionError, ReadTimeout) as e:
                retries += 1
                if retries > self._max_retries:
                    raise RuntimeError(
                        f"Failed to stream {self._url} after {self._max_retries} retries"
                    ) from e
                wait_s = min(30.0, self._retry_backoff_s * retries)
                print(
                    f"    ! stream interrupted at byte {self._pos:,}; "
                    f"retry {retries}/{self._max_retries} in {wait_s:.1f}s"
                )
                time.sleep(wait_s)
                self._open(initial=False)

    def readable(self):
        return True


def _discover_parts(archive: str, token: str) -> list:
    """Probe HuggingFace for split parts (part-aa … part-az). Returns sorted list."""
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    parts = []
    for c in "abcdefghijklmnopqrstuvwxyz":
        suffix = f"part-a{c}"
        url = f"{BASE_URL}/{archive}.tar.{suffix}"
        r = session.head(url, timeout=30, allow_redirects=True)
        if r.status_code == 404:
            break
        parts.append(suffix)
    return parts


def open_tar_stream(archive: str, token: str, local_tar: str = None):
    """Return a tarfile opened in streaming mode.

    Uses local file if provided, otherwise streams from HuggingFace.
    Handles both single-file (.tar) and split (.tar.part-aa …) archives.
    """
    import tarfile

    if local_tar:
        print(f"  Using local tar: {local_tar}")
        return tarfile.open(local_tar, mode="r:")  # seekable local file → faster

    # Try single-file archive first
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    url = f"{BASE_URL}/{archive}.tar"
    rh = session.head(url, timeout=30, allow_redirects=True)

    if rh.status_code == 200:
        print(f"  Streaming from HuggingFace: {url}")
        stream = _ResumableHTTPStream(url, token)
        return tarfile.open(fileobj=stream, mode="r|")

    # Fall back to split archive
    parts = _discover_parts(archive, token)
    if not parts:
        raise FileNotFoundError(
            f"Archive not found as single file or split parts: {archive}"
        )
    print(f"  Split archive detected ({len(parts)} parts): {archive}")
    stream = _MultiPartStream(archive, parts, token)
    return tarfile.open(fileobj=stream, mode="r|")


def process_archive(archive: str, token: str, output_dir: Path,
                    fps: float = 8.0, max_episodes: int = None,
                    include_partial: bool = False, resume: bool = True,
                    local_tar: str = None) -> int:
    """
    Stream a tar archive, extract failures, save keyframes + manifest.

    Two-pass for mixed archives (needs index_mappings.json to filter):
      Pass 1: collect index_mappings.json + embedding filenames (to build index→uuid)
      Pass 2: re-open and extract only failure NPZs

    Single-pass for all-failure archives (no filtering needed).
    """
    manifest_path = output_dir / "manifests" / f"{archive}.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Look up category from catalog (fall back to "unknown")
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from datasets_catalog import DATASETS
        _archive_category = next(
            (d["category"] for d in DATASETS if d["archive"] == archive), "unknown"
        )
    except Exception:
        _archive_category = "unknown"

    # Resume: load already-done episode IDs
    done = set()
    if resume and manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                done.add(json.loads(line)["episode_id"])
        if done:
            print(f"  Resume: {len(done)} episodes already done")

    is_all_failures = archive in ALL_FAILURES_ARCHIVES
    processed = 0

    if is_all_failures:
        # ── Two passes: all episodes are failures but we need task labels ───
        # Pass 1 (fast): stream to find index_mappings.json at end of archive
        print(f"  All-failures archive — pass 1: reading metadata …")
        idx_to_task = {}
        with open_tar_stream(archive, token, local_tar) as tf:
            for member in tf:
                if member.name.endswith("index_mappings.json") and member.size > 0:
                    raw = tf.extractfile(member).read()
                    idx_map = json.loads(raw)
                    for task, indices in idx_map.get("task_indices", {}).items():
                        for i in indices:
                            idx_to_task[i] = task
                # All other members: tarfile reads and discards the data

        print(f"  Pass 2: extracting all NPZs …")
        with open_tar_stream(archive, token, local_tar) as tf:
            episode_counter = 0

            for member in tf:
                if not member.name.endswith(".npz"):
                    continue

                # UUID from filename
                m = re.search(r"trajectory_([0-9a-f\-]+)\.npz$", member.name)
                if not m:
                    continue
                uuid = m.group(1)
                episode_id = f"ep_{episode_counter:06d}_{uuid}"
                episode_counter += 1

                if episode_id in done:
                    continue
                if max_episodes and processed >= max_episodes:
                    break

                raw_npz = tf.extractfile(member).read()
                npz = np.load(io.BytesIO(raw_npz), allow_pickle=True)
                frames_array = npz["frames"]
                kf, src_idx = extract_keyframes(frames_array, 8)

                ep_dir = output_dir / "keyframes" / archive / episode_id
                save_keyframes(kf, src_idx, fps, ep_dir)

                task = idx_to_task.get(episode_counter - 1, "unknown task")
                entry = {
                    "episode_id":      episode_id,
                    "archive":         archive,
                    "category":        _archive_category,
                    "task":            task,
                    "robometer_label": "failure",
                    "n_source_frames": int(frames_array.shape[0]),
                    "keyframes_dir":   str(ep_dir.relative_to(output_dir)),
                }
                with open(manifest_path, "a") as mf:
                    mf.write(json.dumps(entry) + "\n")

                processed += 1
                if processed % 200 == 0:
                    print(f"  {processed} episodes extracted")

    else:
        # ── Two-pass for mixed archives ─────────────────────────────────────
        # Pass 1: get index_mappings.json + build index→uuid from embeddings
        print(f"  Pass 1: collecting metadata …")

        if local_tar:
            # Local file: seek directly
            import tarfile
            with tarfile.open(local_tar, "r:") as tf:
                idx_map_raw = None
                idx_to_uuid = {}
                emb_pat = re.compile(r"trajectory_(\d+)_([0-9a-f\-]+)_embeddings\.pt$")
                for member in tf:
                    if "index_mappings" in member.name and member.size > 0:
                        idx_map_raw = tf.extractfile(member).read()
                    m = emb_pat.search(member.name)
                    if m:
                        idx_to_uuid[int(m.group(1))] = m.group(2)
        else:
            # Stream: must read all members to reach the metadata at the end
            idx_map_raw = None
            idx_to_uuid = {}
            emb_pat = re.compile(r"trajectory_(\d+)_([0-9a-f\-]+)_embeddings\.pt$")
            with open_tar_stream(archive, token, local_tar) as tf:
                for member in tf:
                    if "index_mappings" in member.name and member.size > 0:
                        idx_map_raw = tf.extractfile(member).read()
                        continue
                    m = emb_pat.search(member.name)
                    if m:
                        idx_to_uuid[int(m.group(1))] = m.group(2)
                    # Skip NPZ data (tarfile reads and discards)

        if idx_map_raw is None:
            print("  WARNING: index_mappings.json not found — skipping")
            return 0

        idx_map = json.loads(idx_map_raw)
        qi = idx_map.get("quality_indices", {})
        failure_set = set(qi.get("failure", []))
        if include_partial:
            failure_set |= set(qi.get("partial_success", []))

        idx_to_task = {}
        for task, indices in idx_map.get("task_indices", {}).items():
            for i in indices:
                idx_to_task[i] = task

        idx_to_source = {}
        for src, indices in idx_map.get("source_indices", {}).items():
            for i in indices:
                idx_to_source[i] = src

        failure_uuids = {idx_to_uuid[i]: i for i in failure_set if i in idx_to_uuid}
        print(f"  {len(failure_uuids)} failure episodes to extract")

        # Pass 2: stream again, extract only failure NPZs
        print(f"  Pass 2: extracting failure NPZs …")
        npz_pat = re.compile(r"trajectory_([0-9a-f\-]+)\.npz$")

        with open_tar_stream(archive, token, local_tar) as tf:
            for member in tf:
                m = npz_pat.search(member.name)
                if not m:
                    continue

                uuid = m.group(1)
                if uuid not in failure_uuids:
                    continue  # skip — tarfile reads and discards the data

                traj_idx = failure_uuids[uuid]
                episode_id = f"ep_{traj_idx:06d}_{uuid}"

                if episode_id in done:
                    continue
                if max_episodes and processed >= max_episodes:
                    break

                raw_npz = tf.extractfile(member).read()
                npz = np.load(io.BytesIO(raw_npz), allow_pickle=True)
                frames_array = npz["frames"]
                kf, src_idx = extract_keyframes(frames_array, 8)

                ep_dir = output_dir / "keyframes" / archive / episode_id
                save_keyframes(kf, src_idx, fps, ep_dir)

                entry = {
                    "episode_id":      episode_id,
                    "archive":         archive,
                    "category":        _archive_category,
                    "source_dataset":  idx_to_source.get(traj_idx, "unknown"),
                    "task":            idx_to_task.get(traj_idx, "unknown task"),
                    "robometer_label": "failure",
                    "n_source_frames": int(frames_array.shape[0]),
                    "keyframes_dir":   str(ep_dir.relative_to(output_dir)),
                }
                with open(manifest_path, "a") as mf:
                    mf.write(json.dumps(entry) + "\n")

                processed += 1
                if processed % 200 == 0:
                    print(f"  {processed}/{len(failure_uuids)} episodes extracted")

    print(f"  Done: {processed} new episodes")
    return processed


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="all")
    parser.add_argument("--output-dir", default=str(DATA_DIR))
    parser.add_argument("--hf-token", default=HF_TOKEN)
    parser.add_argument("--local-tar", default=None,
                        help="Path to a locally cached .tar file (skips download)")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Stop after N episodes (for testing)")
    parser.add_argument("--include-partial", action="store_true")
    parser.add_argument("--skip-humans", action="store_true",
                        help="Skip human/egocentric archives (robot prompts don't apply)")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if not args.hf_token and not args.local_tar:
        print("ERROR: set --hf-token or HF_TOKEN, or use --local-tar")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.archive == "all":
        sys.path.insert(0, str(Path(__file__).parent))
        from datasets_catalog import list_datasets
        archives = [d["archive"] for d in list_datasets(skip_done=True, no_splits=True)]
    else:
        archives = [args.archive]

    total = 0
    for archive in archives:
        if archive in SKIP_ARCHIVES:
            print(f"\nSKIP {archive} (MetaWorld — own version exists)")
            continue
        if args.skip_humans and archive in HUMAN_ARCHIVES:
            print(f"\nSKIP {archive} (human egocentric)")
            continue

        print(f"\n{'='*60}")
        print(f"Archive: {archive}")
        print(f"{'='*60}")

        local = args.local_tar if (args.archive != "all") else None

        try:
            n = process_archive(
                archive=archive,
                token=args.hf_token,
                output_dir=output_dir,
                fps=args.fps,
                max_episodes=args.max_episodes,
                include_partial=args.include_partial,
                resume=not args.no_resume,
                local_tar=local,
            )
            total += n
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print(f"\nTotal new episodes: {total}")
    print(f"Manifests: {output_dir}/manifests/")
    print(f"\nNext: python score_robometer_failures.py --archive all --data-dir {output_dir}")


if __name__ == "__main__":
    main()
