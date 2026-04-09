#!/usr/bin/env python3
"""
score_robometer_failures.py — Stage 2 of the RoboMeter failure scoring pipeline.

Reads the manifest JSONL produced by extract_failures.py and runs the two-stage
VLM+LLM scoring pipeline (same as DROID failures: Qwen3.5-35B → Qwen3-32B).

Produces one output JSONL per archive with the 1-4 granular progress score,
plus the intermediate VLM frame descriptions.

Usage:
    # Score one archive
    python score_robometer_failures.py \\
        --archive jesbu1_oxe_rfm_oxe_bridge_v2 \\
        --data-dir /data/robometer_failures

    # Score all extracted archives
    python score_robometer_failures.py \\
        --archive all \\
        --data-dir /data/robometer_failures

    # Two-stage split (useful for large archives or limited GPU time)
    python score_robometer_failures.py --archive X --data-dir /data --stage 1  # VLM only
    python score_robometer_failures.py --archive X --data-dir /data --stage 2  # LLM only

Environment variables:
    VLM_MODEL_ID   default: Qwen/Qwen3.5-35B-A3B
    LLM_MODEL_ID   default: Qwen/Qwen3-32B
    HF_TOKEN       HuggingFace token for model downloads
"""

import argparse
import json
import os
import re
import sys
import time
import gc
import base64
from pathlib import Path

import torch


# ────────────────────────────────────────────────────────────────────────────
# Config (override via env vars)
# ────────────────────────────────────────────────────────────────────────────

VLM_MODEL_ID = os.environ.get("VLM_MODEL_ID", "Qwen/Qwen3.5-35B-A3B")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "Qwen/Qwen3-32B")
VLM_MAX_MODEL_LEN = int(os.environ.get("VLM_MAX_MODEL_LEN", "16384"))

# Batch size for VLM inference (lower if OOM)
VLM_BATCH_SIZE = int(os.environ.get("VLM_BATCH_SIZE", "32"))


# ────────────────────────────────────────────────────────────────────────────
# Import prompts and helpers from score_keyframes.py
# ────────────────────────────────────────────────────────────────────────────
# score_keyframes.py lives one directory up in droid_failures/
_DROID_DIR = Path(__file__).parent.parent / "droid_failures"
sys.path.insert(0, str(_DROID_DIR))

try:
    from score_keyframes import (
        VLM_FRAME_PROMPT,
        VLM_FRAME_PROMPT_MULTI,
        LLM_GRADE_PROMPT_SIMPLE,
        LLM_GRADE_PROMPT_MULTI,
        TASK_DECOMPOSE_PROMPT,
        MULTI_OBJECT_PHRASES,
        is_multi_step_task,
        describe_frames_batch,
        grade_frame,
        decompose_task,
        load_vlm,
        load_llm,
    )
    print("Imported scoring functions from score_keyframes.py")
except ImportError as e:
    print(f"WARNING: Could not import from score_keyframes.py: {e}")
    print("Falling back to bundled implementations")
    # If running standalone (e.g. in a different environment), the prompts
    # and logic are duplicated below. See score_keyframes.py for the canonical version.
    raise


# ────────────────────────────────────────────────────────────────────────────
# Episode loading from manifest
# ────────────────────────────────────────────────────────────────────────────

def load_manifest(manifest_path: Path) -> list[dict]:
    episodes = []
    with open(manifest_path) as f:
        for line in f:
            episodes.append(json.loads(line.strip()))
    return episodes


def get_frame_paths(episode: dict, data_dir: Path) -> list[tuple[int, float, Path]]:
    """Return list of (frame_idx, timestamp, path) for an episode's keyframes."""
    kf_dir = data_dir / episode["keyframes_dir"]
    frames = []
    for p in sorted(kf_dir.glob("frame_*.jpg")):
        # filename: frame_N_T.XXs.jpg
        m = re.match(r"frame_(\d+)_(\d+\.\d+)s\.jpg", p.name)
        if m:
            frames.append((int(m.group(1)), float(m.group(2)), p))
    return sorted(frames, key=lambda x: x[0])


# ────────────────────────────────────────────────────────────────────────────
# Stage 1: VLM frame descriptions
# ────────────────────────────────────────────────────────────────────────────

def run_vlm_stage(episodes: list[dict], data_dir: Path,
                  descriptions_path: Path, resume: bool = True):
    """Describe all keyframes with VLM. Writes to descriptions_path (JSONL)."""
    # Load already-done
    done = {}
    if resume and descriptions_path.exists():
        with open(descriptions_path) as f:
            for line in f:
                entry = json.loads(line)
                # Support both flat {"episode_id": ...} and nested {"episode": {"episode_id": ...}}
                ep_id = entry.get("episode_id") or entry.get("episode", {}).get("episode_id")
                if ep_id:
                    done[ep_id] = entry
        print(f"  VLM stage: {len(done)} episodes already described")

    todo = [ep for ep in episodes if ep["episode_id"] not in done]
    if not todo:
        print("  VLM stage: all episodes already done")
        return

    print(f"  VLM stage: {len(todo)} episodes to describe")

    # Build flat work items: (ep_id, task, frame_path, frame_idx, timestamp)
    work_items = []
    ep_frame_counts = {}
    for ep in todo:
        frames = get_frame_paths(ep, data_dir)
        ep_frame_counts[ep["episode_id"]] = len(frames)
        for frame_idx, timestamp, frame_path in frames:
            work_items.append((ep["episode_id"], ep["task"], str(frame_path), frame_idx, timestamp))

    # Load VLM
    vlm = load_vlm()

    # Process in batches
    descriptions_by_ep = {ep["episode_id"]: {"episode": ep, "frames": []} for ep in todo}

    with open(descriptions_path, "a") as out_f:
        for batch_start in range(0, len(work_items), VLM_BATCH_SIZE):
            batch = work_items[batch_start: batch_start + VLM_BATCH_SIZE]
            results = describe_frames_batch(batch, vlm)

            for ep_id, frame_idx, timestamp, description in results:
                descriptions_by_ep[ep_id]["frames"].append({
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                    "description": description,
                })

            # Check if any episodes are complete
            completed = []
            for ep_id, ep_data in descriptions_by_ep.items():
                expected = ep_frame_counts[ep_id]
                if len(ep_data["frames"]) >= expected:
                    completed.append(ep_id)

            for ep_id in completed:
                ep_data = descriptions_by_ep.pop(ep_id)
                ep_data["frames"].sort(key=lambda x: x["frame_idx"])
                out_f.write(json.dumps(ep_data) + "\n")
                out_f.flush()

            n_done = batch_start + len(batch)
            print(f"  VLM: {n_done}/{len(work_items)} frames ({n_done*100//len(work_items)}%)")

    # Write any remaining incomplete episodes
    if descriptions_by_ep:
        with open(descriptions_path, "a") as out_f:
            for ep_id, ep_data in descriptions_by_ep.items():
                ep_data["frames"].sort(key=lambda x: x["frame_idx"])
                out_f.write(json.dumps(ep_data) + "\n")

    del vlm
    gc.collect()
    torch.cuda.empty_cache()
    print("  VLM stage complete")


# ────────────────────────────────────────────────────────────────────────────
# Stage 2: LLM grading
# ────────────────────────────────────────────────────────────────────────────

def run_llm_stage(descriptions_path: Path, scores_path: Path, resume: bool = True):
    """Grade each episode with LLM. Reads descriptions_path, writes scores_path."""
    done = set()
    if resume and scores_path.exists():
        with open(scores_path) as f:
            for line in f:
                entry = json.loads(line)
                done.add(entry["episode_id"])
        print(f"  LLM stage: {len(done)} episodes already graded")

    # Load descriptions
    episodes_with_descriptions = []
    with open(descriptions_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry["episode"]["episode_id"] not in done:
                episodes_with_descriptions.append(entry)

    if not episodes_with_descriptions:
        print("  LLM stage: all episodes already graded")
        return

    print(f"  LLM stage: {len(episodes_with_descriptions)} episodes to grade")

    llm_model, llm_tokenizer = load_llm()

    with open(scores_path, "a") as out_f:
        for i, entry in enumerate(episodes_with_descriptions):
            ep = entry["episode"]
            ep_id = ep["episode_id"]
            task = ep["task"]
            frames_list = entry["frames"]  # [{frame_idx, timestamp, description}]

            # Optionally decompose multi-step tasks
            substeps = None
            if is_multi_step_task(task):
                substeps = decompose_task(task, llm_model, llm_tokenizer)

            # Grade frame-by-frame (cumulative context)
            frames_so_far = []
            for frame_data in frames_list:
                frames_so_far.append({
                    "timestamp": frame_data["timestamp"],
                    "description": frame_data["description"],
                    "score": None,
                })
                score = grade_frame(
                    task=task,
                    substeps=substeps,
                    frames_so_far=frames_so_far,
                    total_frames=len(frames_list),
                    llm_model=llm_model,
                    llm_tokenizer=llm_tokenizer,
                )
                frames_so_far[-1]["score"] = score

            # Final episode score = score of last frame
            final_score = frames_so_far[-1]["score"]

            result = {
                "episode_id": ep_id,
                "archive": ep["archive"],
                "category": ep.get("category", "unknown"),
                "task": task,
                "robometer_label": ep.get("robometer_label", "failure"),
                "score": final_score,
                "frames": [
                    {
                        "frame_idx": frames_list[j].get("frame_idx", j),
                        "timestamp": frames_list[j]["timestamp"],
                        "description": frames_list[j]["description"],
                        "score": f["score"],
                    }
                    for j, f in enumerate(frames_so_far)
                ],
            }
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

            if (i + 1) % 50 == 0:
                print(f"  LLM: {i+1}/{len(episodes_with_descriptions)} episodes graded")

    del llm_model
    gc.collect()
    torch.cuda.empty_cache()
    print("  LLM stage complete")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Score RoboMeter failures with VLM+LLM")
    parser.add_argument("--archive", required=True,
                        help="Archive name or 'all' for everything with a manifest")
    parser.add_argument("--data-dir", default="/data/robometer_failures",
                        help="Root data directory (same as extract_failures.py output)")
    parser.add_argument("--stage", type=int, choices=[1, 2, 0], default=0,
                        help="0=both stages, 1=VLM only, 2=LLM only (default: 0)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Reprocess from scratch (ignore existing outputs)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manifests_dir = data_dir / "manifests"
    scores_dir = data_dir / "scores"
    descriptions_dir = data_dir / "vlm_descriptions"
    scores_dir.mkdir(parents=True, exist_ok=True)
    descriptions_dir.mkdir(parents=True, exist_ok=True)

    if args.archive == "all":
        manifests = sorted(manifests_dir.glob("*.jsonl"))
        archives = [m.stem for m in manifests]
    else:
        archives = [args.archive]

    for archive in archives:
        manifest_path = manifests_dir / f"{archive}.jsonl"
        if not manifest_path.exists():
            print(f"\nNo manifest for {archive} — run extract_failures.py first")
            continue

        descriptions_path = descriptions_dir / f"{archive}_descriptions.jsonl"
        scores_path = scores_dir / f"{archive}_scored.jsonl"

        episodes = load_manifest(manifest_path)
        print(f"\n{'='*60}")
        print(f"Archive: {archive}  ({len(episodes)} failure episodes)")
        print(f"{'='*60}")

        if args.stage in (0, 1):
            print("\n[Stage 1] VLM frame descriptions")
            run_vlm_stage(
                episodes=episodes,
                data_dir=data_dir,
                descriptions_path=descriptions_path,
                resume=not args.no_resume,
            )

        if args.stage in (0, 2):
            if not descriptions_path.exists():
                print(f"  ERROR: descriptions file missing — run stage 1 first")
                continue
            print("\n[Stage 2] LLM grading")
            run_llm_stage(
                descriptions_path=descriptions_path,
                scores_path=scores_path,
                resume=not args.no_resume,
            )

        if scores_path.exists():
            # Quick summary
            from collections import Counter
            scores = []
            with open(scores_path) as f:
                for line in f:
                    raw = json.loads(line).get("score")
                    scores.append(raw[0] if isinstance(raw, list) else raw)
            dist = dict(sorted(Counter(scores).items()))
            print(f"\nScore distribution for {archive}:")
            for s, c in dist.items():
                print(f"  Score {s}: {c} episodes")


if __name__ == "__main__":
    main()
