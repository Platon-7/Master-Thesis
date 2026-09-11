#!/usr/bin/env python3
"""
Frame Drop Experiment for RoboReward
=====================================

Tests whether RoboReward overfits on video length (number of frames).

For each video, we:
  1. Extract all frames from the original MP4
  2. Run inference on the full frame list (baseline)
  3. Uniformly drop 20%, 40%, 60% of frames and run inference on each subset
  4. Compare scores to see if fewer frames change the model's judgement

Frames are fed as a list of PIL images (not as temporary MP4 files).
From our investigation (investigate_video_vs_frames_91697.txt), feeding a list
of frames vs an MP4 path produces equivalent results — the only difference is a
minor double-resize artifact (max pixel diff 27, mean 0.34) and synthetic
timestamps, neither of which meaningfully affect predictions.

FPS: We use 60.0 as the sampling FPS for Qwen, matching the native video FPS
of the DROID pillow case videos (best setting for this dataset).

Usage:
    python scripts/frame_drop_experiment.py [--drop-rates 0.2 0.4 0.6]
"""

import os
import sys
import glob
import json
import re
import argparse
import cv2
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRATCH_DIR = os.environ.get("SCRATCH_DIR", f"/var/scratch/{os.environ['USER']}")

# Paths
SUCCESS_PATH = os.path.join(SCRATCH_DIR, "my_success_case")
FAILURE_PATH = os.path.join(SCRATCH_DIR, "my_failure_case")
MODEL_CACHE = os.path.join(SCRATCH_DIR, "hf_cache/models--teetone--RoboReward-8B/snapshots")
OUTPUT_DIR = os.path.join(SCRATCH_DIR, "roboreward_results")

TASK_INSTRUCTION = "rearrange pillows on sofa"
VIDEO_PATTERN = "22246076.mp4"
QWEN_FPS = 60.0       # Native FPS of DROID pillow videos (best for this dataset)
SCORE_THRESHOLD = 3.5  # >= 3.5 = success, < 3.5 = failure

REWARD_PROMPT = """You are a reward model. Score the robot's task completion in the video.
Task: {task_instruction}

Scoring criteria:
1 = Complete failure / No progress
2 = Minimal progress, mostly unsuccessful
3 = Partial completion / Some success
4 = Mostly successful / Minor issues
5 = Perfect task completion

Respond with ONLY a single number (1-5)."""


# ============================================================================
# MODEL
# ============================================================================

class RoboRewardModel:
    """Wrapper for RoboReward model inference."""

    def __init__(self, cache_path: str):
        print("Loading RoboReward Model...")
        snapshot_subfolders = sorted(glob.glob(os.path.join(cache_path, "*")))
        model_path = snapshot_subfolders[0]

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            device_map="auto",
            trust_remote_code=False
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        print(f"Model loaded from: {model_path}")

    def predict_video(self, video_path: str, task_instruction: str, fps: float = QWEN_FPS) -> Tuple[float, int]:
        """Run inference on an MP4 file. Returns (score, num_frames_used)."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": fps},
                {"type": "text", "text": REWARD_PROMPT.format(task_instruction=task_instruction)}
            ]
        }]
        return self._run_inference(messages)

    def predict_frames(self, frames: List[Image.Image], task_instruction: str, fps: float = QWEN_FPS) -> Tuple[float, int]:
        """Run inference on a list of PIL frames. Returns (score, num_frames_used)."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": frames, "fps": fps},
                {"type": "text", "text": REWARD_PROMPT.format(task_instruction=task_instruction)}
            ]
        }]
        return self._run_inference(messages)

    def _run_inference(self, messages: list) -> Tuple[float, int]:
        """Shared inference logic for both video and frame-list inputs."""
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        num_frames = video_inputs[0].shape[0] if video_inputs else 0

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=16)

        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        match = re.search(r'([1-5](?:\.\d+)?)', response.strip())
        if match:
            score = float(match.group(1))
        else:
            numbers = re.findall(r'\d+\.?\d*', response)
            score = min(max(float(numbers[0]), 1.0), 5.0) if numbers else 3.0

        return score, num_frames


# ============================================================================
# VIDEO UTILITIES
# ============================================================================

def get_video_info(path: str) -> Tuple[float, int, float]:
    """Returns (fps, total_frames, duration_seconds)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps > 0 else 0
    cap.release()
    return fps, total, duration


def extract_all_frames(video_path: str) -> List[Image.Image]:
    """Extract all frames from a video as PIL Images."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV returns BGR, convert to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


def drop_frames_uniformly(frames: List[Image.Image], drop_rate: float) -> List[Image.Image]:
    """
    Uniformly drop a fraction of frames.

    Keeps frames at evenly spaced indices so the remaining frames
    still cover the full temporal extent of the video.
    """
    n_total = len(frames)
    n_keep = max(1, int(n_total * (1.0 - drop_rate)))
    keep_indices = np.linspace(0, n_total - 1, n_keep, dtype=int)
    return [frames[i] for i in keep_indices]


def find_videos(base_path: str, pattern: str = "*.mp4") -> List[str]:
    """Recursively find videos matching pattern."""
    return glob.glob(os.path.join(base_path, "**", pattern), recursive=True)


# ============================================================================
# EXPERIMENT
# ============================================================================

def run_experiment(drop_rates: List[float]):
    """Run the full frame-drop experiment."""
    print("=" * 80)
    print("  FRAME DROP EXPERIMENT — Does RoboReward overfit on video length?")
    print("=" * 80)
    print(f"  Task: {TASK_INSTRUCTION}")
    print(f"  Qwen sampling FPS: {QWEN_FPS}")
    print(f"  Score threshold: {SCORE_THRESHOLD}")
    print(f"  Drop rates: {drop_rates}")
    print(f"  Method: PIL frame lists (no temp MP4 re-encoding)")

    # Find videos
    success_videos = sorted(find_videos(SUCCESS_PATH, VIDEO_PATTERN))
    failure_videos = sorted(find_videos(FAILURE_PATH, VIDEO_PATTERN))
    print(f"\n  Found {len(success_videos)} success videos, {len(failure_videos)} failure videos")

    if not success_videos and not failure_videos:
        print("ERROR: No videos found!")
        return

    # Load model
    model = RoboRewardModel(MODEL_CACHE)

    # Conditions: original (full MP4) + each drop rate (frame lists)
    drop_labels = [f"drop_{int(r*100)}pct" for r in drop_rates]
    all_condition_labels = ["original"] + drop_labels

    all_results = []
    all_videos = (
        [(v, "success") for v in success_videos] +
        [(v, "failure") for v in failure_videos]
    )

    for vid_idx, (video_path, actual_label) in enumerate(all_videos, 1):
        orig_fps, orig_frames, orig_duration = get_video_info(video_path)
        short_name = "/".join(video_path.split("/")[-3:])
        print(f"\n{'─'*70}")
        print(f"[{vid_idx}/{len(all_videos)}] {short_name}")
        print(f"  actual={actual_label}  fps={orig_fps}  frames={orig_frames}  dur={orig_duration:.2f}s")

        video_result = {
            "video_path": video_path,
            "actual_label": actual_label,
            "original_fps": orig_fps,
            "original_frames": orig_frames,
            "original_duration": orig_duration,
            "conditions": {}
        }

        # --- Baseline: original full MP4 (uses the MP4-path pipeline) ---
        score, qwen_frames = model.predict_video(video_path, TASK_INSTRUCTION, fps=QWEN_FPS)
        predicted = "success" if score >= SCORE_THRESHOLD else "failure"
        correct = predicted == actual_label

        video_result["conditions"]["original"] = {
            "drop_rate": 0.0,
            "input_frames": orig_frames,
            "qwen_frames_used": qwen_frames,
            "score": score,
            "predicted": predicted,
            "correct": correct
        }
        status = "OK" if correct else "WRONG"
        print(f"       original: input_frames={orig_frames:4d}  "
              f"qwen_frames={qwen_frames:3d}  score={score:.1f} -> {predicted:7s} [{status}]")

        # --- Extract all frames once, then drop subsets ---
        all_frames = extract_all_frames(video_path)

        for drop_rate, cond_label in zip(drop_rates, drop_labels):
            subset = drop_frames_uniformly(all_frames, drop_rate)
            score, qwen_frames = model.predict_frames(subset, TASK_INSTRUCTION, fps=QWEN_FPS)
            predicted = "success" if score >= SCORE_THRESHOLD else "failure"
            correct = predicted == actual_label

            video_result["conditions"][cond_label] = {
                "drop_rate": drop_rate,
                "input_frames": len(subset),
                "qwen_frames_used": qwen_frames,
                "score": score,
                "predicted": predicted,
                "correct": correct
            }
            status = "OK" if correct else "WRONG"
            print(f"  {cond_label:>14s}: input_frames={len(subset):4d}  "
                  f"qwen_frames={qwen_frames:3d}  score={score:.1f} -> {predicted:7s} [{status}]")

        all_results.append(video_result)

    # ── Summary ──
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    header = f"{'Condition':<18s} {'Accuracy':>8s} {'TP':>4s} {'FN':>4s} {'FP':>4s} {'TN':>4s} {'Avg Score':>10s} {'Avg QFrames':>11s}"
    print(f"\n{header}")
    print("-" * len(header))

    summary = {}
    for cond_label in all_condition_labels:
        tp = fn = fp = tn = 0
        scores = []
        qframes = []
        for r in all_results:
            c = r["conditions"][cond_label]
            scores.append(c["score"])
            qframes.append(c["qwen_frames_used"])
            actual = r["actual_label"]
            pred = c["predicted"]
            if actual == "success" and pred == "success": tp += 1
            elif actual == "success" and pred == "failure": fn += 1
            elif actual == "failure" and pred == "success": fp += 1
            else: tn += 1

        total = tp + fn + fp + tn
        acc = (tp + tn) / total if total > 0 else 0
        avg_score = np.mean(scores)
        avg_qf = np.mean(qframes)

        print(f"{cond_label:<18s} {acc:>7.1%} {tp:>4d} {fn:>4d} {fp:>4d} {tn:>4d} {avg_score:>10.2f} {avg_qf:>11.1f}")

        summary[cond_label] = {
            "accuracy": round(acc, 4),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "avg_score": round(float(avg_score), 2),
            "avg_qwen_frames": round(float(avg_qf), 1)
        }

    # ── Score change analysis ──
    print(f"\n{'─'*70}")
    print("  SCORE STABILITY (per-video score changes vs original)")
    print(f"{'─'*70}")

    for cond_label in drop_labels:
        deltas = []
        flips = 0
        for r in all_results:
            orig_score = r["conditions"]["original"]["score"]
            cond_score = r["conditions"][cond_label]["score"]
            deltas.append(cond_score - orig_score)
            if r["conditions"]["original"]["predicted"] != r["conditions"][cond_label]["predicted"]:
                flips += 1

        deltas = np.array(deltas)
        print(f"\n  {cond_label}:")
        print(f"    Mean score change:   {deltas.mean():+.3f}")
        print(f"    Std of change:       {deltas.std():.3f}")
        print(f"    Max increase:        {deltas.max():+.1f}")
        print(f"    Max decrease:        {deltas.min():+.1f}")
        print(f"    Label flips:         {flips}/{len(all_results)} ({flips/len(all_results):.1%})")

    # ── Save results ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"frame_drop_experiment_{timestamp}.json")

    output = {
        "experiment_info": {
            "description": "Test whether RoboReward overfits on video length/number of frames",
            "method": "Baseline uses MP4 path; drop conditions use PIL frame lists",
            "task_instruction": TASK_INSTRUCTION,
            "qwen_sampling_fps": QWEN_FPS,
            "score_threshold": SCORE_THRESHOLD,
            "drop_rates": drop_rates,
            "n_success_videos": len(success_videos),
            "n_failure_videos": len(failure_videos),
            "timestamp": datetime.now().isoformat()
        },
        "summary": summary,
        "per_video_results": all_results
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {out_path}")
    print("\nExperiment complete!")


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Frame Drop Experiment for RoboReward")
    parser.add_argument(
        '--drop-rates', nargs='+', type=float, default=[0.2, 0.4, 0.6],
        help='Fraction of frames to drop (default: 0.2 0.4 0.6)'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.drop_rates)
