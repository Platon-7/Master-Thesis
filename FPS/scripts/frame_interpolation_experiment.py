#!/usr/bin/env python3
"""
Frame Interpolation Experiment for RoboReward
===============================================

Tests whether adding interpolated frames affects RoboReward's predictions.

For each video, we:
  1. Extract all frames from the original MP4
  2. Run inference on the full frame list (baseline)
  3. Add interpolated frames (2x, 3x, 4x total) using linear blending
     between consecutive frames
  4. Compare scores to see if more frames change the model's judgement

This is the inverse of frame_drop_experiment.py — instead of removing frames,
we add synthetic ones in between existing frames.

Interpolation method: linear pixel blending between consecutive frames.
For multiplier M, we insert M-1 evenly spaced blended frames between each
pair of consecutive original frames.

Usage:
    python scripts/frame_interpolation_experiment.py [--multipliers 2 3 4]
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
import qwen_vl_utils.vision_process as _vp

# Override Qwen's default frame cap (768) so interpolated frames actually
# reach the model. 2x TitanRTX can handle up to ~2500 frames safely.
MAX_INPUT_FRAMES = 2500
_vp.FPS_MAX_FRAMES = MAX_INPUT_FRAMES


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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


def interpolate_frames(frames: List[Image.Image], multiplier: int) -> List[Image.Image]:
    """
    Insert linearly interpolated frames between each pair of consecutive frames.

    For multiplier M, each pair of consecutive frames produces M frames total
    (the original first frame + M-1 blended intermediates). The last original
    frame is appended at the end.

    Result has approximately len(frames) * multiplier frames.
    """
    if multiplier <= 1 or len(frames) < 2:
        return frames

    result = []
    for i in range(len(frames) - 1):
        arr_a = np.array(frames[i], dtype=np.float32)
        arr_b = np.array(frames[i + 1], dtype=np.float32)

        for j in range(multiplier):
            alpha = j / multiplier
            blended = ((1.0 - alpha) * arr_a + alpha * arr_b).astype(np.uint8)
            result.append(Image.fromarray(blended))

    # Append the very last frame
    result.append(frames[-1])
    return result


def find_videos(base_path: str, pattern: str = "*.mp4") -> List[str]:
    """Recursively find videos matching pattern."""
    return glob.glob(os.path.join(base_path, "**", pattern), recursive=True)


# ============================================================================
# EXPERIMENT
# ============================================================================

def run_experiment(multipliers: List[int]):
    """Run the full frame interpolation experiment."""
    print("=" * 80)
    print("  FRAME INTERPOLATION EXPERIMENT — Effect of adding synthetic frames")
    print("=" * 80)
    print(f"  Task: {TASK_INSTRUCTION}")
    print(f"  Qwen sampling FPS: {QWEN_FPS}")
    print(f"  Score threshold: {SCORE_THRESHOLD}")
    print(f"  Multipliers: {multipliers}")
    print(f"  Method: Linear pixel blending between consecutive frames")

    # Find videos
    success_videos = sorted(find_videos(SUCCESS_PATH, VIDEO_PATTERN))
    failure_videos = sorted(find_videos(FAILURE_PATH, VIDEO_PATTERN))
    print(f"\n  Found {len(success_videos)} success videos, {len(failure_videos)} failure videos")

    if not success_videos and not failure_videos:
        print("ERROR: No videos found!")
        return

    # Load model
    model = RoboRewardModel(MODEL_CACHE)

    # Conditions: original + each multiplier
    interp_labels = [f"interp_{m}x" for m in multipliers]
    all_condition_labels = ["original"] + interp_labels

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

        # --- Baseline: original full MP4 ---
        score, qwen_frames = model.predict_video(video_path, TASK_INSTRUCTION, fps=QWEN_FPS)
        predicted = "success" if score >= SCORE_THRESHOLD else "failure"
        correct = predicted == actual_label

        video_result["conditions"]["original"] = {
            "multiplier": 1,
            "input_frames": orig_frames,
            "qwen_frames_used": qwen_frames,
            "score": score,
            "predicted": predicted,
            "correct": correct
        }
        status = "OK" if correct else "WRONG"
        print(f"       original: input_frames={orig_frames:4d}  "
              f"qwen_frames={qwen_frames:3d}  score={score:.1f} -> {predicted:7s} [{status}]")

        # --- Extract all frames once, then interpolate ---
        all_frames = extract_all_frames(video_path)

        for mult, cond_label in zip(multipliers, interp_labels):
            interpolated = interpolate_frames(all_frames, mult)
            # Cap frame count to avoid OOM — uniformly subsample if needed
            if len(interpolated) > MAX_INPUT_FRAMES:
                indices = np.linspace(0, len(interpolated) - 1, MAX_INPUT_FRAMES, dtype=int)
                interpolated = [interpolated[i] for i in indices]
            score, qwen_frames = model.predict_frames(interpolated, TASK_INSTRUCTION, fps=QWEN_FPS)
            predicted = "success" if score >= SCORE_THRESHOLD else "failure"
            correct = predicted == actual_label

            video_result["conditions"][cond_label] = {
                "multiplier": mult,
                "input_frames": len(interpolated),
                "qwen_frames_used": qwen_frames,
                "score": score,
                "predicted": predicted,
                "correct": correct
            }
            status = "OK" if correct else "WRONG"
            print(f"  {cond_label:>14s}: input_frames={len(interpolated):4d}  "
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

    for cond_label in interp_labels:
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
    out_path = os.path.join(OUTPUT_DIR, f"frame_interpolation_experiment_{timestamp}.json")

    output = {
        "experiment_info": {
            "description": "Test whether adding interpolated frames affects RoboReward predictions",
            "method": "Baseline uses MP4 path; interpolation conditions use PIL frame lists with linear blending",
            "task_instruction": TASK_INSTRUCTION,
            "qwen_sampling_fps": QWEN_FPS,
            "score_threshold": SCORE_THRESHOLD,
            "multipliers": multipliers,
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
    parser = argparse.ArgumentParser(description="Frame Interpolation Experiment for RoboReward")
    parser.add_argument(
        '--multipliers', nargs='+', type=int, default=[2, 3, 4],
        help='Frame count multipliers (default: 2 3 4)'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.multipliers)
