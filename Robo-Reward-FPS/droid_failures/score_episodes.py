#!/usr/bin/env python3
"""
Score DROID failure episodes with RoboReward-8B (1-5 scale).

Reads the relabeled task descriptions from qwen3_relabeled.jsonl and scores
each (episode, camera) pair using RoboReward-8B. Optionally verifies scores.

For each episode we score all 3 cameras (ext1, ext2, wrist) independently.

Output: scored_episodes.jsonl with fields:
  video_key, episode_id, camera, task, score, verified, score_output, verify_output

Usage:
    python score_episodes.py --device cuda:0
    python score_episodes.py --device cuda:0 --skip-verify   # faster, no verification pass
    python score_episodes.py --device cuda:0 --start-from 500
"""

import os
import json
import re
import argparse
import glob
import torch
from datetime import datetime


# ============================================================
# Paths
# ============================================================

RELABELED_PATH = "/var/scratch/pkarageo/droid_failures_labeled/qwen3_relabeled.jsonl"
VIDEO_DIR = "/var/scratch/pkarageo/droid_failures_processed"
OUTPUT_PATH = "/var/scratch/pkarageo/droid_failures_labeled/scored_episodes_v3.jsonl"

# ============================================================
# Prompts (from RoboReward paper)
# ============================================================

SCORING_PROMPT = """Rubric for end-of-episode progress (judge only the final state without time limits):
1 - No Success: Final state shows no goal-relevant change for the command.
2 - Minimal Progress: Final state shows a small but insufficient change toward the goal (e.g., robot moves toward the correct object or grasps it but fails the task).
3 - Partial Completion: The final state shows good progress toward the goal but violates more than one requirement or a major requirement.
4 - Near Completion: Final state is correct in region and intent but misses a single minor requirement.
5 - Perfect Completion: Final state satisfies all requirements.

Given the task, first briefly state what the robot does (1-2 sentences), then assign a score.
Format: <reasoning> ANSWER: <score>"""

VERIFICATION_PROMPT = """You are a strict annotation validator for short-horizon ROBOT TASK videos.

INPUTS
- TASK DESCRIPTION: {task}
- PROVIDED SCORE: {score}
(must be one of 1,2,3,4,5)

YOUR JOB (single-pass):
1) Watch the video.
2) Using the rubric below, evaluate the robot's actions during the task and the final state, and determine whether the PROVIDED SCORE is correct.
3) Check that the TASK DESCRIPTION is coherent and refers to the correct items/actions visible in the video.
4) Decision basis and rejection criteria:
- If the TASK DESCRIPTION is incoherent with the video (e.g., wrong primary object identity, inconsistent relation, key details not visible, or rule conflicts), you MUST output ANSWER: FALSE.
- Otherwise, output ANSWER: TRUE if the PROVIDED SCORE matches the rubric for the video; output ANSWER: FALSE if it does not match.

RUBRIC (end-of-episode, discrete progress):
1 - No Success: Final state shows no goal-relevant change for the command (e.g., wrong object was handled)
2 - Minimal Progress: Final state shows a small but insufficient change toward the goal.
3 - Partial Completion: The final state shows good progress toward the goal but violates more than one requirement or a major requirement.
4 - Near Completion: The final state is correct in region and intent but misses a single minor requirement.
5 - Perfect Completion: Fully satisfies the task as described; ALL requirements met.

MINOR VS MAJOR REQUIREMENTS (clarification for 3 vs 4):
- Treat a requirement as MINOR when the PRIMARY object identity and PRIMARY spatial relation to the reference object are satisfied, but an AUXILIARY constraint is missed.
- Examples of AUXILIARY constraints that can be MINOR: remaining on the same support surface (e.g., keeping a cloth directly under the object), holding/releasing the gripper at the end, small orientation or placement tolerances, or cosmetic positioning details that do not change the core relation.
- Therefore, if the target object is correctly placed relative to the specified reference and all core elements are correct, but an auxiliary constraint like keeping the original cloth under the object is missed, prefer score 4 (Near Completion) rather than 3 (Partial Completion).
- Use score 3 only when multiple constraints are missed or when the missed constraint is CENTRAL to the task identity (e.g., wrong object identity or the primary spatial relation is incorrect).

PERSPECTIVE & COMMON PITFALLS (apply strictly):
- Left/Right: Use the ROBOT BASE perspective. If the base is not visible, use the CAMERA perspective.
- Object naming: REJECT if the description mislabels key items.
- Irrelevant objects: Ignore objects not needed to perform the task (do NOT reject just for extra/unused items).
- Visibility: If key details to judge success are not visible or the video is too unclear, REJECT.

TASK SANITY CHECKS (MUST REJECT if any apply):
- Task string is empty, placeholder, or nonsense and not a valid task description/command.
- Task does not reference any visible objects/relations in the video or cannot be grounded.

OUTPUT FORMAT (STRICT)
1) First, write your reasoning (few sentences) that clearly states whether the PROVIDED SCORE matches the rubric and why.
2) Then on a new line write exactly ONE of the following without any formatting and nothing else:
ANSWER: TRUE
or
ANSWER: FALSE"""


# ============================================================
# Model loading
# ============================================================

def load_roboreward(device="cuda:0"):
    """Load RoboReward-4B (fine-tuned Qwen3-VL) with 4-bit quantization."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))

    # Try RoboReward-4B first, then fall back to 8B
    model_name = "teetone/RoboReward-4B"
    for model_dir_name in ["models--teetone--RoboReward-4B", "models--teetone--RoboReward-8B"]:
        for subdir in [".", "hub"]:
            cache_path = os.path.join(hf_home, subdir, model_dir_name, "snapshots")
            if os.path.exists(cache_path):
                snapshots = sorted(glob.glob(os.path.join(cache_path, "*")))
                if snapshots:
                    model_name = snapshots[0]
                    break
        else:
            continue
        break

    print(f"Loading RoboReward from: {model_name}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    gpu_id = int(device.split(":")[1]) if ":" in device else 0
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map={"": gpu_id},
        trust_remote_code=False,
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)

    print("RoboReward loaded.")
    return model, processor, process_vision_info


# ============================================================
# Inference
# ============================================================

def score_video(video_path, task, model, processor, process_vision_info):
    """Score a video using RoboReward. Returns (score, raw_output)."""
    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "fps": 1.0},
            {"type": "text", "text": SCORING_PROMPT + f"\n\nTask: {task}"},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    del inputs, output_ids, generated_ids, image_inputs, video_inputs
    torch.cuda.empty_cache()

    # Parse score
    score_match = re.search(r"ANSWER:\s*([1-5])", output_text)
    if not score_match:
        score_match = re.search(r"\b([1-5])\b", output_text)

    score = int(score_match.group(1)) if score_match else 1
    return score, output_text.strip()


def verify_score(video_path, task, score, model, processor, process_vision_info):
    """Verify a (video, task, score) triple. Returns (is_valid, explanation)."""
    verification_text = VERIFICATION_PROMPT.format(task=task, score=score)

    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "fps": 1.0},
            {"type": "text", "text": verification_text},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    del inputs, output_ids, generated_ids, image_inputs, video_inputs
    torch.cuda.empty_cache()

    is_valid = "ANSWER: TRUE" in output_text.upper()
    return is_valid, output_text.strip()


# ============================================================
# Main pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Score DROID episodes with RoboReward-8B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip verification pass (2x faster)")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Skip first N work items (for manual resume)")
    args = parser.parse_args()

    # ----------------------------------------------------------
    # 1. Load relabeled episodes
    # ----------------------------------------------------------
    print("Loading relabeled episodes...")
    episodes = {}
    with open(RELABELED_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            episodes[d["episode"]] = d["new_label"]

    print(f"  Episodes with labels: {len(episodes)}")

    # ----------------------------------------------------------
    # 2. Build work list: (episode, camera, video_path, task)
    # ----------------------------------------------------------
    print("Building work list...")
    work = []
    missing_videos = 0

    for episode, task in sorted(episodes.items()):
        for camera in ["ext1", "ext2", "wrist"]:
            video_path = os.path.join(VIDEO_DIR, f"{episode}_{camera}.mp4")
            if os.path.exists(video_path):
                work.append({
                    "episode": episode,
                    "camera": camera,
                    "video_path": video_path,
                    "task": task,
                })
            else:
                missing_videos += 1

    print(f"  Total scoring tasks: {len(work)}")
    print(f"  Missing videos: {missing_videos}")

    # ----------------------------------------------------------
    # 3. Resume support
    # ----------------------------------------------------------
    already_done = set()
    if os.path.exists(OUTPUT_PATH):
        good_lines = []
        with open(OUTPUT_PATH) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    already_done.add(d["video_key"])
                    good_lines.append(line)
                except json.JSONDecodeError:
                    print(f"  WARNING: Corrupt line {line_num}, removing.")
        # Rewrite clean file
        with open(OUTPUT_PATH, "w") as f:
            for line in good_lines:
                f.write(line + "\n")
        print(f"  Already scored: {len(already_done)}")

    # Filter work list
    work = [w for w in work if f"{w['episode']}_{w['camera']}" not in already_done]

    if args.start_from > 0:
        work = work[args.start_from:]
        print(f"  Skipping first {args.start_from} items")

    print(f"  Remaining to score: {len(work)}")

    if not work:
        print("Nothing to do!")
        return

    # ----------------------------------------------------------
    # 4. Load model
    # ----------------------------------------------------------
    model, processor, pvi = load_roboreward(args.device)

    # ----------------------------------------------------------
    # 5. Score all episodes
    # ----------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"SCORING {len(work)} videos {'(no verification)' if args.skip_verify else '(with verification)'}")
    print(f"{'='*70}\n")

    scored = 0
    verified_true = 0
    verified_false = 0
    errors = 0
    start_time = datetime.now()

    with open(OUTPUT_PATH, "a") as f_out:
        for i, w in enumerate(work):
            video_key = f"{w['episode']}_{w['camera']}"

            try:
                # Score
                score, score_output = score_video(
                    w["video_path"], w["task"], model, processor, pvi
                )

                # Verify (optional)
                is_valid = None
                verify_output = ""
                if not args.skip_verify:
                    is_valid, verify_output = verify_score(
                        w["video_path"], w["task"], score, model, processor, pvi
                    )
                    if is_valid:
                        verified_true += 1
                    else:
                        verified_false += 1

                result = {
                    "video_key": video_key,
                    "episode_id": w["episode"],
                    "camera": w["camera"],
                    "task": w["task"],
                    "score": score,
                    "score_output": score_output[-300:],
                    "verified": is_valid,
                    "verify_output": verify_output[-300:] if verify_output else "",
                }

                f_out.write(json.dumps(result) + "\n")
                f_out.flush()
                scored += 1

            except Exception as e:
                print(f"  ERROR [{i+1}] {video_key}: {e}")
                errors += 1
                continue

            # Progress logging
            if scored <= 10 or scored % 100 == 0:
                elapsed = (datetime.now() - start_time).total_seconds() / 3600
                rate = scored / elapsed if elapsed > 0 else 0
                remaining = (len(work) - i - 1) / rate if rate > 0 else 0
                v_str = ""
                if not args.skip_verify:
                    v_str = f" verified={is_valid}"
                print(f"  [{scored}/{len(work)}] {video_key}: score={score}{v_str}")
                if scored % 100 == 0:
                    print(f"  === Rate: {rate:.0f}/hr | ETA: {remaining:.1f}h | "
                          f"Errors: {errors} ===")

    # ----------------------------------------------------------
    # 6. Summary
    # ----------------------------------------------------------
    elapsed = (datetime.now() - start_time).total_seconds() / 3600
    print(f"\n{'='*70}")
    print("SCORING COMPLETE")
    print(f"{'='*70}")
    print(f"  Total scored: {scored}")
    print(f"  Errors: {errors}")
    print(f"  Time: {elapsed:.1f} hours")
    if not args.skip_verify:
        print(f"  Verified TRUE: {verified_true}")
        print(f"  Verified FALSE: {verified_false}")
    print(f"  Output: {OUTPUT_PATH}")

    # Quick score distribution
    scores = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    total = 0
    with open(OUTPUT_PATH) as f:
        for line in f:
            d = json.loads(line.strip())
            scores[d["score"]] += 1
            total += 1
    print(f"\n  Score distribution (all {total} entries):")
    for s in range(1, 6):
        pct = scores[s] / total * 100 if total > 0 else 0
        print(f"    {s}: {scores[s]:>6} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
