#!/usr/bin/env python3
"""
Test: Reverse Counterfactual Relabeling for DROID Failures
==========================================================

For 4 test videos (2 bad task descriptions, 2 good), compare:
  (A) RoboReward score with original DROID task label
  (B) VLM-generated task description from video analysis (Qwen3-VL-8B-Instruct)
  (C) RoboReward score with VLM-generated task label

Pipeline per video:
  1. Qwen3-VL-8B-Instruct: Video Analysis (paper A.2.2 prompt) -> scene description
  2. Qwen3-VL-8B-Instruct: Generate specific task description from scene
  3. RoboReward-8B: Score with OLD label
  4. RoboReward-8B: Score with NEW label

Usage:
    python test_relabel.py --video-dir /path/to/test_relabel --device cuda:0
"""

import os
import json
import re
import argparse
import torch
from datetime import datetime


# ============================================================
# Test episodes
# ============================================================

TEST_EPISODES = [
    {
        "video": "AUTOLab_2023-07-08_Sat_Jul__8_08-58-33_2023_ext1.mp4",
        "old_task": "Do any task, and then reset the scene.",
        "lab": "AUTOLab",
        "quality": "BAD",
    },
    {
        "video": "TRI_2023-08-22_Tue_Aug_22_11-45-42_2023_ext1.mp4",
        "old_task": "Fold, spread out, or clump object (ex: cloth, charging cord, clothes)",
        "lab": "TRI",
        "quality": "BAD",
    },
    {
        "video": "RPL_2023-05-23_Tue_May_23_12-08-11_2023_ext1.mp4",
        "old_task": "Pnp-Table-To-Cab-Cans",
        "lab": "RPL",
        "quality": "GOOD",
    },
    {
        "video": "ILIAD_2023-05-18_Thu_May_18_11-06-03_2023_ext1.mp4",
        "old_task": "Open or close hinged object (ex: hinged door, microwave, oven)",
        "lab": "ILIAD",
        "quality": "GOOD",
    },
]


# ============================================================
# Prompts (from RoboReward paper)
# ============================================================

# Paper A.2.2 - Video Analysis prompt
VIDEO_ANALYSIS_PROMPT = """You are analyzing a video of a robot performing a short-horizon manipulation task.
Describe the starting scene and objects visible. Be sure to mention objects in the task description.
Describe thoroughly object positions, locations and their relations to each other and to the robot for the starting scene.
Use the ROBOT BASE perspective for left/right; if the base is not visible, use the CAMERA perspective.
Then describe, step by step, what the robot does from start to end, focusing on the final state.
Be concrete and factual. Do not invent objects that are not visible.
Pay special attention to the last frame, especially when describing the final state.
Task description: {task}
Output sections:
1) Scene and objects
2) Robot actions step by step
3) Final state summary including the final state of the robot and relevant objects given the task description."""

# Task generation prompt - infer intended task from a failure video
TASK_GENERATION_PROMPT = """You just watched a video of a robot ATTEMPTING a manipulation task. The robot FAILED — it did not complete the task successfully. Your job is to infer what task the robot was TRYING to accomplish, NOT what it actually did.

You are also given the original task label from the dataset, but it may be vague or generic (e.g., "Do any task" or "Move object to a new position"). You must produce a MORE SPECIFIC task description based on what you observe in the video.

Original task label: {old_task}

Instructions:
- Look at the objects the robot interacts with or reaches toward
- Look at the robot's trajectory and infer the intended goal
- Be specific: name the actual objects (colors, shapes, types) and the intended destination or action
- Describe the INTENDED task, not the outcome. For example, if the robot reaches for a cup but drops it, the task is "Pick up the cup" not "Drop the cup"
- Start with a verb (e.g., "Pick up", "Move", "Fold", "Open", "Place")
- Keep it under 15 words
- Output ONLY the task command on a single line, nothing else."""

# Paper A.2.2 - Scoring rubric
SCORING_PROMPT = """Given the task, assign a discrete progress score reward (1,2,3,4,5) for the robot in the video in the format: ANSWER: <score>

Rubric for end-of-episode progress (judge only the final state without time limits):
1 - No Success: Final state shows no goal-relevant change for the command.
2 - Minimal Progress: Final state shows a small but insufficient change toward the goal.
3 - Partial Completion: The final state shows good progress toward the goal but violates more than one requirement or a major requirement.
4 - Near Completion: Final state is correct in region and intent but misses a single minor requirement.
5 - Perfect Completion: Final state satisfies all requirements.

Task: {task}

Provide brief reasoning, then write ANSWER: <score>"""


# ============================================================
# Model loading
# ============================================================

def load_qwen3_vl(device="cuda:0"):
    """Load Qwen3-VL-8B-Instruct (base, NOT fine-tuned) for video analysis + task generation."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))

    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    for subdir in ["hub", "."]:
        cache_path = os.path.join(hf_home, subdir, "models--Qwen--Qwen3-VL-8B-Instruct", "snapshots")
        if os.path.exists(cache_path):
            import glob
            snapshots = sorted(glob.glob(os.path.join(cache_path, "*")))
            if snapshots:
                model_name = snapshots[0]
                break

    print(f"Loading Qwen3-VL-8B-Instruct from: {model_name}")

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
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)

    print("Qwen3-VL-8B-Instruct loaded.")
    return model, processor, process_vision_info


def load_roboreward(device="cuda:0"):
    """Load RoboReward-8B (fine-tuned Qwen3-VL) for scoring."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))

    model_name = "teetone/RoboReward-8B"
    for subdir in [".", "hub"]:
        cache_path = os.path.join(hf_home, subdir, "models--teetone--RoboReward-8B", "snapshots")
        if os.path.exists(cache_path):
            import glob
            snapshots = sorted(glob.glob(os.path.join(cache_path, "*")))
            if snapshots:
                model_name = snapshots[0]
                break

    print(f"Loading RoboReward-8B from: {model_name}")

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
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)

    print("RoboReward-8B loaded.")
    return model, processor, process_vision_info


# ============================================================
# Inference helpers
# ============================================================

def vlm_inference(video_path, prompt, model, processor, process_vision_info, max_new_tokens=512):
    """Run VLM inference on a video with a text prompt. Returns generated text."""
    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "fps": 10.0},
            {"type": "text", "text": prompt},
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
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    del inputs, output_ids, generated_ids, image_inputs, video_inputs
    torch.cuda.empty_cache()

    return output_text


def parse_score(text):
    """Extract score 1-5 from model output."""
    match = re.search(r"ANSWER:\s*([1-5])", text)
    if not match:
        match = re.search(r"\b([1-5])\b", text)
    return int(match.group(1)) if match else None


# ============================================================
# Main test
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Test reverse counterfactual relabeling")
    parser.add_argument("--video-dir", type=str,
                        default="/home/pkarageo/master-thesis/Robo-Reward-FPS/test_relabel",
                        help="Directory with test videos")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    print("=" * 70)
    print("Test: Reverse Counterfactual Relabeling")
    print("=" * 70)

    # Step 1: Load Qwen3-VL-8B-Instruct for video analysis + task generation
    print("\n--- Loading Qwen3-VL-8B-Instruct (video analysis) ---")
    qwen_model, qwen_processor, qwen_pvi = load_qwen3_vl(args.device)

    results = []

    for ep in TEST_EPISODES:
        video_path = os.path.join(args.video_dir, ep["video"])
        if not os.path.exists(video_path):
            print(f"SKIP: {video_path} not found")
            continue

        print(f"\n{'=' * 70}")
        print(f"Video: {ep['video']}")
        print(f"Lab: {ep['lab']} | Quality: {ep['quality']}")
        print(f"Old task: {ep['old_task']}")
        print("-" * 70)

        # 1a. Video Analysis with old task (paper A.2.2)
        print("\n[Qwen3-VL] Video Analysis...")
        analysis_prompt = VIDEO_ANALYSIS_PROMPT.format(task=ep["old_task"])
        analysis = vlm_inference(video_path, analysis_prompt, qwen_model, qwen_processor, qwen_pvi)
        print(f"Analysis:\n{analysis[:500]}")

        # 1b. Generate new task description
        print("\n[Qwen3-VL] Generating task description...")
        task_gen_prompt = TASK_GENERATION_PROMPT.format(old_task=ep["old_task"])
        new_task = vlm_inference(video_path, task_gen_prompt, qwen_model, qwen_processor, qwen_pvi, max_new_tokens=50)
        new_task = new_task.strip().strip('"').strip("'")
        print(f"Generated task: {new_task}")

        results.append({
            "video": ep["video"],
            "lab": ep["lab"],
            "quality": ep["quality"],
            "old_task": ep["old_task"],
            "new_task": new_task,
            "analysis": analysis,
        })

    # Free Qwen3-VL memory before loading RoboReward
    print("\n--- Unloading Qwen3-VL-8B-Instruct ---")
    del qwen_model, qwen_processor, qwen_pvi
    torch.cuda.empty_cache()

    # Step 2: Load RoboReward-8B for scoring
    print("\n--- Loading RoboReward-8B (scoring) ---")
    rr_model, rr_processor, rr_pvi = load_roboreward(args.device)

    for r in results:
        video_path = os.path.join(args.video_dir, r["video"])

        print(f"\n{'=' * 70}")
        print(f"Video: {r['video']} ({r['quality']} description)")
        print(f"Old task: {r['old_task']}")
        print(f"New task: {r['new_task']}")
        print("-" * 70)

        # 2a. Score with OLD task
        print("\n[RoboReward] Scoring with OLD task...")
        old_scoring_prompt = SCORING_PROMPT.format(task=r["old_task"])
        old_output = vlm_inference(video_path, old_scoring_prompt, rr_model, rr_processor, rr_pvi, max_new_tokens=256)
        old_score = parse_score(old_output)
        print(f"Old score: {old_score}")
        print(f"Reasoning: {old_output[:300]}")

        # 2b. Score with NEW task
        print("\n[RoboReward] Scoring with NEW task...")
        new_scoring_prompt = SCORING_PROMPT.format(task=r["new_task"])
        new_output = vlm_inference(video_path, new_scoring_prompt, rr_model, rr_processor, rr_pvi, max_new_tokens=256)
        new_score = parse_score(new_output)
        print(f"New score: {new_score}")
        print(f"Reasoning: {new_output[:300]}")

        r["old_score"] = old_score
        r["old_score_output"] = old_output
        r["new_score"] = new_score
        r["new_score_output"] = new_output

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Video':<20} {'Quality':<6} {'Old Task':<40} {'Old':>3} {'New Task':<40} {'New':>3}")
    print("-" * 120)
    for r in results:
        vid_short = r["video"][:18]
        old_t = r["old_task"][:38]
        new_t = r["new_task"][:38]
        print(f"{vid_short:<20} {r['quality']:<6} {old_t:<40} {r['old_score']:>3} {new_t:<40} {r['new_score']:>3}")

    # Save full results
    out_path = os.path.join(args.video_dir, "test_relabel_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
