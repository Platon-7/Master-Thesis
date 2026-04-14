#!/usr/bin/env python3
"""
Label DROID Failure Videos with VLM Scores
============================================

Two-phase pipeline:
  Phase 1: Clean task descriptions using Qwen3-4B-Instruct (text-only)
  Phase 2: Score videos using Qwen3-VL-8B + verify with VLM

Scoring uses the RoboReward rubric (1-5 discrete scale).
Verification uses the paper's verification prompt (Appendix A.2.3).

Usage:
    # Phase 1 only (task cleanup)
    python label_droid_failures.py --phase 1 \
        --input-dir /var/scratch/pkarageo/droid_failures_raw \
        --output-dir /var/scratch/pkarageo/droid_failures_labeled

    # Phase 2 only (VLM scoring + verification)
    python label_droid_failures.py --phase 2 \
        --video-dir /var/scratch/pkarageo/droid_failures_processed \
        --output-dir /var/scratch/pkarageo/droid_failures_labeled

    # Both phases
    python label_droid_failures.py --phase both \
        --input-dir /var/scratch/pkarageo/droid_failures_raw \
        --video-dir /var/scratch/pkarageo/droid_failures_processed \
        --output-dir /var/scratch/pkarageo/droid_failures_labeled

Author: Platon Karageorgis
"""

import os
import json
import re
import argparse
import glob
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
from datetime import datetime


# ============================================================
# Phase 1: Task Description Cleanup
# ============================================================

TASK_CLEANUP_PROMPT = """Rewrite the following task description to correct grammar and spelling only.
Fix capitalization errors by beginning a sentence or phrase with a capital letter.
Do not change meaning.
Task description: {task}
Return only the corrected text."""

VAGUE_TASK_PATTERNS = [
    r"do anything you like",
    r"do any .*tasks? consecutively",
    r"do something",
    r"perform any task",
    r"free play",
    r"do whatever",
]


def is_vague_task(task: str) -> bool:
    """Check if a task description is too vague to use."""
    task_lower = task.lower().strip()
    for pattern in VAGUE_TASK_PATTERNS:
        if re.search(pattern, task_lower):
            return True
    return False


def load_text_model(device: str = "cuda:0"):
    """Load Qwen3-4B-Instruct for text-only task cleanup."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))

    # Qwen3-4B-Instruct-2507 (same model used in the RoboReward paper, Section A.2.1)
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    cache_path = os.path.join(hf_home, "hub", "models--Qwen--Qwen3-4B-Instruct-2507", "snapshots")
    if os.path.exists(cache_path):
        snapshots = sorted(glob.glob(os.path.join(cache_path, "*")))
        if snapshots:
            model_name = snapshots[0]

    print(f"Loading text model from: {model_name}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    gpu_id = int(device.split(":")[1]) if ":" in device else 0
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map={"": gpu_id},
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("Text model loaded.")
    return model, tokenizer


def clean_task_description(task: str, model, tokenizer) -> Optional[str]:
    """Clean a single task description using the text model.

    Returns cleaned task or None if unusable.
    """
    # Quick filter for obviously vague tasks
    if is_vague_task(task):
        return None

    prompt = TASK_CLEANUP_PROMPT.format(task=task)
    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )
    # Decode only the generated tokens
    response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # Take only the first line of the response
    cleaned = response.split("\n")[0].strip()
    # Remove quotes if the model wrapped it
    cleaned = cleaned.strip('"').strip("'").strip()

    if not cleaned or len(cleaned) < 3:
        return None

    return cleaned


def run_phase1(input_dir: str, output_dir: str, text_device: str = "cuda:0"):
    """Phase 1: Clean task descriptions from metadata."""
    print(f"\n{'='*60}")
    print("Phase 1: Task Description Cleanup")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    # Load manifest
    manifest_path = os.path.join(input_dir, "download_manifest.json")
    if not os.path.exists(manifest_path):
        print(f"ERROR: No download manifest at {manifest_path}")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    episodes = manifest.get("completed_episodes", {})
    print(f"Total episodes: {len(episodes)}")

    # Load text model
    model, tokenizer = load_text_model(text_device)

    # Resume from checkpoint if exists
    output_path = os.path.join(output_dir, "cleaned_tasks.json")
    cleaned_tasks = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            cleaned_tasks = json.load(f)
        print(f"Resuming Phase 1: {len(cleaned_tasks)} already cleaned")

    filtered_count = 0
    error_count = 0

    for i, (episode_id, info) in enumerate(episodes.items()):
        if episode_id in cleaned_tasks:
            continue

        raw_task = info.get("task", "")
        if not raw_task:
            filtered_count += 1
            continue

        try:
            cleaned = clean_task_description(raw_task, model, tokenizer)
        except Exception as e:
            print(f"  Error cleaning task for {episode_id}: {e}")
            error_count += 1
            continue

        if cleaned is None:
            filtered_count += 1
            if i < 20:
                print(f"  FILTERED: '{raw_task[:60]}...'")
        else:
            cleaned_tasks[episode_id] = {
                "raw_task": raw_task,
                "cleaned_task": cleaned,
            }
            if i < 20:
                print(f"  '{raw_task[:40]}' -> '{cleaned[:40]}'")

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(episodes)} | Kept: {len(cleaned_tasks)} | Filtered: {filtered_count}")

        # Checkpoint every 500 episodes
        if (i + 1) % 500 == 0:
            with open(output_path, "w") as f:
                json.dump(cleaned_tasks, f, indent=2)

    # Final save
    with open(output_path, "w") as f:
        json.dump(cleaned_tasks, f, indent=2)

    print(f"\nPhase 1 complete:")
    print(f"  Kept: {len(cleaned_tasks)}")
    print(f"  Filtered (vague/unusable): {filtered_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output: {output_path}")


# ============================================================
# Phase 2: VLM Scoring + Verification
# ============================================================

SCORING_PROMPT = """Given the task, assign a discrete progress score reward (1,2,3,4,5) for the robot in the video in the format: ANSWER: <score>
Rubric for end-of-episode progress (judge only the final state without time limits):
1 - No Success: Final state shows no goal-relevant change for the command.
2 - Minimal Progress: Final state shows a small but insufficient change toward the goal.
3 - Partial Completion: The final state shows good progress toward the goal but violates more than one requirement or a major requirement.
4 - Near Completion: Final state is correct in region and intent but misses a single minor requirement.
5 - Perfect Completion: Final state satisfies all requirements."""

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


def load_vlm_model(device: str = "cuda:1"):
    """Load Qwen3-VL-8B for video scoring."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))

    # Try RoboReward-8B first (fine-tuned Qwen3-VL), then base Qwen3-VL
    # Check both {hf_home}/{name} and {hf_home}/hub/{name} layouts
    model_name = "teetone/RoboReward-8B"
    for model_dir_name in ["models--teetone--RoboReward-8B", "models--Qwen--Qwen3-VL-8B-Instruct"]:
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

    print(f"Loading VLM from: {model_name}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    gpu_id = int(device.split(":")[1]) if ":" in device else 1
    vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map={"": gpu_id},
        trust_remote_code=False,
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)

    print("VLM loaded.")
    return vlm_model, processor, process_vision_info


def extract_frames_at_1fps(video_path: str, max_frames: int = 32) -> List[Image.Image]:
    """Extract frames at 1 FPS from a video, up to max_frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        return []

    # Sample at 1 FPS
    frame_interval = max(1, int(fps))
    target_indices = list(range(0, total_frames, frame_interval))

    # Subsample if too many
    if len(target_indices) > max_frames:
        indices = np.linspace(0, len(target_indices) - 1, max_frames, dtype=int)
        target_indices = [target_indices[i] for i in indices]

    frames = []
    for idx in target_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


def score_video(
    video_path: str,
    task: str,
    vlm_model,
    processor,
    process_vision_info,
    max_frames: int = 32,
) -> Tuple[int, str]:
    """Score a video using the VLM. Returns (score, raw_output)."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SCORING_PROMPT}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "fps": 1.0},
            {"type": "text", "text": f"Task: {task}"},
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
    ).to(vlm_model.device)

    with torch.no_grad():
        output_ids = vlm_model.generate(**inputs, max_new_tokens=128, do_sample=False)
        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    del inputs, output_ids, generated_ids, image_inputs, video_inputs
    torch.cuda.empty_cache()

    # Parse score
    score_match = re.search(r"ANSWER:\s*([1-5])", output_text)
    if not score_match:
        score_match = re.search(r"\b([1-5])\b", output_text)

    score = int(score_match.group(1)) if score_match else 1
    return score, output_text


def verify_score(
    video_path: str,
    task: str,
    score: int,
    vlm_model,
    processor,
    process_vision_info,
    max_frames: int = 32,
) -> Tuple[bool, str]:
    """Verify a (video, task, score) triple using the VLM. Returns (is_valid, explanation)."""
    verification_text = VERIFICATION_PROMPT.format(
        task=task,
        score=score,
    )

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
    ).to(vlm_model.device)

    with torch.no_grad():
        output_ids = vlm_model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    del inputs, output_ids, generated_ids, image_inputs, video_inputs
    torch.cuda.empty_cache()

    # Parse verification result
    is_valid = "ANSWER: TRUE" in output_text.upper()
    return is_valid, output_text


def run_phase2(
    video_dir: str,
    output_dir: str,
    vlm_device: str = "cuda:1",
    max_frames: int = 32,
):
    """Phase 2: VLM scoring and verification."""
    print(f"\n{'='*60}")
    print("Phase 2: VLM Scoring + Verification")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    # Load cleaned tasks from Phase 1
    cleaned_tasks_path = os.path.join(output_dir, "cleaned_tasks.json")
    if not os.path.exists(cleaned_tasks_path):
        print(f"ERROR: No cleaned tasks at {cleaned_tasks_path}. Run Phase 1 first.")
        return

    with open(cleaned_tasks_path) as f:
        cleaned_tasks = json.load(f)

    print(f"Episodes with cleaned tasks: {len(cleaned_tasks)}")

    # Find processed videos
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    print(f"Processed videos found: {len(video_files)}")

    # Build mapping: episode_id -> list of (camera, video_path)
    episode_videos = {}
    for vf in video_files:
        basename = os.path.basename(vf).replace(".mp4", "")
        # Format: {episode_id}_{camera}
        # Camera is the last part after the final underscore
        parts = basename.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in ("ext1", "ext2", "wrist"):
            episode_id = parts[0]
            camera = parts[1]
        else:
            continue

        if episode_id not in episode_videos:
            episode_videos[episode_id] = []
        episode_videos[episode_id].append((camera, vf))

    # Find episodes with both cleaned tasks and videos
    scorable = set(cleaned_tasks.keys()) & set(episode_videos.keys())
    print(f"Scorable episodes (have both tasks and videos): {len(scorable)}")

    # Load VLM
    vlm_model, processor, process_vision_info_fn = load_vlm_model(vlm_device)

    # Load checkpoint if resuming
    results_path = os.path.join(output_dir, "scored_results.jsonl")
    completed = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                entry = json.loads(line)
                completed.add(entry["video_key"])
        print(f"Resuming: {len(completed)} already scored")

    # Score and verify
    scored_count = 0
    verified_count = 0
    rejected_count = 0

    with open(results_path, "a") as f_out:
        for i, episode_id in enumerate(sorted(scorable)):
            task = cleaned_tasks[episode_id]["cleaned_task"]
            cameras = episode_videos[episode_id]

            for camera, video_path in cameras:
                video_key = f"{episode_id}_{camera}"
                if video_key in completed:
                    continue

                try:
                    # Score
                    score, score_output = score_video(
                        video_path, task, vlm_model, processor, process_vision_info_fn, max_frames
                    )
                    scored_count += 1

                    # Verify
                    is_valid, verify_output = verify_score(
                        video_path, task, score, vlm_model, processor, process_vision_info_fn, max_frames
                    )

                    result = {
                        "video_key": video_key,
                        "episode_id": episode_id,
                        "camera": camera,
                        "video_path": video_path,
                        "task": task,
                        "raw_task": cleaned_tasks[episode_id]["raw_task"],
                        "score": score,
                        "score_output": score_output[-300:],  # Truncate
                        "verified": is_valid,
                        "verify_output": verify_output[-300:],  # Truncate
                        "timestamp": datetime.now().isoformat(),
                    }

                    f_out.write(json.dumps(result) + "\n")
                    f_out.flush()

                    if is_valid:
                        verified_count += 1
                    else:
                        rejected_count += 1

                    if scored_count <= 20 or scored_count % 20 == 0:
                        print(f"  [{scored_count}] {video_key}: score={score}, verified={is_valid}")
                        print(f"       Task: {task[:50]}")

                except Exception as e:
                    print(f"  ERROR on {video_key}: {e}")
                    continue

            if (i + 1) % 20 == 0:
                print(f"  Episodes: {i+1}/{len(scorable)} | Scored: {scored_count} | "
                      f"Verified: {verified_count} | Rejected: {rejected_count}")

    print(f"\n{'='*60}")
    print("Phase 2 complete:")
    print(f"  Total scored: {scored_count}")
    print(f"  Verified (kept): {verified_count}")
    print(f"  Rejected: {rejected_count}")
    print(f"  Results: {results_path}")
    print(f"{'='*60}")


# ============================================================
# Phase 3: VLM Label Verification (Qwen3.5 via vLLM)
# ============================================================

LABEL_VERIFY_PROMPT_STRONG = """Watch this video of a robot manipulation episode.

The following task description was provided by a human annotator:
"{label}"

Your job is to verify whether this description is CORRECT. Only change it if it is factually WRONG — for example, if it names the wrong object, the wrong action, or describes something that does not happen in the video.

Do NOT change the label just to add detail, specificity, or color names. "Stack 3 cups together" is perfectly fine — do not rewrite it as "Stack the red cup on the blue cup." Generic but correct descriptions should be kept as-is.

Reply with exactly one line in this format:
LABEL: <task description>"""

LABEL_VERIFY_PROMPT_WEAK = """Watch this video of a robot manipulation episode.

An AI annotator suggested the following task description with {certainty} certainty:
"{label}"

Your job is to watch the video and either verify this description or suggest a better one.

A good task description must:
1. Accurately name the objects the robot interacts with (e.g., "red cup", "white cloth", "blender lid")
2. Accurately name the action (e.g., "stack", "fold", "pour into", "hang on")
3. Be a single imperative sentence (e.g., "Stack the red cup on the blue cup.")
4. Not reference objects or actions that do not appear in the video

If the description is correct, reply with it unchanged.
If it is wrong or vague, reply with a corrected version.

Reply with exactly one line in this format:
LABEL: <task description>"""

# Map categories to certainty levels for the prompt
CATEGORY_CERTAINTY = {
    "SPECIFIC": "strong",
    "BORDERLINE_REVIEW": "medium",
    "GENERIC_REVIEW": "weak",
    "UNLABELED": "weak",
}


def run_phase3(
    relabeled_path: str,
    video_dir: str,
    output_path: str,
    batch_size: int = 8,
):
    """Phase 3: Verify/fix labels using Qwen3.5 VLM via vLLM (full video at 1 FPS)."""
    import time
    import gc
    from vllm import LLM, SamplingParams

    print(f"\n{'='*60}")
    print("Phase 3: VLM Label Verification (Qwen3.5, video @ 1 FPS)")
    print(f"{'='*60}\n")

    # Load current labels
    episodes = []
    with open(relabeled_path) as f:
        for line in f:
            episodes.append(json.loads(line.strip()))
    print(f"Total episodes: {len(episodes)}")

    # Build lookup for original data
    ep_lookup = {ep["episode"]: ep for ep in episodes}

    # Resume from checkpoint — read line by line to handle partial writes
    completed = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    completed.add(d["episode"])
                except json.JSONDecodeError:
                    # Partial write from crash — skip corrupted line
                    continue
        print(f"Resuming: {len(completed)} already verified")

    # Build work list: only episodes not yet verified
    work = []
    for ep in episodes:
        ep_id = ep["episode"]
        if ep_id in completed:
            continue
        label = ep.get("new_label") or ep.get("old_label") or ""
        if not label:
            continue
        video_path = os.path.join(video_dir, f"{ep_id}_ext1.mp4")
        if not os.path.exists(video_path):
            continue
        category = ep.get("category", "GENERIC_REVIEW")
        certainty = CATEGORY_CERTAINTY.get(category, "weak")
        work.append((ep_id, label, video_path, category, certainty))

    print(f"Episodes to verify: {len(work)}")
    by_cert = {}
    for _, _, _, _, c in work:
        by_cert[c] = by_cert.get(c, 0) + 1
    for c, n in sorted(by_cert.items()):
        print(f"  {c}: {n}")
    if not work:
        print("Nothing to do!")
        return

    # Load VLM
    vlm_model_id = os.environ.get("VLM_MODEL_ID", "Qwen/Qwen3.5-VL-3B-Instruct")
    model_id = vlm_model_id
    snapshots_dir = os.path.join(model_id, "snapshots")
    if os.path.exists(snapshots_dir):
        snaps = os.listdir(snapshots_dir)
        if snaps:
            model_id = os.path.join(snapshots_dir, snaps[0])

    num_gpus = torch.cuda.device_count()
    print(f"Loading VLM: {model_id} ({num_gpus} GPUs)")

    vlm = LLM(
        model=model_id,
        dtype="bfloat16",
        tensor_parallel_size=num_gpus,
        max_model_len=int(os.environ.get("VLM_MAX_MODEL_LEN", 16384)),
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        trust_remote_code=True,
        allowed_local_media_path="/",
    )
    print("VLM loaded.")

    sampling_params = SamplingParams(max_tokens=128, temperature=0)

    # Process in batches — flush after every entry for crash safety
    t0 = time.time()
    verified_count = 0
    changed_count = 0
    error_count = 0

    for batch_start in range(0, len(work), batch_size):
        batch = work[batch_start:batch_start + batch_size]

        conversations = []
        for ep_id, label, video_path, category, certainty in batch:
            if certainty == "strong":
                prompt = LABEL_VERIFY_PROMPT_STRONG.format(label=label)
            else:
                prompt = LABEL_VERIFY_PROMPT_WEAK.format(label=label, certainty=certainty)
            messages = [
                {"role": "user", "content": [
                    {"type": "video_url", "video_url": {"url": f"file://{video_path}"}},
                    {"type": "text", "text": prompt},
                ]},
            ]
            conversations.append(messages)

        chat_kwargs = {}
        if any(k in vlm_model_id.lower() for k in ["qwen", "glm"]):
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

        try:
            outputs = vlm.chat(
                messages=conversations,
                sampling_params=sampling_params,
                **chat_kwargs,
            )
        except Exception as e:
            print(f"  ERROR on batch starting at {batch_start}: {e}")
            error_count += len(batch)
            continue

        with open(output_path, "a") as fout:
            for (ep_id, label, video_path, category, certainty), output in zip(batch, outputs):
                raw_text = output.outputs[0].text
                # Strip thinking tags
                clean = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
                clean = re.sub(r"<think>.*", "", clean, flags=re.DOTALL)
                clean = re.sub(r"</think>", "", clean)
                clean = re.sub(r"<\|[^|]*\|>", "", clean)
                clean = clean.strip()

                # Parse LABEL: line
                new_label = label  # default: keep original
                match = re.search(r"LABEL:\s*(.+)", clean)
                if match:
                    candidate = match.group(1).strip().strip('"').strip("'").strip()
                    if candidate and len(candidate) >= 3:
                        new_label = candidate

                was_changed = new_label.lower().strip(".") != label.lower().strip(".")
                if was_changed:
                    changed_count += 1

                orig = ep_lookup.get(ep_id, {})
                entry = {
                    "episode": ep_id,
                    "old_label": orig.get("old_label", ""),
                    "phase2_label": label,
                    "new_label": new_label,
                    "verified_changed": was_changed,
                    "certainty": certainty,
                    "vlm_raw": clean[:200],
                    "category": orig.get("category", ""),
                    "tier": orig.get("tier", ""),
                }
                fout.write(json.dumps(entry) + "\n")
                fout.flush()  # flush every entry — crash safe
                verified_count += 1

        elapsed_so_far = time.time() - t0
        rate = verified_count / elapsed_so_far if elapsed_so_far > 0 else 0
        eta = (len(work) - verified_count) / rate if rate > 0 else 0
        print(f"  [{verified_count}/{len(work)}] batch done | changed: {changed_count} | "
              f"errors: {error_count} | {rate:.1f} ep/s | ETA: {eta/60:.0f}min")

    elapsed = time.time() - t0
    print(f"\nPhase 3 complete:")
    print(f"  Verified: {verified_count}")
    print(f"  Changed: {changed_count}")
    print(f"  Errors: {error_count}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/3600:.1f}h)")
    print(f"  Output: {output_path}")

    del vlm
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Label DROID failure videos")
    parser.add_argument("--phase", type=str, choices=["1", "2", "3", "both"], default="both",
                        help="Which phase to run")
    parser.add_argument("--input-dir", type=str,
                        default="/var/scratch/pkarageo/droid_failures_raw",
                        help="Raw download directory (for Phase 1)")
    parser.add_argument("--video-dir", type=str,
                        default="/home/pkarageorgis/droid_failures/videos",
                        help="Video directory (for Phase 2 & 3)")
    parser.add_argument("--output-dir", type=str,
                        default="/var/scratch/pkarageo/droid_failures_labeled",
                        help="Output directory for labels")
    parser.add_argument("--relabeled-path", type=str,
                        default="/home/pkarageorgis/droid_failures/qwen3_relabeled.jsonl",
                        help="Path to relabeled JSONL (for Phase 3)")
    parser.add_argument("--phase3-output", type=str,
                        default="/home/pkarageorgis/droid_failures/qwen3_5_verified.jsonl",
                        help="Output path for Phase 3 verified labels")
    parser.add_argument("--text-device", type=str, default="cuda:0",
                        help="Device for text model (Phase 1)")
    parser.add_argument("--vlm-device", type=str, default="cuda:1",
                        help="Device for VLM (Phase 2)")
    parser.add_argument("--max-frames", type=int, default=32,
                        help="Max frames per video for VLM")
    args = parser.parse_args()

    if args.phase in ("1", "both"):
        run_phase1(args.input_dir, args.output_dir, args.text_device)

    if args.phase in ("2", "both"):
        run_phase2(args.video_dir, args.output_dir, args.vlm_device, args.max_frames)

    if args.phase == "3":
        run_phase3(args.relabeled_path, args.video_dir, args.phase3_output)


if __name__ == "__main__":
    main()
