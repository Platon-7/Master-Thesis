#!/usr/bin/env python3
"""
Two-stage scoring of DROID failure episodes:
  Stage 1 (VLM): Qwen3-VL-30B-A3B-Instruct describes the video given the task
  Stage 2 (LLM): Qwen3-14B grades 1-4 from the text description + rubric

Sequential loading:
  Stage 1: VLM bf16 (no quantization), describe ext1 videos, save to disk, unload
  Stage 2: LLM bf16 (no quantization), grade from descriptions

Only ext1 camera is described by the VLM — the description and score are
copied to ext2 and wrist since all cameras show the same episode.

Usage:
    python score_two_stage.py --sample 500
    python score_two_stage.py                  # full dataset
"""

import os
import json
import re
import time
import argparse
import gc
import torch
from pathlib import Path

# ============================================================
# Paths (override via environment variables for different clusters)
# ============================================================

RELABELED_PATH = os.environ.get(
    "RELABELED_PATH",
    "/home/pkarageorgis/droid_failures/qwen3_relabeled.jsonl",
)
VIDEO_DIR = os.environ.get(
    "VIDEO_DIR",
    "/home/pkarageorgis/droid_failures/videos",
)
OUTPUT_PATH = os.environ.get(
    "OUTPUT_PATH",
    "/home/pkarageorgis/droid_failures/scored_two_stage.jsonl",
)
DESC_PATH = os.environ.get(
    "DESC_PATH",
    "/home/pkarageorgis/droid_failures/scored_two_stage_descriptions.jsonl",
)

VLM_MODEL_ID = os.environ.get("VLM_MODEL_ID", "Qwen/Qwen3-VL-30B-A3B-Instruct")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "Qwen/Qwen3-14B")

# Camera used for VLM description (score copied to other cameras)
DESCRIBE_CAMERA = "ext1"
ALL_CAMERAS = ["ext1", "ext2", "wrist"]

# ============================================================
# Prompts
# ============================================================

VLM_DESCRIBE_PROMPT = """You are watching a video of a robot attempting a manipulation task.
The commanded task is: {task}

Describe what happens step by step in 3-5 sentences:
1. What are the relevant objects on the scene?
2. What does the robot do, in order? (e.g., moves toward X, grasps X, lifts X, moves toward Y, places X on Y)
3. Where does the robot end up and what is the state of the objects at the end?

Be specific and factual. Only describe what you observe."""

LLM_GRADE_PROMPT = """You are grading how much progress a robot made toward completing a task, based on a description of what happened in the video.

Task: {task}

Video description:
{description}

Score the robot's progress from 1 to 4. These are all failed episodes — the task was NOT completed successfully, so a score of 5 is not possible. Your job is to assess how far the robot got before failing.

Consider the ENTIRE sequence of actions, not just the final frame. Progress is cumulative: if the robot approaches the right object, grasps it, and moves it partway, that counts even if the final placement fails. However, if the robot undoes its own progress (e.g., picks up an object then puts it back and moves away), the score should reflect only the net progress.

Scoring guide:
1 - No progress: The robot did not perform any action relevant to the task (e.g., didn't move, moved to a completely wrong object, or only performed unrelated motions).
2 - Early progress: The robot began the task — it approached or made contact with the correct object, but did not advance further (e.g., moved toward the cup but didn't grasp it, or grasped it but immediately dropped it).
3 - Partial progress: The robot completed some meaningful steps — it interacted with the correct object and made progress, but failed at a critical step or missed a major requirement (e.g., grasped the cup and moved it but poured in the wrong location).
4 - Near completion: The robot performed most of the task correctly but fell short on a minor detail (e.g., poured water toward the glass but spilled some, or placed the object slightly off-target).

Reply with exactly one line in this format: ANSWER: <score> because <one sentence reason>"""

# ============================================================
# Model loading
# ============================================================

def find_snapshot(model_path):
    """Find the actual model snapshot directory (for HF cache layout)."""
    snapshots_dir = os.path.join(model_path, "snapshots")
    if os.path.exists(snapshots_dir):
        snaps = os.listdir(snapshots_dir)
        if snaps:
            return os.path.join(snapshots_dir, snaps[0])
    return model_path


def load_vlm():
    """Load VLM in bf16 (no quantization) to preserve spatial reasoning."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info

    model_id = VLM_MODEL_ID
    # If it looks like a local cache path, resolve snapshot
    if os.path.exists(model_id):
        model_id = find_snapshot(model_id)

    print(f"Loading VLM: {model_id} (bf16, device_map=auto)")

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print("VLM loaded (bf16)")
    return model, processor, process_vision_info


def load_llm():
    """Load LLM in bf16 (no quantization)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = LLM_MODEL_ID
    if os.path.exists(model_id):
        model_id = find_snapshot(model_id)

    print(f"Loading LLM: {model_id} (bf16, device_map=auto)")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("LLM loaded (bf16)")
    return model, tokenizer


# ============================================================
# Inference
# ============================================================

def describe_video(video_path, task, vlm_model, vlm_processor, process_vision_info):
    """Stage 1: VLM describes what happens in the video."""
    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "fps": 1.0},
            {"type": "text", "text": VLM_DESCRIBE_PROMPT.format(task=task)},
        ]},
    ]

    text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = vlm_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    first_device = next(vlm_model.parameters()).device
    inputs = inputs.to(first_device)

    with torch.no_grad():
        output_ids = vlm_model.generate(**inputs, max_new_tokens=512, do_sample=False)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        description = vlm_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return description


def grade_description(task, description, llm_model, llm_tokenizer):
    """Stage 2: LLM grades 1-4 based on the text description."""
    prompt = LLM_GRADE_PROMPT.format(task=task, description=description)

    messages = [{"role": "user", "content": prompt}]
    text = llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    first_device = next(llm_model.parameters()).device
    inputs = llm_tokenizer(text, return_tensors="pt").to(first_device)

    with torch.no_grad():
        output_ids = llm_model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
        )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        output = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    score = None
    match = re.search(r"ANSWER:\s*(\d)", output)
    if match:
        score = int(match.group(1))
        if score < 1 or score > 4:
            score = None

    return score, output


# ============================================================
# Main pipeline
# ============================================================

def load_episodes():
    """Load relabeled episodes and build work list."""
    episodes = {}
    with open(RELABELED_PATH) as f:
        for line in f:
            d = json.loads(line.strip())
            ep_id = d["episode"]
            task = d.get("new_label", d.get("old_label", ""))
            episodes[ep_id] = task
    return episodes


def load_completed():
    """Load already-scored episode IDs for resume."""
    done = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    done.add(d["episode_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def load_existing_descriptions():
    """Load already-generated descriptions for resume."""
    existing = {}
    if os.path.exists(DESC_PATH):
        with open(DESC_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    existing[d["episode_id"]] = d
                except (json.JSONDecodeError, KeyError):
                    continue
    return existing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0,
                        help="If >0, only process this many episodes (for testing)")
    args = parser.parse_args()

    print("=" * 60)
    print("Two-stage scoring: VLM (describe) + LLM (grade)")
    print(f"  VLM: {VLM_MODEL_ID}")
    print(f"  LLM: {LLM_MODEL_ID}")
    print(f"  Camera for description: {DESCRIBE_CAMERA}")
    print(f"  Score copied to: {ALL_CAMERAS}")
    print("=" * 60)

    # Load episodes
    episodes = load_episodes()
    print(f"Episodes with labels: {len(episodes)}")

    # Build work list (ext1 only for VLM)
    work = []
    missing = 0
    for ep_id, task in episodes.items():
        video_path = os.path.join(VIDEO_DIR, f"{ep_id}_{DESCRIBE_CAMERA}.mp4")
        if os.path.exists(video_path):
            work.append((ep_id, task, video_path))
        else:
            missing += 1

    print(f"Episodes with ext1 video: {len(work)}")
    print(f"Missing ext1 videos: {missing}")

    # Resume: check what's already scored
    done = load_completed()
    work = [w for w in work if w[0] not in done]
    print(f"Already scored: {len(done)}")
    print(f"Remaining: {len(work)}")

    # Sample mode
    if args.sample > 0 and len(work) > args.sample:
        import random
        random.seed(42)
        work = random.sample(work, args.sample)
        print(f"Sampled {args.sample} episodes for testing")

    if not work:
        print("Nothing to do!")
        return

    # Load existing descriptions (for resume between stages)
    existing_descs = load_existing_descriptions()
    print(f"Existing descriptions: {len(existing_descs)}")

    # ==========================================
    # STAGE 1: VLM describes ext1 videos
    # ==========================================
    work_needing_desc = [w for w in work if w[0] not in existing_descs]

    if work_needing_desc:
        print(f"\n{'=' * 60}")
        print(f"Stage 1: VLM describing {len(work_needing_desc)} episodes...")
        print(f"{'=' * 60}")

        vlm_model, vlm_processor, process_vision_info = load_vlm()

        t0 = time.time()
        described = 0
        errors_s1 = 0

        with open(DESC_PATH, "a") as fout:
            for ep_id, task, video_path in work_needing_desc:
                try:
                    description = describe_video(
                        video_path, task, vlm_model, vlm_processor, process_vision_info
                    )

                    entry = {
                        "episode_id": ep_id,
                        "task": task,
                        "description": description,
                    }

                    fout.write(json.dumps(entry) + "\n")
                    fout.flush()
                    existing_descs[ep_id] = entry
                    described += 1

                except Exception as e:
                    errors_s1 += 1
                    print(f"  S1 ERROR {ep_id}: {e}")
                    continue

                if described <= 10 or described % 50 == 0:
                    elapsed = time.time() - t0
                    rate = described / elapsed * 3600
                    remaining = len(work_needing_desc) - described - errors_s1
                    eta = remaining / (described / elapsed) if described > 0 else 0
                    print(f"  S1 [{described}/{len(work_needing_desc)}] "
                          f"rate={rate:.0f}/hr ETA={eta/3600:.1f}h errors={errors_s1}")

        elapsed = time.time() - t0
        print(f"Stage 1 done: {described} described, {errors_s1} errors, "
              f"{elapsed/3600:.1f}h")

        # Free VLM memory
        del vlm_model, vlm_processor
        gc.collect()
        torch.cuda.empty_cache()
        print("VLM unloaded, GPU memory freed.")
    else:
        print(f"\nAll {len(work)} descriptions already exist, skipping Stage 1.")

    # ==========================================
    # STAGE 2: LLM grades all descriptions
    # ==========================================
    work_needing_grade = [w for w in work if w[0] in existing_descs and w[0] not in done]

    if not work_needing_grade:
        print("Nothing to grade!")
        return

    print(f"\n{'=' * 60}")
    print(f"Stage 2: LLM grading {len(work_needing_grade)} episodes...")
    print(f"{'=' * 60}")

    llm_model, llm_tokenizer = load_llm()

    t0 = time.time()
    scored = 0
    errors_s2 = 0

    with open(OUTPUT_PATH, "a") as fout:
        for ep_id, task, video_path in work_needing_grade:
            desc_entry = existing_descs.get(ep_id)
            if not desc_entry:
                continue

            try:
                score, grade_output = grade_description(
                    task, desc_entry["description"], llm_model, llm_tokenizer
                )

                if score is None:
                    score = 1  # fallback

                # Write one result per camera, same description and score
                for cam in ALL_CAMERAS:
                    result = {
                        "video_key": f"{ep_id}_{cam}",
                        "episode_id": ep_id,
                        "camera": cam,
                        "task": task,
                        "description": desc_entry["description"],
                        "score": score,
                        "grade_output": grade_output,
                    }
                    fout.write(json.dumps(result) + "\n")

                fout.flush()
                scored += 1

            except Exception as e:
                errors_s2 += 1
                print(f"  S2 ERROR {ep_id}: {e}")
                continue

            if scored <= 10 or scored % 50 == 0:
                elapsed = time.time() - t0
                rate = scored / elapsed * 3600
                remaining = len(work_needing_grade) - scored - errors_s2
                eta = remaining / (scored / elapsed) if scored > 0 else 0
                print(f"  S2 [{scored}/{len(work_needing_grade)}] score={score} "
                      f"rate={rate:.0f}/hr ETA={eta/3600:.1f}h errors={errors_s2}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done. Scored: {scored} episodes ({scored * len(ALL_CAMERAS)} video entries)")
    print(f"Errors: {errors_s2}")
    print(f"Time: {elapsed/3600:.1f}h, Rate: {scored/elapsed*3600:.0f}/hr")
    print(f"Output: {OUTPUT_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
