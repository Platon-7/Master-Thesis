#!/usr/bin/env python3
"""
Keyframe-based scoring: feed individual frames to a VLM, get per-frame descriptions,
then use an LLM to grade each frame's progress toward the task.

Usage:
    python score_keyframes.py --keyframes-dir /path/to/keyframes
"""

import os
import json
import re
import time
import argparse
import gc
import base64
import torch
from pathlib import Path

# ============================================================
# Paths (override via environment variables)
# ============================================================

RELABELED_PATH = os.environ.get(
    "RELABELED_PATH",
    "/home/pkarageorgis/DAS-5/Master-Thesis/Robo-Reward-FPS/droid_failures/qwen3_relabeled.jsonl",
)
KEYFRAMES_DIR = os.environ.get(
    "KEYFRAMES_DIR",
    "/home/pkarageorgis/DAS-5/Master-Thesis/Robo-Reward-FPS/droid_failures/keyframes",
)
OUTPUT_PATH = os.environ.get(
    "OUTPUT_PATH",
    "/home/pkarageorgis/DAS-5/Master-Thesis/Robo-Reward-FPS/droid_failures/scored_keyframes.jsonl",
)

VLM_MODEL_ID = os.environ.get("VLM_MODEL_ID", "nvidia/Cosmos-Reason2-8B")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "Qwen/Qwen3-32B")

# ============================================================
# Task description cleanup
# ============================================================

def clean_task_description(task: str) -> str:
    """Clean a DROID generic_task string: remove examples, fix grammar."""
    # Remove parenthetical examples: "(ex: towel on hook, clothes over chair)"
    task = re.sub(r"\s*\(ex:.*?\)", "", task)
    task = re.sub(r"\s*\(e\.g\..*?\)", "", task)
    # Strip trailing/leading whitespace and periods, then add final period
    task = task.strip().rstrip(".")
    # Capitalize first letter
    if task:
        task = task[0].upper() + task[1:]
    # Fix common issues
    task = re.sub(r"\s+", " ", task)  # double spaces
    task = task + "."
    return task


# ============================================================
# Prompts
# ============================================================

VLM_FRAME_PROMPT = """Task: {task}

Look at this image and answer each question in 1 sentence. Do not guess intentions or say what the robot is "about to do".

1. List the main objects relevant to the task and their state (up to 5 objects). Count multiples. (e.g., "3 cups upright on table, 1 cup in gripper", "cloth folded in half on table", "lid off, next to blender")
2. What is the gripper doing RIGHT NOW? (e.g., "gripping the red cup", "empty and open", "closed around the cloth edge", "not visible")
3. Compared to the task goal, what has been accomplished and what has not? (e.g., "1 of 4 cups stacked, 3 remain", "cloth lifted but not folded", "towel is on the rack")"""

# VLM prompt for multi-object tasks (instructs VLM to count objects and track collection progress)
VLM_FRAME_PROMPT_MULTI = """Task: {task}

This task most likely involves manipulating MULTIPLE objects. Focus on tracking overall progress across all of them.

1. List ALL relevant objects and their current state. If there are multiple instances of the same object, count them explicitly — both those already processed and those remaining. (e.g., "3 cups stacked, 2 still on table", "1 nail polish in purse, 3 remaining on table", "cloth pieces: 2 in box, 4 on table")
2. What is the gripper doing RIGHT NOW? (e.g., "gripping the red cup", "empty and open", "not visible")
3. How many objects have been successfully moved/processed toward the goal vs how many remain? Give a count if possible."""

# Pre-classified multi-object task phrases (substring match against cleaned task description).
# Rule: a task is multi-object if it requires picking up and moving MULTIPLE distinct objects.
# Single-object tasks — even complex ones (fold cloth, erase whiteboard, pour coffee,
# zip jacket, hang towel) — are NOT in this list.
MULTI_OBJECT_PHRASES = [
    # Explicit collection/sorting operations
    "clean up",            # clean up the table — many objects
    "clear objects",       # clear objects from the table
    "clear the clutter",   # clutter implies many objects
    "sort ",               # sort disks into cups by color
    "fill box with",       # fill box with cloth pieces
    "cloth pieces",        # multiple pieces
    "put objects",         # put objects into the bag
    "put stuff",           # put stuff in/out of bag
    "put small stuff",     # put small stuff into drawer
    "put ropes",           # put ropes in box (ropes = plural)
    "put each rope",       # put each rope in one by one
    "move items",          # move items out of bin (items = plural)
    "move the pods",       # move pods from one container to another
    "move rubber bands",   # rubber bands in/out of bag
    "move two objects",    # move two objects into container
    "move three",          # move three red discs
    "spread out objects",  # "unclump and spread out objects" (not "fold, spread out, or clump object")
    "unclump",             # unclump objects
    "clump objects",       # clump objects (cloth, rope, plush toy) — multi-object
    "nail polish",         # multiple nail polishes (verified from video)
    "place disks",         # place disks in bowl (disks = plural)
    "place ropes",         # place ropes in basket
    "place items",         # place items in box
    "place objects",       # place objects in drawer/cabinet
    "place all ",          # place all 5 circles
    "crockery",            # crockery = multiple dishes
    "wrap pen",            # pen + towel + container — sequential multi-object
    "unwrap objects",      # unwrap objects from cloth — multiple objects
    "pen in cloth",        # place pen in cloth... and put in box
    # Stacking/unstacking multiple objects
    "stack cups",          # stack cups (plural)
    "stack the cups",      # stack the cups with same styles
    "stack the cup with",  # variant spelling
    "stack blocks",        # stack blocks (plural)
    "stack dishes",        # stack dishes (plural)
    "stack bowls",         # stack bowls (plural)
    "stack pieces",        # stack pieces on top of one another
    "stack objects",       # stack objects on top of each other
    "stack or unstack",    # stack or unstack cups/objects/lego
    "stack and unstack",   # stack and unstack objects
    "stack / unstack",     # variant
    "stack/unstack",       # variant
    "stacking ",           # stacking objects, stacking bowls/cups
    "stack 2 ",            # stack 2 bowls
    "stack 3 ",            # stack 3 cups
    "stack 4",             # stack 4 cups / 4cups
    "stack two ",          # stack two cups/bowls
    "stack three ",        # stack three cups
    "stack cup with",      # stack cup with 2 to cup with 3
    "stack tupperware",    # stack tupperware (multiple pieces)
    # Swap/rearrange operations (multiple objects)
    "swap objects",        # swap objects between locations
    "rearrange",           # rearrange items/objects/pillows
    "arrange objects",     # arrange objects in a line
    "sweep objects",       # sweep objects into drawer
    # Bagging/container operations with multiple objects
    "bagging",             # bagging/unbagging objects
    "put the items",       # put the items in the box
    "put tea bags",        # put tea bags into tea box
    "put laundry",         # put laundry in basket
    "put cables",          # put cables into storage
    "take everything",     # take everything out and put back
    "take clothes",        # take clothes and objects out of basket
    "take items out of box and then", # take items out and put back
    "take toys",           # take toys out of the toy bag
    "take cloth pieces",   # take cloth pieces out of box/container
    # Explicit multi-object tasks (verified from frames)
    "put 3 ",              # put 3 stuff into bag (multiple toys)
    "hang bag and towel",  # bag + towel = 2 objects
    "pack the bottle in the box with bubble wrap", # bottle + wrap + box
    "smaller ball in cup and bigger ball", # 2 balls, 2 targets
    "red and green block", # 2 blocks
    "slot pieces",         # multiple track pieces
    "open a drawer and take some items", # open + take multiple items
    "open the box and take some tissues", # open + take multiple
    "unload dishwasher",   # multiple silverware items
    "load dishwasher",     # multiple items
    "serve each cup",      # multiple cups + forks
    "mess up the office",  # multiple objects
    "remove all objects",  # remove all objects
    "add objects",         # add objects into container
    "get objects",         # get objects from container
    "open/close cabinet",  # open cabinet + place/pick objects
    "pick up cloth, wipe all objects", # cloth + multiple objects
    "cups and place pen",  # stack cups + place pen
    "sort food",           # sort food boxes
    "circular tokens",     # pour circular tokens into bowl (multiple tokens)
    "reaarrange",          # typo variant of rearrange
    "cabineet",            # typo variant of cabinet (open/close cabineet)
    "insert 2 ",           # insert 2 orange donut disks
]


def is_multi_step_task(task_description):
    """Return True if the task involves manipulating multiple distinct objects."""
    t = task_description.lower()
    return any(phrase in t for phrase in MULTI_OBJECT_PHRASES)


# Step 0: Decompose task into ordered sub-steps (only for multi-step tasks)
TASK_DECOMPOSE_PROMPT = """Break the following robot manipulation task into 3-6 ordered sub-steps that a robot must complete sequentially. Each sub-step should be a concrete, observable action.

Task: {task}

Rules:
- List exactly the physical sub-steps in order.
- Each sub-step must be independently verifiable from a camera image.
- Do NOT include "start" or "finish" as sub-steps.
- Keep each sub-step to one sentence.

Reply with a numbered list and nothing else. Example for "stack the red cup on the blue cup":
1. Move gripper toward the red cup.
2. Grasp the red cup.
3. Lift the red cup off the table.
4. Move the red cup above the blue cup.
5. Lower the red cup onto the blue cup.
6. Release the red cup."""

# Prompt for SIMPLE tasks (v1 — the best-performing prompt for single-step tasks)
LLM_GRADE_PROMPT_SIMPLE = """You are grading a robot's cumulative progress toward completing a task. You will see descriptions of keyframes from a failed episode.

Task: {task}

This is frame {frame_num} of {total_frames}.{prev_score_text}

{previous_context}Current frame (frame {frame_num}, {current_timestamp}s):
{current_description}

Score the robot's cumulative progress AS OF THIS FRAME from 1 to 4:
1 - No progress: The robot never interacted with the correct task-relevant object. Moving, hovering, or grasping the WRONG object counts as 1. If the task is "stack cups" but the robot only touches a bottle, that is 1.
2 - Early progress: The robot approached or contacted the correct object but did not advance beyond that (e.g., moved toward the cup but didn't grasp, or grasped it but immediately dropped it).
3 - Partial progress: The robot completed at least one meaningful sub-step of the task (e.g., grasped the correct object AND moved it toward the goal, but didn't finish).
4 - Near completion: The robot completed MOST sub-steps of the task but failed at the very end (e.g., grasped + moved + nearly placed but dropped; lid was almost on but slid off).

How to handle ambiguity:
- Frame 1 MUST be scored 1. The robot cannot have completed any meaningful action by the very first keyframe. If the description suggests otherwise, it is a VLM hallucination — score 1.
- Frame 2 can be scored at most 2. If the description suggests score 3 or 4, it is almost certainly a VLM hallucination — score 1 or 2.
- Your score should be >= your previous score UNLESS there is clear evidence that progress was REVERSED (the robot actively undid a completed step — e.g., it had the pen in the gripper but now the pen is back on the table, farther from the goal).
- Do NOT lower the score just because the current description is less detailed than a previous one. VLM descriptions can be inconsistent, and objects may be occluded by the robot arm.
- If in doubt, keep your previous score.

All episodes are failures — the task was NOT completed, so a score of 5 is not possible.

Reply with exactly one line: ANSWER: <score> because <one sentence reason>"""

# Prompt for MULTI-STEP tasks (v6 — relaxed thresholds to reduce negative bias)
LLM_GRADE_PROMPT_MULTI = """You are grading a robot's cumulative progress toward completing a multi-step task. You will see descriptions of keyframes from a failed episode.

Task: {task}

The task consists of these ordered sub-steps:
{substeps}

This is frame {frame_num} of {total_frames}.{prev_score_text}

{previous_context}Current frame (frame {frame_num}, {current_timestamp}s):
{current_description}

This is a MULTI-STEP task. Score based on what fraction of the sub-steps above have been completed:
1 - No progress: The robot never interacted with the correct task-relevant object(s).
2 - Early progress: The robot completed only the very first sub-step(s) — e.g., approached and grasped the correct object but made no further advancement. Grasping alone is a 2, not a 3.
3 - Partial progress: The robot completed roughly ONE THIRD of the sub-steps and made meaningful advancement beyond just grasping. For a 6-step task, completing ~2 steps earns a 3. Grasping + clearly moving the object toward the goal counts as a 3.
4 - Near completion: The robot completed roughly TWO THIRDS or more of the sub-steps. Only the last step(s) remain. For a 6-step task, completing 4+ steps earns a 4.

How to handle ambiguity:
- Frame 1 MUST be scored 1. The robot cannot have completed any sub-step by the very first keyframe. If the description suggests otherwise, it is a VLM hallucination — score 1.
- Frame 2 can be scored at most 2. If the description suggests score 3 or 4, it is almost certainly a VLM hallucination — score 1 or 2.
- Your score should be >= your previous score UNLESS there is clear evidence that progress was REVERSED.
- Do NOT lower the score just because the current description is less detailed than a previous one. VLM descriptions can be inconsistent, and objects may be occluded by the robot arm.
- If in doubt, keep your previous score.

All episodes are failures — the task was NOT completed, so a score of 5 is not possible.

Reply with exactly one line: ANSWER: <score> because <one sentence reason>"""

# Alias for backward compatibility — grade_frame picks the right one
LLM_GRADE_PROMPT = LLM_GRADE_PROMPT_SIMPLE

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
    """Load VLM via vLLM for batched image inference."""
    from vllm import LLM

    model_id = VLM_MODEL_ID
    if os.path.exists(model_id):
        model_id = find_snapshot(model_id)

    num_gpus = torch.cuda.device_count()
    print(f"Loading VLM: {model_id} (vLLM, {num_gpus} GPUs)")

    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        tensor_parallel_size=num_gpus,
        max_model_len=int(os.environ.get("VLM_MAX_MODEL_LEN", 16384)),
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        trust_remote_code=True,
        allowed_local_media_path="/",
        enable_prefix_caching=True,
    )
    print("VLM loaded (vLLM)")
    return llm


def load_llm():
    """Load LLM via vLLM (supports MoE models like Qwen3.5) or transformers fallback."""
    from vllm import LLM as vLLM
    from transformers import AutoTokenizer

    model_id = LLM_MODEL_ID
    if os.path.exists(model_id):
        model_id = find_snapshot(model_id)

    num_gpus = torch.cuda.device_count()
    print(f"Loading LLM: {model_id} (vLLM, {num_gpus} GPUs)")

    model = vLLM(
        model=model_id,
        dtype="bfloat16",
        tensor_parallel_size=num_gpus,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        trust_remote_code=True,
        enable_prefix_caching=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print("LLM loaded (vLLM)")
    return model, tokenizer


# ============================================================
# Inference
# ============================================================

def decompose_task(task, llm_model, llm_tokenizer):
    """Decompose a task into ordered sub-steps using the LLM."""
    from vllm import SamplingParams

    prompt = TASK_DECOMPOSE_PROMPT.format(task=task)
    messages = [{"role": "user", "content": prompt}]
    text = llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    params = SamplingParams(max_tokens=300, temperature=0)
    outputs = llm_model.generate([text], sampling_params=params)
    output = outputs[0].outputs[0].text

    return output.strip()


def grade_frame(task, substeps, frames_so_far, total_frames, llm_model, llm_tokenizer):
    """Stage 2: LLM grades progress for the current frame with context from previous frames.

    Uses LLM_GRADE_PROMPT_SIMPLE for simple tasks and LLM_GRADE_PROMPT_MULTI for multi-step tasks.
    """
    current = frames_so_far[-1]
    frame_num = len(frames_so_far)

    # Build previous frames context (include scores for graded frames)
    if len(frames_so_far) > 1:
        lines = []
        for i, fr in enumerate(frames_so_far[:-1]):
            score_str = f" [your score: {fr['score']}]" if fr.get('score') is not None else ""
            lines.append(f"Frame {i+1} of {total_frames} ({fr['timestamp']:.1f}s){score_str}:\n{fr['description']}")
        previous_context = "Previous frames:\n" + "\n\n".join(lines) + "\n\n"
    else:
        previous_context = ""

    # Tell the LLM what its previous score was
    prev_score = frames_so_far[-2].get('score') if len(frames_so_far) > 1 else None
    if prev_score is not None:
        prev_score_text = f" Your score for the previous frame was {prev_score}."
    else:
        prev_score_text = ""

    # Pick the right prompt based on task complexity
    if is_multi_step_task(task) and substeps:
        prompt = LLM_GRADE_PROMPT_MULTI.format(
            task=task,
            substeps=substeps,
            frame_num=frame_num,
            total_frames=total_frames,
            previous_context=previous_context,
            prev_score_text=prev_score_text,
            current_timestamp=f"{current['timestamp']:.1f}",
            current_description=current["description"],
        )
    else:
        prompt = LLM_GRADE_PROMPT_SIMPLE.format(
            task=task,
            frame_num=frame_num,
            total_frames=total_frames,
            previous_context=previous_context,
            prev_score_text=prev_score_text,
            current_timestamp=f"{current['timestamp']:.1f}",
            current_description=current["description"],
        )

    from vllm import SamplingParams

    messages = [{"role": "user", "content": prompt}]
    text = llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    params = SamplingParams(max_tokens=80, temperature=0)
    outputs = llm_model.generate([text], sampling_params=params)
    output = outputs[0].outputs[0].text

    score = None
    match = re.search(r"ANSWER:\s*(\d)", output)
    if match:
        score = int(match.group(1))
        if score < 1 or score > 4:
            score = None

    return score, output


def describe_frames_batch(work_items, vlm, force_simple_prompt=False):
    """Stage 1: VLM describes individual frames in batch.

    work_items: list of (ep_id, task, frame_path, frame_idx, timestamp)
    Returns: list of (ep_id, frame_idx, timestamp, description)
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(max_tokens=256, temperature=0)

    conversations = []
    for ep_id, task, frame_path, frame_idx, timestamp in work_items:
        if force_simple_prompt:
            vlm_prompt_template = VLM_FRAME_PROMPT
        else:
            vlm_prompt_template = VLM_FRAME_PROMPT_MULTI if is_multi_step_task(task) else VLM_FRAME_PROMPT
        prompt = vlm_prompt_template.format(task=task)
        messages = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"file://{frame_path}"}},
                {"type": "text", "text": prompt},
            ]},
        ]
        conversations.append(messages)

    chat_kwargs = {}
    if any(k in VLM_MODEL_ID.lower() for k in ["qwen", "glm"]):
        chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

    outputs = vlm.chat(
        messages=conversations,
        sampling_params=sampling_params,
        **chat_kwargs,
    )

    results = []
    for (ep_id, task, frame_path, frame_idx, timestamp), output in zip(work_items, outputs):
        raw_text = output.outputs[0].text
        # Strip leaked thinking/special tokens
        description = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
        description = re.sub(r"<think>.*", "", description, flags=re.DOTALL)
        description = re.sub(r"</think>", "", description)
        description = re.sub(r"<\|[^|]*\|>", "", description)
        description = description.strip()
        results.append((ep_id, frame_idx, timestamp, description))

    return results


def describe_frames_claude(work_items, force_simple_prompt=False):
    """Stage 1 via Claude API (vision). No GPU needed.

    work_items: list of (ep_id, task, frame_path, frame_idx, timestamp)
    Returns: list of (ep_id, frame_idx, timestamp, description)
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY env var not set")

    client = anthropic.Anthropic(api_key=api_key)
    model = VLM_MODEL_ID  # e.g. "claude-sonnet-4-6"

    results = []
    for i, (ep_id, task, frame_path, frame_idx, timestamp) in enumerate(work_items):
        if force_simple_prompt:
            vlm_prompt_template = VLM_FRAME_PROMPT
        else:
            vlm_prompt_template = VLM_FRAME_PROMPT_MULTI if is_multi_step_task(task) else VLM_FRAME_PROMPT
        prompt_text = vlm_prompt_template.format(task=task)

        with open(frame_path, "rb") as f:
            img_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

        message = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                    {"type": "text", "text": prompt_text},
                ],
            }],
        )
        description = message.content[0].text.strip()
        results.append((ep_id, frame_idx, timestamp, description))

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Claude VLM: {i+1}/{len(work_items)} frames described")
        time.sleep(0.1)  # light rate-limit buffer

    return results


# ============================================================
# Main pipeline
# ============================================================

def extract_episode_id(dir_name: str) -> str:
    """Extract the plain episode ID from a keyframe directory name.

    Keyframe dirs are named {ep_id}__{task_slug} where ep_id ends with _YYYY.
    Episode IDs themselves can contain __ (e.g. Dec__5 for single-digit days).
    """
    m = re.match(r'^(.+_\d{4})__(.*)', dir_name)
    if m:
        return m.group(1)
    return dir_name  # fallback: return as-is


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframes-dir", type=str, default=KEYFRAMES_DIR)
    parser.add_argument("--episodes-file", type=str, default=None)
    parser.add_argument("--videos-dir", type=str, default=None,
                        help="If provided, extract keyframes from videos in this directory for episodes missing from --keyframes-dir.")
    parser.add_argument("--descriptions-file", type=str, default=None,
                        help="Skip Stage 1; load pre-existing frame descriptions from this JSONL file and run Stage 2 only.")
    parser.add_argument("--skip-grading", action="store_true",
                        help="Skip Stage 2 (LLM grading), only run VLM descriptions.")
    parser.add_argument("--force-simple-prompt", action="store_true",
                        help="Force VLM to use the simple prompt for all episodes (ignore multi-step classification).")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="Which shard to process (0-indexed).")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards to split the episode list into.")
    args = parser.parse_args()

    keyframes_dir = args.keyframes_dir

    print("=" * 60)
    print("Keyframe scoring: per-frame VLM description + LLM grading")
    print(f"  VLM: {VLM_MODEL_ID}")
    print(f"  LLM: {LLM_MODEL_ID}")
    print(f"  Keyframes: {keyframes_dir}")
    print(f"  Output: {OUTPUT_PATH}")
    if args.descriptions_file:
        print(f"  Descriptions file: {args.descriptions_file} (skipping Stage 1)")
    if args.force_simple_prompt:
        print("  VLM prompt: SIMPLE (forced, ignoring multi-step classification)")
    else:
        print("  VLM prompt: AUTO (multi-step prompt for multi-step tasks)")
    if args.num_shards > 1:
        print(f"  Shard: {args.shard_id + 1} / {args.num_shards}")
    print("=" * 60)

    # Load already-completed episodes for resume support
    already_done = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    already_done.add(d["episode_id"])
                except Exception:
                    pass
        if already_done:
            print(f"Resuming: {len(already_done)} episodes already in {OUTPUT_PATH}, skipping them.")

    # Load task descriptions (use original DROID generic_task)
    tasks = {}
    with open(RELABELED_PATH) as f:
        for line in f:
            d = json.loads(line.strip())
            ep_id = d["episode"]
            task = d.get("generic_task", d.get("new_label", d.get("old_label", "")))
            tasks[ep_id] = task

    # Clean task descriptions: remove examples in parentheses, fix grammar
    tasks = {ep: clean_task_description(t) for ep, t in tasks.items()}

    # ==========================================
    # STAGE 1: VLM describes each frame
    # ==========================================
    if args.descriptions_file:
        # Load pre-existing descriptions (skip VLM)
        print(f"\nLoading pre-existing descriptions from {args.descriptions_file}")
        episode_frames = {}
        ep_dirs = []
        with open(args.descriptions_file) as f:
            for line in f:
                d = json.loads(line.strip())
                ep_id = d["episode_id"]
                ep_dirs.append(ep_id)
                episode_frames[ep_id] = d["frames"]
                if ep_id not in tasks:
                    tasks[ep_id] = d.get("task", "unknown task")
        # Skip already-done episodes (resume)
        if already_done:
            ep_dirs = [ep for ep in ep_dirs if ep not in already_done]
            episode_frames = {ep: episode_frames[ep] for ep in ep_dirs if ep in episode_frames}
        print(f"Episodes to process: {len(ep_dirs)} (skipped {len(already_done)} already done)")
    else:
        # Find episodes with keyframes (dirs named {ep_id}__{task_slug})
        kf_dirs = sorted([
            d for d in os.listdir(keyframes_dir)
            if os.path.isdir(os.path.join(keyframes_dir, d))
        ])
        # Build mapping: plain_ep_id -> dir_name
        ep_id_to_dir = {extract_episode_id(d): d for d in kf_dirs}

        if args.episodes_file:
            with open(args.episodes_file) as f:
                allowed = set(line.strip() for line in f if line.strip())
        elif args.videos_dir:
            # Derive episode IDs from video filenames
            video_eps = set()
            for fname in os.listdir(args.videos_dir):
                for suffix in ["_ext1.mp4", "_ext2.mp4", ".mp4"]:
                    if fname.endswith(suffix):
                        video_eps.add(fname[: -len(suffix)])
                        break
            allowed = video_eps
        else:
            allowed = set(ep_id_to_dir.keys())

        # If videos-dir provided, extract keyframes for episodes not yet in keyframes-dir
        if args.videos_dir:
            import cv2
            missing = [ep for ep in allowed if ep not in ep_id_to_dir]
            if missing:
                print(f"Extracting keyframes for {len(missing)} episodes from {args.videos_dir}")
            for ep in missing:
                task_str = tasks.get(ep, "unknown_task")
                task_slug = re.sub(r'[^a-zA-Z0-9]+', '_', task_str).strip('_')[:60]
                dir_name = f"{ep}__{task_slug}"
                ep_kf_dir = os.path.join(keyframes_dir, dir_name)
                os.makedirs(ep_kf_dir, exist_ok=True)

                # Find video (prefer ext1)
                video_path = None
                for suffix in [f"{ep}_ext1.mp4", f"{ep}_ext2.mp4", f"{ep}.mp4"]:
                    candidate = os.path.join(args.videos_dir, suffix)
                    if os.path.exists(candidate):
                        video_path = candidate
                        break
                if video_path is None:
                    print(f"  WARNING: no video found for {ep}, skipping")
                    continue

                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                n_kf = 8
                indices = [int(i * (total_frames - 1) / (n_kf - 1)) for i in range(n_kf)]
                saved = 0
                for i, idx in enumerate(indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    ts = idx / fps
                    out_path = os.path.join(ep_kf_dir, f"frame_{i}_{ts:.2f}s.jpg")
                    cv2.imwrite(out_path, frame)
                    saved += 1
                cap.release()
                ep_id_to_dir[ep] = dir_name
                print(f"  Extracted {saved} frames for {ep} -> {dir_name}")

        # Filter to requested episodes
        ep_ids = [ep for ep in sorted(allowed) if ep in ep_id_to_dir]

        # Apply sharding
        if args.num_shards > 1:
            ep_ids = [ep for i, ep in enumerate(ep_ids) if i % args.num_shards == args.shard_id]
            print(f"Shard {args.shard_id}/{args.num_shards}: {len(ep_ids)} episodes")

        # Skip already-done episodes (resume)
        ep_ids = [ep for ep in ep_ids if ep not in already_done]

        # ep_dirs is the list of episode IDs (plain), not dir names — keep for downstream
        ep_dirs = ep_ids

        print(f"Episodes to process: {len(ep_dirs)} (skipped {len(already_done)} already done)")

        # Build work list: (ep_id, task, frame_path, frame_idx, timestamp)
        work_items = []
        for ep_id in ep_dirs:
            task = tasks.get(ep_id, "unknown task")
            dir_name = ep_id_to_dir[ep_id]
            ep_path = os.path.join(keyframes_dir, dir_name)
            frames = sorted(os.listdir(ep_path))
            for fname in frames:
                if not fname.endswith('.jpg'):
                    continue
                # Parse frame_N_Ts.jpg
                parts = fname.replace('.jpg', '').split('_')
                frame_idx = int(parts[1])
                timestamp = float(parts[2].replace('s', ''))
                frame_path = os.path.join(ep_path, fname)
                work_items.append((ep_id, task, frame_path, frame_idx, timestamp))

        print(f"Total frames to describe: {len(work_items)}")

        if not work_items:
            print("Nothing to do!")
            return

        print(f"\n{'=' * 60}")
        print(f"Stage 1: VLM describing {len(work_items)} frames...")
        print(f"{'=' * 60}")

        t0 = time.time()
        if VLM_MODEL_ID.startswith("claude-"):
            print(f"Using Claude API for VLM (no GPU needed)")
            all_results = describe_frames_claude(work_items, force_simple_prompt=args.force_simple_prompt)
        else:
            vlm = load_vlm()
            all_results = describe_frames_batch(work_items, vlm, force_simple_prompt=args.force_simple_prompt)
            del vlm
            gc.collect()
            torch.cuda.empty_cache()
            print("VLM unloaded.")
        elapsed = time.time() - t0
        print(f"Stage 1 done: {len(all_results)} frames described in {elapsed:.1f}s")

        # Group by episode
        episode_frames = {}
        for ep_id, frame_idx, timestamp, description in all_results:
            if ep_id not in episode_frames:
                episode_frames[ep_id] = []
            episode_frames[ep_id].append({
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "description": description,
            })

        # Sort frames within each episode
        for ep_id in episode_frames:
            episode_frames[ep_id].sort(key=lambda x: x["frame_idx"])

    # ==========================================
    # STAGE 2: LLM grades each frame
    # ==========================================
    if not args.skip_grading:
        print(f"\n{'=' * 60}")
        total_frames = sum(len(fs) for fs in episode_frames.values())
        print(f"Stage 2: LLM grading {total_frames} frames...")
        print(f"{'=' * 60}")

        llm_model, llm_tokenizer = load_llm()

        # Decompose only multi-step tasks into sub-steps
        unique_tasks = set(tasks.get(ep_id, "unknown task") for ep_id in ep_dirs)
        multi_tasks = [t for t in unique_tasks if is_multi_step_task(t)]
        simple_tasks = [t for t in unique_tasks if not is_multi_step_task(t)]
        print(f"  Tasks: {len(simple_tasks)} simple, {len(multi_tasks)} multi-step")
        print(f"  Decomposing {len(multi_tasks)} multi-step tasks...")
        task_substeps = {}
        for task_str in sorted(multi_tasks):
            substeps = decompose_task(task_str, llm_model, llm_tokenizer)
            task_substeps[task_str] = substeps
            print(f"    [MULTI] {task_str[:40]} => {substeps.count(chr(10))+1} steps")
        for task_str in sorted(simple_tasks):
            print(f"    [SIMPLE] {task_str[:40]}")

        t0 = time.time()
        graded = 0
        total_to_grade = sum(len(episode_frames.get(ep_id, [])) for ep_id in ep_dirs)
        with open(OUTPUT_PATH, "a") as fout:
            for ep_id in ep_dirs:
                task = tasks.get(ep_id, "unknown task")
                substeps = task_substeps.get(task, "1. Complete the task.")
                frames_list = episode_frames.get(ep_id, [])
                n_frames = len(frames_list)
                for i, fr in enumerate(frames_list):
                    frames_so_far = frames_list[:i + 1]
                    score, raw = grade_frame(
                        task, substeps, frames_so_far, n_frames,
                        llm_model, llm_tokenizer,
                    )
                    fr["score"] = score
                    fr["llm_raw"] = raw
                    graded += 1

                # Episode score = final frame score
                frames = episode_frames.get(ep_id, [])
                final_score = frames[-1]["score"] if frames else None

                # Write immediately (incremental, crash-safe)
                entry = {
                    "episode_id": ep_id,
                    "task": task,
                    "prompt_type": "multi" if is_multi_step_task(task) else "simple",
                    "substeps": task_substeps.get(task, ""),
                    "score": final_score,
                    "frames": frames,
                }
                fout.write(json.dumps(entry) + "\n")
                fout.flush()

                print(f"  [{graded}/{total_to_grade}] {ep_id[:50]} | final_score={final_score}")

        elapsed = time.time() - t0
        print(f"Stage 2 done: {graded} frames graded in {elapsed:.1f}s")

        del llm_model, llm_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    elif args.skip_grading:
        # VLM-only run: write descriptions incrementally
        if 'task_substeps' not in locals():
            task_substeps = {}
        with open(OUTPUT_PATH, "a") as fout:
            for ep_id in ep_dirs:
                task = tasks.get(ep_id, "unknown task")
                frames = episode_frames.get(ep_id, [])
                entry = {
                    "episode_id": ep_id,
                    "task": task,
                    "prompt_type": "multi" if is_multi_step_task(task) else "simple",
                    "substeps": task_substeps.get(task, ""),
                    "score": None,
                    "frames": frames,
                }
                fout.write(json.dumps(entry) + "\n")
                fout.flush()

    print(f"\nResults written to: {OUTPUT_PATH}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("Episode scores (final frame):")
    print(f"{'=' * 60}")
    for ep_id in ep_dirs:
        task = tasks.get(ep_id, "?")
        frames = episode_frames.get(ep_id, [])
        final_score = frames[-1].get("score", "?") if frames else "?"
        per_frame = [str(fr.get("score", "?")) for fr in frames]
        print(f"  {ep_id[:50]}  [{' → '.join(per_frame)}]  final={final_score}  task={task[:40]}")


if __name__ == "__main__":
    main()
