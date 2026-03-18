#!/usr/bin/env python3
"""
Relabel ALL ~5,500 DROID failure episodes using Qwen3-VL-8B-Instruct.

Each episode is shown to Qwen3 as a video, along with context about the
task. The strength of the suggestion varies by category:

  TIER 1 — SPECIFIC + BORDERLINE (2,088 episodes):
    Strong suggestion. These already have good labels from text cleanup.
    Qwen3 sees the video + the label and is told: "This label was verified
    by a human reviewer and is likely correct. Keep it unless the video
    clearly shows something different."

  TIER 2 — GENERIC_REVIEW verified PASS + FAIL (1,255 episodes):
    Regular suggestion. A reviewer proposed a label from visual inspection.
    Qwen3 sees the video + the suggestion: "A reviewer suggested this label.
    Use it if it matches what you see, otherwise write a better one."

  TIER 3 — Unlabeled remaining (789 episodes):
    Freestyle with format guidance. Qwen3 sees only the generic DROID label
    and must infer a specific task from the video, with examples of good
    task descriptions to match the style.

Output: A new JSONL file with one entry per episode containing the final
task description.

Usage:
    python relabel_all_qwen3.py --device cuda:0 [--start-from N]
"""

import os
import json
import argparse
import torch
from datetime import datetime


# ============================================================
# Paths
# ============================================================

MANIFEST_PATH = "/var/scratch/pkarageo/droid_failures_labeled/review_manifest.json"
LABELS_PATH = "/var/scratch/pkarageo/droid_failures_labeled/episode_labels.jsonl"
CLEANED_TASKS_PATH = "/var/scratch/pkarageo/droid_failures_labeled/cleaned_tasks.json"
VERIFY_PATH = "/var/scratch/pkarageo/droid_failures_labeled/verify_batch_20260309_160434.json"
VIDEO_DIR = "/var/scratch/pkarageo/droid_failures_processed"
OUTPUT_PATH = "/var/scratch/pkarageo/droid_failures_labeled/qwen3_relabeled.jsonl"

# ============================================================
# RoboReward-style example task descriptions
# ============================================================

ROBOREWARD_EXAMPLES = """\
Examples of task descriptions from the RoboReward dataset (use this style):
  "Pour"
  "Open door"
  "Close the laptop."
  "Pick up the red object from the table"
  "Folding a cloth"
  "Place dishes in the dish rack"
  "Open the Fridge"
  "Take the lid off the pot"
  "Close the drawer."
  "Rotate the faucet spout so it points forward."
  "Pick up the yellow cup and stack it on the other cup."
  "Place the apple fruit in the sink."
  "Push down the blue button to turn off the blue light"
  "Wipe the paper with the dry erase marker."
"""

# ============================================================
# Verb constraints (from fix_labels_qwen3.py)
# ============================================================

VERB_CONSTRAINTS = {
    "Fold, spread out, or clump object": {
        "verb": "Fold",
        "hint": 'The action is most likely "Fold". Identify WHAT is being folded.',
    },
    "Hang or unhang object": {
        "verb": "Hang",
        "hint": 'The action is most likely "Hang". Identify WHAT is being hung and WHERE.',
    },
    "Move lid on or off of container": {
        "verb": "Put",
        "hint": 'The action is most likely "Put the lid on ...". Identify the CONTAINER.',
    },
    "Press button": {
        "verb": "Press",
        "hint": 'The action is most likely "Press". Identify WHAT is being pressed.',
    },
    "Turn twistable object": {
        "verb": "Turn",
        "hint": 'The action is most likely "Turn". Identify WHAT is being turned.',
    },
    "Use cloth to clean something": {
        "verb": "Clean",
        "hint": 'The action is most likely "Clean ... with the cloth". Identify WHAT SURFACE is being cleaned.',
    },
    "Use cup to pour something granular": {
        "verb": "Pour",
        "hint": 'The action is most likely "Pour". Identify WHAT is being poured.',
    },
    "Use object to pick up something": {
        "verb": None,
        "hint": 'The action is "Pick up" or "Scoop". Identify the OBJECT and the TOOL.',
    },
    "Use object to stir something": {
        "verb": "Stir",
        "hint": 'The action is most likely "Stir". Identify WHAT is being stirred and IN what container.',
    },
    "Open or close hinged object": {
        "verb": None,
        "hint": 'The action is either "Open" or "Close". Look at the first and last frames to decide direction. Identify the specific object (door, microwave, oven, box, book, laptop, etc.).',
    },
    "Open or close slidable objects": {
        "verb": None,
        "hint": 'The action is either "Open" or "Close". Identify the specific object (drawer, sliding door, etc.).',
    },
    "Open or close curtain": {
        "verb": None,
        "hint": 'The action is either "Open" or "Close". Identify the specific object (curtain, blinds, shades, etc.).',
    },
}


def get_verb_hint(generic_task):
    """Get verb constraint hint for a generic DROID task."""
    prefix = generic_task.split("(")[0].strip()
    for key, val in VERB_CONSTRAINTS.items():
        if prefix.startswith(key):
            return val["hint"]
    return None


# ============================================================
# Prompts for each tier
# ============================================================

def build_tier1_prompt(current_label, generic_task):
    """SPECIFIC + BORDERLINE: strong suggestion."""
    return f"""\
You are looking at a video of a robot arm attempting a manipulation task. \
The robot may have failed — that is expected. Your job is to write a short, \
specific task description for what the robot was TRYING to do.

A human reviewer already wrote this task description after careful review:
  "{current_label}"

This label is likely correct. KEEP IT unless the video CLEARLY shows \
something different. If you keep it, output it exactly as written. If \
you change it, explain nothing — just output the corrected version.

{ROBOREWARD_EXAMPLES}

Rules:
- Start with a verb.
- Be specific: name the actual objects visible in the video.
- End with a period.
- Keep it under 15 words.
- Output ONLY the task description on a single line, nothing else.

Your answer:"""


def build_tier2_prompt(suggestion, generic_task):
    """GENERIC_REVIEW (passed or failed verification): regular suggestion."""
    verb_hint = get_verb_hint(generic_task) or ""
    verb_line = f"\nHint about the action: {verb_hint}" if verb_hint else ""

    return f"""\
You are looking at a video of a robot arm attempting a manipulation task. \
The robot may have failed — that is expected. Your job is to write a short, \
specific task description for what the robot was TRYING to do.

The original dataset label is generic: "{generic_task}"

A reviewer suggested this more specific description:
  "{suggestion}"

Use this suggestion if it matches what you see in the video. If the objects \
or action described don't match the video, write a better description based \
on what you actually see.{verb_line}

{ROBOREWARD_EXAMPLES}

Rules:
- Start with a verb.
- Be specific: name the actual objects visible in the video.
- End with a period.
- Keep it under 15 words.
- Output ONLY the task description on a single line, nothing else.

Your answer:"""


def build_tier3_prompt(generic_task):
    """Unlabeled: freestyle with format guidance."""
    verb_hint = get_verb_hint(generic_task) or ""
    verb_line = f"\nHint about the action: {verb_hint}" if verb_hint else ""

    return f"""\
You are looking at a video of a robot arm attempting a manipulation task. \
The robot may have failed — that is expected. Your job is to write a short, \
specific task description for what the robot was TRYING to do.

The original dataset label is generic: "{generic_task}"

You must replace it with a SPECIFIC description based on what you see in \
the video. Look at what objects are in the scene and what the robot arm \
reaches toward or interacts with.{verb_line}

{ROBOREWARD_EXAMPLES}

Rules:
- Start with a verb.
- Be specific: name the actual objects visible in the video.
- End with a period.
- Keep it under 15 words.
- Output ONLY the task description on a single line, nothing else.

Your answer:"""


# ============================================================
# Model loading
# ============================================================

def load_qwen3_vl(device="cuda:0"):
    """Load Qwen3-VL-8B-Instruct with 4-bit quantization."""
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


# ============================================================
# Inference
# ============================================================

def vlm_query(video_path, prompt, model, processor, process_vision_info):
    """Run Qwen3-VL on a video with a prompt. Returns generated text."""
    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "fps": 5.0},
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
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
        )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    del inputs, output_ids, generated_ids, image_inputs, video_inputs
    torch.cuda.empty_cache()

    # Clean: first line only, strip quotes, ensure period
    result = output_text.strip().split("\n")[0].strip().strip('"').strip("'").strip()
    if result and not result.endswith("."):
        result += "."

    return result


# ============================================================
# Build work list
# ============================================================

def build_work_list():
    """Build the complete list of episodes to process, with tier info."""
    # Load all data
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    labeled = {}
    with open(LABELS_PATH) as f:
        for line in f:
            d = json.loads(line)
            labeled[d['episode']] = d

    with open(CLEANED_TASKS_PATH) as f:
        cleaned_tasks = json.load(f)

    # Load verification results
    verified = {}
    if os.path.exists(VERIFY_PATH):
        with open(VERIFY_PATH) as f:
            for r in json.load(f):
                verified[r['group_id']] = r['verdict']

    # Build manifest lookup: episode -> group
    ep_to_group = {}
    for g in manifest:
        for ep in g['episodes']:
            ep_to_group[ep] = g

    manifest_eps = set(ep_to_group.keys())

    work = []

    # TIER 1: SPECIFIC + BORDERLINE (strong suggestion)
    for ep, d in labeled.items():
        if d['category'] in ('SPECIFIC', 'BORDERLINE_REVIEW'):
            generic_task = d.get('original_task', '')
            work.append({
                'episode': ep,
                'tier': 1,
                'current_label': d['new_label'],
                'generic_task': generic_task,
                'category': d['category'],
            })

    # TIER 2: GENERIC_REVIEW (regular suggestion — both manifest and non-manifest)
    for ep, d in labeled.items():
        if d['category'] == 'GENERIC_REVIEW':
            entry = {
                'episode': ep,
                'tier': 2,
                'current_label': d['new_label'],
                'generic_task': d.get('original_task', ''),
                'category': d['category'],
            }
            # Add manifest/verification info if available
            if ep in ep_to_group:
                g = ep_to_group[ep]
                entry['group_id'] = g['group_id']
                entry['generic_task'] = g['task']
                if g['group_id'] in verified:
                    entry['verify_verdict'] = verified[g['group_id']]
            work.append(entry)

    # TIER 3: Unlabeled manifest episodes (freestyle)
    for g in manifest:
        for ep in g['episodes']:
            if ep not in labeled:
                work.append({
                    'episode': ep,
                    'tier': 3,
                    'current_label': None,
                    'generic_task': g['task'],
                    'category': 'UNLABELED',
                    'group_id': g['group_id'],
                })

    # Sort: tier 1 first (fast — strong suggestion), then tier 2, then tier 3
    work.sort(key=lambda x: (x['tier'], x['episode']))

    return work


# ============================================================
# Main processing loop
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Relabel all DROID failures with Qwen3-VL")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Resume from this index (0-based)")
    args = parser.parse_args()

    # Build work list
    print("Building work list...")
    work = build_work_list()

    tier_counts = {1: 0, 2: 0, 3: 0}
    for w in work:
        tier_counts[w['tier']] += 1
    print(f"Total episodes: {len(work)}")
    print(f"  Tier 1 (SPECIFIC+BORDERLINE, strong suggestion): {tier_counts[1]}")
    print(f"  Tier 2 (GENERIC_REVIEW, regular suggestion):     {tier_counts[2]}")
    print(f"  Tier 3 (Unlabeled, freestyle):                   {tier_counts[3]}")

    # Load already-processed episodes (for resume support)
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
                    already_done.add(d['episode'])
                    good_lines.append(line)
                except json.JSONDecodeError:
                    print(f"WARNING: Corrupt line {line_num} in output file, removing it.")
        # Rewrite file without corrupt lines
        with open(OUTPUT_PATH, 'w') as f:
            for line in good_lines:
                f.write(line + '\n')
        print(f"Already processed (from previous run): {len(already_done)}")

    # Load model
    model, processor, pvi = load_qwen3_vl(args.device)

    # Process
    processed = 0
    skipped_done = 0
    skipped_no_video = 0
    start_time = datetime.now()

    for i, w in enumerate(work):
        if i < args.start_from:
            continue

        ep = w['episode']

        # Skip if already processed
        if ep in already_done:
            skipped_done += 1
            continue

        # Find video
        video_path = os.path.join(VIDEO_DIR, f"{ep}_ext1.mp4")
        if not os.path.exists(video_path):
            # Try ext2
            video_path = os.path.join(VIDEO_DIR, f"{ep}_ext2.mp4")
            if not os.path.exists(video_path):
                skipped_no_video += 1
                if skipped_no_video <= 10:
                    print(f"  [{i}] SKIP (no video): {ep}")
                continue

        # Build prompt based on tier
        if w['tier'] == 1:
            prompt = build_tier1_prompt(w['current_label'], w['generic_task'])
        elif w['tier'] == 2:
            prompt = build_tier2_prompt(w['current_label'], w['generic_task'])
        else:
            prompt = build_tier3_prompt(w['generic_task'])

        # Query VLM
        try:
            new_label = vlm_query(video_path, prompt, model, processor, pvi)
        except Exception as e:
            print(f"  [{i}] ERROR on {ep}: {e}")
            continue

        # Write result
        result = {
            'episode': ep,
            'tier': w['tier'],
            'category': w['category'],
            'generic_task': w['generic_task'],
            'old_label': w['current_label'],
            'new_label': new_label,
            'changed': w['current_label'] is None or new_label.lower() != w['current_label'].lower(),
        }
        if 'group_id' in w:
            result['group_id'] = w['group_id']
        if 'verify_verdict' in w:
            result['verify_verdict'] = w['verify_verdict']

        with open(OUTPUT_PATH, 'a') as f:
            f.write(json.dumps(result) + '\n')
            f.flush()
        already_done.add(ep)
        processed += 1

        # Log
        changed_str = " [CHANGED]" if result['changed'] else ""
        tier_str = f"T{w['tier']}"
        if processed <= 20 or processed % 100 == 0:
            print(f"  [{i}/{len(work)}] {tier_str} {ep[:40]}... -> {new_label}{changed_str}")

        # Progress every 250
        if processed % 250 == 0 and processed > 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = processed / elapsed * 3600
            remaining = len(work) - i
            eta_hours = remaining / rate if rate > 0 else 0
            print(f"\n  === PROGRESS: {processed} done, {skipped_done} skipped (already done), "
                  f"{skipped_no_video} skipped (no video) ===")
            print(f"  === Rate: {rate:.0f} eps/hour, ETA: {eta_hours:.1f} hours ===\n")

    # Final summary
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*70}")
    print(f"DONE")
    print(f"{'='*70}")
    print(f"Processed: {processed}")
    print(f"Skipped (already done): {skipped_done}")
    print(f"Skipped (no video): {skipped_no_video}")
    print(f"Time: {elapsed/3600:.1f} hours")
    print(f"Output: {OUTPUT_PATH}")

    # Quick stats on changes
    if os.path.exists(OUTPUT_PATH):
        changed = 0
        total = 0
        with open(OUTPUT_PATH) as f:
            for line in f:
                d = json.loads(line)
                total += 1
                if d.get('changed'):
                    changed += 1
        print(f"\nTotal in output: {total}")
        print(f"Changed from original: {changed} ({changed/total*100:.1f}%)")
        print(f"Kept original: {total - changed} ({(total-changed)/total*100:.1f}%)")


if __name__ == "__main__":
    main()
