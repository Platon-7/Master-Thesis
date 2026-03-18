#!/usr/bin/env python3
"""
Fix generic DROID task descriptions using Qwen3-VL-8B-Instruct.

The DROID dataset contains many episodes with vague task descriptions like
"Fold, spread out, or clump object (ex: cloth, charging cord, clothes)" or
"Open or close hinged object (ex: hinged door, microwave, oven, book)".
These are too ambiguous for a reward model — it can't judge task completion
if it doesn't know what the task actually is.

This script loads Qwen3-VL-8B-Instruct, shows it the video, and asks it to
produce a specific, concise task description based on what it sees. For
example, "Fold the paper." or "Close the blinds."

Usage:
    # Test mode: run on 6 validation samples and compare to human labels
    python fix_labels_qwen3.py --mode test --device cuda:0

    # Batch mode: process all remaining unlabeled manifest episodes
    python fix_labels_qwen3.py --mode batch --device cuda:0
"""

import os
import json
import re
import argparse
import torch
from datetime import datetime


# ============================================================
# Paths
# ============================================================

MANIFEST_PATH = "/var/scratch/pkarageo/droid_failures_labeled/review_manifest.json"
LABELS_PATH = "/var/scratch/pkarageo/droid_failures_labeled/episode_labels.jsonl"
VIDEO_DIR = "/var/scratch/pkarageo/droid_failures_processed"

# ============================================================
# Verb constraints per task type
# ============================================================
# When we know the verb from the generic label, we tell Qwen3 directly.
# For ambiguous cases (open/close), we tell it to decide from the video.

VERB_CONSTRAINTS = {
    "Fold, spread out, or clump object": {
        "verb": "Fold",
        "instruction": 'The action is "Fold". You only need to identify WHAT is being folded.',
        "template": "Fold the {object}.",
    },
    "Hang or unhang object": {
        "verb": "Hang",
        "instruction": 'The action is "Hang". You only need to identify WHAT is being hung and optionally WHERE.',
        "template": "Hang the {object}.",
    },
    "Move lid on or off of container": {
        "verb": "Put",
        "instruction": 'The action is "Put the lid on ...". You only need to identify the CONTAINER (pot, cup, jar, pan, bowl, etc.).',
        "template": "Put the lid on the {object}.",
    },
    "Press button": {
        "verb": "Press",
        "instruction": 'The action is "Press". You only need to identify WHAT is being pressed (light switch, microwave button, stove button, etc.).',
        "template": "Press the {object}.",
    },
    "Turn twistable object": {
        "verb": "Turn",
        "instruction": 'The action is "Turn". You only need to identify WHAT is being turned (faucet knob, stove knob, oven knob, door handle, lamp switch, etc.).',
        "template": "Turn the {object}.",
    },
    "Use cloth to clean something": {
        "verb": "Clean",
        "instruction": 'The action is "Clean ... with the cloth". You only need to identify WHAT SURFACE is being cleaned (table, counter, plate, sink, window, etc.).',
        "template": "Clean the {object} with the cloth.",
    },
    "Use cup to pour something granular": {
        "verb": "Pour",
        "instruction": 'The action is "Pour". You only need to identify WHAT is being poured and/or FROM/INTO what container.',
        "template": "Pour the {object} from the cup.",
    },
    "Use object to pick up something": {
        "verb": None,  # could be "Pick up" or "Scoop"
        "instruction": 'The action is either "Pick up ... with the ..." or "Scoop ... with the ...". Identify the OBJECT being picked up and the TOOL being used (spoon, spatula, fork, tongs, etc.).',
        "template": "{verb} the {object} with the {tool}.",
    },
    "Use object to stir something": {
        "verb": "Stir",
        "instruction": 'The action is "Stir". You only need to identify WHAT is being stirred and IN what container (food in the bowl, contents in the cup, pasta in the pot, etc.).',
        "template": "Stir the {object}.",
    },
    # Ambiguous verb cases — Qwen3 must decide open vs close
    "Open or close hinged object": {
        "verb": None,
        "instruction": 'The action is either "Open" or "Close". Look at the first and last frames to decide which direction the object moves. If the object (door, lid, microwave, oven, book, box, laptop, etc.) moves from a closed/shut state toward an open state, the action is "Open". If it moves from open toward closed, the action is "Close". Then identify the specific object.',
        "template": "{verb} the {object}.",
    },
    "Open or close slidable objects": {
        "verb": None,
        "instruction": 'The action is either "Open" or "Close". Look at the first and last frames. If a drawer, sliding door, or similar object is being pulled out/slid open, the action is "Open". If pushed in/slid shut, the action is "Close". Identify the specific object (drawer, sliding door, toaster, etc.).',
        "template": "{verb} the {object}.",
    },
    "Open or close curtain": {
        "verb": None,
        "instruction": 'The action is either "Open" or "Close". Look at the first and last frames. If the curtain/blinds/shades are being drawn shut or lowered, the action is "Close". If being pulled open or raised, the action is "Open". Identify the specific object (curtain, blinds, shades, etc.).',
        "template": "{verb} the {object}.",
    },
}


def get_verb_key(old_task):
    """Match a generic task string to a VERB_CONSTRAINTS key."""
    old_prefix = old_task.split("(")[0].strip()
    for key in VERB_CONSTRAINTS:
        if old_prefix.startswith(key):
            return key
    return None


# ============================================================
# Prompt builder
# ============================================================

def build_prompt(old_task):
    """Build a task-specific prompt using verb constraints when available."""
    verb_key = get_verb_key(old_task)
    constraint = VERB_CONSTRAINTS.get(verb_key) if verb_key else None

    if constraint:
        verb_section = constraint["instruction"]
        if constraint["verb"]:
            verb_note = f'\nThe verb is already known: "{constraint["verb"]}". Do NOT use a different verb.'
        else:
            verb_note = "\nYou must decide the correct verb from the video."
    else:
        verb_section = "Determine the action from the video."
        verb_note = ""

    return f"""\
You are looking at a video of a robot arm attempting a manipulation task. \
The robot may have FAILED — it may not have completed the task. Your job is \
to figure out what the robot was TRYING to do and write a short, specific \
task description.

The dataset provides a generic task label. You must replace it with a \
SPECIFIC description based on what you see in the video.

Generic label: {old_task}

STEP 1 — IDENTIFY THE ACTION:
{verb_section}{verb_note}

STEP 2 — IDENTIFY THE OBJECTS:
Look carefully at ALL frames in the video. Focus on:
- What objects are on the table/counter/surface?
- What does the robot arm reach toward, grasp, or touch?
- What changes between the first frame and the last frame?
Name objects specifically: "the towel", "the microwave", "the faucet knob", \
"the blue cloth", "the headphones", etc.

STEP 3 — WRITE THE TASK:
Combine the action and the object(s) into a short imperative command.
Rules:
- Start with a verb.
- Name the specific object(s) you identified.
- End with a period.
- Keep it under 10 words.
- Do NOT repeat the generic label.
- Do NOT include parenthetical examples.
- Do NOT explain your reasoning.

Examples of GOOD outputs:
  "Fold the towel."
  "Open the microwave."
  "Close the drawer."
  "Turn the faucet knob."
  "Clean the counter with the cloth."
  "Pour the rice from the cup."
  "Hang the headphones."
  "Press the light switch."
  "Close the blinds."
  "Put the lid on the pot."
  "Stir the food in the bowl."
  "Pick up the bread with the spatula."

Output ONLY the task command on a single line, nothing else.

Your answer:"""


# ============================================================
# 6 validation samples (human-labeled ground truth)
# ============================================================

TEST_SAMPLES = [
    {
        "group_id": "g424",
        "episode": "ILIAD_2023-07-29_Sat_Jul_29_14-25-15_2023",
        "old_task": "Hang or unhang object (ex: towel on hook, clothes over chair)",
        "human_label": "Hang the headphones.",
        "video": "ILIAD_2023-07-29_Sat_Jul_29_14-25-15_2023_ext1.mp4",
    },
    {
        "group_id": "g446",
        "episode": "TRI_2023-10-16_Mon_Oct_16_16-29-57_2023",
        "old_task": "Open or close curtain (ex: blanket on bed, shower curtain, shades)",
        "human_label": "Close the blinds.",
        "video": "TRI_2023-10-16_Mon_Oct_16_16-29-57_2023_ext1.mp4",
    },
    {
        "group_id": "g416",
        "episode": "TRI_2023-10-19_Thu_Oct_19_11-47-37_2023",
        "old_task": "Fold, spread out, or clump object (ex: cloth, charging cord, clothes)",
        "human_label": "Fold the paper.",
        "video": "TRI_2023-10-19_Thu_Oct_19_11-47-37_2023_ext1.mp4",
    },
    {
        "group_id": "g520",
        "episode": "ILIAD_2023-07-19_Wed_Jul_19_17-56-35_2023",
        "old_task": "Use cloth to clean something (ex: table, window, mirror, plate)",
        "human_label": "Clean the counter with the cloth.",
        "video": "ILIAD_2023-07-19_Wed_Jul_19_17-56-35_2023_ext1.mp4",
    },
    {
        "group_id": "g018",
        "episode": "TRI_2023-10-19_Thu_Oct_19_16-59-25_2023",
        "old_task": "Turn twistable object (ex: faucets, lamps, stove knobs)",
        "human_label": "Turn the faucet knob.",
        "video": "TRI_2023-10-19_Thu_Oct_19_16-59-25_2023_ext1.mp4",
    },
    {
        "group_id": "g037",
        "episode": "IPRL_2023-10-26_Thu_Oct_26_22-03-57_2023",
        "old_task": "Open or close hinged object (ex: hinged door, microwave, oven, book, dryer, toilet, box)",
        "human_label": "Open the microwave.",
        "video": "IPRL_2023-10-26_Thu_Oct_26_22-03-57_2023_ext1.mp4",
    },
]


# ============================================================
# Model loading (same pattern as test_relabel.py)
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

def fix_task_label(video_path, old_task, model, processor, process_vision_info):
    """Ask Qwen3-VL to produce a specific task description from a video."""
    prompt = build_prompt(old_task)

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
            do_sample=False,  # greedy for reproducibility
        )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    del inputs, output_ids, generated_ids, image_inputs, video_inputs
    torch.cuda.empty_cache()

    # Clean up: take only the first line, strip quotes, ensure period at end
    result = output_text.strip().split("\n")[0].strip().strip('"').strip("'").strip()
    if result and not result.endswith("."):
        result += "."

    return result


# ============================================================
# Test mode: compare Qwen3 outputs to human labels
# ============================================================

def run_test(model, processor, process_vision_info):
    """Run on 5 validation samples and compare to human labels."""
    print("\n" + "=" * 70)
    print("TEST MODE: Comparing Qwen3-VL output to human labels")
    print("=" * 70)

    results = []
    for sample in TEST_SAMPLES:
        video_path = os.path.join(VIDEO_DIR, sample["video"])
        if not os.path.exists(video_path):
            print(f"SKIP: {video_path} not found")
            continue

        print(f"\n--- {sample['group_id']} ({sample['episode'][:30]}...) ---")
        print(f"  Generic label: {sample['old_task']}")

        new_label = fix_task_label(video_path, sample["old_task"], model, processor, process_vision_info)

        print(f"  Qwen3 output:  {new_label}")
        print(f"  Human label:   {sample['human_label']}")

        match = new_label.lower().strip(".") == sample["human_label"].lower().strip(".")
        print(f"  Exact match:   {'YES' if match else 'NO'}")

        results.append({
            "group_id": sample["group_id"],
            "old_task": sample["old_task"],
            "human_label": sample["human_label"],
            "qwen3_label": new_label,
            "exact_match": match,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Group':<8} {'Generic Label':<35} {'Human':<30} {'Qwen3':<30} {'Match'}")
    print("-" * 140)
    for r in results:
        short_old = r["old_task"].split("(")[0].strip()[:33]
        print(f"{r['group_id']:<8} {short_old:<35} {r['human_label']:<30} {r['qwen3_label']:<30} {'YES' if r['exact_match'] else 'NO'}")

    exact = sum(1 for r in results if r["exact_match"])
    print(f"\nExact matches: {exact}/{len(results)}")

    # Save results
    out_path = os.path.join(
        os.path.dirname(LABELS_PATH),
        f"qwen3_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_path}")

    return results


# ============================================================
# Batch mode: process all remaining unlabeled manifest episodes
# ============================================================

def run_batch(model, processor, process_vision_info):
    """Process all unlabeled manifest groups and write labels."""
    print("\n" + "=" * 70)
    print("BATCH MODE: Processing all remaining unlabeled groups")
    print("=" * 70)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    labeled = set()
    with open(LABELS_PATH) as f:
        for line in f:
            labeled.add(json.loads(line)["episode"])

    # Collect unlabeled groups
    unlabeled_groups = []
    for g in manifest:
        if any(ep not in labeled for ep in g["episodes"]):
            unlabeled_groups.append(g)

    total_eps = sum(
        sum(1 for ep in g["episodes"] if ep not in labeled)
        for g in unlabeled_groups
    )
    print(f"Unlabeled groups: {len(unlabeled_groups)}")
    print(f"Unlabeled episodes: {total_eps}")

    processed = 0
    written = 0

    for gi, g in enumerate(unlabeled_groups):
        gid = g["group_id"]
        old_task = g["task"]

        # Use the representative episode's video for label inference
        video_path = g["video_path"]
        if not os.path.exists(video_path):
            # Try to find any episode video
            for ep in g["episodes"]:
                alt = os.path.join(VIDEO_DIR, f"{ep}_ext1.mp4")
                if os.path.exists(alt):
                    video_path = alt
                    break
            else:
                print(f"  [{gi+1}/{len(unlabeled_groups)}] {gid}: NO VIDEO FOUND, skipping")
                continue

        print(f"  [{gi+1}/{len(unlabeled_groups)}] {gid} ({g['count']} eps, {g['lab']}/{g['date']}): {old_task[:50]}...")

        new_label = fix_task_label(video_path, old_task, model, processor, process_vision_info)
        print(f"    -> {new_label}")

        # Write labels for all unlabeled episodes in this group
        with open(LABELS_PATH, "a") as f:
            for ep in g["episodes"]:
                if ep not in labeled:
                    entry = {
                        "episode": ep,
                        "original_task": old_task,
                        "new_label": new_label,
                        "category": "GENERIC_REVIEW",
                        "keep": True,
                        "failed": False,
                        "reasoning": f"Qwen3-VL auto-fix: {g['lab']}/{g['date']} subgroup",
                    }
                    f.write(json.dumps(entry) + "\n")
                    labeled.add(ep)
                    written += 1

        processed += 1

        # Progress update every 25 groups
        if (gi + 1) % 25 == 0:
            print(f"  --- Progress: {gi+1}/{len(unlabeled_groups)} groups, {written} episodes written ---")

    print(f"\nDone. Processed {processed} groups, wrote {written} episode labels.")
    print(f"Total labeled episodes now: {len(labeled)}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Fix generic DROID task labels with Qwen3-VL")
    parser.add_argument("--mode", choices=["test", "batch"], default="test",
                        help="'test' = 6 validation samples, 'batch' = all remaining")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    model, processor, pvi = load_qwen3_vl(args.device)

    if args.mode == "test":
        run_test(model, processor, pvi)
    else:
        run_batch(model, processor, pvi)


if __name__ == "__main__":
    main()
