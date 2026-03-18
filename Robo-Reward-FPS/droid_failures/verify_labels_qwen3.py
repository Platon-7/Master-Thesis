#!/usr/bin/env python3
"""
Verify GENERIC_REVIEW labels using Qwen3-VL-8B-Instruct.

For each labeled group, show Qwen3 the video and the assigned label,
and ask: "Does this video show [label]?" If Qwen3 says No, flag it.

This is a binary verification task — much easier and more reliable than
asking Qwen3 to generate labels from scratch.

Usage:
    # Test mode: verify 6 known samples
    python verify_labels_qwen3.py --mode test --device cuda:0

    # Batch mode: verify all 212 completed GENERIC_REVIEW groups
    python verify_labels_qwen3.py --mode batch --device cuda:0
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
RESULTS_DIR = "/var/scratch/pkarageo/droid_failures_labeled"

# ============================================================
# Verification prompt
# ============================================================

VERIFY_PROMPT = """\
You are looking at a video of a robot arm in a real-world environment \
(kitchen, lab, desk, etc.). The robot is ATTEMPTING a manipulation task \
but may have FAILED — it may not have completed the action successfully. \
That is expected. Your job is NOT to judge success. Your job is to verify \
whether the task description is plausible given the scene.

Task description: "{label}"

Answer this single question:

OBJECT CHECK: Is the object mentioned in the task description \
("{object_hint}") actually visible somewhere in the video frames? \
Look carefully at all frames. The object must be clearly present in the \
scene — on the table, counter, shelf, or wherever. It does NOT need to \
be successfully manipulated by the robot. The robot may fail, drop it, \
or barely touch it. That is fine. We only care whether the object EXISTS \
in the scene.

Examples:
- "Fold the towel." -> Is there a towel visible? (yes/no)
- "Open the microwave." -> Is there a microwave visible? (yes/no)
- "Turn the faucet knob." -> Is there a faucet with a knob visible? (yes/no)
- "Clean the counter with the cloth." -> Is there a counter visible? (yes/no)

Based on your answer:
- PASS: The object IS visible in the scene.
- FAIL: The object is NOT visible anywhere in the video.
- UNSURE: The video is too blurry/dark to tell.

Format your answer EXACTLY like this (2 lines):
OBJECT: [yes/no] [brief reason — what do you actually see?]
VERDICT: [PASS/FAIL/UNSURE]"""


def extract_object_hint(label):
    """Extract the main object from a label for the prompt."""
    # "Fold the towel." -> "towel"
    # "Open the microwave." -> "microwave"
    # "Clean the counter with the cloth." -> "counter"
    # "Put the lid on the pot." -> "pot"
    words = label.strip(".").split()
    if len(words) >= 3:
        # Skip verb + "the", take the next noun(s)
        # Handle "with the cloth" suffix
        core = label.strip(".")
        if " with the " in core:
            core = core.split(" with the ")[0]
        if " from the " in core:
            core = core.split(" from the ")[0]
        if " on the " in core:
            # "Put the lid on the pot" -> "pot"
            core = core.split(" on the ")[-1]
        elif " in the " in core:
            core = core.split(" in the ")[-1]
        else:
            # "Fold the towel" -> "towel"
            # "Open the microwave" -> "microwave"
            parts = core.split()
            if len(parts) >= 3:
                core = " ".join(parts[2:])  # drop verb + "the"
            else:
                core = parts[-1]
        return core.strip()
    return label.strip(".")


# ============================================================
# Test samples — includes known good AND the known bad (g037)
# ============================================================

TEST_SAMPLES = [
    # Known GOOD labels (should PASS)
    {
        "group_id": "g008",
        "label": "Fold the cloth.",
        "video": "RAD_2023-11-18_Sat_Nov_18_18-10-20_2023_ext1.mp4",
        "expected": "PASS",
    },
    {
        "group_id": "g005",
        "label": "Press the microwave button.",
        "video": "RAD_2023-12-19_Tue_Dec_19_10-09-36_2023_ext1.mp4",
        "expected": "PASS",
    },
    {
        "group_id": "g053",
        "label": "Close the drawer.",
        "video": "AUTOLab_2023-09-05_Tue_Sep__5_09-19-23_2023_ext1.mp4",
        "expected": "PASS",
    },
    # Known BAD label (should FAIL) — no microwave in this video
    {
        "group_id": "g037",
        "label": "Open the microwave.",
        "video": "IPRL_2023-10-26_Thu_Oct_26_22-03-57_2023_ext1.mp4",
        "expected": "FAIL",
    },
    # Borderline — human said "counter", could be "sink"
    {
        "group_id": "g520",
        "label": "Clean the counter with the cloth.",
        "video": "ILIAD_2023-07-19_Wed_Jul_19_17-56-35_2023_ext1.mp4",
        "expected": "PASS",
    },
    # Open/close — human said "Open the microwave" for a different group
    {
        "group_id": "g051",
        "label": "Open the microwave.",
        "video": "RAD_2023-12-17_Sun_Dec_17_17-07-07_2023_ext1.mp4",
        "expected": "PASS",
    },
]


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

def verify_label(video_path, label, model, processor, process_vision_info):
    """Ask Qwen3-VL whether a label matches a video. Returns (verdict, full_output)."""
    object_hint = extract_object_hint(label)
    prompt = VERIFY_PROMPT.format(label=label, object_hint=object_hint)

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
            max_new_tokens=200,
            do_sample=False,
        )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    del inputs, output_ids, generated_ids, image_inputs, video_inputs
    torch.cuda.empty_cache()

    # Parse verdict
    verdict = "UNSURE"
    for line in output_text.strip().split("\n"):
        line = line.strip()
        if line.startswith("VERDICT:"):
            v = line.split(":", 1)[1].strip().upper()
            if "PASS" in v:
                verdict = "PASS"
            elif "FAIL" in v:
                verdict = "FAIL"
            else:
                verdict = "UNSURE"
            break

    return verdict, output_text.strip()


# ============================================================
# Test mode
# ============================================================

def run_test(model, processor, process_vision_info):
    """Verify 6 test samples (3 known good, 1 known bad, 2 borderline)."""
    print("\n" + "=" * 70)
    print("TEST MODE: Verifying labels against videos")
    print("=" * 70)

    results = []
    for sample in TEST_SAMPLES:
        video_path = os.path.join(VIDEO_DIR, sample["video"])
        if not os.path.exists(video_path):
            print(f"SKIP: {video_path} not found")
            continue

        print(f"\n--- {sample['group_id']} ---")
        print(f"  Label:    {sample['label']}")
        print(f"  Expected: {sample['expected']}")

        verdict, output = verify_label(video_path, sample["label"], model, processor, process_vision_info)

        print(f"  Verdict:  {verdict}")
        print(f"  Output:\n    {output.replace(chr(10), chr(10) + '    ')}")

        correct = verdict == sample["expected"]
        print(f"  Correct:  {'YES' if correct else 'NO'}")

        results.append({
            "group_id": sample["group_id"],
            "label": sample["label"],
            "expected": sample["expected"],
            "verdict": verdict,
            "correct": correct,
            "output": output,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Group':<12} {'Label':<35} {'Expected':<10} {'Verdict':<10} {'Correct'}")
    print("-" * 85)
    for r in results:
        print(f"{r['group_id']:<12} {r['label']:<35} {r['expected']:<10} {r['verdict']:<10} {'YES' if r['correct'] else 'NO'}")

    correct_count = sum(1 for r in results if r["correct"])
    print(f"\nCorrect: {correct_count}/{len(results)}")

    out_path = os.path.join(RESULTS_DIR, f"verify_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_path}")


# ============================================================
# Batch mode
# ============================================================

def run_batch(model, processor, process_vision_info):
    """Verify all 212 completed GENERIC_REVIEW groups."""
    print("\n" + "=" * 70)
    print("BATCH MODE: Verifying all GENERIC_REVIEW labels")
    print("=" * 70)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    labeled = {}
    with open(LABELS_PATH) as f:
        for line in f:
            d = json.loads(line)
            labeled[d['episode']] = d

    # Collect completed GENERIC_REVIEW groups
    groups_to_verify = []
    for g in manifest:
        if all(ep in labeled for ep in g['episodes']):
            sample = labeled[g['episodes'][0]]
            if sample.get('category') == 'GENERIC_REVIEW':
                groups_to_verify.append({
                    "group_id": g["group_id"],
                    "label": sample["new_label"],
                    "video_path": g["video_path"],
                    "task": g["task"],
                    "lab": g["lab"],
                    "date": g["date"],
                    "count": g["count"],
                })

    print(f"Groups to verify: {len(groups_to_verify)}")
    print(f"Episodes covered: {sum(g['count'] for g in groups_to_verify)}")

    results = []
    counts = {"PASS": 0, "FAIL": 0, "UNSURE": 0}

    for gi, g in enumerate(groups_to_verify):
        video_path = g["video_path"]
        if not os.path.exists(video_path):
            # Try alternative
            ep_base = os.path.basename(video_path).replace("_ext1.mp4", "")
            for cam in ["ext1", "ext2", "wrist"]:
                alt = os.path.join(VIDEO_DIR, f"{ep_base}_{cam}.mp4")
                if os.path.exists(alt):
                    video_path = alt
                    break
            else:
                print(f"  [{gi+1}/{len(groups_to_verify)}] {g['group_id']}: NO VIDEO, skipping")
                continue

        verdict, output = verify_label(video_path, g["label"], model, processor, process_vision_info)
        counts[verdict] += 1

        status = "OK" if verdict == "PASS" else f"**{verdict}**"
        print(f"  [{gi+1}/{len(groups_to_verify)}] {g['group_id']} ({g['count']}ep): "
              f"{g['label']:<40} -> {status}")

        if verdict != "PASS":
            # Print full output for failures
            print(f"    Qwen3: {output[:200]}")

        results.append({
            "group_id": g["group_id"],
            "label": g["label"],
            "task": g["task"],
            "lab": g["lab"],
            "date": g["date"],
            "count": g["count"],
            "verdict": verdict,
            "output": output,
        })

        # Progress every 25
        if (gi + 1) % 25 == 0:
            print(f"  --- Progress: {gi+1}/{len(groups_to_verify)} | "
                  f"PASS={counts['PASS']} FAIL={counts['FAIL']} UNSURE={counts['UNSURE']} ---")

    # Final summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    total = len(results)
    print(f"Total groups verified: {total}")
    print(f"  PASS:   {counts['PASS']} ({counts['PASS']/total*100:.1f}%)")
    print(f"  FAIL:   {counts['FAIL']} ({counts['FAIL']/total*100:.1f}%)")
    print(f"  UNSURE: {counts['UNSURE']} ({counts['UNSURE']/total*100:.1f}%)")

    failed_eps = sum(r["count"] for r in results if r["verdict"] == "FAIL")
    unsure_eps = sum(r["count"] for r in results if r["verdict"] == "UNSURE")
    print(f"\nEpisodes affected by FAIL: {failed_eps}")
    print(f"Episodes affected by UNSURE: {unsure_eps}")

    if counts["FAIL"] > 0:
        print(f"\nFAILED GROUPS:")
        print(f"{'Group':<8} {'Eps':>4} {'Label':<40} {'Task Type'}")
        print("-" * 100)
        for r in results:
            if r["verdict"] == "FAIL":
                short_task = r["task"].split("(")[0].strip()[:30]
                print(f"{r['group_id']:<8} {r['count']:>4} {r['label']:<40} {short_task}")

    # Save full results
    out_path = os.path.join(RESULTS_DIR, f"verify_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Verify DROID labels with Qwen3-VL")
    parser.add_argument("--mode", choices=["test", "batch"], default="test",
                        help="'test' = 6 test samples, 'batch' = all 212 groups")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    model, processor, pvi = load_qwen3_vl(args.device)

    if args.mode == "test":
        run_test(model, processor, pvi)
    else:
        run_batch(model, processor, pvi)


if __name__ == "__main__":
    main()
