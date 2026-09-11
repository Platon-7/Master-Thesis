#!/usr/bin/env python3
"""Why does RoboDopamine score 0.0 in the live loop but 0.2-1.0 in the sanity check?

Reproduces the LIVE input shape: the buffer pads a growing prefix up to
max_frames=16 by repeating frames (linspace-with-repeats), so the adapter sees a
16-frame clip whose first and last frames may be identical early in an episode.
Prints the model's raw text so we can see whether parse_hop is failing or the
model genuinely emits a zero hop.
"""
import numpy as np, h5py, os
from PIL import Image

H5 = "/scratch-shared/pkarageorgis1/maniskill_assets/demos_converted/PullCube-v1_bc_clip.h5"
TASK = "Pull the cube to the goal region"
GOAL = os.environ.get("ROBODOPAMINE_GOAL", "")


def pad_to(frames, n=16):
    """Same rule as RobometerReplayBuffer._pad_to_max_frames."""
    if len(frames) >= n:
        return frames
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return frames[idx]


def main():
    from robometer_policy_learning.utils.baseline_reward_adapters import (
        load_baseline_reward_model, _to_pil,
    )
    from robometer_policy_learning.utils.robodopamine_utils import RoboDopamineScorer, parse_hop
    _, _, _, rm = load_baseline_reward_model(
        "robodopamine", "tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview", "cuda")
    goal = Image.open(GOAL).convert("RGB") if GOAL and os.path.exists(GOAL) else None
    print(f"goal image: {GOAL if goal is not None else '<none>'}", flush=True)

    f = h5py.File(H5, "r")
    k = list(f["data"].keys())[0]
    frames = np.array(f["data"][k]["obs"]["image"])
    print(f"demo {k}: T={len(frames)}", flush=True)

    # prefixes exactly like the live loop: 1,2,4,8 real frames, each padded to 16
    for n_real in (1, 2, 4, 8, len(frames)):
        pref = pad_to(frames[:max(1, n_real)], 16)
        first, last = _to_pil(pref[0]), _to_pil(pref[-1])
        same = np.array_equal(np.asarray(first), np.asarray(last))
        scorer = RoboDopamineScorer(
            rm.model, rm.processor, task=TASK,
            goal_img=goal if goal is not None else first,
            ref_start_img=first, eval_mode="forward", max_new_tokens=512)
        raw_txt = None
        try:
            # call the model directly so we can see the text before parsing
            import torch
            from qwen_vl_utils import process_vision_info
            msgs, _ = scorer._build_messages([first]*3, [last]*3)
            text = scorer.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, _ = process_vision_info(msgs)
            inp = scorer.processor(text=[text], images=imgs, padding=True, return_tensors="pt")
            dev = next(scorer.model.parameters()).device
            inp = {kk: (v.to(dev) if hasattr(v, "to") else v) for kk, v in inp.items()}
            with torch.inference_mode():
                gen = scorer.model.generate(**inp, max_new_tokens=512, do_sample=False)
            raw_txt = scorer.processor.tokenizer.decode(
                gen[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)
        except Exception as e:
            raw_txt = f"<generate failed: {type(e).__name__}: {e}>"
        hop = parse_hop(raw_txt) if isinstance(raw_txt, str) else None
        print(f"\n--- prefix {n_real} real frames (first==last: {same}) ---", flush=True)
        print(f"raw output ({len(raw_txt)} chars): {raw_txt[:300]!r}", flush=True)
        print(f"parse_hop -> {hop}", flush=True)


if __name__ == "__main__":
    main()
