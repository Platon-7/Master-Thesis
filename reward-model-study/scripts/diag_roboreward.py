"""Decisive diagnostic: is RoboReward-4B actually SEEING our video, or answering
from text prior? Feed visually extreme inputs; inspect processed tensors; check
whether the output score varies with pixels. If black/white/noise/real all give
the SAME score AND video_grid_thw is empty -> feeding bug. If scores vary ->
pixels are ingested (then near-constant on MetaWorld = OOD)."""
import os, sys, re
import numpy as np

_REPO = os.environ.get("MT_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for sub in ("MetaWorld/metaworld_repo", "MetaWorld", "vlm_ibrl_v3", "Robometer"):
    sys.path.insert(0, os.path.join(_REPO, sub))

from env.metaworld_wrapper import MetaWorldEnv
from env.roboreward_utils import get_roboreward_4b, prompt_roboreward
from env.vlm_prompts import METAWORLD_TASK_DESCRIPTIONS as TASK_STR, roboreward_prompt
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

TASK = TASK_STR["CoffeePush"]
_PARSE = re.compile(r"ANSWER:\s*([1-5])")


def build_messages(frames):
    prompt = f"{roboreward_prompt}\n\nTask: {TASK}"
    content = [
        {"type": "video", "video": frames, "sample_fps": 20,
         "video_metadata": {"duration": len(frames) / 20.0}},
        {"type": "text", "text": prompt},
    ]
    return [
        {"role": "system", "content": [{"type": "text", "text": roboreward_prompt}]},
        {"role": "user", "content": content},
    ]


def inspect_inputs(model, processor, frames, label):
    """Replicate prompt_roboreward's preprocessing and print tensor shapes."""
    messages = build_messages(frames)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
    if video_inputs:
        videos, vmeta = zip(*video_inputs); videos = list(videos); vmeta = list(vmeta)
    else:
        videos, vmeta = None, None
    inputs = processor(text=[text], images=image_inputs or None, videos=videos,
                       video_metadata=vmeta, padding=True, return_tensors="pt", **video_kwargs)
    keys = list(inputs.keys())
    vgt = inputs.get("video_grid_thw", None)
    pvv = inputs.get("pixel_values_videos", None)
    print(f"  [{label}] video_inputs={'yes' if video_inputs else 'NONE'} keys={keys}")
    print(f"  [{label}] video_grid_thw={None if vgt is None else vgt.tolist()} "
          f"pixel_values_videos.shape={None if pvv is None else tuple(pvv.shape)} "
          f"input_ids.len={inputs['input_ids'].shape[-1]}", flush=True)


def score(model, processor, frames):
    kw = dict(max_new_tokens=16, do_sample=False, top_p=1.0, top_k=0, temperature=0)
    raw = prompt_roboreward(model=model, processor=processor,
                            messages=build_messages(frames), debug=False, prompt_kwargs=kw)
    m = _PARSE.search(raw or "")
    return (int(m.group(1)) if m else None), (raw or "")[-60:]


def main():
    print("[diag] loading RoboReward-4B ...", flush=True)
    model, processor = get_roboreward_4b()
    N = 16
    black = [Image.fromarray(np.zeros((224, 224, 3), np.uint8)) for _ in range(N)]
    white = [Image.fromarray(np.full((224, 224, 3), 255, np.uint8)) for _ in range(N)]
    rng = np.random.RandomState(0)
    noise = [Image.fromarray(rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)) for _ in range(N)]
    # a real oracle success + failure rollout on CoffeePush
    env = MetaWorldEnv("CoffeePush", camera_name="corner2", width=224, height=224)
    def roll(noisy):
        env.reset(); fr = []; succ = 0
        for t in range(160):
            a = env.get_heuristic_action()
            if noisy: a = (a + np.random.uniform(-1, 1, size=a.shape) * 0.8).clip(-1, 1)
            _, _, done, info = env.step(a)
            fr.append(Image.fromarray(env.render(camera_name="corner2", width=224, height=224)))
            if int(info.get("success", 0)) == 1: succ = 1
            if done: break
        return fr, succ
    real_succ, s1 = roll(False)
    real_fail, s2 = roll(True)
    print(f"[diag] real rollouts: succ_label={s1} fail_label={s2} (len {len(real_succ)}/{len(real_fail)})", flush=True)

    print("\n=== TENSOR INSPECTION (are pixels tokenized?) ===", flush=True)
    for lab, fr in [("black", black), ("real_succ", real_succ)]:
        inspect_inputs(model, processor, fr, lab)

    print("\n=== SCORE vs VISUAL CONTENT (does output track pixels?) ===", flush=True)
    for lab, fr in [("black", black), ("white", white), ("noise", noise),
                    ("real_succ", real_succ), ("real_fail", real_fail)]:
        sc, tail = score(model, processor, fr)
        print(f"  {lab:10s} -> ANSWER={sc}   ...{tail!r}", flush=True)
    print("\n[diag] If black/white/noise/real all identical -> NOT seeing pixels (feeding bug).", flush=True)


if __name__ == "__main__":
    main()
