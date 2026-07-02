"""Fix-test: does subsampling to a fixed frame count restore spatial resolution
and success discrimination for RoboReward-4B? Score real succ/fail CoffeePush
rollouts at 8/16/32 frames, print grid_thw + score."""
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

TASK = TASK_STR["CoffeePush"]
_PARSE = re.compile(r"ANSWER:\s*([1-5])")


def subsample(frames, n):
    if len(frames) <= n: return frames
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx]


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


def grid_of(processor, frames):
    messages = build_messages(frames)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ii, vi, vk = process_vision_info(messages, image_patch_size=16,
                                     return_video_kwargs=True, return_video_metadata=True)
    if vi: videos, vmeta = zip(*vi); videos = list(videos); vmeta = list(vmeta)
    else: videos, vmeta = None, None
    inputs = processor(text=[text], images=ii or None, videos=videos,
                       video_metadata=vmeta, padding=True, return_tensors="pt", **vk)
    vgt = inputs.get("video_grid_thw", None)
    return None if vgt is None else vgt.tolist()


def score(model, processor, frames):
    kw = dict(max_new_tokens=16, do_sample=False, top_p=1.0, top_k=0, temperature=0)
    raw = prompt_roboreward(model=model, processor=processor,
                            messages=build_messages(frames), debug=False, prompt_kwargs=kw)
    m = _PARSE.search(raw or "")
    return int(m.group(1)) if m else None


def main():
    print("[diag2] loading RoboReward-4B ...", flush=True)
    model, processor = get_roboreward_4b()
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
    # collect 3 success + 3 failure rollouts
    succs, fails = [], []
    while len(succs) < 3 or len(fails) < 3:
        fr, sc = roll(len(fails) < 3 and (len(succs) >= 3 or np.random.rand() < 0.5))
        if sc and len(succs) < 3: succs.append(fr)
        elif not sc and len(fails) < 3: fails.append(fr)
    print(f"[diag2] got {len(succs)} succ / {len(fails)} fail rollouts (len~{len(succs[0])})", flush=True)

    for n in [8, 16, 32, 64]:
        g = grid_of(processor, subsample(succs[0], n))
        ss = [score(model, processor, subsample(f, n)) for f in succs]
        fs = [score(model, processor, subsample(f, n)) for f in fails]
        print(f"  nframes={n:3d}  grid_thw={g}  succ_scores={ss}  fail_scores={fs}", flush=True)
    print("\n[diag2] want: succ_scores > fail_scores at some n, and grid spatial >> 4x4.", flush=True)


if __name__ == "__main__":
    main()
