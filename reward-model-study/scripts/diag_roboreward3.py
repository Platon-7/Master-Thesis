"""Fairness check: does a fuller rubric prompt unlock RoboReward-4B success
discrimination on MetaWorld, or is it genuinely OOD? 16-frame subsample (readable
res confirmed), 4 succ + 4 fail CoffeePush rollouts, several prompt variants."""
import os, sys, re
import numpy as np

_REPO = os.environ.get("MT_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for sub in ("MetaWorld/metaworld_repo", "MetaWorld", "vlm_ibrl_v3", "Robometer"):
    sys.path.insert(0, os.path.join(_REPO, sub))

from env.metaworld_wrapper import MetaWorldEnv
from env.roboreward_utils import get_roboreward_4b, prompt_roboreward
from env.vlm_prompts import METAWORLD_TASK_DESCRIPTIONS as TASK_STR, roboreward_prompt
from PIL import Image

TASK = TASK_STR["CoffeePush"]
_PARSE = re.compile(r"ANSWER:\s*([1-5])")

TERSE = roboreward_prompt
RUBRIC = (
    "You are evaluating a robot performing a manipulation task from a video. "
    "Assign a discrete progress score:\n"
    "1: No success / no meaningful progress\n"
    "2: Minimal progress\n"
    "3: Partial completion\n"
    "4: Near completion\n"
    "5: Task fully completed (perfect success)\n"
    "Respond in the format: ANSWER: <score>"
)
PROMPTS = {"terse": TERSE, "rubric": RUBRIC}


def subsample(frames, n=16):
    if len(frames) <= n: return frames
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx]


def score(model, processor, frames, sysp, usertext):
    content = [
        {"type": "video", "video": frames, "sample_fps": 20,
         "video_metadata": {"duration": len(frames) / 20.0}},
        {"type": "text", "text": usertext},
    ]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": sysp}]},
        {"role": "user", "content": content},
    ]
    kw = dict(max_new_tokens=16, do_sample=False, top_p=1.0, top_k=0, temperature=0)
    raw = prompt_roboreward(model=model, processor=processor, messages=messages,
                            debug=False, prompt_kwargs=kw)
    m = _PARSE.search(raw or "")
    return int(m.group(1)) if m else None


def main():
    print("[diag3] loading RoboReward-4B ...", flush=True)
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
    succs, fails = [], []
    while len(succs) < 4 or len(fails) < 4:
        fr, sc = roll(len(fails) < 4 and (len(succs) >= 4 or np.random.rand() < 0.5))
        if sc and len(succs) < 4: succs.append(subsample(fr))
        elif not sc and len(fails) < 4: fails.append(subsample(fr))
    print(f"[diag3] {len(succs)} succ / {len(fails)} fail (16-frame)", flush=True)

    for pname, ptext in PROMPTS.items():
        # variant A: prompt in system, "Task: ..." in user
        usertext = f"{ptext}\n\nTask: {TASK}"
        ss = [score(model, processor, f, ptext, usertext) for f in succs]
        fs = [score(model, processor, f, ptext, usertext) for f in fails]
        print(f"  [{pname:6s}] succ={ss}  fail={fs}  "
              f"(succ_mean={np.mean([x for x in ss if x]):.2f} fail_mean={np.mean([x for x in fs if x]):.2f})",
              flush=True)
    print("\n[diag3] if rubric gives succ>fail clearly -> use rubric prompt; else OOD.", flush=True)


if __name__ == "__main__":
    main()
