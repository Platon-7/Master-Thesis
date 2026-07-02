"""Score RoboReward-4B on the EXACT reward feed the RL VLM sees: the dual-render
`corner2_default` view (un-zoomed v3 default, the reward model's in-domain view),
with V3_CORNER2_ZOOM=1 like the RL jobs. Compare against `corner2` (the zoomed
POLICY feed) and an eval clip (positive control). Save frames for visual compare."""
import os, sys, re
os.environ["V3_CORNER2_ZOOM"] = "1"   # MUST be set BEFORE MetaWorldEnv init (RL config)
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
OUT = "/shared/home/PKA4388/scratchpad_rr"; os.makedirs(OUT, exist_ok=True)
RES = int(os.environ.get("RR_RES", "240"))
# corner2_default = the REWARD feed (what the VLM actually sees); corner2 = zoomed POLICY feed
CAMERAS = ["corner2_default", "corner2"]
_PARSE = re.compile(r"ANSWER:\s*([1-5])")


def subsample(frames, n=16):
    if len(frames) <= n: return frames
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx]


def score(model, processor, frames):
    imgs = [Image.fromarray(np.asarray(f).astype(np.uint8)) for f in subsample(frames, 16)]
    content = [
        {"type": "video", "video": imgs, "sample_fps": 20,
         "video_metadata": {"duration": len(imgs) / 20.0}},
        {"type": "text", "text": f"{roboreward_prompt}\n\nTask: {TASK}"},
    ]
    messages = [{"role": "system", "content": [{"type": "text", "text": roboreward_prompt}]},
                {"role": "user", "content": content}]
    kw = dict(max_new_tokens=24, do_sample=False, top_p=1.0, top_k=0, temperature=0)
    raw = prompt_roboreward(model=model, processor=processor, messages=messages, debug=False, prompt_kwargs=kw)
    m = _PARSE.search(raw or ""); return int(m.group(1)) if m else None


def main():
    print(f"[view] V3_CORNER2_ZOOM={os.environ.get('V3_CORNER2_ZOOM')} cameras={CAMERAS} res={RES}", flush=True)
    env = MetaWorldEnv("CoffeePush", camera_name="corner2", width=RES, height=RES)

    def roll_multicam(noisy):
        env.reset(); frames = {c: [] for c in CAMERAS}; succ = 0
        for t in range(160):
            a = env.get_heuristic_action()
            if noisy: a = (a + np.random.uniform(-1, 1, size=a.shape) * 0.8).clip(-1, 1)
            _, _, done, info = env.step(a)
            for c in CAMERAS:
                frames[c].append(env.render(camera_name=c, width=RES, height=RES))
            if int(info.get("success", 0)) == 1: succ = 1
            if done: break
        return frames, succ

    succ_rolls, fail_rolls = [], []
    while len(succ_rolls) < 3 or len(fail_rolls) < 3:
        fr, sc = roll_multicam(len(fail_rolls) < 3 and (len(succ_rolls) >= 3 or np.random.rand() < 0.5))
        if sc and len(succ_rolls) < 3: succ_rolls.append(fr)
        elif not sc and len(fail_rolls) < 3: fail_rolls.append(fr)
    print("[view] collected 3 succ / 3 fail multicam rollouts", flush=True)

    model, processor = get_roboreward_4b()
    # positive control: an eval clip (known to discriminate)
    import glob
    EV = "/shared/home/PKA4388/robometer_frames_hf_full_step2/_fsx_PKA4388_robometer_frames_hf_full_eval_metaworld_raw_robometer_frames_eval_metaworld/frames"
    ev_s = glob.glob(EV + "/*coffee_push_v3_success*")
    ev_f = glob.glob(EV + "/*coffee_push_v3_score1*")
    if ev_s and ev_f:
        s = score(model, processor, list(np.load(ev_s[0])["frames"]))
        f = score(model, processor, list(np.load(ev_f[0])["frames"]))
        print(f"  [EVAL-CLIP control] succ_clip={s}  fail_clip={f}", flush=True)

    for c in CAMERAS:
        # save last frame of first success + a 4-frame strip for visual compare
        Image.fromarray(succ_rolls[0][c][-1]).save(f"{OUT}/ours_{c}_coffee_succ_last.png")
        fr = succ_rolls[0][c]
        strip = np.concatenate([fr[i] for i in np.linspace(0, len(fr)-1, 4).astype(int)], axis=1)
        Image.fromarray(strip).save(f"{OUT}/ours_{c}_coffee_succ_strip.png")
        ss = [score(model, processor, r[c]) for r in succ_rolls]
        fs = [score(model, processor, r[c]) for r in fail_rolls]
        print(f"  [{c:16s}] succ={ss} fail={fs}", flush=True)
    print(f"\n[view] frames -> {OUT}/ours_<cam>_coffee_succ_*.png", flush=True)


if __name__ == "__main__":
    main()
