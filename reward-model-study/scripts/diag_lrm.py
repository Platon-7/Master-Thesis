"""Validate the LRM Absolute-Progress scorer on MetaWorld BEFORE any RL:
render oracle SUCCESS + FAILURE CoffeePush rollouts (single view = the reward feed
corner2_default), score progress along each, and confirm success-progress climbs
toward ~1 while failure stays low. If they separate, wire the RL reward."""
import os, sys
os.environ.setdefault("V3_CORNER2_ZOOM", "1")   # RL render config
import numpy as np

_REPO = os.environ.get("MT_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for sub in ("MetaWorld/metaworld_repo", "MetaWorld", "vlm_ibrl_v3"):
    sys.path.insert(0, os.path.join(_REPO, sub))

from env.metaworld_wrapper import MetaWorldEnv
from env.lrm_utils import get_lrm_progress, LRMProgressScorer
from env.vlm_prompts import METAWORLD_TASK_DESCRIPTIONS as TASK_STR
from PIL import Image

TASK = TASK_STR["CoffeePush"]
RES = int(os.environ.get("LRM_RES", "240"))
STRIDE = int(os.environ.get("LRM_STRIDE", "20"))
CAM = os.environ.get("LRM_CAM", "corner2_default")


def main():
    print(f"[lrm] loading LRM-progress (subfolder='progress') res={RES} cam={CAM}", flush=True)
    env = MetaWorldEnv("CoffeePush", camera_name="corner2", width=RES, height=RES)

    def roll(noisy):
        env.reset(); frames = []; succ = 0
        for t in range(160):
            a = env.get_heuristic_action()
            if noisy: a = (a + np.random.uniform(-1, 1, size=a.shape) * 0.8).clip(-1, 1)
            _, _, done, info = env.step(a)
            frames.append(Image.fromarray(env.render(camera_name=CAM, width=RES, height=RES)))
            if int(info.get("success", 0)) == 1: succ = 1
            if done: break
        return frames, succ

    succ_rolls, fail_rolls = [], []
    while len(succ_rolls) < 2 or len(fail_rolls) < 2:
        fr, sc = roll(len(fail_rolls) < 2 and (len(succ_rolls) >= 2 or np.random.rand() < 0.5))
        if sc and len(succ_rolls) < 2: succ_rolls.append(fr)
        elif not sc and len(fail_rolls) < 2: fail_rolls.append(fr)
    print(f"[lrm] {len(succ_rolls)} succ / {len(fail_rolls)} fail rollouts", flush=True)

    model, processor = get_lrm_progress()

    def curve(fr, label):
        # zero-shot initial anchor = this rollout's own frame 0
        scorer = LRMProgressScorer(model, processor, task=TASK, initial_img=fr[0])
        vals = []
        for t in range(0, len(fr), STRIDE):
            p = scorer.score(fr[t], debug=(t == 0 and label == "succ0"))
            vals.append(round(p, 2))
        print(f"  [{label}] progress@stride: {vals}  final={vals[-1] if vals else None}", flush=True)
        return vals[-1] if vals else None

    print("\n=== SUCCESS (want climbing toward ~1) ===", flush=True)
    sfin = [curve(r, f"succ{i}") for i, r in enumerate(succ_rolls)]
    print("=== FAILURE (want staying low) ===", flush=True)
    ffin = [curve(r, f"fail{i}") for i, r in enumerate(fail_rolls)]
    sfin = [x for x in sfin if x is not None]; ffin = [x for x in ffin if x is not None]
    if sfin and ffin:
        print(f"\n[lrm] SUMMARY succ_final~{np.mean(sfin):.2f} fail_final~{np.mean(ffin):.2f} "
              f"separation={np.mean(sfin)-np.mean(ffin):+.2f}", flush=True)
    else:
        print("[lrm] insufficient data", flush=True)


if __name__ == "__main__":
    main()
