"""Validate the RoboDopamine GRM scorer on MetaWorld BEFORE any RL:
1) probe which MetaWorld cameras render (pick 3 for multi-view),
2) render oracle SUCCESS + FAILURE CoffeePush rollouts in 3-view,
3) score forward-mode progress along each trajectory (every N frames),
4) confirm success-progress climbs toward 1 while failure stays low.

If success and failure separate, the scorer works and we can wire the RL reward."""
import os, sys
os.environ.setdefault("V3_CORNER2_ZOOM", "1")   # RL render config (policy=zoomed corner2)
import numpy as np

_REPO = os.environ.get("MT_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for sub in ("MetaWorld/metaworld_repo", "MetaWorld", "vlm_ibrl_v3", "Robometer"):
    sys.path.insert(0, os.path.join(_REPO, sub))

from env.metaworld_wrapper import MetaWorldEnv
from env.robodopamine_utils import get_robodopamine, RoboDopamineScorer, accumulate_progress
from env.vlm_prompts import METAWORLD_TASK_DESCRIPTIONS as TASK_STR
from PIL import Image

TASK = TASK_STR["CoffeePush"]
MODEL_PATH = os.environ.get("ROBODOPAMINE_PATH", "tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview")
RES = int(os.environ.get("RD_RES", "240"))
EVAL_MODE = os.environ.get("RD_EVAL_MODE", "forward")
STRIDE = int(os.environ.get("RD_STRIDE", "20"))     # score every STRIDE frames (authors sample sparsely)
# front + two "wrist" views; overridable. We probe availability first.
CAM_CANDIDATES = os.environ.get(
    "RD_CAMS", "corner2_default,gripperPOV,behindGripper,corner3,corner4,topview").split(",")


def probe_cameras(env):
    """Return the first 3 camera names that render without error (front first)."""
    ok = []
    for c in CAM_CANDIDATES:
        try:
            env.render(camera_name=c, width=RES, height=RES)
            ok.append(c)
            print(f"  camera OK: {c}", flush=True)
        except Exception as e:
            print(f"  camera FAIL: {c} ({type(e).__name__})", flush=True)
        if len(ok) >= 3:
            break
    return ok


def main():
    print(f"[rd] model={MODEL_PATH} mode={EVAL_MODE} res={RES} stride={STRIDE}", flush=True)
    env = MetaWorldEnv("CoffeePush", camera_name="corner2", width=RES, height=RES)
    print("[rd] probing cameras ...", flush=True)
    cams = probe_cameras(env)
    if len(cams) < 3:
        cams = (cams + cams + cams)[:3]   # single/dual-view fallback: repeat
    print(f"[rd] using 3 views: {cams}", flush=True)

    def roll_multiview(noisy):
        env.reset(); views = {c: [] for c in cams}; succ = 0
        for t in range(160):
            a = env.get_heuristic_action()
            if noisy: a = (a + np.random.uniform(-1, 1, size=a.shape) * 0.8).clip(-1, 1)
            _, _, done, info = env.step(a)
            for c in cams:
                views[c].append(Image.fromarray(env.render(camera_name=c, width=RES, height=RES)))
            if int(info.get("success", 0)) == 1: succ = 1
            if done: break
        return views, succ

    succ_rolls, fail_rolls = [], []
    while len(succ_rolls) < 2 or len(fail_rolls) < 2:
        v, sc = roll_multiview(len(fail_rolls) < 2 and (len(succ_rolls) >= 2 or np.random.rand() < 0.5))
        if sc and len(succ_rolls) < 2: succ_rolls.append(v)
        elif not sc and len(fail_rolls) < 2: fail_rolls.append(v)
    print(f"[rd] {len(succ_rolls)} succ / {len(fail_rolls)} fail multiview rollouts", flush=True)

    # goal = a success rollout's FINAL front frame (a valid goal/success state)
    goal_img = succ_rolls[0][cams[0]][-1]

    model, processor = get_robodopamine(MODEL_PATH)

    def views_at(roll, t):
        return [roll[c][min(t, len(roll[c]) - 1)] for c in cams]

    def score_traj(roll, label):
        L = len(roll[cams[0]])
        ref_start = roll[cams[0]][0]
        start_views = views_at(roll, 0)
        scorer = RoboDopamineScorer(model, processor, task=TASK, goal_img=goal_img,
                                    ref_start_img=ref_start, eval_mode=EVAL_MODE)
        prev, progs = 0.0, []
        for t in range(0, L, STRIDE):
            before = start_views if EVAL_MODE == "forward" else views_at(roll, max(0, t - STRIDE))
            raw = scorer.score(before, views_at(roll, t), debug=(t == 0 and label.endswith("0")))
            if raw is None:
                print(f"  [{label}] t={t} PARSE-FAIL", flush=True); continue
            prev = accumulate_progress(EVAL_MODE, raw, prev)
            progs.append(round(prev, 2))
        print(f"  [{label}] progress@stride: {progs}  final={progs[-1] if progs else None}", flush=True)
        return progs[-1] if progs else None

    print("\n=== SUCCESS rollouts (want progress climbing toward ~1) ===", flush=True)
    sfin = [score_traj(r, f"succ{i}") for i, r in enumerate(succ_rolls)]
    print("=== FAILURE rollouts (want progress staying low) ===", flush=True)
    ffin = [score_traj(r, f"fail{i}") for i, r in enumerate(fail_rolls)]
    sfin = [x for x in sfin if x is not None]; ffin = [x for x in ffin if x is not None]
    print(f"\n[rd] SUMMARY  succ_final~{np.mean(sfin):.2f}  fail_final~{np.mean(ffin):.2f}  "
          f"separation={np.mean(sfin)-np.mean(ffin):+.2f}" if sfin and ffin else "[rd] insufficient data", flush=True)


if __name__ == "__main__":
    main()
