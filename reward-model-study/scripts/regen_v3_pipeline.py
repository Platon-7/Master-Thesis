"""DECISIVE root-cause test: regenerate fresh CoffeePush v3 successes through
MetaWorld's OWN curated pipeline (_run_success_episode: run oracle to success,
+8 post-success steps, linspace-16 keyframes, corner2 via env.render) and score
them with the SAME Robometer-FT scorer.

Compares three success-generation variants to isolate WHY diag_v3_cameras'
fresh v3 corner2 successes scored only succ_mean 0.33 while the curated eval
frames score 0.77 (both corner2, same engine, same scorer — control AUC 0.985):

  A) curated pipeline, MT1 frozen tasks (cycled)      -> exact curated reproduction
  B) curated pipeline, random goals (_freeze_rand_vec)-> isolates goal distribution
  C) fixed-175-step rollout, random goals (= diag)    -> the collapsed baseline

If A ~ 0.77 -> the collapse is trajectory/keyframe STRUCTURE (truncate at
success+8 vs run-on), not rendering; the reward model recognizes properly
structured v3 successes. If A also ~0.33 -> something else about freshly
generated frames.
"""
import os
import sys
import numpy as np

sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/MetaWorld")
sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl_v3")
sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer")

import metaworld
from generate_failures import (
    MetaworldWrapper, calibrate_task, SCRIPTED_POLICIES, PUSH_SWEEP_TASKS,
)
from generate_dataset import _run_success_episode
from env.robometer_utils import get_robometer_4b
from PIL import Image

TASK_ID = "coffee-push-v3"
TASK = "Push a mug under a coffee machine."
MODEL = "/scratch-shared/pkarageorgis1/Robometer_FT_consolidated/run1_icl_ours_step3000"
CAM = "corner2"
N = 25


def keyframes_to_pils(keyframes):
    out = []
    for kf in keyframes:
        img = kf["images"][CAM]                     # (H,W,3) uint8, already flipped
        out.append(Image.fromarray(np.asarray(img).astype(np.uint8)))
    return out


def gen_curated(env_cls, tasks, policy, calib, sc, n, random_goals):
    sps, nfail = [], 0
    i = 0
    while len(sps) < n and i < n * 6:
        task = tasks[i % len(tasks)]
        i += 1
        env = MetaworldWrapper(env_cls, task, render_mode="rgb_array", camera_name=CAM)
        if random_goals:
            env.env._freeze_rand_vec = False
        res = _run_success_episode(env, policy, "push_sweep", calib, [CAM], task_name=TASK_ID)
        if res is None:
            nfail += 1
            continue
        keyframes, n_steps = res
        pil = keyframes_to_pils(keyframes)
        sp = float(sc(pil, task=TASK, icl_frames=None)["success_prob"])
        sps.append(sp)
        if len(sps) % 5 == 0:
            print(f"    {len(sps)}/{n}  last sp={sp:.3f} (n_steps={n_steps})", flush=True)
    return np.array(sps), nfail


def gen_fixed175(env_cls, tasks, policy, sc, n):
    """Variant C: my diag rollout — fixed 175 steps, random goals, linspace-16."""
    from env.metaworld_wrapper import MetaWorldEnv
    env = MetaWorldEnv("CoffeePush", camera_name=CAM, width=240, height=240)
    sps = []
    tries = 0
    while len(sps) < n and tries < n * 6:
        tries += 1
        env.reset()
        succ = 0
        frames = []
        for t in range(175):
            a = env.get_heuristic_action()
            o, r, d, info = env.step(a)
            if int(info.get("success", 0)) == 1:
                succ = 1
            frames.append(env.render(camera_name=CAM, width=240, height=240))
            if d:
                break
        if not succ:
            continue
        idx = np.linspace(0, len(frames) - 1, 16).round().astype(int)
        pil = [Image.fromarray(frames[k]) for k in idx]
        sps.append(float(sc(pil, task=TASK, icl_frames=None)["success_prob"]))
        if len(sps) % 5 == 0:
            print(f"    C {len(sps)}/{n}", flush=True)
    return np.array(sps)


def main():
    mt1 = metaworld.MT1(TASK_ID, seed=42)
    env_cls = mt1.train_classes[TASK_ID]
    tasks = list(mt1.train_tasks)
    policy = SCRIPTED_POLICIES[TASK_ID]()
    assert TASK_ID in PUSH_SWEEP_TASKS
    calib = calibrate_task(env_cls, tasks[0], policy)
    print(f"calibrated; {len(tasks)} tasks; category=push_sweep", flush=True)

    sc = get_robometer_4b(model_path=MODEL)
    print(f"scorer max_frames={getattr(sc,'max_frames','?')}", flush=True)

    print("\n[A] curated pipeline, MT1 frozen tasks (cycled)…", flush=True)
    A, Afail = gen_curated(env_cls, tasks, policy, calib, sc, N, random_goals=False)
    print("\n[B] curated pipeline, random goals…", flush=True)
    B, Bfail = gen_curated(env_cls, tasks, policy, calib, sc, N, random_goals=True)
    print("\n[C] fixed-175 rollout, random goals (= diag_v3_cameras)…", flush=True)
    C = gen_fixed175(env_cls, tasks, policy, sc, N)

    print("\n==== fresh CoffeePush v3 successes: success_prob by generation method ====")
    print(f"  ref curated eval frames : succ_mean=0.772  (control AUC 0.985)")
    print(f"  A curated pipe, frozen  : n={len(A)} mean={A.mean():.3f} std={A.std():.3f} "
          f"min={A.min():.3f} max={A.max():.3f}  (skipped {Afail})")
    print(f"  B curated pipe, random  : n={len(B)} mean={B.mean():.3f} std={B.std():.3f} "
          f"min={B.min():.3f} max={B.max():.3f}  (skipped {Bfail})")
    print(f"  C fixed-175, random     : n={len(C)} mean={C.mean():.3f} std={C.std():.3f} "
          f"min={C.min():.3f} max={C.max():.3f}")
    print("\n  A~0.77 -> collapse is trajectory/keyframe STRUCTURE (truncate at success+8),")
    print("  not rendering. C~0.33 reproduces the diag collapse.")


if __name__ == "__main__":
    main()
