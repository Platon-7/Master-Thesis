"""Offline discrimination check for RoboReward-4B on Robomimic (handoff rule #1).

Rolls out N_EP BC episodes (full, non-terminating), then scores each with RoboReward-4B via
the EXACT env path (16-frame subsample, final frame kept, max_new_tokens=24, 1-5 -> [0,1]).
Reports success-vs-failure score distributions + a threshold sweep. RoboReward is a scorer
(not a detector), so the signal is: do GT-success episodes score HIGHER than failures?
"""
import os
import numpy as np
import torch

import train_bc
from env.robosuite_wrapper import PixelRobosuite
from env.robosuite_vlm_env import tensor_to_pil, single_prompt_eval, roboreward_to_reward
from env.roboreward_utils import get_roboreward_4b, prompt_roboreward
from env.vlm_prompts import roboreward_prompt, ROBOMIMIC_TASK_DESCRIPTIONS
from common_utils import ibrl_utils as utils

ENV = os.environ.get("DIAG_ENV", "PickPlaceCan")
EP_LEN = int(os.environ.get("DIAG_EP_LEN", "200"))
N_EP = int(os.environ.get("N_EP", "20"))
BC = os.environ.get("DIAG_BC", "release/model/robomimic/can/model0.pt")
NFR = int(os.environ.get("ROBOREWARD_NFRAMES", "16"))


def main():
    task = ROBOMIMIC_TASK_DESCRIPTIONS[ENV]
    print(f"[roboreward-calib] env={ENV} ep_len={EP_LEN} n_ep={N_EP} nframes={NFR}", flush=True)
    model, processor = get_roboreward_4b()
    model.eval()
    prompt = f"{roboreward_prompt}\n\nTask: {task}"

    policy, _, ep = train_bc.load_model(BC, "cuda")
    ep = dict(ep)
    pol = list(ep["rl_cameras"])
    ep["camera_names"] = list(dict.fromkeys(pol + ["agentview"]))
    ep["episode_length"] = EP_LEN
    ep["end_on_success"] = 0
    env = PixelRobosuite(**ep)

    def score(video):
        vid = video
        if len(vid) > NFR:
            idx = np.linspace(0, len(vid) - 1, NFR).round().astype(int)
            vid = [vid[i] for i in idx]
        out = single_prompt_eval(
            prompt_vlm=prompt_roboreward, model=model, processor=processor,
            system_prompt=roboreward_prompt, prompt=prompt, frames=vid,
            debug=False, use_video=True, max_new_tokens=24,
        )
        return roboreward_to_reward(out)

    results = []
    for e in range(N_EP):
        np.random.seed(300 + e)
        obs, hi = env.reset()
        video = [tensor_to_pil(hi["agentview"])]
        gt = 0
        with torch.no_grad(), utils.eval_mode(policy):
            for t in range(1, EP_LEN + 1):
                a = policy.act(obs, eval_mode=True)
                obs, r, term, succ, hi = env.step(a)
                video.append(tensor_to_pil(hi["agentview"]))
                gt = max(gt, int(r > 0))
                if term:
                    break
        s = score(video)
        results.append((gt, s))
        print(f"  ep{e:02d}: GT={'S' if gt else 'F'} roboreward={s:.3f}", flush=True)

    labels = np.array([g for g, s in results])
    scores = np.array([s for g, s in results])
    succ = scores[labels == 1]
    fail = scores[labels == 0]

    def d(v):
        return f"mean={v.mean():.3f} min={v.min():.2f} max={v.max():.2f}" if len(v) else "none"

    print(f"\n=== RoboReward-4B on {ENV} (n_succ={len(succ)} n_fail={len(fail)}) ===", flush=True)
    print(f"  SUCCESS: [{d(succ)}]", flush=True)
    print(f"  FAIL   : [{d(fail)}]", flush=True)
    if len(succ) and len(fail):
        print(f"  separation (succ_mean - fail_mean) = {succ.mean() - fail.mean():+.3f}", flush=True)
    print("  threshold sweep (score > thr):", flush=True)
    for thr in [0.25, 0.50, 0.75]:
        fire = scores > thr
        tpr = float(fire[labels == 1].mean()) if (labels == 1).any() else float("nan")
        fpr = float(fire[labels == 0].mean()) if (labels == 0).any() else float("nan")
        print(f"    thr={thr}: TPR={tpr:.2f} FPR={fpr:.2f}", flush=True)
    print("[roboreward-calib] done", flush=True)


if __name__ == "__main__":
    main()
