"""Per-step reward-trace diagnostic: roll out the BC policy in Can, score the
GROWING agentview video every step (exactly like the live critic), but run FULL
episodes (no early termination) so we see the entire success_prob / progress curve
relative to the true GT-success moment. Tells us which lever to pull.

Usage: python jobs/diag_reward_trace.py <bc_model0.pt>
Env: ROBOMETER_FT_PATH (scorer ckpt), N_EP (default 8), THR (default 0.61).
"""
import os, sys
import numpy as np
import torch

import train_bc
from env.robosuite_wrapper import PixelRobosuite
from env.robometer_utils import get_robometer_4b
from env.vlm_prompts import ROBOMIMIC_TASK_DESCRIPTIONS
from env.robosuite_vlm_env import tensor_to_pil
from common_utils import ibrl_utils as utils


def main():
    bc = sys.argv[1]
    ckpt = os.environ["ROBOMETER_FT_PATH"]
    n_ep = int(os.environ.get("N_EP", "8"))
    thr = float(os.environ.get("THR", "0.61"))

    policy, _, ep = train_bc.load_model(bc, "cuda")
    env_name = ep["env_name"]
    pol = list(ep["rl_cameras"])
    ep = dict(ep)
    ep["camera_names"] = list(dict.fromkeys(pol + ["agentview"]))
    ep["rl_cameras"] = pol
    ep["episode_length"] = 200
    ep["end_on_success"] = 0            # full episodes — see post-success decay
    env = PixelRobosuite(**ep)
    scorer = get_robometer_4b(model_path=ckpt)
    task = ROBOMIMIC_TASK_DESCRIPTIONS[env_name]
    print(f"[diag] env={env_name} thr={thr} n_ep={n_ep} ckpt={ckpt}", flush=True)

    succ_at_gt, cross_on_succ, cross_on_fail, sp_at_90 = [], 0, 0, {"succ": [], "fail": []}
    for e in range(n_ep):
        np.random.seed(100 + e)
        obs, hi = env.reset()
        video = [tensor_to_pil(hi["agentview"])]
        sps, prs = [], []
        gt_step = None
        with torch.no_grad(), utils.eval_mode(policy):
            for t in range(1, 201):
                a = policy.act(obs, eval_mode=True)
                obs, r, term, succ, hi = env.step(a)
                video.append(tensor_to_pil(hi["agentview"]))
                out = scorer(video, task=task)
                sps.append(float(out["success_prob"])); prs.append(float(out["progress_reward"]))
                if r > 0 and gt_step is None:
                    gt_step = t
                if term:
                    break
        sps, prs = np.array(sps), np.array(prs)
        is_succ = gt_step is not None
        crossed = np.where(sps > thr)[0]
        first_cross = int(crossed[0]) + 1 if len(crossed) else None
        sp90 = float(sps[89]) if len(sps) > 89 else float("nan")
        sp_at_90["succ" if is_succ else "fail"].append(sp90)
        if is_succ:
            succ_at_gt.append(float(sps[min(gt_step - 1, len(sps) - 1)]))
            cross_on_succ += int(first_cross is not None)
        else:
            cross_on_fail += int(first_cross is not None)
        print(f"\nEP{e}: GT={'SUCCESS@'+str(gt_step) if is_succ else 'FAIL'} len={len(sps)} "
              f"| success_head: max={sps.max():.2f}@{int(sps.argmax())+1} final={sps[-1]:.2f} "
              f"first_cross>{thr}={first_cross} sp@gate90={sp90:.2f} "
              f"| progress: max={prs.max():.2f} final={prs[-1]:.2f}", flush=True)
        # trace around the GT-success moment (or whole ep if failure), every 2 steps
        if is_succ:
            lo, hi2 = max(0, gt_step - 6), min(len(sps), gt_step + 20)
            tag = "around GT-success"
        else:
            lo, hi2 = 0, len(sps)
            tag = "failure ep (spurious spikes?)"
        pts = [f"t{i+1}:sp{sps[i]:.2f}/pr{prs[i]:.2f}" for i in range(lo, hi2, max(1, (hi2 - lo) // 12))]
        print(f"   {tag}: " + " ".join(pts), flush=True)

    print("\n===== SUMMARY =====", flush=True)
    print(f"success_prob AT the GT-success step (succ eps): "
          f"{[round(x,2) for x in succ_at_gt]}  mean={np.mean(succ_at_gt) if succ_at_gt else float('nan'):.2f}", flush=True)
    print(f"crossed {thr} on SUCCESS eps: {cross_on_succ}/{len(succ_at_gt)} | "
          f"crossed {thr} on FAILURE eps: {cross_on_fail}/{n_ep-len(succ_at_gt)} (these = false fires)", flush=True)
    print(f"success_prob @ step90 (gate): succ={[round(x,2) for x in sp_at_90['succ']]} "
          f"fail={[round(x,2) for x in sp_at_90['fail']]}", flush=True)


if __name__ == "__main__":
    main()
