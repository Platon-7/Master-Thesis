"""Two diagnostics in one:
  PHASE A — frame-count freak-out test: score the SAME clip with N=1..32 frames.
            Tests Chris's "<16 frames -> OOD -> spurious success" hypothesis.
            Uses saved rung-3 rollout videos (no env needed).
  PHASE B — robomimic causal threshold: roll out N_EP Can episodes, score the
            GROWING video every step, take per-episode MAX success_prob, sweep
            thresholds for causal-TPR/FPR (same protocol as the metaworld 0.85).
Env: ROBOMETER_FT_PATH, N_EP (default 24).
"""
import os
import numpy as np
import torch

import train_bc
from env.robosuite_wrapper import PixelRobosuite
from env.robometer_utils import get_robometer_4b
from env.vlm_prompts import ROBOMIMIC_TASK_DESCRIPTIONS
from env.robosuite_vlm_env import tensor_to_pil
from common_utils import ibrl_utils as utils

ENV = "PickPlaceCan"


def main():
    ckpt = os.environ["ROBOMETER_FT_PATH"]
    scorer = get_robometer_4b(model_path=ckpt)
    task = ROBOMIMIC_TASK_DESCRIPTIONS[ENV]
    print(f"[diag] ckpt={ckpt}", flush=True)

    # ===================== PHASE A: frame-count sensitivity =====================
    print("\n##### PHASE A: frame-count freak-out test #####", flush=True)
    npz = "../reward-model-study/results/robomimic/rollouts_can.npz"
    if os.path.exists(npz):
        d = np.load(npz, allow_pickle=True)
        labels = d["labels"]
        eps = [d[k] for k in sorted([k for k in d.files if k.startswith("ep")], key=lambda s: int(s[2:]))]
        si = [i for i, l in enumerate(labels) if l == 1][0]
        fi = [i for i, l in enumerate(labels) if l == 0][0]
        for tag, idx in [("SUCCESS", si), ("FAILURE", fi)]:
            vid = eps[idx]
            frames = [vid[j] for j in range(vid.shape[0])]
            print(f"\n  {tag} ep ({len(frames)}f). UNIFORM N-frame subsample of the full clip:", flush=True)
            for N in [1, 2, 4, 8, 16, 32]:
                if N > len(frames):
                    continue
                ix = np.linspace(0, len(frames) - 1, N).round().astype(int)
                out = scorer([frames[j] for j in ix], task=task)
                print(f"    N={N:2d}: success_prob={out['success_prob']:.3f} progress={out['progress_reward']:.3f}", flush=True)
            print(f"  {tag} ep. FIRST-N frames only (early/pre-success content — the real OOD test):", flush=True)
            for N in [1, 2, 4, 8, 16]:
                if N > len(frames):
                    continue
                out = scorer(frames[:N], task=task)
                print(f"    first-{N:2d}: success_prob={out['success_prob']:.3f}", flush=True)
    else:
        print("  (rollouts_can.npz not found — skipping Phase A)", flush=True)

    # ===================== PHASE B: causal threshold =====================
    print("\n##### PHASE B: robomimic-Can causal threshold #####", flush=True)
    n_ep = int(os.environ.get("N_EP", "24"))
    policy, _, ep = train_bc.load_model("release/model/robomimic/can/model0.pt", "cuda")
    pol = list(ep["rl_cameras"])
    ep = dict(ep)
    ep["camera_names"] = list(dict.fromkeys(pol + ["agentview"]))
    ep["rl_cameras"] = pol
    ep["episode_length"] = 200
    ep["end_on_success"] = 0
    env = PixelRobosuite(**ep)

    labels, maxsp = [], []
    for e in range(n_ep):
        np.random.seed(200 + e)
        obs, hi = env.reset()
        video = [tensor_to_pil(hi["agentview"])]
        gt, sps = 0, []
        with torch.no_grad(), utils.eval_mode(policy):
            for t in range(1, 201):
                a = policy.act(obs, eval_mode=True)
                obs, r, term, succ, hi = env.step(a)
                video.append(tensor_to_pil(hi["agentview"]))
                sps.append(float(scorer(video, task=task)["success_prob"]))
                gt = max(gt, int(r > 0))
                if term:
                    break
        labels.append(gt); maxsp.append(max(sps))
        print(f"  ep{e:02d}: GT={'S' if gt else 'F'} max_success_prob={max(sps):.2f}", flush=True)

    labels, maxsp = np.array(labels), np.array(maxsp)
    nS, nF = int((labels == 1).sum()), int((labels == 0).sum())
    print(f"\n  causal sweep (n_succ={nS} n_fail={nF}):  cTPR=frac succ eps that EVER cross; cFPR=frac fail eps that EVER cross", flush=True)
    print(f"  {'thr':>5} {'cTPR':>6} {'cFPR':>6} {'J':>6}", flush=True)
    best = None
    for thr in np.arange(0.05, 0.96, 0.05):
        fire = maxsp > thr
        ctpr = float(fire[labels == 1].mean()) if nS else float("nan")
        cfpr = float(fire[labels == 0].mean()) if nF else float("nan")
        j = ctpr - cfpr
        if best is None or j > best[0]:
            best = (j, round(float(thr), 2), round(ctpr, 2), round(cfpr, 2))
        print(f"  {thr:5.2f} {ctpr:6.2f} {cfpr:6.2f} {j:6.2f}", flush=True)
    print(f"\n  ROBOMIMIC-CAN causal threshold = {best[1]}  (cTPR={best[2]} cFPR={best[3]} J={best[0]:.2f})  "
          f"[compare: CoffeePush 0.85→J0.47, BoxClose 0.85→J0.87]", flush=True)
    print("[diag] done", flush=True)


if __name__ == "__main__":
    main()
