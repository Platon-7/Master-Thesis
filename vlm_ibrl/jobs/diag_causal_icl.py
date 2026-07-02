"""Causal-threshold calibration WITH ICL (for Curve B / run1).
Rolls out N_EP BC Can episodes, scores the growing agentview video every step
through the scorer WITH icl_frames (the 1 demo), takes per-episode MAX success_prob,
and sweeps thresholds for causal TPR/FPR — the same protocol that gave run2=0.80,
so the result is directly comparable.
Env: ROBOMETER_FT_PATH (run1), ROBOMETER_ICL_DEMO_PATH (the demo), ROBOMETER_ICL_FRAMES, N_EP.
"""
import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image

import train_bc
from env.robosuite_wrapper import PixelRobosuite
from env.robometer_utils import get_robometer_4b
from env.vlm_prompts import ROBOMIMIC_TASK_DESCRIPTIONS
from env.robosuite_vlm_env import tensor_to_pil
from common_utils import ibrl_utils as utils

ENV = "PickPlaceCan"


def load_icl():
    p = os.environ.get("ROBOMETER_ICL_DEMO_PATH", "")
    if not p:
        return None
    idx = int(os.environ.get("ROBOMETER_ICL_DEMO_IDX", "0"))
    n = int(os.environ.get("ROBOMETER_ICL_FRAMES", "16"))
    avail = sorted(q for q in Path(p).iterdir() if q.name.startswith(f"{idx}_") and q.suffix == ".png")
    picks = np.linspace(0, len(avail) - 1, n).round().astype(int)
    return [np.asarray(Image.open(avail[i]).convert("RGB"), dtype=np.uint8) for i in picks]


def main():
    ckpt = os.environ["ROBOMETER_FT_PATH"]
    n_ep = int(os.environ.get("N_EP", "24"))
    scorer = get_robometer_4b(model_path=ckpt)
    icl = load_icl()
    task = ROBOMIMIC_TASK_DESCRIPTIONS[ENV]
    print(f"[diag-icl] ckpt={ckpt} ICL={'ON('+str(len(icl))+'f)' if icl else 'OFF'} n_ep={n_ep}", flush=True)

    policy, _, ep = train_bc.load_model("release/model/robomimic/can/model0.pt", "cuda")
    pol = list(ep["rl_cameras"]); ep = dict(ep)
    ep["camera_names"] = list(dict.fromkeys(pol + ["agentview"])); ep["rl_cameras"] = pol
    ep["episode_length"] = 200; ep["end_on_success"] = 0
    env = PixelRobosuite(**ep)

    labels, maxsp = [], []
    for e in range(n_ep):
        np.random.seed(300 + e)
        obs, hi = env.reset()
        video = [tensor_to_pil(hi["agentview"])]
        gt, sps = 0, []
        with torch.no_grad(), utils.eval_mode(policy):
            for t in range(1, 201):
                a = policy.act(obs, eval_mode=True)
                obs, r, term, succ, hi = env.step(a)
                video.append(tensor_to_pil(hi["agentview"]))
                sps.append(float(scorer(video, task=task, icl_frames=icl)["success_prob"]))
                gt = max(gt, int(r > 0))
                if term:
                    break
        labels.append(gt); maxsp.append(max(sps))
        print(f"  ep{e:02d}: GT={'S' if gt else 'F'} max_sp={max(sps):.2f}", flush=True)

    labels, maxsp = np.array(labels), np.array(maxsp)
    nS, nF = int((labels == 1).sum()), int((labels == 0).sum())
    print(f"\n  causal sweep (n_succ={nS} n_fail={nF}):", flush=True)
    best = None
    for thr in np.arange(0.05, 0.96, 0.05):
        fire = maxsp > thr
        ctpr = float(fire[labels == 1].mean()) if nS else float("nan")
        cfpr = float(fire[labels == 0].mean()) if nF else float("nan")
        j = ctpr - cfpr
        if best is None or j > best[0]:
            best = (j, round(float(thr), 2), round(ctpr, 2), round(cfpr, 2))
        print(f"  thr={thr:.2f} cTPR={ctpr:.2f} cFPR={cfpr:.2f} J={j:.2f}", flush=True)
    print(f"\n  RUN1+ICL causal threshold = {best[1]} (cTPR={best[2]} cFPR={best[3]} J={best[0]:.2f}) "
          f"[run2 no-ICL was 0.80]", flush=True)
    print("[diag-icl] done", flush=True)


if __name__ == "__main__":
    main()
