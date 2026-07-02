"""Generalized causal-threshold + gate calibration (same protocol that gave Can run2=0.80).

Rolls out N_EP BC episodes, scores the growing agentview video every step with the
reward model (optionally with ICL), records per-episode the full success_prob series +
GT outcome + GT-success step. Then:
  (1) causal THRESHOLD sweep on per-episode MAX success_prob (TPR/FPR, best Youden J);
  (2) GATE guidance: at the chosen threshold, when does success_prob first cross it for
      GT-fail (fake fires) vs GT-success (real fires) episodes, and when do real GT
      successes actually occur — so min_ep_steps can be set between fake-fires and real ones.

Env vars: ROBOMETER_FT_PATH (reward model), DIAG_ENV (default PickPlaceCan),
          DIAG_BC (BC policy .pt), DIAG_EP_LEN (default 200), N_EP (default 24),
          optionally ROBOMETER_ICL_DEMO_PATH/IDX/FRAMES for ICL.
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

ENV = os.environ.get("DIAG_ENV", "PickPlaceCan")
BC_PATH = os.environ.get("DIAG_BC", "release/model/robomimic/can/model0.pt")
EP_LEN = int(os.environ.get("DIAG_EP_LEN", "200"))
N_EP = int(os.environ.get("N_EP", "24"))


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
    scorer = get_robometer_4b(model_path=ckpt)
    icl = load_icl()
    task = ROBOMIMIC_TASK_DESCRIPTIONS[ENV]
    print(f"[calib] env={ENV} ckpt={ckpt} ICL={'ON' if icl else 'OFF'} n_ep={N_EP} ep_len={EP_LEN}", flush=True)
    print(f"        bc={BC_PATH}", flush=True)

    policy, _, ep = train_bc.load_model(BC_PATH, "cuda")
    pol = list(ep["rl_cameras"]); ep = dict(ep)
    ep["camera_names"] = list(dict.fromkeys(pol + ["agentview"])); ep["rl_cameras"] = pol
    ep["episode_length"] = EP_LEN; ep["end_on_success"] = 0
    env = PixelRobosuite(**ep)

    episodes = []  # dict per ep: gt, gt_step, sps (np array)
    for e in range(N_EP):
        np.random.seed(300 + e)
        obs, hi = env.reset()
        video = [tensor_to_pil(hi["agentview"])]
        gt, gt_step, sps = 0, -1, []
        with torch.no_grad(), utils.eval_mode(policy):
            for t in range(1, EP_LEN + 1):
                a = policy.act(obs, eval_mode=True)
                obs, r, term, succ, hi = env.step(a)
                video.append(tensor_to_pil(hi["agentview"]))
                sps.append(float(scorer(video, task=task, icl_frames=icl)["success_prob"]))
                if r > 0 and gt_step < 0:
                    gt_step = t
                gt = max(gt, int(r > 0))
                if term:
                    break
        sps = np.array(sps)
        episodes.append({"gt": gt, "gt_step": gt_step, "sps": sps})
        pk = int(sps.argmax()) + 1
        print(f"  ep{e:02d}: GT={'S' if gt else 'F'} gt_step={gt_step} len={len(sps)} "
              f"max_sp={sps.max():.2f}@step{pk}", flush=True)

    labels = np.array([d["gt"] for d in episodes])
    maxsp = np.array([d["sps"].max() for d in episodes])
    nS, nF = int((labels == 1).sum()), int((labels == 0).sum())

    print(f"\n=== (1) causal THRESHOLD sweep (n_succ={nS} n_fail={nF}) ===", flush=True)
    best = None
    for thr in np.arange(0.05, 0.96, 0.05):
        fire = maxsp > thr
        ctpr = float(fire[labels == 1].mean()) if nS else float("nan")
        cfpr = float(fire[labels == 0].mean()) if nF else float("nan")
        j = ctpr - cfpr
        if best is None or j > best[0]:
            best = (j, round(float(thr), 2), round(ctpr, 2), round(cfpr, 2))
        print(f"  thr={thr:.2f} cTPR={ctpr:.2f} cFPR={cfpr:.2f} J={j:.2f}", flush=True)
    bthr = best[1]
    print(f"  --> best causal threshold = {bthr} (cTPR={best[2]} cFPR={best[3]} J={best[0]:.2f}) "
          f"[Can run2 was 0.80]", flush=True)

    def first_cross(sps, thr):
        idx = np.where(sps > thr)[0]
        return int(idx[0]) + 1 if len(idx) else -1

    print(f"\n=== (2) GATE guidance at thr={bthr} (and at 0.80) ===", flush=True)
    for use_thr in sorted({bthr, 0.80}):
        fc_S = [first_cross(d["sps"], use_thr) for d in episodes if d["gt"] == 1]
        fc_F = [first_cross(d["sps"], use_thr) for d in episodes if d["gt"] == 0]
        gtS = [d["gt_step"] for d in episodes if d["gt"] == 1 and d["gt_step"] > 0]
        fcS = [x for x in fc_S if x > 0]
        fcF = [x for x in fc_F if x > 0]
        def fmt(v): return f"min={min(v)} med={int(np.median(v))} max={max(v)}" if v else "none"
        print(f"  thr={use_thr}: REAL fires (GT-success) step [{fmt(fcS)}] (n={len(fcS)}/{nS}) | "
              f"FAKE fires (GT-fail) step [{fmt(fcF)}] (n={len(fcF)}/{nF}) | "
              f"GT-success actual step [{fmt(gtS)}]", flush=True)
        if fcF and fcS:
            print(f"     -> a gate (min_ep_steps) between {max(fcF) if fcF else 0} (latest fake) and "
                  f"{min(fcS)} (earliest real) cleanly separates; if they overlap, no gate fully separates.", flush=True)
    print("[calib] done", flush=True)


if __name__ == "__main__":
    main()
