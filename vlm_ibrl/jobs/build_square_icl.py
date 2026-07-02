"""Build a 224x224 Square ICL demo folder from a successful BC rollout.

Mirrors the Can ICL folder (icl_demo_can_median): rolls out the Square BC policy until a
ground-truth success, then saves NFRAMES evenly-spaced agentview frames (rendered at 224)
up to the success step, named 0_000.png ... so load_icl() picks them up.
"""
import os
from pathlib import Path
import numpy as np
import torch

import train_bc
from env.robosuite_wrapper import PixelRobosuite
from env.robosuite_vlm_env import tensor_to_pil
from common_utils import ibrl_utils as utils

OUT = os.environ.get("ICL_OUT", "/shared/home/PKA4388/vlm_ibrl_runs/icl_demo_square_median")
BC = os.environ.get("DIAG_BC", "release/model/robomimic/square/model0.pt")
EP_LEN = int(os.environ.get("DIAG_EP_LEN", "300"))
NFR = int(os.environ.get("ICL_NFRAMES", "32"))


def main():
    policy, _, ep = train_bc.load_model(BC, "cuda")
    ep = dict(ep)
    pol = list(ep["rl_cameras"])
    ep["camera_names"] = list(dict.fromkeys(pol + ["agentview"]))
    ep["episode_length"] = EP_LEN
    ep["end_on_success"] = 0
    env = PixelRobosuite(**ep)
    Path(OUT).mkdir(parents=True, exist_ok=True)

    for e in range(40):
        np.random.seed(1000 + e)
        obs, hi = env.reset()
        frames = [tensor_to_pil(hi["agentview"])]
        gt_step = -1
        with torch.no_grad(), utils.eval_mode(policy):
            for t in range(1, EP_LEN + 1):
                a = policy.act(obs, eval_mode=True)
                obs, r, term, succ, hi = env.step(a)
                frames.append(tensor_to_pil(hi["agentview"]))
                if r > 0 and gt_step < 0:
                    gt_step = t
                if term:
                    break
        if gt_step > 0:
            seq = frames[:gt_step + 1]
            picks = np.linspace(0, len(seq) - 1, NFR).round().astype(int)
            for i, pi in enumerate(picks):
                seq[int(pi)].save(f"{OUT}/0_{i:03d}.png")
            w, h = seq[0].size
            print(f"[build_icl] ep{e}: success@{gt_step}, saved {NFR} frames ({w}x{h}) to {OUT}", flush=True)
            return
        print(f"[build_icl] ep{e}: no success (len {len(frames)})", flush=True)
    raise SystemExit("[build_icl] FAILED: no BC success in 40 tries")


if __name__ == "__main__":
    main()
