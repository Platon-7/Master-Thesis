"""Curve-B root-cause + threshold test.

Question: is run1+ICL's success_prob reliable on the RL policy's (OOD) states, or
does it overlap real-success scores so that NO calibration threshold separates them?

Method: roll out (a) the FAILED 907 RL policy (OOD states, ~all GT-fail) and (b) the
BC policy (in-distribution, ~half GT-success) for N_EP full 200-step episodes each.
Score every growing agentview video causally (every KSTEP frames) with BOTH the
run1+ICL head (ICL on, via the workspace's own train_env.scorer) and the run2 head
(no ICL). Report per-episode GT vs causal-max success_prob for both heads, then a
threshold-separability summary.

Env: RL_CKPT (907 latest.pt, with its cfg.yaml), RUN2_PATH (run2_noicl ckpt),
     ROBOMETER_FT_PATH=run1_icl + ROBOMETER_ICL_DEMO_PATH/IDX/FRAMES (so the workspace
     builds the run1+ICL scorer), N_EP, SAVE_ROOT.
"""
import os
import copy
import numpy as np
import torch
import pyrallis
from pathlib import Path
from PIL import Image

import train_bc
import train_vlm_rl
from train_vlm_rl import Workspace, MainConfig
from env.robosuite_wrapper import PixelRobosuite
from env.robometer_utils import get_robometer_4b
from env.vlm_prompts import ROBOMIMIC_TASK_DESCRIPTIONS
from env.robosuite_vlm_env import tensor_to_pil
from common_utils import ibrl_utils as utils

ENV = "PickPlaceCan"
TASK = ROBOMIMIC_TASK_DESCRIPTIONS[ENV]
KSTEP = 10                       # causal score every 10 frames (cuts cost 10x)
N_EP = int(os.environ.get("N_EP", "8"))
SAVE_ROOT = os.environ.get("SAVE_ROOT", "/shared/home/PKA4388/vlm_ibrl_runs/diag_curveB_ood")


def build_eval_env(ep_params):
    ep = dict(ep_params)
    pol = list(ep["rl_cameras"])
    ep["camera_names"] = list(dict.fromkeys(pol + ["agentview"]))
    ep["rl_cameras"] = pol
    ep["episode_length"] = 200
    ep["end_on_success"] = 0      # run the FULL trajectory regardless of GT success
    return PixelRobosuite(**ep)


def rollout(agent, env, n_ep, seed0, tag):
    """Full 200-step rollouts; save agentview frames per ep. Returns [(gt, gt_step, dir, nframes)]."""
    out = []
    for e in range(n_ep):
        np.random.seed(seed0 + e)
        obs, hi = env.reset()
        frames = [tensor_to_pil(hi["agentview"])]
        gt, gt_step = 0, -1
        with torch.no_grad(), utils.eval_mode(agent):
            for t in range(1, 201):
                a = agent.act(obs, eval_mode=True)
                obs, r, term, succ, hi = env.step(a)
                frames.append(tensor_to_pil(hi["agentview"]))
                if r > 0 and gt_step < 0:
                    gt_step = t
                gt = max(gt, int(r > 0))
        d = Path(SAVE_ROOT) / f"{tag}_{e:02d}"
        d.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            f.save(d / f"{i:03d}.png")
        out.append((gt, gt_step, str(d), len(frames)))
        print(f"  [{tag}] ep{e:02d}: GT={'S' if gt else 'F'} gt_step={gt_step} len={len(frames)}", flush=True)
    return out


def causal_max(scorer, frames_dir, n_frames, icl):
    frames = [np.asarray(Image.open(Path(frames_dir) / f"{i:03d}.png").convert("RGB"), dtype=np.uint8)
              for i in range(n_frames)]
    mx, last = 0.0, 0.0
    for t in range(KSTEP, n_frames + 1, KSTEP):
        sp = float(scorer(frames[:t], task=TASK, icl_frames=icl)["success_prob"])
        mx = max(mx, sp)
        last = sp
    return mx, last


def score_all(scorer, icl, tag, rl_eps, bc_eps):
    res = {}
    for name, eps in [("RL", rl_eps), ("BC", bc_eps)]:
        for i, (gt, gtstep, d, nf) in enumerate(eps):
            mx, last = causal_max(scorer, d, nf, icl)
            res[(name, i)] = (gt, mx, last)
            print(f"  [{tag}] {name}{i:02d} GT={'S' if gt else 'F'} maxSP={mx:.2f} lastSP={last:.2f}", flush=True)
    return res


def summarize(res, name):
    for src in ["BC", "RL"]:
        S = [v[1] for k, v in res.items() if k[0] == src and v[0] == 1]
        F = [v[1] for k, v in res.items() if k[0] == src and v[0] == 0]
        ms = f"{np.mean(S):.2f}" if S else "--"
        mf = f"{np.mean(F):.2f}" if F else "--"
        print(f"  [{name}] {src}: GT-success maxSP n={len(S)} mean={ms} (vals={[round(x,2) for x in S]}) | "
              f"GT-fail maxSP n={len(F)} mean={mf} (vals={[round(x,2) for x in F]})", flush=True)


def main():
    rl_ckpt = os.environ["RL_CKPT"]
    run2 = os.environ["RUN2_PATH"]

    # --- inline load_model so we keep the workspace handle (gives the run1+ICL scorer) ---
    cfg_path = os.path.join(os.path.dirname(rl_ckpt), "cfg.yaml")
    cfg = pyrallis.load(MainConfig, open(cfg_path, "r"))  # type: ignore
    cfg.preload_num_data = 0
    ws = Workspace(cfg, from_main=False)
    ws.agent.load_state_dict(torch.load(rl_ckpt))
    if cfg.bc_policy:
        ws.agent.add_bc_policy(copy.deepcopy(train_bc._load_model(cfg.bc_policy, ws.eval_env, "cuda")))
    rl_agent = ws.agent.to("cuda")
    scorer1 = ws.train_env.scorer
    icl1 = ws.train_env.icl_frames
    print(f"[diag-ood] run1+ICL scorer ready (icl={len(icl1) if icl1 is not None else 0}f) rl_ckpt={rl_ckpt}", flush=True)

    bc_agent, _, bc_ep = train_bc.load_model(cfg.bc_policy, "cuda")
    rl_env = build_eval_env(ws.eval_env_params)
    bc_env = build_eval_env(bc_ep)

    # ---- Pass 1: rollouts (saves frames) ----
    print("=== rollouts ===", flush=True)
    rl_eps = rollout(rl_agent, rl_env, N_EP, 500, "RL")
    bc_eps = rollout(bc_agent, bc_env, N_EP, 700, "BC")

    # ---- score the SAME states with the 2x2: {run1,run2} x {ICL,noICL} ----
    print("=== run1 + ICL  (the FAILED config) ===", flush=True)
    r1_icl = score_all(scorer1, icl1, "run1+ICL", rl_eps, bc_eps)
    print("=== run1, no ICL ===", flush=True)
    r1_noicl = score_all(scorer1, None, "run1-noICL", rl_eps, bc_eps)

    # free run1, load run2
    del scorer1, ws, rl_agent
    torch.cuda.empty_cache()
    scorer2 = get_robometer_4b(model_path=run2)
    print("=== run2, no ICL  (the WORKING config) ===", flush=True)
    r2_noicl = score_all(scorer2, None, "run2-noICL", rl_eps, bc_eps)
    print("=== run2 + ICL  (run2 was NOT ICL-trained; perturbation check) ===", flush=True)
    r2_icl = score_all(scorer2, icl1, "run2+ICL", rl_eps, bc_eps)

    print("\n=== SUMMARY (causal-max success_prob by policy x GT) ===", flush=True)
    for res, name in [(r1_icl, "run1+ICL "), (r1_noicl, "run1-noICL"),
                      (r2_noicl, "run2-noICL"), (r2_icl, "run2+ICL ")]:
        summarize(res, name)
    print("\nReading:", flush=True)
    print(" * The CONTROL is run2-noICL on RL GT-fail states: this is the WORKING reward; its maxSP "
          "there should be LOW (it correctly says 'not success').", flush=True)
    print(" * If run1+ICL maxSP on RL GT-fail states is HIGH -> it is fooled on OOD states.", flush=True)
    print(" * Compare run1+ICL vs run1-noICL: if noICL is fine but +ICL is fooled -> ICL-at-runtime is "
          "the culprit (ICL hurts OOD even though it helps in-distribution).", flush=True)
    print(" * Compare run1-noICL vs run2-noICL: if run1-noICL is also fooled -> it's the run1 checkpoint, "
          "not ICL.", flush=True)
    print("[diag-ood] done", flush=True)


if __name__ == "__main__":
    main()
