"""Fair (unconfounded) OOD discrimination test for the 2x2 {run1,run2} x {ICL,noICL}.

Motivation: the earlier diag (945) scored OOD states from the 907 policy, which was trained to
maximize run1+ICL's reward -> those states are ADVERSARIAL to run1+ICL, so the cross-model OOD
comparison is circular. Here the OOD states come from NOISY-BC rollouts (BC actions + Gaussian
noise) -- neutral to every reward model. We then score clean-BC (real successes) and noisy-BC
(neutral OOD, ~all GT-fail) with all four configs and report separation.

Reading: a config that keeps real-success score HIGH while pushing neutral-OOD-fail score LOW is
the genuinely more discriminative / OOD-robust reward. That is the fair basis to pick the 1-demo
reward model.

Env: ROBOMETER_FT_PATH=run1_icl, RUN2_PATH=run2_noicl, ROBOMETER_ICL_DEMO_PATH/IDX/FRAMES, N_EP, NOISE.
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
TASK = ROBOMIMIC_TASK_DESCRIPTIONS[ENV]
KSTEP = 10
N_EP = int(os.environ.get("N_EP", "10"))
NOISE = float(os.environ.get("NOISE", "0.4"))     # action-noise std for the neutral OOD set
BC_PATH = os.environ.get("BC_PATH", "release/model/robomimic/can/model0.pt")
SAVE_ROOT = os.environ.get("SAVE_ROOT", "/shared/home/PKA4388/vlm_ibrl_runs/diag_fairsep")


def load_icl():
    p = os.environ.get("ROBOMETER_ICL_DEMO_PATH", "")
    if not p:
        return None
    idx = int(os.environ.get("ROBOMETER_ICL_DEMO_IDX", "0"))
    n = int(os.environ.get("ROBOMETER_ICL_FRAMES", "16"))
    avail = sorted(q for q in Path(p).iterdir() if q.name.startswith(f"{idx}_") and q.suffix == ".png")
    picks = np.linspace(0, len(avail) - 1, n).round().astype(int)
    return [np.asarray(Image.open(avail[i]).convert("RGB"), dtype=np.uint8) for i in picks]


def build_eval_env(ep_params):
    ep = dict(ep_params)
    pol = list(ep["rl_cameras"])
    ep["camera_names"] = list(dict.fromkeys(pol + ["agentview"]))
    ep["rl_cameras"] = pol
    ep["episode_length"] = 200
    ep["end_on_success"] = 0
    return PixelRobosuite(**ep)


def rollout_noisy(agent, env, n_ep, seed0, noise, tag):
    out = []
    for e in range(n_ep):
        np.random.seed(seed0 + e)
        torch.manual_seed(seed0 + e)
        obs, hi = env.reset()
        frames = [tensor_to_pil(hi["agentview"])]
        gt = 0
        with torch.no_grad(), utils.eval_mode(agent):
            for t in range(1, 201):
                a = agent.act(obs, eval_mode=True)
                if noise > 0:
                    a = torch.clamp(a + noise * torch.randn_like(a), -1.0, 1.0)
                obs, r, term, succ, hi = env.step(a)
                frames.append(tensor_to_pil(hi["agentview"]))
                gt = max(gt, int(r > 0))
        d = Path(SAVE_ROOT) / f"{tag}_{e:02d}"
        d.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            f.save(d / f"{i:03d}.png")
        out.append((gt, str(d), len(frames)))
        print(f"  [{tag}] ep{e:02d}: GT={'S' if gt else 'F'} len={len(frames)}", flush=True)
    return out


def causal_max(scorer, frames_dir, n_frames, icl):
    frames = [np.asarray(Image.open(Path(frames_dir) / f"{i:03d}.png").convert("RGB"), dtype=np.uint8)
              for i in range(n_frames)]
    mx = 0.0
    for t in range(KSTEP, n_frames + 1, KSTEP):
        mx = max(mx, float(scorer(frames[:t], task=TASK, icl_frames=icl)["success_prob"]))
    return mx


def score_all(scorer, icl, tag, clean_eps, ood_eps):
    res = {}
    for name, eps in [("CLEAN", clean_eps), ("OOD", ood_eps)]:
        for i, (gt, d, nf) in enumerate(eps):
            mx = causal_max(scorer, d, nf, icl)
            res[(name, i)] = (gt, mx)
            print(f"  [{tag}] {name}{i:02d} GT={'S' if gt else 'F'} maxSP={mx:.2f}", flush=True)
    return res


def summarize(res, name):
    cs = [v[1] for k, v in res.items() if k[0] == "CLEAN" and v[0] == 1]   # real successes
    of = [v[1] for k, v in res.items() if k[0] == "OOD" and v[0] == 0]     # neutral OOD failures
    ms = f"{np.mean(cs):.2f}" if cs else "--"
    mf = f"{np.mean(of):.2f}" if of else "--"
    gap = f"{np.mean(cs) - np.mean(of):.2f}" if (cs and of) else "--"
    worst_ok = f"{min(cs):.2f}" if cs else "--"
    worst_bad = f"{max(of):.2f}" if of else "--"
    print(f"  [{name}] real-success n={len(cs)} mean={ms} (min={worst_ok}) | "
          f"neutral-OOD-fail n={len(of)} mean={mf} (max={worst_bad}) | GAP={gap} | "
          f"separable={'YES' if (cs and of and min(cs) > max(of)) else 'no/overlap'}", flush=True)


def main():
    # Model set: "name:path,name:path,...". Default = the Robometer pair (run1/run2).
    # NOTE: run a Qwen3.5 set (run4/run5) in a SEPARATE process from the Robometer
    # set — get_robometer_4b's Qwen3.5 dispatch is sys.path/import-order based, so
    # mixing Qwen3-VL (Robometer) and Qwen3.5 loads in one process is unsafe.
    models_env = os.environ.get("MODELS", "").strip()
    if models_env:
        models = [tuple(m.split(":", 1)) for m in models_env.split(",") if m.strip()]
    else:
        models = [("run1", os.environ["ROBOMETER_FT_PATH"]), ("run2", os.environ["RUN2_PATH"])]
    icl = load_icl()
    print(f"[fairsep] models={[m[0] for m in models]} N_EP={N_EP} NOISE={NOISE} "
          f"icl={len(icl) if icl else 0}f", flush=True)
    for n, p in models:
        print(f"          {n} -> {p}", flush=True)

    bc_agent, _, bc_ep = train_bc.load_model(BC_PATH, "cuda")
    env = build_eval_env(bc_ep)

    print("=== rollouts (deterministic: BC clean seed700, noisy seed900) ===", flush=True)
    clean_eps = rollout_noisy(bc_agent, env, N_EP, 700, 0.0, "CLEAN")
    ood_eps = rollout_noisy(bc_agent, env, N_EP, 900, NOISE, "OOD")

    results = []
    for name, path in models:
        scorer = get_robometer_4b(model_path=path)
        for tag, icl_arg in [("+ICL", icl), ("-noICL", None)]:
            label = f"{name}{tag}"
            print(f"=== {label} ===", flush=True)
            results.append((label, score_all(scorer, icl_arg, label, clean_eps, ood_eps)))
        del scorer
        torch.cuda.empty_cache()

    print("\n=== FAIR SUMMARY (real-success vs NEUTRAL OOD-fail; bigger GAP + separable = better reward) ===", flush=True)
    for label, res in results:
        summarize(res, label.ljust(11))
    print("[fairsep] done", flush=True)


if __name__ == "__main__":
    main()
