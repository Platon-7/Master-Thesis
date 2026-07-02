"""Two Square probes, one rollout harness.

Rolls out N_EP episodes (POLICY=bc behavior-cloning, or POLICY=rl loading a trained
agent checkpoint), in a NON-terminating env (end_on_success=0, full EP_LEN), scoring the
growing agentview video every step with run2 -> records per-episode the full
success_prob AND progress_reward series + GT outcome + GT-success step. Then:

  PROBE A (dense-progress viability): does the progress head separate real successes from
    failures, and can a FAILURE episode reach a high progress value (i.e. is dense progress
    itself hackable)? -> max/end progress distributions for success vs fail + best separation.

  PROBE B (gate idea): on THIS policy's own rollouts, when does success_prob first cross each
    threshold for GT-fail (fake fires) vs GT-success (real fires)? If fakes and reals are
    temporally separable a min_ep_steps gate can work; if they overlap, no gate works.

Env vars: POLICY (bc|rl), ROBOMETER_FT_PATH (run2, used by env+scorer), DIAG_ENV,
          DIAG_BC (bc mode), AGENT_CKPT (rl mode, dir must hold cfg.yaml),
          DIAG_EP_LEN (default 300), N_EP (default 24).
"""
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from env.robosuite_wrapper import PixelRobosuite
from env.vlm_prompts import ROBOMIMIC_TASK_DESCRIPTIONS
from env.robosuite_vlm_env import tensor_to_pil
from common_utils import ibrl_utils as utils

ENV = os.environ.get("DIAG_ENV", "NutAssemblySquare")
EP_LEN = int(os.environ.get("DIAG_EP_LEN", "300"))
N_EP = int(os.environ.get("N_EP", "24"))
MODE = os.environ.get("POLICY", "bc").lower()
THR_LIST = [0.80, 0.85, 0.90]


def load_icl():
    """Optional ICL frames for run1 (mirrors diag_causal_calib.load_icl)."""
    p = os.environ.get("ROBOMETER_ICL_DEMO_PATH", "")
    if not p:
        return None
    idx = int(os.environ.get("ROBOMETER_ICL_DEMO_IDX", "0"))
    n = int(os.environ.get("ROBOMETER_ICL_FRAMES", "16"))
    avail = sorted(q for q in Path(p).iterdir() if q.name.startswith(f"{idx}_") and q.suffix == ".png")
    picks = np.linspace(0, len(avail) - 1, n).round().astype(int)
    return [np.asarray(Image.open(avail[i]).convert("RGB"), dtype=np.uint8) for i in picks]


def build_env(ep):
    ep = dict(ep)
    pol = list(ep["rl_cameras"])
    ep["camera_names"] = list(dict.fromkeys(pol + ["agentview"]))
    ep["rl_cameras"] = pol
    ep["episode_length"] = EP_LEN
    ep["end_on_success"] = 0
    return PixelRobosuite(**ep)


def load_policy_and_scorer():
    """Return (policy, scorer, task, icl, ep). RL mode reuses the env's loaded run2."""
    if MODE == "rl":
        import pyrallis
        import train_bc
        from train_vlm_rl import Workspace, MainConfig
        ckpt = os.environ["AGENT_CKPT"]
        cfg_path = os.path.join(os.path.dirname(ckpt), "cfg.yaml")
        cfg = pyrallis.load(MainConfig, open(cfg_path, "r"))  # type: ignore
        cfg.preload_num_data = 0
        ws = Workspace(cfg, from_main=False)
        policy = ws.agent
        policy.load_state_dict(torch.load(ckpt))
        if cfg.bc_policy:
            bc = train_bc._load_model(cfg.bc_policy, ws.eval_env, "cuda")
            policy.add_bc_policy(bc)
        policy = policy.to("cuda")
        # reuse the reward model the VLM env already loaded (no second copy)
        return policy, ws.train_env.scorer, ws.train_env.task_description, ws.train_env.icl_frames, ws.eval_env_params
    else:
        import train_bc
        from env.robometer_utils import get_robometer_4b
        bc_path = os.environ.get("DIAG_BC", "release/model/robomimic/square/model0.pt")
        scorer = get_robometer_4b(model_path=os.environ["ROBOMETER_FT_PATH"])
        policy, _, ep = train_bc.load_model(bc_path, "cuda")
        return policy, scorer, ROBOMIMIC_TASK_DESCRIPTIONS[ENV], load_icl(), ep


def main():
    print(f"[probes] MODE={MODE} env={ENV} ep_len={EP_LEN} n_ep={N_EP} rm={os.environ.get('ROBOMETER_FT_PATH')}", flush=True)
    policy, scorer, task, icl, ep = load_policy_and_scorer()
    env = build_env(ep)

    episodes = []
    for e in range(N_EP):
        np.random.seed(300 + e)
        obs, hi = env.reset()
        video = [tensor_to_pil(hi["agentview"])]
        gt, gt_step, sps, pgs = 0, -1, [], []
        with torch.no_grad(), utils.eval_mode(policy):
            for t in range(1, EP_LEN + 1):
                a = policy.act(obs, eval_mode=True)
                obs, r, term, succ, hi = env.step(a)
                video.append(tensor_to_pil(hi["agentview"]))
                out = scorer(video, task=task, icl_frames=icl)
                sps.append(float(out["success_prob"]))
                pgs.append(float(out["progress_reward"]))
                if r > 0 and gt_step < 0:
                    gt_step = t
                gt = max(gt, int(r > 0))
                if term:
                    break
        sps, pgs = np.array(sps), np.array(pgs)
        episodes.append({"gt": gt, "gt_step": gt_step, "sps": sps, "pgs": pgs})
        print(f"  ep{e:02d}: GT={'S' if gt else 'F'} gt_step={gt_step} len={len(sps)} "
              f"max_sp={sps.max():.2f}@{int(sps.argmax()) + 1} "
              f"max_pg={pgs.max():.2f} end_pg={pgs[-10:].mean():.2f}", flush=True)

    labels = np.array([d["gt"] for d in episodes])
    nS, nF = int((labels == 1).sum()), int((labels == 0).sum())
    maxpg = np.array([d["pgs"].max() for d in episodes])
    endpg = np.array([d["pgs"][-10:].mean() for d in episodes])
    maxsp = np.array([d["sps"].max() for d in episodes])

    def dist(v):
        return f"min={min(v):.2f} med={np.median(v):.2f} max={max(v):.2f}" if len(v) else "none"

    print(f"\n================ MODE={MODE}  n_succ={nS} n_fail={nF} ================", flush=True)
    print("\n=== PROBE A: progress head (is dense progress hackable?) ===", flush=True)
    print(f"  max  progress | SUCCESS: [{dist(maxpg[labels == 1])}] | FAIL: [{dist(maxpg[labels == 0])}]", flush=True)
    print(f"  end  progress | SUCCESS: [{dist(endpg[labels == 1])}] | FAIL: [{dist(endpg[labels == 0])}]", flush=True)
    best = None
    for thr in np.arange(0.10, 0.96, 0.05):
        fire = maxpg > thr
        tpr = float(fire[labels == 1].mean()) if nS else float("nan")
        fpr = float(fire[labels == 0].mean()) if nF else float("nan")
        j = tpr - fpr
        if best is None or (j == j and j > best[0]):
            best = (j, round(float(thr), 2), round(tpr, 2), round(fpr, 2))
    if best:
        print(f"  best progress separation: thr={best[1]} TPR={best[2]} FPR={best[3]} J={best[0]:.2f}", flush=True)
    print("  read: if FAIL max-progress is high (close to success), dense reward is hackable too;", flush=True)
    print("        if FAIL stays low and success climbs, the progress head is worth the RL run.", flush=True)

    def first_cross(sps, thr):
        idx = np.where(sps > thr)[0]
        return int(idx[0]) + 1 if len(idx) else -1

    print("\n=== PROBE B: success-head gate timing on THIS policy ===", flush=True)
    for thr in THR_LIST:
        fcS = [x for x in (first_cross(d["sps"], thr) for d in episodes if d["gt"] == 1) if x > 0]
        fcF = [x for x in (first_cross(d["sps"], thr) for d in episodes if d["gt"] == 0) if x > 0]
        gtS = [d["gt_step"] for d in episodes if d["gt"] == 1 and d["gt_step"] > 0]
        def fmt(v):
            return f"min={min(v)} med={int(np.median(v))} max={max(v)}" if v else "none"
        print(f"  thr={thr}: REAL fires [{fmt(fcS)}] (n={len(fcS)}/{nS}) | "
              f"FAKE fires [{fmt(fcF)}] (n={len(fcF)}/{nF}) | GT-success step [{fmt(gtS)}]", flush=True)
        if fcF and fcS:
            sep = max(fcF) < min(fcS)
            print(f"     -> latest fake={max(fcF)}, earliest real={min(fcS)}: "
                  f"{'SEPARABLE by a gate' if sep else 'OVERLAP -> no min_ep_steps gate separates'}", flush=True)
    print("[probes] done", flush=True)


if __name__ == "__main__":
    main()
