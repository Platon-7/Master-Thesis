"""DEFINITIVE scoring-path A/B.

The IBRL env scores reward by calling the scorer with the FULL episode video
(vlm_envs.py:433: scorer(self.current_video, icl=self.icl_frames)) and letting
the scorer subsample internally to max_frames. Our E2 test pre-subsampled to 16
frames ourselves. Those two paths gave different AUC on the same BC policy
(0.66 inline vs 0.88 direct). This script scores EVERY clip BOTH ways, holding
policy / ICL / model / clip identical, so the only variable is the scoring path.

Per episode we record:
  - sp_inline : scorer(full current_video, icl=env.icl_frames)   [exact env path]
  - sp_direct : scorer(linspace-subsample-to-16, icl=env.icl_frames)
  - env_success (ground truth)

If sp_inline == sp_direct per episode  => no scoring-path bug; the prior 0.88/0.66
gap was ICL/seed/sampling between the E2 and sp-dist harnesses.
If they differ                          => the env scoring path is the culprit.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pyrallis
import torch

sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl")
sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl/mw_main")
sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer")

from common_utils import ibrl_utils as utils
from train_rl_vlm_mw import MainConfig, Workspace

TASK = "push a mug under a coffee machine"
N = 16


def subsample(frames, n):
    if len(frames) <= n:
        return frames
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx]


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    a = np.concatenate([np.asarray(pos), np.asarray(neg)])
    order = np.argsort(a, kind="mergesort"); sa = a[order]; rk = np.empty(len(a)); i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        rk[i:j + 1] = (i + j + 2) / 2; i = j + 1
    inv = np.empty_like(order, dtype=float); inv[order] = rk
    return float((inv[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--n-episodes", type=int, default=60)
    ap.add_argument("--policy", choices=["bc", "rl"], default="bc")
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    cfg = pyrallis.load(MainConfig, open(args.run_dir / "cfg.yaml"))
    cfg.use_wb = 0
    cfg.save_dir = str(args.run_dir.parent / f"scoreab_{args.run_dir.name}")
    ws = Workspace(cfg)
    env = ws.train_env
    scorer = env.scorer
    if args.policy == "rl":
        sd = torch.load(str(args.run_dir / "latest.pt"), map_location="cuda", weights_only=False)
        ws.agent.load_state_dict(sd); actor = ws.agent; print("policy = RL-trained")
    else:
        actor = ws.bc_policy if ws.bc_policy is not None else ws.agent; print("policy = BC")

    icl = env.icl_frames  # whatever the env loaded (release demo via ROBOMETER_ICL_DEMO_PATH)
    print(f"env.icl_frames: {None if icl is None else len(icl)} frames")

    rows = []
    with torch.no_grad(), utils.eval_mode(actor):
        for ep in range(args.n_episodes):
            np.random.seed(2000 + ep)
            obs, image_obs = env.reset()
            terminal = False; succ = 0
            while not terminal:
                action = actor.act(obs, eval_mode=True).numpy()
                obs, reward, terminal, success, image_obs = env.step(action)
                if success:
                    succ = 1
            full = [np.asarray(f) for f in env.current_video]
            sp_inline = float(scorer(full, task=TASK, icl_frames=icl)["success_prob"])
            sp_direct = float(scorer(subsample(full, N), task=TASK, icl_frames=icl)["success_prob"])
            rows.append({"episode": ep, "env_success": succ,
                         "n_full": len(full), "sp_inline": sp_inline, "sp_direct": sp_direct})
            if (ep + 1) % 15 == 0:
                print(f"  {ep+1}/{args.n_episodes}  succ={sum(r['env_success'] for r in rows)}")

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "env_success", "n_full", "sp_inline", "sp_direct"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {args.out_csv}")

    si = [r["sp_inline"] for r in rows]; sd = [r["sp_direct"] for r in rows]
    lab = [r["env_success"] for r in rows]
    diff = np.abs(np.array(si) - np.array(sd))
    print("\n==================== RESULT ====================")
    print(f"per-episode |sp_inline - sp_direct|: mean={diff.mean():.4f} max={diff.max():.4f} median={np.median(diff):.4f}")
    ps_i = [si[k] for k in range(len(lab)) if lab[k] == 1]; ng_i = [si[k] for k in range(len(lab)) if lab[k] == 0]
    ps_d = [sd[k] for k in range(len(lab)) if lab[k] == 1]; ng_d = [sd[k] for k in range(len(lab)) if lab[k] == 0]
    print(f"n_success={sum(lab)}  n_failure={len(lab)-sum(lab)}")
    print(f"AUC inline (env path) : {auc(ps_i, ng_i):.3f}   succ_mean={np.mean(ps_i) if ps_i else float('nan'):.4f} fail_mean={np.mean(ng_i) if ng_i else float('nan'):.4f}")
    print(f"AUC direct (16-frame) : {auc(ps_d, ng_d):.3f}   succ_mean={np.mean(ps_d) if ps_d else float('nan'):.4f} fail_mean={np.mean(ng_d) if ng_d else float('nan'):.4f}")


if __name__ == "__main__":
    main()
