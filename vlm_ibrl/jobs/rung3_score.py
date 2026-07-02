"""Rung-3 stage B: score dumped rollout videos with a Robometer checkpoint and
report success-vs-failure separability (the gate metric).

Usage: python jobs/rung3_score.py <dump.npz> <robometer_ckpt_path> <model_tag>
Prints per-episode success_prob/progress and AUC + d' (success head & progress head)
between GT-success and GT-failure episodes.
"""
import json
import sys

import numpy as np
import scipy.stats as st

from env.robometer_utils import get_robometer_4b
from env.vlm_prompts import ROBOMIMIC_TASK_DESCRIPTIONS


def auc(labels, scores):
    labels = np.asarray(labels); scores = np.asarray(scores, dtype=float)
    P = int((labels == 1).sum()); N = int((labels == 0).sum())
    if P == 0 or N == 0:
        return float("nan")
    r = st.rankdata(scores)
    return float((r[labels == 1].sum() - P * (P + 1) / 2) / (P * N))


def dprime(labels, scores):
    labels = np.asarray(labels); scores = np.asarray(scores, dtype=float)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    sd = np.sqrt(0.5 * (pos.var(ddof=1) + neg.var(ddof=1)))
    return float((pos.mean() - neg.mean()) / sd) if sd > 0 else float("nan")


def main():
    dump_path, ckpt, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    d = np.load(dump_path, allow_pickle=True)
    labels = d["labels"]
    env_name = str(d["env_name"])
    eps = [d[k] for k in sorted([k for k in d.files if k.startswith("ep")],
                                key=lambda s: int(s[2:]))]
    task = ROBOMIMIC_TASK_DESCRIPTIONS[env_name]
    print(f"[score:{tag}] env={env_name} episodes={len(eps)} "
          f"pos={int((labels==1).sum())} neg={int((labels==0).sum())} ckpt={ckpt}", flush=True)

    scorer = get_robometer_4b(model_path=ckpt)
    sp, pr = [], []
    for i, ep in enumerate(eps):
        frames = [ep[j] for j in range(ep.shape[0])]   # list of (H,W,C) uint8
        out = scorer(frames, task=task, episode_id=i)
        sp.append(float(out["success_prob"])); pr.append(float(out["progress_reward"]))
        print(f"[score:{tag}] ep{i:02d} y={int(labels[i])} "
              f"success_prob={sp[-1]:.4f} progress={pr[-1]:.4f}", flush=True)

    sp, pr = np.array(sp), np.array(pr)
    res = dict(
        model=tag, env=env_name, n=len(eps),
        success_auc=round(auc(labels, sp), 4), success_dprime=round(dprime(labels, sp), 4),
        progress_auc=round(auc(labels, pr), 4), progress_dprime=round(dprime(labels, pr), 4),
        succ_mean_pos=round(float(sp[labels == 1].mean()), 4),
        succ_mean_neg=round(float(sp[labels == 0].mean()), 4),
    )
    print(f"[RUNG3-RESULT] {json.dumps(res)}", flush=True)


if __name__ == "__main__":
    main()
