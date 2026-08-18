#!/usr/bin/env python3
"""On-policy AUROC of the reward-model return, from [DISCRIM] training logs.

The offline counterpart is measured on GT-actor rollouts (causal_calib/*.json). This
one is measured on the RL policy's OWN visited distribution, during learning -- the
comparison that tests whether an offline reward metric transfers downstream.

Needs successes to exist: with 0 positives the AUROC is undefined and the run tells
us only that the failure reproduced.
"""
import re, sys, glob
import numpy as np

PAT = re.compile(r"\[DISCRIM\] ep=(\S+) gt_success=(\d) rm_reward_sum=([-0-9.]+) len=(\d+)")

def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    sc = np.r_[pos, neg]
    o = np.argsort(sc); r = np.empty(len(sc)); r[o] = np.arange(1, len(sc) + 1)
    return (r[lab == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

for f in sorted(sys.argv[1:]):
    S, F, lens = [], [], []
    for line in open(f, errors="ignore"):
        m = PAT.search(line)
        if not m:
            continue
        _, gt, rs, ln = m.groups()
        (S if gt == "1" else F).append(float(rs)); lens.append(int(ln))
    n = len(S) + len(F)
    if n == 0:
        print(f"{f.split('/')[-1]:28s} no [DISCRIM] lines yet"); continue
    a = auroc(np.array(S), np.array(F))
    # 95% CI (Hanley-McNeil) so an underpowered result is visible as such
    ci = ""
    if len(S) and len(F) and not np.isnan(a):
        q1 = a / (2 - a); q2 = 2 * a * a / (1 + a)
        se = np.sqrt((a*(1-a) + (len(S)-1)*(q1-a*a) + (len(F)-1)*(q2-a*a)) / (len(S)*len(F)))
        ci = f"  95% CI [{max(0,a-1.96*se):.3f}, {min(1,a+1.96*se):.3f}]"
    print(f"{f.split('/')[-1]:28s} episodes={n:5d} succ={len(S):4d} ({len(S)/n:5.1%})  "
          f"AUROC={a:.3f}{ci}")
    if len(S):
        print(f"{'':28s} mean return  success={np.mean(S):9.3f}  failure={np.mean(F):9.3f}"
              f"   median len={int(np.median(lens))}")
