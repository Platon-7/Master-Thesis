"""On-policy false-positive metric for the ManiSkill PullCube runs.

Each training episode is logged with the reward model's per-step score and the
simulator's ground truth at every step, so reward hacking can be counted rather
than eyeballed off a curve. Per episode we take the peak score the model ever
assigned and whether the task was ever actually solved.

Reported per model, pooled over its five seeds:
  solve rate   fraction of episodes the simulator marks solved
  peak(fail)   mean peak score on episodes that were NEVER solved
  peak(succ)   mean peak score on episodes that were solved
  AUROC        P(peak on a solved episode > peak on an unsolved one), scale free,
               so it is comparable across models whose scores live on different ranges
  FPR@95TPR    put the threshold where 95 percent of solved episodes are caught,
               then the fraction of never-solved episodes that also clear it

Raw peak values are in each model's own units and are not comparable across rows.
AUROC and FPR@95TPR are.
"""
import glob, json, os, re
import numpy as np

R = "/scratch-shared/pkarageorgis1/roboref_runs"
# Seed sets matching the paper figure. run2 drew from a larger pool; the rest are 0-4.
SEEDS = {"run2": [3, 9, 0, 1, 2]}
DEFAULT = [0, 1, 2, 3, 4]
MODELS = [("run2", "RoboRef"), ("run3", "Robometer + Failures"), ("base", "Robometer-4B"),
          ("robodopamine", "RoboDopamine"), ("roboreward", "RoboReward-4B"), ("lrm", "LRM-TRI")]


def dirs_for(model, seed):
    """All legs for one seed, oldest first, skipping empty ones."""
    out = []
    for d in sorted(glob.glob(f"{R}/ms_PullCube-v1_{model}_dense_s{seed}_*")):
        if not re.match(rf".*_s{seed}_\d+_\d+$", d):
            continue
        p = f"{d}/episodes.jsonl"
        if os.path.exists(p) and os.path.getsize(p) > 0:
            out.append(p)
    return out


def load(paths, min_eps=500):
    """-> (peak score, solved) per episode, concatenated over a seed's legs."""
    s, y = [], []
    for p in paths:
        n = 0
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            sc = r.get("score_max")
            if sc is None:
                sp = r.get("score_per_step") or []
                sc = max(sp) if sp else None
            if sc is None:
                continue
            s.append(float(sc)); y.append(int(r.get("gt_solved_anytime", 0))); n += 1
        if n < min_eps:
            print(f"    (skipped short leg {os.path.basename(os.path.dirname(p))}, {n} eps)")
    return np.array(s), np.array(y)


def auroc(s, y):
    """Tie-aware. RoboReward emits only 5 distinct values and LRM 6, so ordinal
    ranks would silently break ties in whatever order argsort happened to give."""
    from scipy.stats import rankdata
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))       # averages tied ranks
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def fpr_at_tpr(s, y, tpr=0.95):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    t = np.quantile(pos, 1 - tpr)
    return float((neg >= t).mean())


def n_levels(s):
    return len(np.unique(np.round(s, 6)))


print(f"{'model':22s} {'eps':>7} {'solve':>7} {'peak(fail)':>11} {'peak(succ)':>11} {'AUROC':>7} {'FPR@95':>7} {'levels':>7}")
print("-" * 82)
for key, name in MODELS:
    S, Y = [], []
    for sd in SEEDS.get(key, DEFAULT):
        p = dirs_for(key, sd)
        if not p:
            print(f"  {name}: seed {sd} missing"); continue
        a, b = load(p)
        S.append(a); Y.append(b)
    if not S:
        continue
    s = np.concatenate(S); y = np.concatenate(Y)
    print(f"{name:22s} {len(s):7d} {y.mean():7.3f} {s[y==0].mean():11.4f} "
          f"{(s[y==1].mean() if (y==1).any() else float('nan')):11.4f} "
          f"{auroc(s,y):7.3f} {fpr_at_tpr(s,y):7.3f} {n_levels(s):7d}")
