"""Extend the RoboDopamine curves to the full 300k budget.

The five seeds ran in two legs. Leg 1 covered absolute steps 0..150k; leg 2 resumed
from that checkpoint and ran its own 0..150k, which is absolute 150k..300k. The
existing maniskill_curves.json was built while leg 2 was only ~60k in, which is why
those curves stopped at 210k.

Parses "step N (episode" to track the step and the following "[EVAL] Success Rate: X%"
from each leg's training.log, applies the +150000 offset to leg 2, and merges.
"""
import json, re, glob, os
import numpy as np

D = "/projects/prjs1958/pkarageorgis1/maniskill_figure"
RUNS = "/scratch-shared/pkarageorgis1/roboref_runs"
STEP_RE = re.compile(r"step (\d+) \(episode")
SUCC_RE = re.compile(r"\[EVAL\] Success Rate: ([\d.]+)%")


def parse(log):
    """-> [(step, success_pct)] using the most recent step seen before each eval."""
    out, cur = [], None
    with open(log, errors="ignore") as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                cur = int(m.group(1)); continue
            m = SUCC_RE.search(line)
            if m and cur is not None:
                out.append((cur, float(m.group(1))))
    return out


def leg_dirs(seed):
    """The two legs for a seed, oldest first."""
    ds = sorted(glob.glob(f"{RUNS}/ms_PullCube-v1_robodopamine_dense_s{seed}_*"),
                key=lambda p: os.path.basename(p).split("_")[-2:])
    return [d for d in ds if os.path.exists(f"{d}/training.log")]


cur = json.load(open(f"{D}/maniskill_curves.json"))
for seed in range(5):
    legs = leg_dirs(seed)
    merged = {}
    for d in legs:
        pts = parse(f"{d}/training.log")
        if not pts:
            continue
        # a leg that starts near 0 but whose dir is the later one is the resumed leg
        resumed = "RESUME" in open(f"{d}/config.yaml", errors="ignore").read() if os.path.exists(f"{d}/config.yaml") else False
        off = 0
        if max(p[0] for p in pts) <= 155000 and legs.index(d) > 0:
            off = 150000
        for s, v in pts:
            merged[s + off] = v
        print(f"  seed {seed} leg {os.path.basename(d)[-15:]}: {len(pts)} evals, "
              f"steps {min(p[0] for p in pts)}..{max(p[0] for p in pts)} offset +{off}")
    if merged:
        arr = sorted(merged.items())
        cur[f"robodopamine|s{seed}"] = [[float(s), float(v)] for s, v in arr]
        print(f"  -> robodopamine|s{seed}: {len(arr)} pts, {arr[0][0]}..{arr[-1][0]}, last={arr[-1][1]}%")

json.dump(cur, open(f"{D}/maniskill_curves.json", "w"))
print("\nwrote maniskill_curves.json")
