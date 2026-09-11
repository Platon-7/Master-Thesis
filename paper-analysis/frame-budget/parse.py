"""Parse the per-step policy_ranking dumps.

Each record is one task at one eval step:
  avg_differences        "succ-fail:0.06"                      -> the task's margin
  quality_and_rews_last  "failure:[...],successful:[...]"      -> per-clip last-frame scores
The thesis' panel (a) takes the max margin over a split's tasks; panel (b) takes
rank accuracy, the share of within-task (success, failure) pairs ordered right.
"""
import json, os, re, glob
import numpy as np

def _lists(s):
    """'failure:[a,b],successful:[c,d]' -> {'failure':[a,b],'successful':[c,d]}"""
    out = {}
    for m in re.finditer(r"(\w+):\[([^\]]*)\]", s or ""):
        vals = [float(x) for x in m.group(2).split(",") if x.strip() not in ("", "nan")]
        out[m.group(1)] = vals
    return out

def _margin(s):
    m = re.search(r"succ-fail:\s*(-?[\d.]+)", s or "")
    return float(m.group(1)) if m else None

def load(path, field="quality_and_rews_last"):
    recs = json.load(open(path))
    rows = []
    for r in recs:
        rows.append({"task": r.get("task"),
                     "margin": _margin(r.get("avg_differences")),
                     "scores": _lists(r.get(field))})
    return rows

def widest_margin(rows):
    m = [r["margin"] for r in rows if r["margin"] is not None]
    return (max(m), float(np.mean(m)), len(m)) if m else (float("nan"),)*2 + (0,)

def rank_accuracy(rows):
    ok = tot = 0.0
    for r in rows:
        S, F = r["scores"].get("successful", []), r["scores"].get("failure", [])
        for s in S:
            for f in F:
                tot += 1
                ok += 1.0 if s > f else (0.5 if s == f else 0.0)
    return (ok/tot if tot else float("nan")), int(tot)

def steps(run_dir):
    base = os.path.join(run_dir, "policy_ranking_samples")
    out = []
    for p in glob.glob(os.path.join(base, "step_*")):
        m = re.search(r"step_(\d+)$", p)
        if m: out.append((int(m.group(1)), p))
    return sorted(out)
