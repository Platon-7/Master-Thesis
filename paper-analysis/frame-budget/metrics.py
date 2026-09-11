"""Metrics behind thesis Figure 5.3 (the frame-budget ablation), recomputed from
the training-time policy_ranking dumps rather than from W&B.

Panel (a)  widest success-failure margin per split.
           A task's margin is mean(score | successes) - mean(score | failures);
           the panel plots the max over that split's tasks.
Panel (b)  rank accuracy on DROID over training. Rank accuracy is the fraction
           of (success, failure) pairs inside a task that the score orders right.

"score" is the model's scalar readout of the LAST frame: the expectation of the
progress head's bin distribution, which is what convert_bins_to_continuous does
(softmax over the 10 bin logits, dotted with the bin centres).
"""
import json, os, glob, re
import numpy as np

SPLITS = ["droid", "robometer", "metaworld", "failsafe"]


def _bin_centres(n):
    return np.arange(n) / (n - 1)


def score_last_frame(progress_pred, readout="expectation"):
    """progress_pred is [n_frames][n_bins] of logits. Return the last frame's scalar."""
    a = np.asarray(progress_pred, dtype=float)
    if a.ndim == 1:                      # already scalar per frame
        return float(a[-1])
    logits = a[-1]
    p = np.exp(logits - logits.max()); p /= p.sum()
    c = _bin_centres(len(p))
    if readout == "conditional_mean":    # drop the lowest bin, renormalise
        p2 = p[1:].copy()
        if p2.sum() <= 0:
            return 0.0
        return float((p2 / p2.sum()) @ c[1:])
    return float(p @ c)


def load_records(path):
    with open(path) as f:
        return json.load(f)


def per_task_scores(records, readout="expectation"):
    """task -> {'successful': [...], 'failure': [...]}"""
    out = {}
    for r in records:
        q = r.get("quality_label") or (r.get("metadata") or {}).get("quality_label")
        if q not in ("successful", "failure"):
            continue
        s = score_last_frame(r["progress_pred"], readout)
        out.setdefault(r["task"], {"successful": [], "failure": []})[q].append(s)
    return out


def widest_margin(records, readout="expectation", min_per_side=1):
    """Max over tasks of mean(success) - mean(failure). Also returns the mean margin."""
    margins = []
    for task, d in per_task_scores(records, readout).items():
        if len(d["successful"]) >= min_per_side and len(d["failure"]) >= min_per_side:
            margins.append(np.mean(d["successful"]) - np.mean(d["failure"]))
    if not margins:
        return float("nan"), float("nan"), 0
    return float(np.max(margins)), float(np.mean(margins)), len(margins)


def rank_accuracy(records, readout="expectation"):
    """Fraction of within-task (success, failure) pairs ordered correctly."""
    ok = tot = 0
    for task, d in per_task_scores(records, readout).items():
        for s in d["successful"]:
            for f in d["failure"]:
                tot += 1
                if s > f:
                    ok += 1
                elif s == f:
                    ok += 0.5
    return (ok / tot) if tot else float("nan"), tot


def step_dirs(run_dir):
    """[(step, path)] sorted, from policy_ranking_samples/step_N."""
    base = os.path.join(run_dir, "policy_ranking_samples")
    out = []
    for p in glob.glob(os.path.join(base, "step_*")):
        m = re.search(r"step_(\d+)$", p)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)
