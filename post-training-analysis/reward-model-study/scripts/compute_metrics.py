"""Compute the full metric suite from an eval_dump.py JSONL, for BOTH heads.

Success head (uses succ_last / succ_per_frame vs binary gt_success):
  - AUC
  - TPR @ FPR=5% (and the threshold)
  - dense-ECE (per-frame success_prob calibrated to trajectory outcome)
Progress head (uses prog_per_frame vs per-frame gt_progress):
  - VOC-Pearson r  (mean over trajectories of pearson(pred_progress, gt_progress))
  - Kendall tau    (mean over trajectories of kendall(pred_progress, gt_progress))
Also reports progress-head-as-classifier (prog_last vs gt_success) AUC for completeness.

Usage: python compute_metrics.py dump1.jsonl [dump2.jsonl ...]  -> prints a row per dump
       --csv out.csv to append rows.
"""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path
import numpy as np


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); o = np.argsort(a, kind="mergesort"); sa = a[o]
    rk = np.empty(len(a)); i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        rk[i:j + 1] = (i + j + 2) / 2; i = j + 1
    inv = np.empty_like(o, dtype=float); inv[o] = rk
    return float((inv[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def tpr_at_fpr(scores, labels, target_fpr=0.05):
    s = np.asarray(scores); y = np.asarray(labels)
    pos = s[y == 1]; neg = s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan"), float("nan")
    # pick the lowest threshold with FPR <= target
    for tau in np.unique(neg)[::-1]:
        fpr = (neg > tau).mean()
        if fpr <= target_fpr:
            tpr = (pos > tau).mean()
            return float(tpr), float(tau)
    return 0.0, float(neg.max())


def ece(probs, labels, n_bins=10):
    probs = np.asarray(probs); labels = np.asarray(labels)
    if not len(probs):
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1); tot = 0.0; n = len(probs)
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        m = (probs >= lo) & ((probs < hi) if k < n_bins - 1 else (probs <= hi))
        c = int(m.sum())
        if c:
            tot += (c / n) * abs(labels[m].mean() - probs[m].mean())
    return float(tot)


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def kendall(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    if n < 2:
        return np.nan
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            if s > 0: c += 1
            elif s < 0: d += 1
    tot = c + d
    return float((c - d) / tot) if tot else np.nan


def metrics_for_dump(path):
    rows = [json.loads(l) for l in open(path)]
    gt = np.array([r["gt_success"] for r in rows])
    # success head
    succ_last = np.array([r["succ_last"] for r in rows])
    sh_auc = auc(succ_last[gt == 1], succ_last[gt == 0])
    sh_tpr, sh_tau = tpr_at_fpr(succ_last, gt, 0.05)
    # dense ECE: per-frame success_prob, label = trajectory outcome
    dprobs, dlabels = [], []
    for r in rows:
        spf = r.get("succ_per_frame") or [r["succ_last"]]
        dprobs.extend(spf); dlabels.extend([r["gt_success"]] * len(spf))
    sh_ece = ece(dprobs, dlabels)
    # progress head: per-trajectory correlation vs gt_progress.
    # The model subsamples to its max_frames (8 for baseline, 16 for FT), so
    # pred has K<=len(gt) points at linspace positions. Resample gt onto the K
    # prediction positions before correlating (NOT gt[:K], which misaligns).
    pe, ke = [], []
    for r in rows:
        ppf = np.asarray(r.get("prog_per_frame") or [r["prog_last"]], float)
        gpr = np.asarray(r.get("gt_progress") or [], float)
        if len(ppf) >= 2 and len(gpr) >= 2:
            pos = np.linspace(0, len(gpr) - 1, len(ppf))
            gpr_aligned = np.interp(pos, np.arange(len(gpr)), gpr)
            pe.append(pearson(ppf, gpr_aligned)); ke.append(kendall(ppf, gpr_aligned))
    ph_pearson = float(np.nanmean(pe)) if pe else np.nan
    ph_kendall = float(np.nanmean(ke)) if ke else np.nan
    # progress-head-as-classifier
    prog_last = np.array([r["prog_last"] for r in rows])
    ph_auc = auc(prog_last[gt == 1], prog_last[gt == 0])
    return {
        "dump": Path(path).stem, "n": len(rows), "n_pos": int((gt == 1).sum()), "n_neg": int((gt == 0).sum()),
        "succ_AUC": round(sh_auc, 4), "succ_TPR@5FPR": round(sh_tpr, 4), "succ_tau@5FPR": round(sh_tau, 4),
        "succ_denseECE": round(sh_ece, 4),
        "prog_VOC_pearson": round(ph_pearson, 4), "prog_kendall": round(ph_kendall, 4),
        "prog_AUC": round(ph_auc, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    results = [metrics_for_dump(d) for d in args.dumps]
    cols = list(results[0].keys())
    print("  ".join(f"{c}" for c in cols))
    for r in results:
        print("  ".join(f"{r[c]}" for c in cols))
    if args.csv:
        new = not Path(args.csv).exists()
        with open(args.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if new: w.writeheader()
            for r in results: w.writerow(r)
        print(f"\nappended {len(results)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
