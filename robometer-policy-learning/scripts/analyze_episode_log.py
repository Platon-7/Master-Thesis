#!/usr/bin/env python3
"""On-policy reward-hacking metrics from episodes.jsonl.

Two regimes, two metrics -- deliberately not one metric forced across both.

DENSE / NO-TERMINATION (ManiSkill headline runs). The model never makes a
success *decision*, so a thresholded FP rate would be an offline reconstruction
of a decision that does not exist. What breaks these runs is overoptimisation of
a shaped reward: the policy farms VLM return without solving. Quantified by

    d'_onpolicy = (mean vlm_return | solved - mean vlm_return | unsolved) / pooled_sd
    farm_ratio  = p95(vlm_return | unsolved) / median(vlm_return | solved)
    rho         = Spearman(vlm_return, gt_solved_anytime)

All three are scale-free, so a rubric scorer, a dense progress model and a
detector can appear in one table. farm_ratio ~ 0.89 means unsolved episodes
reach 89% of what solving pays -- the farmable basin, as a number.

DETECTOR (MetaWorld, or ManiSkill with use_success_detection=true)

    FP_rate = #(fired & !gt_solved_at_fire) / #episodes
    TP_rate = #(fired &  gt_solved_at_fire) / #episodes
    miss    = #(!fired & gt_solved_anytime) / #episodes
    lead_time = mean(gt_first_solve_step - fire_step) over true positives

FP is swept over a threshold GRID rather than reported at one point: the same
model gives false_rate 1.00 at thr=0.10 and 0.00 at thr=0.6875, so a single
number without provenance is uninterpretable. The grid is recomputed from
score_per_step, so no re-run is needed to change the operating point.

Usage:
    python scripts/analyze_episode_log.py RUN_DIR/episodes.jsonl [--window 500]
    python scripts/analyze_episode_log.py 'runs/*/episodes.jsonl' --compare
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st


def load(path):
    out = []
    with open(path, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _spearman(x, y):
    """Rank correlation with average ranks for ties (no scipy dependency)."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    if len(x) < 3:
        return float("nan")
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def _pctl(v, q):
    if not v:
        return float("nan")
    v = sorted(v)
    i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
    return v[i]


def dense_metrics(recs):
    """Overoptimisation metrics -- the ManiSkill headline."""
    ret = [r["vlm_return"] for r in recs]
    gt = [r["gt_solved_anytime"] for r in recs]
    solved = [v for v, g in zip(ret, gt) if g]
    unsolved = [v for v, g in zip(ret, gt) if not g]
    out = {
        "n_episodes": len(recs),
        "gt_success_rate": (sum(gt) / len(gt)) if gt else float("nan"),
        "mean_return_solved": st.mean(solved) if solved else float("nan"),
        "mean_return_unsolved": st.mean(unsolved) if unsolved else float("nan"),
        "d_prime_onpolicy": float("nan"),
        "farm_ratio": float("nan"),
        "rho_spearman": _spearman(ret, [float(g) for g in gt]),
    }
    if len(solved) >= 2 and len(unsolved) >= 2:
        sd = math.sqrt((st.pvariance(solved) + st.pvariance(unsolved)) / 2.0)
        if sd > 0:
            out["d_prime_onpolicy"] = (st.mean(solved) - st.mean(unsolved)) / sd
    if solved and unsolved:
        med = st.median(solved)
        if med:
            out["farm_ratio"] = _pctl(unsolved, 0.95) / med
    return out


def detector_metrics(recs):
    """FP/TP/miss/lead_time as actually deployed (only if a detector ran)."""
    live = [r for r in recs if r.get("detection_enabled")]
    if not live:
        return None
    n = len(live)
    fp = sum(1 for r in live if r.get("fired") and not r.get("gt_solved_at_fire"))
    tp = sum(1 for r in live if r.get("fired") and r.get("gt_solved_at_fire"))
    miss = sum(1 for r in live if not r.get("fired") and r.get("gt_solved_anytime"))
    gated = sum(1 for r in live if r.get("gate_suppressed"))
    leads = [
        r["gt_first_solve_step"] - r["fire_step"]
        for r in live
        if r.get("fired") and r.get("gt_solved_anytime") and r.get("fire_step") is not None
        and r.get("gt_first_solve_step") is not None
    ]
    return {
        "n_episodes": n,
        "FP_rate": fp / n,
        "TP_rate": tp / n,
        "miss_rate": miss / n,
        "gate_suppressed_rate": gated / n,
        "lead_time_mean": st.mean(leads) if leads else float("nan"),
        "threshold": live[0].get("threshold"),
        "threshold_source": live[0].get("threshold_source"),
    }


def fp_grid(recs, grid, duration=1, min_ep_steps=0):
    """Recompute FP/TP offline at each threshold from sp_per_step.

    Applies the same 'sustained for `duration` steps, gated by `min_ep_steps`'
    rule the live detector uses, so a dense run can be placed on the same axis
    as a detector run without ever having fired.
    """
    rows = []
    for thr in grid:
        fp = tp = fired = 0
        for r in recs:
            sp, gt = r.get("sp_per_step") or [], r.get("gt_per_step") or []
            run = 0
            hit = None
            for t, v in enumerate(sp):
                run = run + 1 if v > thr else 0
                if run >= duration and t >= min_ep_steps:
                    hit = t
                    break
            if hit is None:
                continue
            fired += 1
            if hit < len(gt) and gt[hit]:
                tp += 1
            else:
                fp += 1
        n = max(1, len(recs))
        rows.append((thr, fired / n, fp / n, tp / n))
    return rows


def fmt(d):
    return "  ".join(
        f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items()
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="episodes.jsonl (glob ok)")
    ap.add_argument("--window", type=int, default=0,
                    help="also report metrics per rolling window of N episodes")
    ap.add_argument("--grid", default="0.05,0.10,0.20,0.35,0.50,0.6875,0.80,0.90")
    args = ap.parse_args()

    for path in sorted(glob.glob(args.path)):
        recs = load(path)
        if not recs:
            print(f"{path}: no records")
            continue
        print(f"\n=== {path}  ({len(recs)} episodes) ===")
        print("[dense / overoptimisation]  " + fmt(dense_metrics(recs)))
        det = detector_metrics(recs)
        print("[detector as deployed]      " + (fmt(det) if det else "detection disabled in this run"))

        grid = [float(x) for x in args.grid.split(",")]
        print("[offline FP sweep from sp_per_step]")
        print(f"    {'thr':>8} {'fire_rate':>10} {'FP_rate':>9} {'TP_rate':>9}")
        for thr, fr, fp, tp in fp_grid(recs, grid):
            print(f"    {thr:8.4f} {fr:10.3f} {fp:9.3f} {tp:9.3f}")

        if args.window:
            print(f"[per {args.window}-episode window: return vs GT divergence]")
            for i in range(0, len(recs), args.window):
                w = recs[i:i + args.window]
                if len(w) < max(20, args.window // 5):
                    break
                m = dense_metrics(w)
                print(f"    eps {i:6d}-{i+len(w):6d}  gt={m['gt_success_rate']:.3f}  "
                      f"ret_unsolved={m['mean_return_unsolved']:.2f}  "
                      f"ret_solved={m['mean_return_solved']:.2f}  "
                      f"farm_ratio={m['farm_ratio']:.3f}  d'={m['d_prime_onpolicy']:.3f}")


if __name__ == "__main__":
    main()
