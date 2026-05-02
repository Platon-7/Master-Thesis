"""Verify the CORN-head pretrain produced a sensibly-calibrated head.

Three diagnostic checks per the post-Apr-30 plan:

  1. Per-level distribution: bucket frames by their true ordinal label y ∈ {1..5}
     (decoded from `target_progress`) and report mean σ(z_5) per bucket.
     Want monotone increasing means: y=1 → σ near 0 ... y=5 → σ near 1.

  2. Per-threshold cumulative sanity: for every frame, check
        σ(z_2) ≥ σ(z_3) ≥ σ(z_4) ≥ σ(z_5).
     CORN's cumulative thresholds are P(y ≥ 2) ≥ P(y ≥ 3) ≥ ... by construction,
     so any frame violating this means the head is broken.

  3. ECE on σ(z_5) vs binary success labels: bucket predictions into 10 bins,
     compute |empirical_rate - mean_pred| weighted by bucket size. Target ≈ 0.05–0.10
     for a well-pretrained head.

Reads `progress_pred_raw` (the 4 CORN logits per frame) from the policy_ranking
eval JSON saved by the trainer at <output_dir>/eval_results/. Re-uses the same
field that final_analysis.py uses for the bake-off eval.

Usage:
    python scripts/verify_corn_pretrain.py <pretrain_output_dir>

Exit code 0 if all three checks pass cleanly, 1 if any check fails the threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def decode_y_from_target(t: float) -> int:
    """Recover ordinal label y ∈ {1..5} from a target_progress decimal in [0, 1]."""
    rubric = {0.0: 1, 0.25: 2, 0.5: 3, 0.75: 4, 1.0: 5}
    if t in rubric:
        return rubric[t]
    return min(5, max(1, int(round(t * 4)) + 1))


def per_level_distribution(records, model_name: str = "pretrained"):
    """Check 1: σ(z_5) bucketed by true ordinal label."""
    bucket_sums = {y: [] for y in (1, 2, 3, 4, 5)}
    for rec in records:
        raw = rec.get("progress_pred_raw")
        if raw is None:
            return None
        arr = np.asarray(raw)
        if arr.ndim != 2 or arr.shape[-1] != 4:
            continue
        sigma_z5 = sigmoid(arr[..., -1])
        targets = np.asarray(rec["target_progress"]).flatten()
        for s, t in zip(sigma_z5, targets):
            y = decode_y_from_target(float(t))
            bucket_sums[y].append(float(s))

    print("  Check 1 — σ(z_5) per true ordinal label")
    means = {}
    for y in sorted(bucket_sums):
        vals = bucket_sums[y]
        if not vals:
            means[y] = None
            continue
        means[y] = np.mean(vals)
        print(f"    y={y}: mean σ(z_5) = {means[y]:.4f}   (n={len(vals)})")

    valid = [(y, m) for y, m in sorted(means.items()) if m is not None]
    monotone = all(valid[i][1] <= valid[i + 1][1] for i in range(len(valid) - 1))
    span = (valid[-1][1] - valid[0][1]) if len(valid) >= 2 else 0.0
    print(f"    monotone increasing: {monotone}    span (max-min): {span:.3f}")
    return {"means": means, "monotone": monotone, "span": span}


def per_threshold_sanity(records):
    """Check 2: σ(z_2) ≥ σ(z_3) ≥ σ(z_4) ≥ σ(z_5) per frame."""
    n_total = 0
    n_violation = 0
    max_gap = 0.0
    for rec in records:
        raw = rec.get("progress_pred_raw")
        if raw is None:
            continue
        arr = np.asarray(raw)
        if arr.ndim != 2 or arr.shape[-1] != 4:
            continue
        sigmas = sigmoid(arr)  # [T, 4] = σ(z_2..z_5)
        n_total += sigmas.shape[0]
        diffs = sigmas[:, :-1] - sigmas[:, 1:]   # σ(z_k) - σ(z_{k+1}); want ≥ 0
        per_frame_violations = (diffs < -1e-6).any(axis=-1)
        n_violation += int(per_frame_violations.sum())
        if (-diffs).max(initial=0.0) > max_gap:
            max_gap = float((-diffs).max(initial=0.0))

    print("  Check 2 — cumulative threshold ordering")
    print(f"    frames checked       : {n_total}")
    print(f"    frames in violation  : {n_violation}  ({100*n_violation/max(n_total,1):.2f}%)")
    print(f"    worst inversion gap  : {max_gap:.4f}  (negative = inverted)")
    return {"n_total": n_total, "n_violation": n_violation, "max_gap": max_gap}


def compute_ece(records, n_bins: int = 10):
    """Check 3: ECE of σ(z_5) vs per-frame binary success labels."""
    probs, labels = [], []
    for rec in records:
        raw = rec.get("progress_pred_raw")
        if raw is None:
            continue
        arr = np.asarray(raw)
        if arr.ndim != 2 or arr.shape[-1] != 4:
            continue
        sigma_z5 = sigmoid(arr[..., -1])
        sl = np.asarray(rec.get("success_labels", [])).flatten()
        n = min(len(sigma_z5), len(sl))
        probs.extend(sigma_z5[:n].tolist())
        labels.extend(sl[:n].astype(int).tolist())
    probs = np.asarray(probs); labels = np.asarray(labels)
    if len(probs) == 0:
        return None

    edges = np.linspace(0, 1, n_bins + 1)
    n = len(probs)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(labels[m].mean() - probs[m].mean())

    print("  Check 3 — ECE on σ(z_5) vs binary success labels")
    print(f"    n_frames       : {n}")
    print(f"    pred range     : [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"    pred median    : {np.median(probs):.4f}")
    print(f"    ECE            : {ece:.4f}   (lower is better; target ≈ 0.05–0.10)")
    return {"n": int(n), "ece": float(ece),
            "pred_min": float(probs.min()), "pred_max": float(probs.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir", help="pretrain output dir (contains eval_results/)")
    ap.add_argument("--split", default="robometer_frames_eval_failsafe",
                    help="which eval split JSON to read")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    # Pretrain config has training.exp_name=corn_head_pretrain → eval_results lives under
    # <output_dir>/<exp_name>/eval_results/policy_ranking_<split>.json
    candidates = [
        out_dir / "eval_results" / f"policy_ranking_{args.split}.json",
        out_dir / "corn_head_pretrain" / "eval_results" / f"policy_ranking_{args.split}.json",
    ]
    eval_json = next((p for p in candidates if p.is_file()), None)
    if eval_json is None:
        print(f"ERROR: could not find eval JSON. Looked in:")
        for p in candidates:
            print(f"  {p}")
        sys.exit(2)

    print(f"Reading {eval_json}")
    records = json.load(open(eval_json))
    print(f"  {len(records)} eval records")

    if not records or records[0].get("progress_pred_raw") is None:
        print("ERROR: eval records do not contain progress_pred_raw — the trainer patch")
        print("       that preserves the 4 CORN logits must be present for this verification.")
        sys.exit(2)
    print()

    # --- Run the three checks ---
    r1 = per_level_distribution(records); print()
    r2 = per_threshold_sanity(records);   print()
    r3 = compute_ece(records);            print()

    # --- Verdict ---
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    fail = []
    if r1 is None or not r1.get("monotone", False):
        fail.append("per-level monotonicity")
    if r1 is not None and r1.get("span", 0) < 0.30:
        fail.append(f"per-level σ(z_5) span < 0.30 (got {r1['span']:.3f})")
    if r2 is not None and r2["n_violation"] > 0.001 * r2["n_total"]:
        fail.append(f"cumulative-threshold violations > 0.1% of frames ({r2['n_violation']}/{r2['n_total']})")
    if r3 is not None and r3["ece"] > 0.20:
        fail.append(f"ECE > 0.20 (got {r3['ece']:.4f})")
    if r3 is not None and (r3["pred_max"] - r3["pred_min"]) < 0.30:
        fail.append(f"σ(z_5) range too narrow (got [{r3['pred_min']:.3f}, {r3['pred_max']:.3f}])")

    if fail:
        print("FAILED checks:")
        for f in fail:
            print(f"  ✗ {f}")
        print()
        print("→ pretrain did NOT meet the verification bar; head is not ready for LoRA bake-off.")
        sys.exit(1)
    else:
        print("✓ all three checks passed.")
        print("→ pretrained head is ready to feed into the loss1_corn LoRA bake-off.")
        sys.exit(0)


if __name__ == "__main__":
    main()
