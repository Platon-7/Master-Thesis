"""Per-episode ICL on/off delta on the held-out test set.

For each model (baseline, loss1, loss2) and each test episode that has a non-null partner
in pairs_index_test.jsonl, compare the prediction with the demo prepended (icl_prob=1.0)
vs. the prediction without (the existing test_eval results from final_analysis.py).

The same physical episode is scored twice. We can therefore compute *per-episode* deltas
without pooled-distribution confounding — the only thing that changed between the two
runs is whether 16 frames of a successful partner trajectory were prepended.

Outputs (results/presentation/):
  table_icl_delta.csv          — per-model summary (n, mean ΔP, sign-correctness, AUC on/off, ECE on/off)
  table_icl_delta_per_source.csv — same but split by data_source family
  fig_icl_delta_scatter.png    — 3-panel scatter (P_off, P_on) coloured by ground-truth label
  fig_icl_delta_hist.png       — 3-panel histogram of ΔP, separated by ground-truth label
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score


OUT = Path("/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer-LoRA/results/presentation")
OUT.mkdir(parents=True, exist_ok=True)

OFF_BASE = Path("/projects/prjs1958/LoRA_weights/test_eval")
ON_BASE  = Path("/projects/prjs1958/LoRA_weights/test_eval_icl")

# Each model's checkpoint dir name under {OFF_BASE,ON_BASE}/<model>/<ckpt>/eval_results/
CKPT_NAME = {
    "baseline": "robometer_lora_loss2_c51_asymmetric",  # baseline reuses L2 preset
    "loss1":    "robometer_lora_loss1_corn_asymmetric",
    "loss2":    "robometer_lora_loss2_c51_asymmetric",
}
JSON_NAME = "policy_ranking_robometer_frames_test.json"

LABELS = {"baseline": "Robometer-4B (baseline)",
          "loss1":    "L1 — CORN asym",
          "loss2":    "L2 — C51 + BCE asym"}
COLORS = {"baseline": "#7F8C8D", "loss1": "#2E86AB", "loss2": "#C0392B"}

PAIRS_INDEX = Path("/scratch-shared/pkarageorgis1/robometer_frames_splits/pairs_index_test.jsonl")


# ---------- I/O ----------

def load_partnered_ids() -> set[str]:
    """Episode IDs in the test set that have a non-null partner in the test pair index."""
    ids = set()
    with PAIRS_INDEX.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("partner_episode_id"):
                ids.add(r["episode_id"])
    return ids


def load_recs(model: str, condition: str):
    """condition ∈ {'off','on'}. Returns list of records or None if path missing."""
    base = OFF_BASE if condition == "off" else ON_BASE
    p = base / model / CKPT_NAME[model] / "eval_results" / JSON_NAME
    if not p.is_file():
        print(f"  [{condition.upper()}] missing: {p}")
        return None
    with p.open() as f:
        return json.load(f)


# ---------- Head-per-model P(success) extraction ----------

def per_frame_p_success(rec, model: str) -> np.ndarray:
    """One scalar in [0,1] per frame. Same logic as final_analysis.py:
       baseline + loss2 → success head sigmoid. loss1 → CORN-decoded progress_pred,
       or σ(z_5) if the raw 4-logit field was preserved (post-Apr-26 trainer patch)."""
    if model == "loss1":
        raw = rec.get("progress_pred_raw")
        if raw is not None:
            arr = np.asarray(raw)
            if arr.ndim == 2 and arr.shape[-1] == 4:
                z5 = arr[..., -1]
                return 1.0 / (1.0 + np.exp(-z5))
        return np.asarray(rec["progress_pred"]).flatten()
    return np.asarray(rec["success_probs"]).flatten()


def per_episode_p_success(rec, model: str) -> float:
    """Aggregate to one number per episode. We use the LAST frame — that's the deciding
    frame for success in Robometer's published ranking metric. Mean of last 4 also gives
    similar trends; last-frame is the operational answer."""
    p = per_frame_p_success(rec, model)
    return float(p[-1]) if len(p) else float("nan")


# ---------- Calibration ----------

def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    n = len(probs)
    if n == 0: return float("nan")
    e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        cnt = m.sum()
        if cnt == 0: continue
        e += (cnt / n) * abs(labels[m].mean() - probs[m].mean())
    return float(e)


# ---------- Build paired arrays ----------

def build_paired(model: str, partnered_ids: set[str], require_partner: bool = True) -> Dict[str, np.ndarray]:
    """Per-episode aligned arrays: P_off, P_on, label, source.

    require_partner=True  → only paired-failure episodes (the demo's effect on FPR direction).
    require_partner=False → all 602 test trajectories (slide-11 aggregate; for the 295 success
                            episodes ICL-on silently falls back to ICL-off because successes
                            have no partner in the test pair index — only the 307 failures
                            actually see a demo)."""
    off = load_recs(model, "off")
    on  = load_recs(model, "on")
    if off is None or on is None:
        return {}

    if require_partner:
        off_by_id = {r["id"]: r for r in off if r["id"] in partnered_ids}
        on_by_id  = {r["id"]: r for r in on  if r["id"] in partnered_ids}
    else:
        off_by_id = {r["id"]: r for r in off}
        on_by_id  = {r["id"]: r for r in on}
    common = sorted(set(off_by_id) & set(on_by_id))

    P_off, P_on, y, src = [], [], [], []
    for eid in common:
        r_off, r_on = off_by_id[eid], on_by_id[eid]
        P_off.append(per_episode_p_success(r_off, model))
        P_on .append(per_episode_p_success(r_on,  model))
        y.append(1 if r_off.get("quality_label") in ("success", "successful") else 0)
        src.append(r_off.get("data_source", "unknown"))
    return {
        "P_off": np.array(P_off),
        "P_on":  np.array(P_on),
        "label": np.array(y, dtype=int),
        "source": np.array(src),
        "n": len(common),
    }


# ---------- Reporting ----------

def safe_auc(y, p):
    if len(set(y)) < 2: return float("nan")
    return float(roc_auc_score(y, p))


def safe_pr_auc(y, p):
    if len(set(y)) < 2: return float("nan")
    return float(average_precision_score(y, p))


def fpr_at(p, y, tau: float = 0.5) -> float:
    """Per-episode FPR at a fixed threshold τ. P >= τ on a true failure (y==0) is a false positive."""
    fail = (y == 0)
    if fail.sum() == 0: return float("nan")
    return float(((p >= tau) & fail).sum() / fail.sum())


def aggregate_row(model: str, d: Dict[str, np.ndarray]) -> dict:
    if not d:
        return {"model": model, "n": 0}
    P_off, P_on, y = d["P_off"], d["P_on"], d["label"]
    delta = P_on - P_off
    # Sign-correctness: did the demo move the prediction in the right direction?
    #   for successes (y=1): want +delta
    #   for failures (y=0): want -delta
    correct = ((y == 1) & (delta > 0)) | ((y == 0) & (delta < 0))
    return {
        "model": model,
        "n": len(P_off),
        "n_succ": int((y == 1).sum()),
        "n_fail": int((y == 0).sum()),
        "mean_P_off": float(P_off.mean()),
        "mean_P_on":  float(P_on.mean()),
        "mean_delta": float(delta.mean()),
        "mean_delta_succ": float(delta[y == 1].mean()) if (y == 1).any() else float("nan"),
        "mean_delta_fail": float(delta[y == 0].mean()) if (y == 0).any() else float("nan"),
        "sign_correct_rate": float(correct.mean()),
        "auc_off": safe_auc(y, P_off),
        "auc_on":  safe_auc(y, P_on),
        "pr_auc_off": safe_pr_auc(y, P_off),
        "pr_auc_on":  safe_pr_auc(y, P_on),
        "fpr05_off": fpr_at(P_off, y, 0.5),
        "fpr05_on":  fpr_at(P_on,  y, 0.5),
        "ece_off": ece(P_off, y),
        "ece_on":  ece(P_on,  y),
    }


# ---------- Headline table (mirrors table_5_test_set_headline.csv schema) ----------

HEADLINE_METRICS = [
    ("ROC-AUC",     "auc"),
    ("FPR @ τ=0.5", "fpr05"),
    ("ECE",         "ece"),
]


def write_headline_table(agg, path):
    """Same pivot as table_5_test_set_headline.csv: rows = metric, cols = model.
    Each metric is repeated three times — ICL-off, ICL-on, Δ — so the reader can scan
    the demo's effect down a single column."""
    cols = [LABELS[m] for m in ["baseline", "loss1", "loss2"]]
    by_model = {r["model"]: r for r in agg}

    def cell(model, key):
        r = by_model.get(model, {})
        v = r.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)): return ""
        return f"{v:.3f}" if isinstance(v, float) else f"{v}"

    with open(path, "w") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + cols)
        # n row first
        w.writerow(["n (paired episodes)"] + [cell(m, "n") for m in ["baseline", "loss1", "loss2"]])
        for label, key in HEADLINE_METRICS:
            row_off = [f"{label} (ICL-off)"]
            row_on  = [f"{label} (ICL-on)"]
            row_d   = [f"Δ {label}"]
            for m in ["baseline", "loss1", "loss2"]:
                r = by_model.get(m, {})
                v_off, v_on = r.get(f"{key}_off"), r.get(f"{key}_on")
                row_off.append("" if v_off is None or (isinstance(v_off, float) and np.isnan(v_off)) else f"{v_off:.3f}")
                row_on .append("" if v_on  is None or (isinstance(v_on,  float) and np.isnan(v_on))  else f"{v_on:.3f}")
                if v_off is not None and v_on is not None and not (np.isnan(v_off) or np.isnan(v_on)):
                    row_d.append(f"{v_on - v_off:+.3f}")
                else:
                    row_d.append("")
            w.writerow(row_off); w.writerow(row_on); w.writerow(row_d)
    print(f"Wrote {path}")


def print_headline_table(agg):
    """Same numbers, ASCII-pretty for the terminal."""
    by_model = {r["model"]: r for r in agg}
    cols = ["baseline", "loss1", "loss2"]
    head = f"{'metric':<22} | " + " | ".join(f"{LABELS[m]:>23}" for m in cols)
    sep  = "-" * len(head)
    print()
    print("ICL-on vs ICL-off headline (paired test-set episodes only)")
    print(sep); print(head); print(sep)
    print(f"{'n (paired)':<22} | " + " | ".join(f"{by_model.get(m, {}).get('n', 0):>23}" for m in cols))
    for label, key in HEADLINE_METRICS:
        for tag, suffix, fmt in [("ICL-off", "_off", "{:>23.3f}"),
                                  ("ICL-on",  "_on",  "{:>23.3f}"),
                                  ("Δ",        "_DELTA", "{:>23}")]:
            row = f"{label+' '+tag:<22} | "
            cells = []
            for m in cols:
                r = by_model.get(m, {})
                if suffix == "_DELTA":
                    v_off, v_on = r.get(key+"_off"), r.get(key+"_on")
                    if v_off is None or v_on is None or np.isnan(v_off) or np.isnan(v_on):
                        cells.append(" " * 23)
                    else:
                        cells.append(f"{v_on - v_off:>+23.3f}")
                else:
                    v = r.get(key + suffix)
                    cells.append(" " * 23 if v is None or (isinstance(v, float) and np.isnan(v))
                                 else fmt.format(v))
            print(row + " | ".join(cells))
        print(sep)


# ---------- Plots ----------

def plot_scatter(rows: Dict[str, dict]):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, m in zip(axes, ["baseline", "loss1", "loss2"]):
        d = rows[m]
        if not d:
            ax.set_title(f"{LABELS[m]}\n(no data yet)")
            ax.axis("off"); continue
        P_off, P_on, y = d["P_off"], d["P_on"], d["label"]
        ax.scatter(P_off[y == 0], P_on[y == 0], s=10, c="#C0392B", alpha=0.6, label="failure")
        ax.scatter(P_off[y == 1], P_on[y == 1], s=10, c="#27AE60", alpha=0.6, label="success")
        lo = min(P_off.min(), P_on.min()); hi = max(P_off.max(), P_on.max())
        ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("P(success)  ICL-off")
        ax.set_ylabel("P(success)  ICL-on")
        ax.set_title(f"{LABELS[m]}  (n={d['n']})")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = OUT / "fig_icl_delta_scatter.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {p}")


def plot_hist(rows):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, m in zip(axes, ["baseline", "loss1", "loss2"]):
        d = rows[m]
        if not d:
            ax.set_title(f"{LABELS[m]}\n(no data yet)"); ax.axis("off"); continue
        delta, y = d["P_on"] - d["P_off"], d["label"]
        bins = np.linspace(min(-0.05, delta.min()), max(0.05, delta.max()), 31)
        ax.hist(delta[y == 0], bins=bins, color="#C0392B", alpha=0.55, label=f"failure (n={int((y==0).sum())})")
        ax.hist(delta[y == 1], bins=bins, color="#27AE60", alpha=0.55, label=f"success (n={int((y==1).sum())})")
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_xlabel("ΔP(success)  =  ICL-on − ICL-off")
        ax.set_title(f"{LABELS[m]}  mean Δ={delta.mean():+.4f}")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = OUT / "fig_icl_delta_hist.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {p}")


# ---------- main ----------

def main():
    partnered = load_partnered_ids()
    print(f"Partnered test episodes (have non-null partner): {len(partnered)}")
    print()

    # --- AGGREGATE view (all 602 test trajectories — slide-11 schema) ---
    agg_all = []
    print("=" * 80)
    print("AGGREGATE — all 602 test trajectories (slide-11 schema)")
    print("Successes have no partner in pairs_index_test, so under ICL-on the success-side")
    print("silently falls back to ICL-off; only the 307 failures actually see a demo.")
    print("=" * 80)
    for m in ["baseline", "loss1", "loss2"]:
        d_all = build_paired(m, partnered, require_partner=False)
        if not d_all:
            agg_all.append({"model": m, "n": 0}); continue
        agg_all.append(aggregate_row(m, d_all))
        last = agg_all[-1]
        print(f"\n{LABELS[m]}:")
        print(f"  n={last['n']}  succ={last['n_succ']}  fail={last['n_fail']}")
        print(f"  AUC      off → on  =  {last['auc_off']:.3f} → {last['auc_on']:.3f}   (Δ {last['auc_on']-last['auc_off']:+.3f})")
        print(f"  FPR @0.5 off → on  =  {last['fpr05_off']:.3f} → {last['fpr05_on']:.3f}   (Δ {last['fpr05_on']-last['fpr05_off']:+.3f})")
        print(f"  ECE      off → on  =  {last['ece_off']:.3f} → {last['ece_on']:.3f}   (Δ {last['ece_on']-last['ece_off']:+.3f})")
    print()

    write_headline_table(agg_all, OUT / "table_icl_headline.csv")
    print_headline_table(agg_all)

    # --- PAIRED-FAILURES view (the 307 partnered failure episodes) ---
    print("\n" + "=" * 80)
    print("PAIRED FAILURES — the 307 failure episodes that got a success demo")
    print("(only direction we can measure: does the demo correctly LOWER P(success)?)")
    print("=" * 80)

    rows, agg = {}, []
    per_source_rows = []
    for m in ["baseline", "loss1", "loss2"]:
        print(f"=== {LABELS[m]} ===")
        d = build_paired(m, partnered)
        rows[m] = d
        if not d:
            print("  no data — skipping")
            agg.append({"model": m, "n": 0})
            continue
        agg.append(aggregate_row(m, d))
        # Per-source breakdown
        for src in sorted(set(d["source"])):
            mask = d["source"] == src
            if mask.sum() < 5: continue
            sub = {k: v[mask] for k, v in d.items() if isinstance(v, np.ndarray)}
            sub["n"] = int(mask.sum())
            r = aggregate_row(m, sub); r["source"] = src
            per_source_rows.append(r)
        last = agg[-1]
        print(f"  n={last['n']}  succ={last['n_succ']}  fail={last['n_fail']}")
        print(f"  ΔP all     = {last['mean_delta']:+.4f}")
        print(f"  ΔP success = {last['mean_delta_succ']:+.4f}   (want > 0)")
        print(f"  ΔP failure = {last['mean_delta_fail']:+.4f}   (want < 0)")
        print(f"  sign-correct rate = {last['sign_correct_rate']:.1%}")
        print(f"  AUC  off → on  =  {last['auc_off']:.3f} → {last['auc_on']:.3f}")
        print(f"  ECE  off → on  =  {last['ece_off']:.3f} → {last['ece_on']:.3f}")
        print()

    # Tables
    csv_path = OUT / "table_icl_delta.csv"
    with csv_path.open("w") as f:
        cols = ["model", "n", "n_succ", "n_fail",
                "mean_P_off", "mean_P_on", "mean_delta",
                "mean_delta_succ", "mean_delta_fail", "sign_correct_rate",
                "auc_off", "auc_on", "ece_off", "ece_on"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in agg: w.writerow({c: r.get(c, "") for c in cols})
    print(f"Wrote {csv_path}")

    if per_source_rows:
        csv_path = OUT / "table_icl_delta_per_source.csv"
        with csv_path.open("w") as f:
            cols = ["model", "source", "n", "n_succ", "n_fail",
                    "mean_delta", "mean_delta_succ", "mean_delta_fail",
                    "sign_correct_rate", "auc_off", "auc_on"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in per_source_rows: w.writerow({c: r.get(c, "") for c in cols})
        print(f"Wrote {csv_path}")

    plot_scatter(rows)
    plot_hist(rows)


if __name__ == "__main__":
    main()
