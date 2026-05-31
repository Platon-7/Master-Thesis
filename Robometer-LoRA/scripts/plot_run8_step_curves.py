"""Regenerate run8_l2_lambda03_kl step-curve plots, 3 panels (no dense ECE), formal titles.

Two plots — same data sources, different trajectory-score head:
  * step_curves_run8_success_head.png   — score = max-frame success_probs (sigmoid binary head)
  * step_curves_run8_progress_head.png  — score = max-frame expected progress (C51-decoded scalar)

Both: 3 panels (AUC / FPR@τ=0.5 / ECE-sparse), 2 lines (ICL on / off).

Reads JSONs at:
  /scratch-shared/pkarageorgis1/LoRA_step_curves_v3/loss2_step{N}_run8_l2_lambda03_kl_{icl}/
      robometer_lora_loss2_c51_asymmetric/eval_results/policy_ranking_robometer_frames_test_v3.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


EVAL_ROOT = Path("/scratch-shared/pkarageorgis1/LoRA_step_curves_v3")
OUT_DIR = Path("/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer-LoRA/results")
RUN = "run8_l2_lambda03_kl"
LOSS = "loss2"
INNER = "robometer_lora_loss2_c51_asymmetric"

DIR_RE = re.compile(rf"^{LOSS}_step(\d+)_{RUN}_(iclon|icloff)$")

ICL_LABELS = {"iclon": "ICL on", "icloff": "ICL off"}
ICL_COLORS = {"iclon": "tab:orange", "icloff": "tab:blue"}


def progress_signal_c51(progress_pred: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Decode C51 logits [T, 10] → expected progress in [0, 1] per frame."""
    pp = np.asarray(progress_pred, dtype=np.float64)
    if pp.ndim == 2:
        ex = np.exp(pp - pp.max(axis=-1, keepdims=True))
        probs = ex / ex.sum(axis=-1, keepdims=True)
        return (probs * np.linspace(0, 1, n_bins)).sum(axis=-1)
    return pp.flatten()


def trajectory_score(rec: dict, head: str) -> float:
    if head == "success":
        return float(np.asarray(rec["success_probs"]).max())
    return float(progress_signal_c51(rec["progress_pred"]).max())


def fpr_at(scores: np.ndarray, labels: np.ndarray, tau: float = 0.5) -> float:
    neg = labels == 0
    if neg.sum() == 0:
        return float("nan")
    return float((scores[neg] > tau).sum() / neg.sum())


def compute_ece_sparse(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Per-frame success_probs vs per-frame success_labels (at-goal-state)."""
    edges = np.linspace(0, 1, n_bins + 1)
    n = len(probs)
    if n == 0:
        return float("nan")
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & ((probs < hi) if i < n_bins - 1 else (probs <= hi))
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        ece += (cnt / n) * abs(labels[mask].mean() - probs[mask].mean())
    return ece


def compute_metrics(eval_json: Path, head: str) -> dict:
    with open(eval_json) as f:
        recs = json.load(f)
    seen = set()
    traj_scores, traj_labels = [], []
    pf_probs, pf_labels = [], []
    for r in recs:
        tid = r["id"]
        if tid in seen:
            continue
        seen.add(tid)
        traj_scores.append(trajectory_score(r, head))
        traj_labels.append(1 if r["quality_label"] == "successful" else 0)
        sp = np.asarray(r["success_probs"], dtype=np.float64).flatten()
        sl = np.asarray(r["success_labels"], dtype=np.float64).flatten()
        n = min(len(sp), len(sl))
        if n:
            pf_probs.extend(sp[:n].tolist())
            pf_labels.extend(sl[:n].astype(int).tolist())
    ts = np.array(traj_scores)
    tl = np.array(traj_labels)
    auc = roc_auc_score(tl, ts) if len(set(tl.tolist())) >= 2 else float("nan")
    return {
        "auc": auc,
        "fpr05": fpr_at(ts, tl, 0.5),
        "ece_sparse": compute_ece_sparse(np.array(pf_probs), np.array(pf_labels), 10),
    }


def discover() -> dict:
    """Return {step: {icl: Path}} for this run."""
    out: dict = defaultdict(dict)
    if not EVAL_ROOT.exists():
        return out
    for d in sorted(EVAL_ROOT.iterdir()):
        if not d.is_dir():
            continue
        m = DIR_RE.match(d.name)
        if not m:
            continue
        step, icl = int(m.group(1)), m.group(2)
        ej = d / INNER / "eval_results" / "policy_ranking_robometer_frames_test_v3.json"
        if ej.exists():
            out[step][icl] = ej
    return out


def plot(head: str, head_label: str, score_descriptor: str, out_path: Path):
    """3-panel figure (AUC / FPR / ECE-sparse), 2 lines (ICL on/off)."""
    data = discover()
    if not data:
        print(f"no eval JSONs under {EVAL_ROOT}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panel_defs = [
        ("auc",        "AUC (higher = better)"),
        ("fpr05",      "FPR @ τ=0.5 (lower = better)"),
        ("ece_sparse", "ECE — sparse-RL view (lower = better)"),
    ]

    cached_metrics: dict = {}
    for ax, (key, title) in zip(axes, panel_defs):
        for icl in ("icloff", "iclon"):
            steps, vals = [], []
            for step in sorted(data.keys()):
                if icl not in data[step]:
                    continue
                ckey = (step, icl)
                if ckey not in cached_metrics:
                    cached_metrics[ckey] = compute_metrics(data[step][icl], head)
                m = cached_metrics[ckey]
                if not np.isnan(m[key]):
                    steps.append(step)
                    vals.append(m[key])
            if steps:
                ax.plot(steps, vals, marker="o", linewidth=2,
                        color=ICL_COLORS[icl], label=ICL_LABELS[icl])
        ax.set_title(title)
        ax.set_xlabel("LoRA training step")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.4)

    # Per-head loss descriptor (success head = BCE; progress head = C51).
    loss_descriptor = (
        "asymmetric C51 + asymmetric BCE, λ = 0.3"
        if head == "success" else "asymmetric C51 progress loss, λ = 0.3"
    )
    fig.suptitle(
        f"{head_label}-head discrimination and calibration on the held-out test set "
        f"across LoRA training steps\n"
        f"({loss_descriptor}; trajectory score = {score_descriptor})",
        fontsize=12, fontweight="bold",
    )
    fig.text(
        0.5, 0.005,
        "ECE — sparse-RL view: P(success) calibrated to per-frame at-goal-state label "
        "(relevant for sparse per-step reward).",
        ha="center", fontsize=9, style="italic",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plot(
        head="success",
        head_label="Success",
        score_descriptor="max-frame success_probs",
        out_path=OUT_DIR / "step_curves_run8_success_head.png",
    )
    plot(
        head="progress",
        head_label="Progress",
        score_descriptor="max-frame expected progress",
        out_path=OUT_DIR / "step_curves_run8_progress_head.png",
    )


if __name__ == "__main__":
    main()
