"""Plot test_v3 step-curves for the KL ablation (kl=0, kl=0.1, kl=0.3).

Reads JSONs dumped by jobs/step_curves_kl.job under
  /scratch-shared/pkarageorgis1/LoRA_step_curves_kl/<run_tag>_step<N>_<icl>/.../eval_results/policy_ranking_robometer_frames_test_v3.json

Computes (per checkpoint × ICL mode):
  - AUC                 — trajectory ROC-AUC of max-frame expected-progress score vs quality_label
  - FPR @ τ=0.5         — false-positive rate on the same trajectory score at threshold 0.5
  - ECE (sparse-RL view) — per-frame `success_probs` calibrated to per-frame `success_labels`
  - ECE (dense-RL view)  — per-frame `success_probs` calibrated to trajectory-level outcome label

Outputs to results/KL/:
  step_curves_kl0_progress_head.png        — 4 panels × 2 lines (ICL on/off), kl=0
  step_curves_kl01_progress_head.png       — same, kl=0.1
  step_curves_kl03_progress_head.png       — same, kl=0.3
  step_curves_kl_comparison_iclon.png      — 4 panels × 3 lines (kl=0/0.1/0.3), ICL on
  step_curves_kl_comparison_icloff.png     — same, ICL off

All loss2_c51 (C51 + BCE asym). Trajectory score = max-frame expected progress
(softmax(progress_pred) · linspace(0,1,10), then max over T frames).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


EVAL_ROOT = Path("/scratch-shared/pkarageorgis1/LoRA_step_curves_kl")
OUT_DIR = Path("/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer-LoRA/results/KL")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIR_RE = re.compile(r"^(kl0|kl01|kl03)_step(\d+)_(iclon|icloff)$")

RUN_LABELS = {"kl0": "kl=0", "kl01": "kl=0.1", "kl03": "kl=0.3"}
RUN_COLORS = {"kl0": "tab:blue", "kl01": "tab:orange", "kl03": "tab:green"}
ICL_LABELS = {"iclon": "ICL on", "icloff": "ICL off"}
ICL_COLORS = {"iclon": "tab:orange", "icloff": "tab:blue"}


def progress_signal_c51(progress_pred: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Decode loss2 C51 logits [T, 10] → expected progress in [0, 1] per frame."""
    pp = np.asarray(progress_pred, dtype=np.float64)
    if pp.ndim == 2:
        ex = np.exp(pp - pp.max(axis=-1, keepdims=True))
        probs = ex / ex.sum(axis=-1, keepdims=True)
        return (probs * np.linspace(0, 1, n_bins)).sum(axis=-1)
    return pp.flatten()


def trajectory_score(rec):
    """Max-frame expected-progress score for a single trajectory record."""
    return float(progress_signal_c51(rec["progress_pred"]).max())


def fpr_at(scores: np.ndarray, labels: np.ndarray, tau: float = 0.5) -> float:
    neg = labels == 0
    if neg.sum() == 0:
        return float("nan")
    return float((scores[neg] > tau).sum() / neg.sum())


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    n = len(probs)
    if n == 0:
        return float("nan")
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (probs >= lo) & (probs < hi)
        else:
            mask = (probs >= lo) & (probs <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        bin_acc = float(labels[mask].mean())
        bin_conf = float(probs[mask].mean())
        ece += (cnt / n) * abs(bin_acc - bin_conf)
    return ece


def compute_metrics_for_eval_file(path: Path) -> dict:
    """Return {'auc', 'fpr05', 'ece_sparse', 'ece_dense'} for one eval JSON."""
    with open(path) as f:
        recs = json.load(f)

    # Trajectory-level: max-frame expected progress
    seen = set()
    traj_scores, traj_labels = [], []
    pf_probs_sparse, pf_labels_sparse = [], []
    pf_probs_dense, pf_labels_dense = [], []
    for r in recs:
        tid = r["id"]
        if tid in seen:
            continue
        seen.add(tid)

        score = trajectory_score(r)
        is_success = 1 if r["quality_label"] == "successful" else 0
        traj_scores.append(score)
        traj_labels.append(is_success)

        # Per-frame success_probs and per-frame labels
        sp = np.asarray(r["success_probs"], dtype=np.float64).flatten()
        sl = np.asarray(r["success_labels"], dtype=np.float64).flatten()
        n = min(len(sp), len(sl))
        if n == 0:
            continue
        # Sparse-RL: per-frame success_probs vs per-frame success_labels
        pf_probs_sparse.extend(sp[:n].tolist())
        pf_labels_sparse.extend(sl[:n].astype(int).tolist())
        # Dense-RL: per-frame success_probs vs TRAJECTORY-LEVEL outcome
        pf_probs_dense.extend(sp[:n].tolist())
        pf_labels_dense.extend([is_success] * n)

    traj_scores = np.array(traj_scores)
    traj_labels = np.array(traj_labels)
    pfps = np.array(pf_probs_sparse)
    pfls = np.array(pf_labels_sparse)
    pfpd = np.array(pf_probs_dense)
    pfld = np.array(pf_labels_dense)

    # AUC only defined when both classes present
    if len(set(traj_labels.tolist())) >= 2:
        auc = roc_auc_score(traj_labels, traj_scores)
    else:
        auc = float("nan")

    return {
        "auc": auc,
        "fpr05": fpr_at(traj_scores, traj_labels, 0.5),
        "ece_sparse": compute_ece(pfps, pfls, 10),
        "ece_dense": compute_ece(pfpd, pfld, 10),
        "n_trajs": int(len(traj_scores)),
    }


def discover_eval_files() -> dict:
    """Map {run_tag: {step: {icl: Path-to-eval-json}}}."""
    out: dict = defaultdict(lambda: defaultdict(dict))
    if not EVAL_ROOT.exists():
        print(f"WARN: {EVAL_ROOT} does not exist yet (no eval results to plot).")
        return out
    for d in sorted(EVAL_ROOT.iterdir()):
        if not d.is_dir():
            continue
        m = DIR_RE.match(d.name)
        if not m:
            continue
        run_tag, step_s, icl = m.group(1), m.group(2), m.group(3)
        # walk one level into the exp subdir, then eval_results
        for inner in d.iterdir():
            ej = inner / "eval_results" / "policy_ranking_robometer_frames_test_v3.json"
            if ej.exists():
                out[run_tag][int(step_s)][icl] = ej
                break
    return out


def plot_per_run(run_tag: str, data: dict, out_path: Path):
    """4 panels (AUC / FPR / ECE-sparse / ECE-dense) × 2 lines (ICL on / off)."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    panel_defs = [
        ("auc",        "AUC (higher = better)"),
        ("fpr05",      "FPR @ τ=0.5 (lower = better)"),
        ("ece_sparse", "ECE — sparse-RL view (lower = better)"),
        ("ece_dense",  "ECE — dense-RL view (lower = better)"),
    ]
    icl_modes = ("icloff", "iclon")

    for ax, (key, title) in zip(axes, panel_defs):
        for icl in icl_modes:
            steps, vals = [], []
            for step in sorted(data.keys()):
                if icl in data[step]:
                    m = compute_metrics_for_eval_file(data[step][icl])
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

    fig.suptitle(
        f"{RUN_LABELS[run_tag]} step trajectory on test_v3\n"
        "Trajectory score = max-frame expected progress (progress head)",
        fontsize=12, fontweight="bold",
    )
    fig.text(
        0.5, 0.005,
        "ECE — sparse-RL view: P(success) calibrated to per-frame at-goal-state label "
        "(relevant for sparse per-step reward).  "
        "ECE — dense-RL view: P(success) calibrated to trajectory-level outcome "
        "(relevant for value-function-style reward).",
        ha="center", fontsize=8, style="italic",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_cross_kl(icl: str, all_data: dict, out_path: Path):
    """4 panels × 3 lines (kl=0/0.1/0.3) for a single ICL condition."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    panel_defs = [
        ("auc",        "AUC (higher = better)"),
        ("fpr05",      "FPR @ τ=0.5 (lower = better)"),
        ("ece_sparse", "ECE — sparse-RL view (lower = better)"),
        ("ece_dense",  "ECE — dense-RL view (lower = better)"),
    ]

    for ax, (key, title) in zip(axes, panel_defs):
        for run_tag in ("kl0", "kl01", "kl03"):
            data = all_data.get(run_tag, {})
            steps, vals = [], []
            for step in sorted(data.keys()):
                if icl in data[step]:
                    m = compute_metrics_for_eval_file(data[step][icl])
                    if not np.isnan(m[key]):
                        steps.append(step)
                        vals.append(m[key])
            if steps:
                ax.plot(steps, vals, marker="o", linewidth=2,
                        color=RUN_COLORS[run_tag], label=RUN_LABELS[run_tag])
        ax.set_title(title)
        ax.set_xlabel("LoRA training step")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.4)

    fig.suptitle(
        f"KL ablation step trajectories on test_v3  —  {ICL_LABELS[icl]}\n"
        "Trajectory score = max-frame expected progress (progress head)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    eval_files = discover_eval_files()
    if not eval_files:
        print("No eval JSONs found yet. Wait for jobs/step_curves_kl.job (job 22671003) to complete.")
        return

    for run_tag, data in eval_files.items():
        n_files = sum(len(v) for v in data.values())
        print(f"{RUN_LABELS[run_tag]}: {len(data)} steps, {n_files} eval files")

    # Per-run plots (mirror the run2 figure)
    for run_tag in ("kl0", "kl01", "kl03"):
        if run_tag not in eval_files:
            continue
        out = OUT_DIR / f"step_curves_{run_tag}_progress_head.png"
        plot_per_run(run_tag, eval_files[run_tag], out)

    # Cross-KL comparison plots (one per ICL condition)
    for icl in ("iclon", "icloff"):
        out = OUT_DIR / f"step_curves_kl_comparison_{icl}.png"
        plot_cross_kl(icl, eval_files, out)


if __name__ == "__main__":
    main()
