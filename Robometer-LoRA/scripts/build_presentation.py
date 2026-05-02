"""Build presentation tables and figures from the bake-off eval dumps.

Reads:
  policy_ranking JSONs in /projects/prjs1958/LoRA_weights/loss{1,2}_<jobid>/.../eval_results/
  policy_ranking_samples/step_*/<split>.json (for per-round trajectory)

Writes:
  results/presentation/table_1_final_results.csv
  results/presentation/table_2_trajectory_per_split.csv
  results/presentation/table_3_headline.csv
  results/presentation/fig_1_failsafe_auc_trajectory.png
  results/presentation/fig_2_per_source_final_bars.png
  results/presentation/fig_3_fpr_per_source_bars.png
  results/presentation/fig_4_per_split_trajectory_grid.png
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


OUT = "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer-LoRA/results/presentation"
os.makedirs(OUT, exist_ok=True)

L1_EVAL = "/projects/prjs1958/LoRA_weights/loss1_22244008/robometer_lora_loss1_corn_asymmetric/eval_results"
L2_EVAL = "/projects/prjs1958/LoRA_weights/loss2_22244009/robometer_lora_loss2_c51_asymmetric/eval_results"
L1_SAMPLES = "/projects/prjs1958/LoRA_weights/loss1_22244008/robometer_lora_loss1_corn_asymmetric/policy_ranking_samples"
L2_SAMPLES = "/projects/prjs1958/LoRA_weights/loss2_22244009/robometer_lora_loss2_c51_asymmetric/policy_ranking_samples"

SPLITS = ["eval_droid", "eval_robometer", "eval_metaworld", "eval_failsafe"]
SPLIT_LABELS = {
    "eval_droid": "DROID",
    "eval_robometer": "Robometer",
    "eval_metaworld": "MetaWorld",
    "eval_failsafe": "Failsafe (deciding)",
}


def s_corn(rec):
    return float(np.array(rec["progress_pred"]).max())


def s_c51(rec, nb=10):
    pp = np.array(rec["progress_pred"])
    if pp.ndim == 2:
        ex = np.exp(pp - pp.max(axis=-1, keepdims=True))
        probs = ex / ex.sum(axis=-1, keepdims=True)
        return float((probs * np.linspace(0, 1, nb)).sum(axis=-1).max())
    return float(pp.max())


def fpr_at(scores, labs, tau=0.5):
    a, l = np.array(scores), np.array(labs)
    fm = l == 0
    return float((a[fm] > tau).sum() / fm.sum()) if fm.sum() else None


def compute(path, fn):
    with open(path) as f:
        recs = json.load(f)
    trajs = defaultdict(list)
    for r in recs:
        trajs[r["id"]].append(r)
    sp, sm, labs = [], [], []
    for tid, g in trajs.items():
        r = max(g, key=lambda x: len(x["progress_pred"]))
        sp.append(fn(r))
        sm.append(float(np.array(r["success_probs"]).max()))
        labs.append(1 if r["quality_label"] == "successful" else 0)
    return dict(
        num=len(labs),
        n_succ=sum(labs),
        n_fail=len(labs) - sum(labs),
        ap=roc_auc_score(labs, sp),
        as_=roc_auc_score(labs, sm),
        fp=fpr_at(sp, labs),
        fs=fpr_at(sm, labs),
    )


# Table 1 — final results
rows = []
final = {}
for split in SPLITS:
    m1 = compute(f"{L1_EVAL}/policy_ranking_robometer_frames_{split}.json", s_corn)
    m2 = compute(f"{L2_EVAL}/policy_ranking_robometer_frames_{split}.json", s_c51)
    final[split] = (m1, m2)
    rows.append([
        split, m1["num"], m1["n_fail"], m1["n_succ"],
        m1["ap"], m1["as_"], m1["fp"], m1["fs"],
        m2["ap"], m2["as_"], m2["fp"], m2["fs"],
    ])

with open(f"{OUT}/table_1_final_results.csv", "w") as f:
    w = csv.writer(f)
    w.writerow([
        "split", "n", "n_fail", "n_succ",
        "L1_AUC_progress", "L1_AUC_success", "L1_FPR_progress", "L1_FPR_success",
        "L2_AUC_progress", "L2_AUC_success", "L2_FPR_progress", "L2_FPR_success",
    ])
    w.writerows(rows)


# Table 2 — per-round trajectory (uses last-frame reward as score)
def parse_qa(s):
    out = {}
    for p in s.split("],"):
        if ":[" in p:
            lab, lst = p.split(":[")
            out[lab] = [float(x) for x in lst.replace("]", "").split(",") if x.strip()]
    return out


def auc_step_dir(step_dir, split):
    fp = os.path.join(step_dir, f"robometer_frames_{split}.json")
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        tasks = json.load(f)
    scores, labs = [], []
    for t in tasks:
        rews = parse_qa(t["quality_and_rews_last"])
        for r in rews.get("failure", []):
            scores.append(r); labs.append(0)
        for r in rews.get("successful", []):
            scores.append(r); labs.append(1)
    if len(set(labs)) < 2:
        return None
    return roc_auc_score(labs, scores)


steps = sorted([int(d.split("_")[1]) for d in os.listdir(L1_SAMPLES) if d.startswith("step_")])
traj_rows = []
traj_data = {"L1": {s: [] for s in SPLITS}, "L2": {s: [] for s in SPLITS}}
for step in steps:
    row = [step]
    for prefix, root in [("L1", L1_SAMPLES), ("L2", L2_SAMPLES)]:
        sd = os.path.join(root, f"step_{step}")
        for split in SPLITS:
            v = auc_step_dir(sd, split)
            row.append(v)
            traj_data[prefix][split].append(v)
    traj_rows.append(row)

with open(f"{OUT}/table_2_trajectory_per_split.csv", "w") as f:
    w = csv.writer(f)
    cols = ["step"]
    for prefix in ["L1", "L2"]:
        for s in SPLITS:
            cols.append(f"{prefix}_{s}_AUC_lastframe")
    w.writerow(cols)
    w.writerows(traj_rows)


# Table 3 — headline
m1_fs, m2_fs = final["eval_failsafe"]
m1_rb, m2_rb = final["eval_robometer"]

with open(f"{OUT}/table_3_headline.csv", "w") as f:
    w = csv.writer(f)
    w.writerow(["metric", "Loss 1 (CORN, ours)", "Loss 2 (C51 + BCE asym, Chris)", "winner"])
    w.writerow(["Failsafe AUC(success-head)", f"{m1_fs['as_']:.3f}", f"{m2_fs['as_']:.3f}", "L2 (+0.025)"])
    w.writerow(["Failsafe AUC(progress)",     f"{m1_fs['ap']:.3f}",  f"{m2_fs['ap']:.3f}",  "L1 (+0.075)"])
    w.writerow(["Failsafe FPR(success) @0.5", f"{m1_fs['fs']:.3f}",  f"{m2_fs['fs']:.3f}",  "tie (both 0.000)"])
    w.writerow(["Robometer FPR(success) @0.5",f"{m1_rb['fs']:.3f}",  f"{m2_rb['fs']:.3f}",  "L1 (huge)"])
    w.writerow(["Failsafe n",                 f"{m1_fs['num']}",     f"{m2_fs['num']}",     ""])


# ---- Figures ----
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"L1": "#2E86AB", "L2": "#E63946"}
LABELS = {"L1": "Loss 1 (CORN, ours)", "L2": "Loss 2 (C51+BCE asym, Chris)"}


# Fig 1: Failsafe AUC trajectory across the 10 eval rounds
fig, ax = plt.subplots(figsize=(8, 5))
for prefix in ["L1", "L2"]:
    ys = traj_data[prefix]["eval_failsafe"]
    ax.plot(steps, ys, marker="o", color=COLORS[prefix], label=LABELS[prefix], linewidth=2)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="random (AUC=0.5)")
ax.set_xlabel("Training step")
ax.set_ylabel("Failsafe AUC (last-frame reward)")
ax.set_title("Failsafe (deciding split) — AUC across training\n[n=30, the cleanest sim ground-truth labels]")
ax.legend(loc="lower right")
ax.set_ylim(0.3, 0.85)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_1_failsafe_auc_trajectory.png", dpi=150)
plt.close()


# Fig 2: Per-source final AUC bar chart (max-frame version)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(SPLITS))
width = 0.35

# AUC(progress)
for ax, key, title in [(axes[0], "ap", "AUC (progress max-frame)"), (axes[1], "as_", "AUC (success-head max-frame)")]:
    l1_vals = [final[s][0][key] for s in SPLITS]
    l2_vals = [final[s][1][key] for s in SPLITS]
    ax.bar(x - width/2, l1_vals, width, label=LABELS["L1"], color=COLORS["L1"])
    ax.bar(x + width/2, l2_vals, width, label=LABELS["L2"], color=COLORS["L2"])
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLITS], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    for i, (v1, v2) in enumerate(zip(l1_vals, l2_vals)):
        ax.text(i - width/2, v1 + 0.01, f"{v1:.2f}", ha="center", fontsize=8)
        ax.text(i + width/2, v2 + 0.01, f"{v2:.2f}", ha="center", fontsize=8)
fig.suptitle("Final results — AUC by source (step 7500)", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_2_per_source_final_bars.png", dpi=150)
plt.close()


# Fig 3: FPR per source (the actually-deciding metric for RL reward use)
fig, ax = plt.subplots(figsize=(9, 5))
l1_fpr = [final[s][0]["fs"] for s in SPLITS]
l2_fpr = [final[s][1]["fs"] for s in SPLITS]
bars1 = ax.bar(x - width/2, l1_fpr, width, label=LABELS["L1"], color=COLORS["L1"])
bars2 = ax.bar(x + width/2, l2_fpr, width, label=LABELS["L2"], color=COLORS["L2"])
ax.axhline(0.05, color="green", linestyle="--", linewidth=0.8, label="deploy bar (5%)")
ax.set_xticks(x)
ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLITS], rotation=20, ha="right")
ax.set_ylabel("FPR @ τ=0.5  (success-head)")
ax.set_title("False-positive rate per source — lower is better\n[the metric that matters for RL reward use]")
ax.legend(loc="upper right")
for i, (v1, v2) in enumerate(zip(l1_fpr, l2_fpr)):
    ax.text(i - width/2, v1 + 0.005, f"{v1:.3f}", ha="center", fontsize=8)
    ax.text(i + width/2, v2 + 0.005, f"{v2:.3f}", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_3_fpr_per_source_bars.png", dpi=150)
plt.close()


# Fig 4: 4-panel trajectory grid (one panel per source)
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
for idx, split in enumerate(SPLITS):
    ax = axes[idx // 2, idx % 2]
    for prefix in ["L1", "L2"]:
        ys = traj_data[prefix][split]
        valid = [(x, y) for x, y in zip(steps, ys) if y is not None]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, marker="o", color=COLORS[prefix], label=LABELS[prefix], linewidth=1.8)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Training step")
    ax.set_ylabel("AUC (last-frame)")
    ax.set_title(SPLIT_LABELS[split])
    ax.set_ylim(0.3, 0.85)
    if idx == 0:
        ax.legend(loc="lower right", fontsize=9)
fig.suptitle("AUC trajectory across eval rounds — by source", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_4_per_split_trajectory_grid.png", dpi=150)
plt.close()


print("Tables and figures written to:", OUT)
for f in sorted(os.listdir(OUT)):
    sz = os.path.getsize(os.path.join(OUT, f))
    print(f"  {f}  ({sz:,} bytes)")
