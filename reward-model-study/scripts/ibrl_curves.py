"""Extract IBRL score curves from local wandb run logs and plot the downstream-RL
story: (A) GT-control trains vs VLM-reward caps, (B) the beta/threshold sweep is
uniformly flat, (C) per-model VLM curves. Reads output.log (offline, no wandb).

  python reward-model-study/scripts/ibrl_curves.py
  -> reward-model-study/figures/rl_*.png  +  results/ibrl_curves.csv
"""
import glob
import json
import os
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WB = Path("/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl/wandb")
FIG = Path("/gpfs/home3/pkarageorgis1/Master-Thesis/reward-model-study/figures")
RES = Path("/gpfs/home3/pkarageorgis1/Master-Thesis/reward-model-study/results")
FIG.mkdir(parents=True, exist_ok=True)

SCORE_RE = re.compile(r"^\s*(\d+):\s*score/score\s*:\s*([\d.]+)", re.M)
TRAIN_RE = re.compile(r"^\s*(\d+):\s*score/train_score\s*\[[^\]]*\]:\s*avg:\s*([\d.]+)", re.M)


def parse_run(run_dir):
    meta = os.path.join(run_dir, "files", "wandb-metadata.json")
    log = os.path.join(run_dir, "files", "output.log")
    if not (os.path.isfile(meta) and os.path.isfile(log)):
        return None
    args = json.load(open(meta)).get("args", [])
    sd = ""
    for i, a in enumerate(args):
        if a == "--save_dir" and i + 1 < len(args):
            sd = os.path.basename(args[i + 1])
    txt = open(log, errors="ignore").read()
    ev = [(int(s), float(v)) for s, v in SCORE_RE.findall(txt)]
    tr = [(int(s), float(v)) for s, v in TRAIN_RE.findall(txt)]
    if not ev and not tr:
        return None
    return {"save_dir": sd, "eval": ev, "train": tr}


runs = [r for r in (parse_run(d) for d in sorted(glob.glob(str(WB / "run-*")))) if r]


def find(substrs, want_min_pts=3, prefer_long=True):
    """Return runs whose save_dir contains all substrs, with enough eval points."""
    out = []
    for r in runs:
        sd = r["save_dir"]
        if all(s in sd for s in substrs) and len(r["eval"]) >= want_min_pts:
            out.append(r)
    if prefer_long:
        out.sort(key=lambda r: -len(r["eval"]))
    return out


def curve(r, kind="eval"):
    pts = sorted(r[kind])
    if not pts:
        return np.array([]), np.array([])
    x, y = zip(*pts)
    return np.array(x) / 1000.0, np.array(y)


GT = "#2ca02c"
VLM4B = "#e0473b"
VLMFT = "#1f6feb"
VLMQ = "#d2a8ff"
GREY = "#999999"

# =====================================================================
# A — GT control trains vs VLM reward caps  (the headline)
# train_score (50-ep running avg) = the honest SUSTAINED signal; eval score/score
# is a noisy 20-ep estimate that spikes (use train to avoid over-claiming).
# =====================================================================
reps = [("coffeepush_robometer_4b", VLM4B, "Robometer-4B + VLM reward"),
        ("coffeepush_robometer_ft", VLMFT, "Robometer-FT + VLM reward"),
        ("coffeepush_qwen35_ft", VLMQ, "Qwen3.5-FT + VLM reward")]
fig, ax = plt.subplots(figsize=(9, 5.2))
gt_labeled = False
for r in find(["coffeepush_gt"], want_min_pts=1):
    x, y = curve(r, "train")
    ax.plot(x, y, color=GT, lw=2.6, alpha=0.95, marker="o", ms=3,
            label="GT reward (3 seeds)" if not gt_labeled else None)
    gt_labeled = True
for key, col, lbl in reps:
    cand = find([key], want_min_pts=1)
    if not cand:
        continue
    x, y = curve(cand[0], "train")
    ax.plot(x, y, color=col, lw=2.2, marker="o", ms=3, label=lbl)
ax.axhline(0.0, color="#ccc", lw=0.6)
ax.set_xlabel("environment steps (×1000)")
ax.set_ylabel("train success rate (50-ep running avg)")
ax.set_title("The loop is capable — the reward is the limiter\n"
             "GT reward trains to 0.48–0.76; every VLM reward caps ≤ 0.15", fontsize=12)
ax.set_ylim(-0.03, 0.9)
ax.legend(fontsize=9, loc="upper left")
ax.grid(ls=":", alpha=0.4)
fig.tight_layout(); fig.savefig(FIG / "rl_gt_vs_vlm.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("wrote rl_gt_vs_vlm.png")

# =====================================================================
# B — beta / threshold sweep is uniformly flat (CoffeePush, Robometer-FT)
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 5.2))
sweep = [r for r in runs if "coffeepush_robometer_ft_b" in r["save_dir"] and len(r["train"]) >= 3]
peak = []
for r in sweep:
    x, y = curve(r, "train")
    m = re.search(r"_b([\d.]+)_t([\d.]+)", r["save_dir"])
    lbl = f"β={m.group(1)} τ={m.group(2)}" if m else r["save_dir"]
    ax.plot(x, y, lw=1.4, alpha=0.7, marker=".", ms=4)
    peak.append((lbl, float(np.max(y)) if len(y) else 0.0))
ax.set_xlabel("environment steps (×1000)")
ax.set_ylabel("eval success rate")
ax.set_title(f"β / threshold sweep on Robometer-FT — all {len(sweep)} configs stay flat\n"
             "(reward shaping cannot rescue a reward with no usable signal)", fontsize=12)
ax.set_ylim(-0.03, 0.9)
ax.grid(ls=":", alpha=0.4)
fig.tight_layout(); fig.savefig(FIG / "rl_beta_tau_sweep.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"wrote rl_beta_tau_sweep.png  ({len(sweep)} configs, peak={max(p for _,p in peak) if peak else 0:.2f})")

# =====================================================================
# C — per-model VLM curves (zoom: everything under ~0.2)
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 5.2))
for key, col, lbl in reps:
    group = find([key], want_min_pts=1)
    for idx, r in enumerate(group[:5]):
        x, y = curve(r, "train")
        ax.plot(x, y, color=col, lw=1.6, alpha=0.65, marker=".", ms=4,
                label=lbl if idx == 0 else None)
ax.set_xlabel("environment steps (×1000)")
ax.set_ylabel("train success rate (50-ep running avg)")
ax.set_title("VLM-reward IBRL by base model — none sustains above ~0.15", fontsize=12)
ax.set_ylim(-0.02, 0.35)
ax.legend(fontsize=9)
ax.grid(ls=":", alpha=0.4)
fig.tight_layout(); fig.savefig(FIG / "rl_vlm_by_model.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("wrote rl_vlm_by_model.png")

# =====================================================================
# D — peak train success by base model  (clean replacement for the tangle)
# =====================================================================
def group_peak(key):
    cand = find([key], want_min_pts=1)
    best = 0.0
    for r in cand:
        _, y = curve(r, "train")
        if len(y):
            best = max(best, float(np.max(y)))
    return best, len(cand)


bars = [("Robometer-4B\n+VLM", group_peak("coffeepush_robometer_4b"), VLM4B),
        ("Robometer-FT\n+VLM", group_peak("coffeepush_robometer_ft"), VLMFT),
        ("Qwen3.5-FT\n+VLM", group_peak("coffeepush_qwen35_ft"), VLMQ),
        ("GT reward\n(control)", group_peak("coffeepush_gt"), GT)]
fig, ax = plt.subplots(figsize=(8, 5.2))
xs = np.arange(len(bars))
vals = [b[1][0] for b in bars]
cols = [b[2] for b in bars]
ax.bar(xs, vals, color=cols, edgecolor="#222", width=0.62)
for x, b in zip(xs, bars):
    ax.text(x, b[1][0] + 0.015, f"{b[1][0]:.2f}\n(n={b[1][1]})", ha="center", va="bottom", fontsize=9)
ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in bars], fontsize=10)
ax.set_ylabel("best train success rate reached")
ax.set_title("Peak IBRL success by reward — every VLM reward ≤ 0.15, GT control 0.76", fontsize=12)
ax.set_ylim(0, 0.9); ax.grid(axis="y", ls=":", alpha=0.4)
fig.tight_layout(); fig.savefig(FIG / "rl_peak_by_model.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("wrote rl_peak_by_model.png")

# =====================================================================
# E — distribution shift: sp-AUC degrades offline -> BC -> IBRL
# (computed from /scratch-shared sp_dist_eval rollout CSVs)
# =====================================================================
SPD = Path("/scratch-shared/pkarageorgis1/sp_dist_eval")


def roll_auc(path, col="sp"):
    if not path.is_file():
        return None
    sp, gt = [], []
    import csv as _csv
    for r in _csv.DictReader(open(path)):
        try:
            sp.append(float(r[col])); gt.append(int(float(r["env_success"])))
        except Exception:
            pass
    sp = np.array(sp); gt = np.array(gt); pos = sp[gt == 1]; neg = sp[gt == 0]
    if not len(pos) or not len(neg):
        return None
    a = np.concatenate([pos, neg]); o = np.argsort(a, kind="mergesort"); rk = np.empty(len(a)); rk[o] = np.arange(1, len(a) + 1)
    return float((rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


stages = ["offline\n(held-out eval)", "BC\nrollouts", "IBRL\nrollouts"]
shift = [
    ("Robometer-FT  (asym+ICL)", VLM4B, 0.85,
     roll_auc(SPD / "bconly_robometer_ft_run1_s3000.csv"), roll_auc(SPD / "robometer_ft_run1_s3000.csv")),
    ("Qwen3.5-FT  (asym+ICL)", VLMQ, 0.83,
     roll_auc(SPD / "bconly_qwen35_ft_run4_s6500.csv"), roll_auc(SPD / "qwen35_ft_run4_s6500.csv")),
    ("Robometer-4B baseline", GREY, None,
     roll_auc(SPD / "bconly_robometer_4b_baseline.csv"), roll_auc(SPD / "robometer_4b_baseline.csv")),
]
fig, ax = plt.subplots(figsize=(8.2, 5.2))
xs = np.arange(3)
for lbl, col, off, bc, ib in shift:
    ys = [off, bc, ib]
    xx = [i for i, y in enumerate(ys) if y is not None]
    yy = [y for y in ys if y is not None]
    ax.plot(xx, yy, color=col, lw=2.4, marker="o", ms=7, label=lbl)
    for i, y in zip(xx, yy):
        ax.text(i, y + 0.012, f"{y:.2f}", ha="center", fontsize=9, color=col)
ax.axhspan(0.0, 0.5, color="#f0f0f0", alpha=0.7, zorder=0)
ax.axhline(0.5, color="#999", ls="--", lw=1)
ax.text(2.0, 0.46, "chance (0.5)", ha="right", va="top", fontsize=9, color="#777")
ax.set_xticks(xs); ax.set_xticklabels(stages, fontsize=10)
ax.set_ylabel("success-prob AUC (reward separates succ/fail)")
ax.set_title("Distribution shift collapses the reward off the demo manifold\n"
             "FT degrades to / below chance on-policy; baseline holds up better", fontsize=12)
ax.set_ylim(0.3, 0.9); ax.legend(fontsize=9, loc="lower left"); ax.grid(axis="y", ls=":", alpha=0.4)
fig.tight_layout(); fig.savefig(FIG / "rl_distribution_shift.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("wrote rl_distribution_shift.png")

# =====================================================================
# tidy CSV of all curves
# =====================================================================
with open(RES / "ibrl_curves.csv", "w") as f:
    f.write("save_dir,kind,step,score\n")
    for r in runs:
        for kind in ("eval", "train"):
            for s, v in r[kind]:
                f.write(f"{r['save_dir']},{kind},{s},{v}\n")
print(f"wrote ibrl_curves.csv  ({len(runs)} runs with curves)")

# summary for the deck
print("\n=== peak eval success by group ===")
for key, _, lbl in reps + [("coffeepush_gt", None, "GT control")]:
    cand = find([key], want_min_pts=1)
    if cand:
        pk = max((np.max(curve(r)[1]) if len(curve(r)[1]) else 0) for r in cand)
        print(f"  {lbl:32} peak={pk:.2f}  (n_runs={len(cand)})")
