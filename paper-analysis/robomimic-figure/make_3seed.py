"""Robomimic PickPlace-Can, 3 seeds: the original thesis seed (recovered from the
Figure 5.10 PDF) plus the two new Snellius seeds."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/projects/prjs1958/pkarageorgis1/robomimic_figure"
old = json.load(open(f"{D}/robomimic_rl.pdf.curves.json"))   # seed 1
new = json.load(open(f"{D}/new_seeds.json"))                 # seeds 2, 3

# original legend name -> new display name
RENAME = {"RoboRef-Asym": "RoboRef", "RoboRef-Std": "Robometer + Failures"}
MODELS = ["RoboRef", "Robometer + Failures", "Robometer-4B-Base",
          "RoboDopamine", "RoboReward", "LRM"]
STYLE = {  # colours kept from the original figure's own legend swatches
    "RoboRef":              ((0.8902,0.1020,0.1098), "s"),
    "Robometer + Failures": ((0.4157,0.2392,0.6039), "^"),
    "Robometer-4B-Base":    ((0.2000,0.6275,0.1725), "D"),
    "RoboDopamine":         ((0.8314,0.6275,0.0902), "v"),
    "RoboReward":           ((0.5490,0.3373,0.2941), "P"),
    "LRM":                  ((0.1500,0.1500,0.1500), "X"),
}
GRID = np.arange(5000, 750001, 5000)
W = 5   # rolling window (25k steps); the thesis figure is already smoothed

def smooth(y, w=W):
    """NaN-safe rolling mean: average only the finite values inside each window, so a
    NaN at an edge cannot eat neighbouring points (which silently trimmed the first
    two grid points off every curve)."""
    y = np.asarray(y, dtype=float)
    if len(y) < 2: return y
    out = np.full(len(y), np.nan)
    h = w // 2
    for i in range(len(y)):
        seg = y[max(0, i-h):min(len(y), i+h+1)]
        seg = seg[np.isfinite(seg)]
        if seg.size: out[i] = seg.mean()
    return out

def series(model):
    """Return (n_seeds, len(GRID)) with NaN where a seed has no data."""
    rows = []
    orig_key = next((k for k in old if k.split("|")[1] == model
                     or RENAME.get(k.split("|")[1]) == model), None)
    if orig_key:
        pts = np.array(old[orig_key]); 
        rows.append(np.interp(GRID, pts[:,0], pts[:,1], left=np.nan, right=np.nan))
    for s in ("s2", "s3"):
        k = f"{model}|{s}"
        if k in new:
            pts = np.array(new[k])
            rows.append(np.interp(GRID, pts[:,0], pts[:,1], left=np.nan, right=np.nan))
    return np.vstack(rows)

fig, ax = plt.subplots(figsize=(11.2, 4.6), dpi=150)
for m in MODELS:
    arr = series(m)
    arr = np.vstack([smooth(r) for r in arr])
    mu = np.nanmean(arr, axis=0)
    n = np.sum(~np.isnan(arr), axis=0)
    sd = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))   # standard error
    ok = ~np.isnan(mu)
    c, mk = STYLE[m]
    ax.fill_between(GRID[ok], (mu-sd)[ok], (mu+sd)[ok], color=c, alpha=0.13, lw=0)
    ax.plot(GRID[ok], mu[ok], color=c, lw=2.0, marker=mk, markersize=4.5,
            markevery=12, label=m, zorder=3)

ax.set_xlim(0, 760000); ax.set_ylim(-0.01, 0.50)
ax.set_xticks(range(0, 700001, 100000))
ax.set_xticklabels(["0","100k","200k","300k","400k","500k","600k","700k"])
ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5])
ax.set_xlabel("Training steps", fontsize=12)
ax.set_ylabel("Task success rate (ground truth)", fontsize=12)
ax.set_title("Robomimic IBRL (PickPlace-Can): policy success by VLM reward model"
             "   (mean $\\pm$ 1 s.e. over 3 seeds)", fontsize=13, fontweight="bold")
ax.grid(True, color="0.9", lw=0.8)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5,-0.16), ncol=3,
          frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(f"{D}/robomimic_3seed.png", bbox_inches="tight")
fig.savefig(f"{D}/robomimic_3seed.pdf", bbox_inches="tight")
print("wrote robomimic_3seed.png / .pdf")
for m in MODELS:
    a = np.vstack([smooth(r) for r in series(m)])
    mu = np.nanmean(a, axis=0)
    print(f"  {m:22s} final mean={mu[-1]:.3f}  peak={np.nanmax(mu):.3f}  seeds={a.shape[0]}")
