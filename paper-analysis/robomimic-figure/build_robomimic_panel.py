"""Rewrite RoboRef's curve in the Robomimic panel of the sparse-reward figure.

Recomputes RoboRef's mean ground-truth success and its standard-error band over
the chosen seeds, with the same smoothing as make_3seed.py, and writes them into
panel (c) of media/tikz_results/sparse-rl.tex in the paper repo. Other curves and
panels are left untouched. The paper uses seeds 1, 3 and 4.

    python build_robomimic_panel.py --tex /path/to/overleaf/media/tikz_results/sparse-rl.tex --seeds 1 3 4
"""
import argparse, json, pickle, re
import numpy as np

D = "/projects/prjs1958/pkarageorgis1/robomimic_figure"
SEED4 = "/projects/prjs1958/pkarageorgis1/vlm_ibrl_robomimic/can_robometer_ft_thr0.8_min120_seed4/log.pkl"
GRID = np.arange(5000, 750001, 5000)

def smooth(y, w=5):
    out = np.full(len(y), np.nan); h = w // 2
    for i in range(len(y)):
        seg = y[max(0, i - h):i + h + 1]; seg = seg[np.isfinite(seg)]
        if seg.size: out[i] = seg.mean()
    return out

def on_grid(pts):
    pts = np.array(pts); return np.interp(GRID, pts[:, 0], pts[:, 1], left=np.nan, right=np.nan)

def seed_curves():
    old = json.load(open(f"{D}/robomimic_rl.pdf.curves.json")); new = json.load(open(f"{D}/new_seeds.json"))
    k1 = [k for k in old if k.split("|")[1] in ("RoboRef-Asym", "RoboRef")][0]
    log = pickle.load(open(SEED4, "rb"))
    s4 = [(r["other/step"], r["score/score"]) for r in log if "score/score" in r]
    return {1: old[k1], 2: new["RoboRef|s2"], 3: new["RoboRef|s3"], 4: s4}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tex", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[1, 3, 4])
    a = ap.parse_args()
    curves = seed_curves(); A = np.vstack([smooth(on_grid(curves[s])) for s in a.seeds])
    mu = np.nanmean(A, 0); se = np.nanstd(A, 0, ddof=1) / np.sqrt(np.sum(~np.isnan(A), 0))
    lo, hi = np.clip(mu - se, 0, None), mu + se
    fmt = lambda xs, ys: " ".join(f"({int(x)},{y:.4f})" for x, y in zip(xs, ys))
    band = fmt(GRID, hi) + " " + fmt(GRID[::-1], lo[::-1]); line = fmt(GRID, mu)
    src = open(a.tex).read(); i = src.index("title={(c) Robomimic") if "title={(c) Robomimic" in src else src.index("title={Robomimic")
    head, panel = src[:i], src[i:]
    panel = re.sub(r"(\\addplot\[draw=none, fill=cAsym[^\]]*\] coordinates \{)[^}]*\}", lambda m: m.group(1) + band + "}", panel, count=1)
    panel = re.sub(r"(\\addplot\[color=cAsym[^\]]*\] coordinates \{)[^}]*\}", lambda m: m.group(1) + line + "}", panel, count=1)
    ymax = 0.05 * np.ceil(np.nanmax(hi) / 0.05 + 0.2)
    panel = re.sub(r"ymax=[0-9.]+", f"ymax={ymax:.2f}", panel, count=1)
    open(a.tex, "w").write(head + panel)
    print(f"seeds {a.seeds}: final mean {mu[-1]:.3f} +- {se[-1]:.3f}, band max {np.nanmax(hi):.3f}, ymax {ymax:.2f}")

if __name__ == "__main__":
    main()
