"""Goodhart curves with the step-level FP metric of Table 3 (tab:hacking).

tau = 5th percentile of scores at successful steps (gt_per_step == 1), pooled over a
model's seeds. FP = fraction of steps in never-solved episodes with score >= tau.
"""
import glob, json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.environ.get("ROBOREF_RUNS", "/scratch-shared/pkarageorgis1/roboref_runs")
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
SEEDS = {"run2": [3, 9, 0, 1, 2]}
DEFAULT = [0, 1, 2, 3, 4]
MODELS = [("run2", "RoboRef"), ("run3", "Robometer + Failures"), ("base", "Robometer-4B"),
          ("robodopamine", "RoboDopamine"), ("roboreward", "RoboReward-4B"), ("lrm", "LRM-TRI")]
BIN = 10_000
DEAD = 0.05


def legs(model, seed):
    out = []
    for d in sorted(glob.glob(f"{R}/ms_PullCube-v1_{model}_dense_s{seed}_*")):
        if re.match(rf".*_s{seed}_\d+_\d+$", d):
            p = f"{d}/episodes.jsonl"
            if os.path.exists(p) and os.path.getsize(p) > 0:
                out.append(p)
    return out


def load_seed(model, seed):
    sc, gt, un, ep_solved, ep_ret = [], [], [], [], []
    for p in legs(model, seed):
        rows = []
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        if len(rows) < 50:          # aborted resume stubs
            continue
        for r in rows:
            s = r.get("score_per_step") or []
            g = r.get("gt_per_step") or [0] * len(s)
            if not s:
                continue
            solved = int(r.get("gt_solved_anytime", 0))
            sc.extend(s); gt.extend(g[:len(s)]); un.extend([1 - solved] * len(s))
            ep_solved.extend([solved] * len(s))
            rw = r.get("reward_per_step") or s
            ep_ret.extend(rw)
    if not sc:
        return None
    return dict(score=np.array(sc, float), gt=np.array(gt, int), unsolved=np.array(un, bool),
                solved=np.array(ep_solved, int), ret=np.array(ep_ret, float))


data, tau = {}, {}
for key, _ in MODELS:
    data[key] = {sd: d for sd in SEEDS.get(key, DEFAULT) if (d := load_seed(key, sd)) is not None}
    pos = np.concatenate([d["score"][d["gt"] == 1] for d in data[key].values()])
    tau[key] = np.quantile(pos, 0.05)

print(f"{'model':22s} {'tau':>7} {'pooled FP':>9} {'solved steps':>12}")
for key, name in MODELS:
    S = np.concatenate([d["score"] for d in data[key].values()])
    U = np.concatenate([d["unsolved"] for d in data[key].values()])
    G = np.concatenate([d["gt"] for d in data[key].values()])
    print(f"{name:22s} {tau[key]:7.4f} {(S[U] >= tau[key]).mean():9.3f} {G.mean():12.3f}")

print(f"\n{'model':22s} {'seed':>4} {'solve 1st20%':>12} {'solve last20%':>13} {'stepFP 1st20%':>13} {'stepFP last20%':>14} {'stepFP all':>10}")
for key, name in MODELS:
    for sd, d in data[key].items():
        n = len(d["score"]); a, z = slice(0, n // 5), slice(4 * n // 5, n)
        def fp(s):
            u = d["unsolved"][s]
            return (d["score"][s][u] >= tau[key]).mean() if u.any() else np.nan
        print(f"{name:22s} {sd:4d} {d['solved'][a].mean():12.3f} {d['solved'][z].mean():13.3f} "
              f"{fp(a):13.3f} {fp(z):14.3f} {fp(slice(0, n)):10.3f}")

cols = ["mean per-step reward\n(proxy SAC maximizes)", "GT solve rate (episodes)",
        "FP, Table 3 definition\n(steps of never-solved eps ≥ τ)"]
fig, ax = plt.subplots(len(MODELS), 3, figsize=(13, 2.3 * len(MODELS)), sharex=True)
for i, (key, name) in enumerate(MODELS):
    for sd, d in data[key].items():
        b = np.arange(len(d["score"])) // BIN
        xs, sv, rt, fp = [], [], [], []
        for k in np.unique(b):
            m = b == k; u = m & d["unsolved"]
            xs.append((k + .5) * BIN); sv.append(d["solved"][m].mean()); rt.append(d["ret"][m].mean())
            fp.append((d["score"][u] >= tau[key]).mean() if u.any() else np.nan)
        sv = np.array(sv)
        dead = sv[-max(1, len(sv) // 5):].mean() < DEAD
        kw = dict(lw=1.4 if dead else 1.0, color="tab:red" if dead else "tab:blue", alpha=.9 if dead else .6)
        ax[i, 0].plot(xs, rt, **kw); ax[i, 1].plot(xs, sv, **kw); ax[i, 2].plot(xs, fp, **kw)
        ax[i, 1].annotate(f"s{sd}", (xs[-1], sv[-1]), fontsize=7, color=kw["color"])
    ax[i, 0].set_ylabel(name, fontsize=10, fontweight="bold")
    ax[i, 1].set_ylim(-.03, 1.03); ax[i, 2].set_ylim(-.03, 1.03)
for j, c in enumerate(cols):
    ax[0, j].set_title(c, fontsize=10); ax[-1, j].set_xlabel("env steps")
fig.suptitle(f"PullCube SAC training, {BIN//1000}k-step bins. Red = seed dead in its last 20%", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/goodhart_curves_stepFP.png", dpi=150)
print(f"\nwrote {OUT}/goodhart_curves_stepFP.png")
