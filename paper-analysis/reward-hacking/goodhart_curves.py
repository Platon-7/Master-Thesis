"""Goodhart curves for the ManiSkill PullCube runs (plot A) and a per-seed scatter (D).

Every training episode is logged with the reward model's per-step score, which is what
SAC maximizes, and the simulator's success flag. Binning the episodes in training order
gives, per seed, the proxy the agent optimized next to the ground truth it should track.
  reward hacking        proxy climbs, success stays flat, false positives rise
  exploration failure   proxy and success both stay flat
Uses the same runs, seeds and leg stitching as onpolicy_fp.py.
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
BIN = 10_000      # env steps per bin
DEAD = 0.05       # a seed whose last-20% solve rate is below this counts as dead


def legs(model, seed):
    out = []
    for d in sorted(glob.glob(f"{R}/ms_PullCube-v1_{model}_dense_s{seed}_*")):
        if not re.match(rf".*_s{seed}_\d+_\d+$", d):
            continue
        p = f"{d}/episodes.jsonl"
        if os.path.exists(p) and os.path.getsize(p) > 0:
            out.append(p)
    return out


def load_leg(p):
    L, S, Y, M, E = [], [], [], [], []
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        sp = r.get("score_per_step") or []
        sc = r.get("score_max")
        if sc is None:
            sc = max(sp) if sp else None
        if sc is None:
            continue
        rw = r.get("reward_per_step") or sp
        L.append(int(r.get("episode_len") or len(sp)))
        S.append(float(sc))
        Y.append(int(r.get("gt_solved_anytime", 0)))
        M.append(float(np.mean(rw)) if rw else float(r.get("vlm_return_mean") or np.nan))
        E.append(r.get("ep"))
    return dict(len=np.array(L), peak=np.array(S), solved=np.array(Y), ret=np.array(M), ep=E)


def load_seed(model, seed):
    parts = []
    for p in legs(model, seed):
        d = load_leg(p)
        if len(d["len"]) < 50:   # aborted resume stubs (e.g. 4 eps), not training
            continue
        print(f"    {os.path.basename(os.path.dirname(p))}: {len(d['len'])} eps, "
              f"ep {d['ep'][0]}..{d['ep'][-1]}, {d['len'].sum():,} env steps")
        parts.append(d)
    if not parts:
        return None
    cat = {k: np.concatenate([d[k] for d in parts]) for k in ("len", "peak", "solved", "ret")}
    cat["steps"] = np.cumsum(cat["len"])
    return cat


def binned(d, tau):
    b = d["steps"] // BIN
    xs, solve, ret, fp, pk_fail = [], [], [], [], []
    for i in np.unique(b):
        m = b == i
        un = m & (d["solved"] == 0)
        xs.append((i + 0.5) * BIN)
        solve.append(d["solved"][m].mean())
        ret.append(d["ret"][m].mean())
        fp.append((d["peak"][un] >= tau).mean() if un.any() and np.isfinite(tau) else np.nan)
        pk_fail.append(d["peak"][un].mean() if un.any() else np.nan)
    return tuple(np.array(v) for v in (xs, solve, ret, fp, pk_fail))


def frac(d, lo, hi):
    n = len(d["len"])
    return slice(int(lo * n), int(hi * n))


os.makedirs(OUT, exist_ok=True)
data = {}
for key, name in MODELS:
    print(name)
    data[key] = {}
    for sd in SEEDS.get(key, DEFAULT):
        d = load_seed(key, sd)
        if d is None:
            print(f"    seed {sd} missing")
            continue
        data[key][sd] = d

# FP threshold per model, as in onpolicy_fp.py: catch 95 percent of solved episodes
tau = {}
for key, _ in MODELS:
    pos = np.concatenate([d["peak"][d["solved"] == 1] for d in data[key].values()] or [np.array([])])
    tau[key] = np.quantile(pos, 0.05) if len(pos) else np.nan

print(f"\n{'model':22s} {'seed':>4} {'steps':>9} {'solve 1st20%':>12} {'solve last20%':>13} "
      f"{'ret 1st':>9} {'ret last':>9} {'pk(fail) 1st':>12} {'pk(fail) last':>13} "
      f"{'FP 1st':>7} {'FP last':>7}")
rows = []
for key, name in MODELS:
    for sd, d in data[key].items():
        a, z = frac(d, 0, .2), frac(d, .8, 1)
        def fpr(s):
            un = d["solved"][s] == 0
            return (d["peak"][s][un] >= tau[key]).mean() if un.any() and np.isfinite(tau[key]) else np.nan
        def pkf(s):
            un = d["solved"][s] == 0
            return d["peak"][s][un].mean() if un.any() else np.nan
        r = dict(key=key, name=name, seed=sd, steps=d["steps"][-1],
                 s0=d["solved"][a].mean(), s1=d["solved"][z].mean(),
                 r0=d["ret"][a].mean(), r1=d["ret"][z].mean(),
                 p0=pkf(a), p1=pkf(z), f0=fpr(a), f1=fpr(z))
        rows.append(r)
        print(f"{name:22s} {sd:4d} {r['steps']:9,d} {r['s0']:12.3f} {r['s1']:13.3f} {r['r0']:9.4f} "
              f"{r['r1']:9.4f} {r['p0']:12.4f} {r['p1']:13.4f} {r['f0']:7.3f} {r['f1']:7.3f}")
print("\nFP threshold (5th pct of peak on solved episodes):",
      {k: round(float(v), 4) for k, v in tau.items()})

# Plot A: one row per model; proxy, ground truth, false-positive rate
cols = ["mean per-step reward\n(proxy SAC maximizes)", "GT solve rate", "false-positive rate\n(unsolved eps above τ)"]
fig, ax = plt.subplots(len(MODELS), 3, figsize=(13, 2.3 * len(MODELS)), sharex=True)
for i, (key, name) in enumerate(MODELS):
    for sd, d in data[key].items():
        x, s, r, f, _ = binned(d, tau[key])
        dead = s[-max(1, len(s) // 5):].mean() < DEAD
        kw = dict(lw=1.4 if dead else 1.0, color="tab:red" if dead else "tab:blue", alpha=.9 if dead else .6)
        ax[i, 0].plot(x, r, **kw)
        ax[i, 1].plot(x, s, **kw)
        ax[i, 2].plot(x, f, **kw)
        ax[i, 1].annotate(f"s{sd}", (x[-1], s[-1]), fontsize=7, color=kw["color"])
    ax[i, 0].set_ylabel(name, fontsize=10, fontweight="bold")
    ax[i, 1].set_ylim(-.03, 1.03)
    ax[i, 2].set_ylim(-.03, 1.03)
for j, c in enumerate(cols):
    ax[0, j].set_title(c, fontsize=10)
    ax[-1, j].set_xlabel("env steps")
fig.suptitle(f"PullCube SAC training, {BIN//1000}k-step bins. Red = seed dead in its last 20%", fontsize=11)
fig.tight_layout()
for ext in ("png",):
    fig.savefig(f"{OUT}/goodhart_curves.{ext}", dpi=150)

# Plot D: per seed, late-training success vs late-training false-positive rate
fig, ax = plt.subplots(figsize=(6, 4.5))
mk = dict(zip([k for k, _ in MODELS], "o s ^ D v P".split()))
for key, name in MODELS:
    rs = [r for r in rows if r["key"] == key]
    ax.scatter([r["s1"] for r in rs], [r["f1"] for r in rs], marker=mk[key], s=45, label=name, alpha=.8)
ax.set_xlabel("GT solve rate, last 20% of training")
ax.set_ylabel("false-positive rate, last 20% of training")
ax.legend(fontsize=8)
fig.tight_layout()
for ext in ("png",):
    fig.savefig(f"{OUT}/seed_scatter.{ext}", dpi=150)
print(f"\nwrote {OUT}/goodhart_curves.png and seed_scatter.png")
