"""Rebuild thesis Figure 5.3 (the frame-budget ablation) for any pair of arms.

  python make_frames_8v16.py --a wandb_run2_8f.json  --a-label "8 frames" \
                             --b wandb_run2_16f.json --b-label "16 frames" \
                             --at best --out frames_8v16_run2

(a) widest success-failure margin per split, grouped bars, at the chosen checkpoint
(b) DROID rank accuracy over training, one line per arm

--at final  use each arm's last eval point
--at best   use each arm's best checkpoint, scored by the mean of the four splits'
            ranking_acc_last, which is the selection rule the runs themselves used
"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPLITS_BAR = ["failsafe", "robometer", "metaworld"]
NICE = {"failsafe": "FailSafe", "robometer": "Robometer", "metaworld": "MetaWorld"}
C8, C16 = "#4C72B0", "#C44E52"


def load(path):
    d = json.load(open(path))
    return d, d["suffix"]


def series(d, kind, split):
    sfx = d["suffix"]
    key = f"eval_p_rank/{kind}/robometer_frames_eval_{split}{sfx}"
    xs, ys = [], []
    for r in d["rows"]:
        if key in r and "_step" in r:
            xs.append(r["_step"]); ys.append(r[key])
    o = np.argsort(xs)
    return np.array(xs)[o], np.array(ys)[o]


def pick_step(d, how):
    steps = sorted({r["_step"] for r in d["rows"] if "_step" in r})
    if how == "final":
        return steps[-1]
    best, bs = -1, steps[-1]
    for s in steps:
        vals = []
        for sp in ["droid", "robometer", "metaworld", "failsafe"]:
            xs, ys = series(d, "ranking_acc_last", sp)
            m = xs == s
            if m.any():
                vals.append(ys[m][0])
        if vals and np.mean(vals) > best:
            best, bs = np.mean(vals), s
    return bs


def value_at(d, kind, split, step):
    xs, ys = series(d, kind, split)
    m = xs == step
    return float(ys[m][0]) if m.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--a-label", default="8 frames")
    ap.add_argument("--b", required=True); ap.add_argument("--b-label", default="16 frames")
    ap.add_argument("--at", choices=["final", "best"], default="best")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", default="frames_8v16")
    args = ap.parse_args()

    A, _ = load(args.a); B, _ = load(args.b)
    sa, sb = pick_step(A, args.at), pick_step(B, args.at)
    print(f"{args.a_label}: step {sa}   {args.b_label}: step {sb}   (--at {args.at})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.1), dpi=200,
                                   gridspec_kw={"width_ratios": [1.0, 1.25]})

    x = np.arange(len(SPLITS_BAR)); w = 0.36
    va = [value_at(A, "max_succ_fail_diff_last", s, sa) for s in SPLITS_BAR]
    vb = [value_at(B, "max_succ_fail_diff_last", s, sb) for s in SPLITS_BAR]
    for i, (v, c, lab, off) in enumerate([(va, C8, args.a_label, -w/2), (vb, C16, args.b_label, w/2)]):
        bars = ax1.bar(x + off, v, w, color=c, label=lab, edgecolor="white", linewidth=0.6)
        for b, val in zip(bars, v):
            ax1.text(b.get_x() + b.get_width()/2, val + 0.008, f"{val:.2f}",
                     ha="center", va="bottom", fontsize=7.5)
    ax1.set_xticks(x); ax1.set_xticklabels([NICE[s] for s in SPLITS_BAR], fontsize=9)
    ax1.set_ylabel("Widest success$-$failure margin", fontsize=9)
    ax1.set_title("(a) Best margin per split", fontsize=10)
    ax1.set_ylim(0, max(max(va), max(vb)) * 1.28)
    ax1.legend(fontsize=8, frameon=False, loc="upper left")
    ax1.grid(True, axis="y", color="0.9", lw=0.7); ax1.set_axisbelow(True)

    for d, c, lab in [(A, C8, args.a_label), (B, C16, args.b_label)]:
        xs, ys = series(d, "ranking_acc_last", "droid")
        ax2.plot(xs, ys, color=c, marker="o", ms=3.5, lw=1.8, label=lab)
        if len(xs):
            ax2.annotate(f"{ys[-1]:.3f}", (xs[-1], ys[-1]), textcoords="offset points",
                         xytext=(4, -1), fontsize=7.5, color=c)
    ax2.axhline(0.5, color="0.6", lw=0.8, ls=":")
    ax2.set_xlabel("Training steps", fontsize=9)
    ax2.set_ylabel("Rank accuracy (DROID)", fontsize=9)
    ax2.set_title("(b) DROID rank accuracy over training", fontsize=10)
    ax2.legend(fontsize=8, frameon=False, loc="lower right")
    ax2.grid(True, color="0.9", lw=0.7); ax2.set_axisbelow(True)

    for a in (ax1, ax2):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.tick_params(labelsize=8)
    if args.title:
        fig.suptitle(args.title, fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
    else:
        fig.tight_layout()
    fig.savefig(args.out + ".pdf"); fig.savefig(args.out + ".png")
    print("wrote", args.out + ".pdf/.png")
    print("\npanel (a) values")
    for s, p, q in zip(SPLITS_BAR, va, vb):
        print(f"  {NICE[s]:11s} {args.a_label}={p:.3f}  {args.b_label}={q:.3f}")


if __name__ == "__main__":
    main()
