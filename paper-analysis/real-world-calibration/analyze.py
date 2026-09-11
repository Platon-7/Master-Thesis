"""Calibrate the success-head stop threshold from scores_{roboref,robometer}.json.

Online rule (same as the sim calibration): stop the episode the first time the
success probability reaches the threshold on K consecutive steps. An episode "fires"
iff its running-K-min series ever reaches thr, so the per-episode score is the max
over t of min(sp[t-K+1..t]). GT success episodes should fire (TP), GT failures should not (FP).
"""
import json, sys, numpy as np
from scipy.stats import rankdata
rng = np.random.default_rng(0)

def auroc(pos, neg):
    s = np.r_[pos, neg]; r = rankdata(s)          # tie-aware (average ranks)
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def boot_ci(pos, neg, n=5000):
    v = [auroc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n)]
    return np.percentile(v, [2.5, 97.5])

def kscore(sp, K):
    sp = np.asarray(sp)
    if len(sp) < K: return sp.min()
    return max(sp[i:i + K].min() for i in range(len(sp) - K + 1))

def first_fire(sp, thr, K):
    run = 0
    for i, v in enumerate(sp):
        run = run + 1 if v >= thr else 0
        if run >= K: return i
    return None

for m, ref in (("roboref", 0.8), ("robometer", 0.6), ("roboref_icl", 0.8)):
    try: D = json.load(open(f"scores_{m}.json"))
    except FileNotFoundError: print(f"\n##### {m}: not finished\n"); continue
    E = D["episodes"]; pos = [e for e in E if e["label"]]; neg = [e for e in E if not e["label"]]
    print(f"\n##### {m}  (max_frames={D['max_frames']}, {len(pos)} success / {len(neg)} fail, task={D['task']!r})")
    fin_p = [e["success_prob"][-1] for e in pos]; fin_n = [e["success_prob"][-1] for e in neg]
    print(f"offline, final-frame success prob: AUROC {auroc(fin_p, fin_n):.3f}  "
          f"[95% CI {boot_ci(fin_p, fin_n)[0]:.2f}, {boot_ci(fin_p, fin_n)[1]:.2f}]  "
          f"mean S {np.mean(fin_p):.3f} / F {np.mean(fin_n):.3f}")
    for K in (1, 3, 5):
        sp_p = np.array([kscore(e["success_prob"], K) for e in pos]); sp_n = np.array([kscore(e["success_prob"], K) for e in neg])
        lo, hi = boot_ci(sp_p, sp_n)
        print(f"\n  online stop, K={K} consecutive steps: AUROC {auroc(sp_p, sp_n):.3f} [95% CI {lo:.2f}, {hi:.2f}]")
        print(f"    per-episode score  S: {np.round(np.sort(sp_p),3).tolist()}")
        print(f"                       F: {np.round(np.sort(sp_n),3).tolist()}")
        grid = np.unique(np.r_[np.round(np.linspace(0.05, 0.99, 95), 3), ref])
        rows = [(t, (sp_p >= t).mean(), (sp_n >= t).mean()) for t in grid]
        j = max(rows, key=lambda r: (r[1] - r[2], r[0]))               # Youden J, ties -> higher thr
        z = [r for r in rows if r[2] == 0]; z0 = min(z, key=lambda r: r[0]) if z else None
        print(f"    best Youden J: thr={j[0]:.2f}  TPR={j[1]:.2f} FPR={j[2]:.2f} (J={j[1]-j[2]:.2f})")
        if z0: print(f"    lowest thr with zero false fires: {z0[0]:.2f}  TPR={z0[1]:.2f}")
        r = [x for x in rows if abs(x[0] - ref) < 1e-9][0]
        print(f"    at reference thr {ref}: TPR={r[1]:.2f} FPR={r[2]:.2f}")
        if K == 1:
            print("    thr   TPR   FPR")
            for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
                print(f"    {t:.2f}  {(sp_p>=t).mean():.2f}  {(sp_n>=t).mean():.2f}")
            thr = j[0]
            fp = [(e["name"], first_fire(e["success_prob"], thr, K), e["T"]) for e in neg]
            tp = [(e["name"], first_fire(e["success_prob"], thr, K), e["T"]) for e in pos]
            f_ok = [f"{s}/{T}" for n, s, T in tp if s is not None]
            print(f"    at thr={thr:.2f}: success eps fire at step/len {f_ok}")
            print(f"                 false fires on {[n.split('_')[-1] for n, s, T in fp if s is not None]} "
                  f"at {[f'{s}/{T}' for n, s, T in fp if s is not None]}")


# ---- matched false-stop comparison across all models (5-step rule and single step) ----
print("\n##### successes caught at a matched false-stop budget")
for K in (1, 5):
    print(f"\n  stop after {K} step(s) in a row")
    for m in ("roboref", "robometer", "roboref_icl"):
        try: E = json.load(open(f"scores_{m}.json"))["episodes"]
        except FileNotFoundError: continue
        lab = np.array([e["label"] for e in E]); s_ = np.array([kscore(e["success_prob"], K) for e in E])
        row = []
        for budget in (0.0, 0.05, 0.10, 0.20, 0.30):
            best = (0.0, None)
            for t in np.unique(s_):
                fpr = (s_[lab == 0] >= t).mean(); tpr = (s_[lab == 1] >= t).mean()
                if fpr <= budget + 1e-9 and tpr > best[0]: best = (tpr, t)
            row.append(f"FPR<={budget:.2f}: {best[0]:.2f}" + (f"@{best[1]:.2f}" if best[1] is not None else ""))
        print(f"    {m:12s} AUROC {auroc(s_[lab==1], s_[lab==0]):.3f} | " + "  ".join(row))
