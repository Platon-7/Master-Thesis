"""Aggregate the mega autonomous-RL sweep: group runs by config (collapsing
seeds), report per-seed final/peak GT-eval scores, flag live configs, and
compare success-head vs progress-head detection. Run anytime:
  python mega_summary.py   (reads the manifest + each run's train.log)
"""
import os, glob, re, collections

MAN = "/shared/home/PKA4388/vlm_ibrl_runs/mega_manifest.tsv"
RUNS = "/shared/home/PKA4388/vlm_ibrl_runs"
ALIVE, PARTIAL = 0.5, 0.15


def scores(jid):
    ds = glob.glob(f"{RUNS}/*_{jid}")
    if not ds:
        return None, None, 0
    tl = os.path.join(ds[0], "train.log")
    if not os.path.exists(tl):
        return None, None, 0
    vals = []
    for line in open(tl, errors="ignore"):
        m = re.match(r"^(\d+): score/score\s*:\s*([0-9.]+)", line)
        if m:
            vals.append((int(m.group(1)), float(m.group(2))))
    if not vals:
        return None, None, 0
    peak = max(v for _, v in vals)
    fin = vals[-1][1]
    return peak, fin, vals[-1][0]


def main():
    rows = [l.rstrip("\n").split("\t") for l in open(MAN) if l.strip() and not l.startswith("#")]
    groups = collections.defaultdict(dict)  # config-key -> seed -> (peak,fin,step)
    for r in rows:
        name, jid, vlm, head, thr, beta, rt, consec, seed = (r + [""] * 9)[:9]
        key = (vlm, head, thr, beta, rt, consec)
        groups[key][seed] = scores(jid)

    summary = []
    for key, seeds in groups.items():
        vlm, head, thr, beta, rt, consec = key
        fins = [v[1] for v in seeds.values() if v and v[1] is not None]
        peaks = [v[0] for v in seeds.values() if v and v[0] is not None]
        if not fins:
            continue
        n_alive = sum(1 for f in fins if f >= ALIVE)
        n_part = sum(1 for f in fins if PARTIAL <= f < ALIVE)
        summary.append((max(fins + [0]), n_alive, n_part, key, seeds, max(peaks + [0])))

    summary.sort(reverse=True)
    print(f"{'='*100}\nMEGA SWEEP — grouped by config (collapsing seeds). ALIVE final>= {ALIVE}, partial>= {PARTIAL}")
    print(f"{'vlm':12} {'detect':8} {'thr':5} {'beta':4} {'rt':3} {'cons':4} | {'per-seed finals':22} | maxfin maxpeak nALIVE")
    print("-" * 100)
    for maxf, na, npart, key, seeds, maxp in summary:
        vlm, head, thr, beta, rt, consec = key
        perseed = " ".join(f"{s}:{(seeds[s][1] if seeds[s] and seeds[s][1] is not None else '-')}" for s in sorted(seeds))
        flag = " <== ALIVE" if na >= 1 else (" (partial)" if npart else "")
        print(f"{vlm[:12]:12} {head:8} {thr:5} {beta:4} {rt:3} {consec:4} | {perseed:22} | {maxf:.2f}  {maxp:.2f}   {na}{flag}")

    # head comparison
    print("\n" + "=" * 60 + "\nDETECTION-HEAD comparison (configs with >=1 alive seed):")
    for vlm in ("robometer_ft", "robometer_4b"):
        for head in ("success", "progress"):
            live = [s for s in summary if s[3][0] == vlm and s[3][1] == head and s[1] >= 1]
            done = sum(1 for s in summary if s[3][0] == vlm and s[3][1] == head)
            print(f"  {vlm[:12]:12} detect={head:8}: {len(live)} live configs (of {done} reported)")


if __name__ == "__main__":
    main()
