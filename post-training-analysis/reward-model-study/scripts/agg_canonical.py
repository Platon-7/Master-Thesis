"""Aggregate repro_canonical/*.jsonl into the fair per-source table."""
import json, glob, os
import numpy as np
def auc(p,n):
    if not len(p) or not len(n): return float('nan')
    a=np.concatenate([p,n]);o=np.argsort(a,kind='mergesort');r=np.empty(len(a));r[o]=np.arange(1,len(a)+1)
    return float((r[:len(p)].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(n)))
D="reward-model-study/results/repro_canonical"
rows={}
for f in sorted(glob.glob(f"{D}/*__*.jsonl")):
    base=os.path.basename(f)[:-6]
    m,s=base.split("__")
    recs=[json.loads(l) for l in open(f) if l.strip()]
    if not recs: continue
    sp=np.array([r['success_prob'] for r in recs]); lab=np.array([r['label'] for r in recs])
    tk=np.array([r.get('task','?') for r in recs])
    g=auc(sp[lab==1],sp[lab==0]); sep=sp[lab==1].mean()-sp[lab==0].mean()
    accs=[]
    for t in np.unique(tk):
        s1=sp[(tk==t)&(lab==1)];s0=sp[(tk==t)&(lab==0)]
        if len(s1) and len(s0):
            accs.append(((s1[:,None]>s0[None,:]).sum()+0.5*(s1[:,None]==s0[None,:]).sum())/(len(s1)*len(s0)))
    wt=np.mean(accs) if accs else float('nan')
    rows.setdefault(m,{})[s]={'auc':g,'wt':wt,'sep':sep,'n':len(recs)}
order=["baseline","run2_asym","run3_std","run4_asym","run5_asym","run6_std"]
srcs=["droid","metaworld","robometer"]
print(f"\n{'model':12} | "+" | ".join(f"{s:^26}" for s in srcs))
print(f"{'':12} | "+" | ".join(f"{'AUC  wtRank  sep   n':^26}" for s in srcs))
print("-"*100)
for m in order:
    if m not in rows: continue
    cells=[]
    for s in srcs:
        if s in rows[m]:
            r=rows[m][s]; cells.append(f"{r['auc']:.3f} {r['wt']:.3f} {r['sep']:+.3f} {r['n']:4d}")
        else: cells.append(f"{'--pending--':^26}")
    print(f"{m:12} | "+" | ".join(f"{c:^26}" for c in cells))
print("\nAUC=global pooled (deployment-realistic) | wtRank=within-task (wandb-style) | sep=mean succ-fail")
