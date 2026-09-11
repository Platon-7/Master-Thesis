"""Progress-head table: TPR@FPR=5% + class separation (raw gap, pooled variance, d'=gap/pooled-sigma).
From correct-env repro_canonical dumps (progress_reward as the success-vs-failure score).
ECE intentionally omitted: frame_labels=None across all sources -> no real per-frame GT progress,
so any dense-ECE would be calibration to a SYNTHETIC ramp = VOC-in-disguise (downstream-blind)."""
import json, os
import numpy as np
def tpr_at_fpr(y,s,fpr=0.05):
    neg=np.sort(s[y==0])[::-1]
    if not len(neg): return np.nan
    thr=neg[min(int(np.ceil(fpr*len(neg)))-1,len(neg)-1)]
    pos=s[y==1]; return float((pos>=thr).mean()) if len(pos) else np.nan
def cell(f):
    if not os.path.isfile(f): return None
    r=[json.loads(l) for l in open(f) if l.strip()]
    if not r or 'progress_reward' not in r[0]: return None
    y=np.array([x['label'] for x in r]); p=np.array([x['progress_reward'] for x in r])
    if len(np.unique(y))<2 or np.isnan(p).any(): return None
    ps,pf=p[y==1],p[y==0]
    gap=ps.mean()-pf.mean(); pv=(ps.var()+pf.var())/2; d=gap/np.sqrt(pv) if pv>0 else np.nan
    return dict(gap=gap,var=pv,dprime=d,tpr5=tpr_at_fpr(y,p,0.05))
D="reward-model-study/results/repro_canonical"
MODELS=[("Robometer-4B (baseline)","baseline"),("4B asym+ICL s4000","run1_fa2"),
        ("4B asym","run2_fa2"),("4B paper-std","run3_fa2"),
        ("Qwen3.5 asym+ICL","run4_asym"),("Qwen3.5 asym","run5_asym"),("Qwen3.5 paper-std","run6_std")]
SRCS=["droid","metaworld","robometer"]
print(f"{'model':24} | {'TPR@5%FPR':>9} | {'raw gap':>8} {'pooled var':>10} {'d-prime':>8}")
print("-"*72)
for label,stem in MODELS:
    cs=[cell(f"{D}/{stem}__{s}.jsonl") for s in SRCS]; cs=[c for c in cs if c]
    if len(cs)<3:
        print(f"{label:24} | (pending {len(cs)}/3 sources)"); continue
    tpr=np.mean([c['tpr5'] for c in cs]); gap=np.mean([c['gap'] for c in cs])
    var=np.mean([c['var'] for c in cs]); d=np.mean([c['dprime'] for c in cs])
    print(f"{label:24} | {tpr:>9.3f} | {gap:>+8.3f} {var:>10.4f} {d:>8.3f}")
print("\nTPR@FPR + d' both use progress_reward as the success-vs-failure score (single-clip).")
print("raw gap & pooled var are d''s components (gap scale-dependent; d' = gap/sqrt(var) is cross-model-fair).")
