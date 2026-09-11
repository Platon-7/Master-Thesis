"""Success-head metrics from repro_canonical success_prob dumps: AUPRC + TPR@FPR=5% + AUC, per source + avg."""
import json, os, sys, glob
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
def tpr_at_fpr(y, s, fpr_target=0.05):
    neg=np.sort(s[y==0])[::-1]
    if len(neg)==0: return float('nan')
    thr=neg[min(int(np.ceil(fpr_target*len(neg)))-1, len(neg)-1)]
    pos=s[y==1]
    return float((pos>=thr).mean()) if len(pos) else float('nan')
def metrics(f):
    r=[json.loads(l) for l in open(f) if l.strip()]
    y=np.array([x['label'] for x in r]); s=np.array([x['success_prob'] for x in r])
    if len(np.unique(y))<2 or np.isnan(s).any(): return None
    return dict(auprc=average_precision_score(y,s), auc=roc_auc_score(y,s),
               tpr5=tpr_at_fpr(y,s,0.05), n=len(r), pos=int(y.sum()))
def cell(model, srcmap):
    rows={}
    for s,f in srcmap.items():
        if os.path.isfile(f):
            m=metrics(f)
            if m: rows[s]=m
    return rows
# pattern: model -> {src: dumpfile}
D="reward-model-study/results/repro_canonical"
# valid dumps: baseline + run4/5/6 (old names); run1/2/3 -> *_fa2__ (new, correct env)
MODELS={"baseline":"baseline","run1_fa2":"run1_fa2","run2_fa2":"run2_fa2","run3_fa2":"run3_fa2","run4_asym":"run4_asym","run5_asym":"run5_asym","run6_std":"run6_std"}
print(f"{'model':12} {'src-avg AUPRC':>13} {'TPR@5%FPR':>11} {'AUC':>7}   per-source AUPRC")
print("-"*80)
for disp,stem in MODELS.items():
    rows=cell(disp,{s:f"{D}/{stem}__{s}.jsonl" for s in ["droid","metaworld","robometer"]})
    if not rows: print(f"{disp:12} (pending — correct-env dump not present)"); continue
    a=np.mean([rows[s]['auprc'] for s in rows]); t=np.mean([rows[s]['tpr5'] for s in rows]); u=np.mean([rows[s]['auc'] for s in rows])
    per=" ".join(f"{s[:3]}={rows[s]['auprc']:.3f}" for s in rows)
    print(f"{disp:12} {a:>13.3f} {t:>11.3f} {u:>7.3f}   {per}")
