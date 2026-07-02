"""Best-case stabilization figure: gate g0.8 + offline-calibrated threshold, run2, 6 seeds.
Row 1: GT success learning curve (eval), mean over seeds + /-1 std band + thin per-seed lines.
Row 2: faithfulness check on TRAINING rollouts — GT success (train_score) vs VLM reward
       collected (train_return). Rise-together = faithful proxy; divergence = reward hacking.
All from train.log (no GPU)."""
import glob, re, statistics
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD = "/shared/home/PKA4388/vlm_ibrl_runs"
CKPT = "run1_icl_ours_step4000"
NAVY="#1F3864"; GREEN="#1E6B2E"; AMBER="#A66000"; GREY="#888888"

def series(d, key):
    """Handles both log formats: '<step>: key : val'  and  '<step>: key [N]: avg: val, ...'"""
    pts=[]
    for line in open(d+"/train.log", errors="ignore"):
        m=re.match(rf"^(\d+): {re.escape(key)}\b\s*(?:\[.*?\])?\s*:\s*(?:avg:\s*)?([0-9.]+)", line)
        if m: pts.append((int(m.group(1)), float(m.group(2))))
    return pts

def collect(task, key):
    """step -> list of per-seed values"""
    agg={}
    for d in sorted(glob.glob(f"{RD}/gatestab_{CKPT}_{task}_*_647_*")):
        for s,v in series(d, key):
            agg.setdefault(s, []).append(v)
    steps=sorted(agg)
    return steps, agg

def mean_band(task, key, min_seeds=2):
    steps, agg = collect(task, key)
    xs=[s for s in steps if len(agg[s])>=min_seeds]
    mean=[statistics.mean(agg[s]) for s in xs]
    std=[statistics.pstdev(agg[s]) if len(agg[s])>1 else 0 for s in xs]
    return np.array(xs)/1000, np.array(mean), np.array(std), agg

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
for col,(task,tlabel) in enumerate([("coffeepush","Coffee-Push"),("boxclose","Box-Close")]):
    # ---- Row 1: GT success learning curve (eval) ----
    ax=axes[0,col]
    xs,mean,std,agg = mean_band(task,"score/score")
    # per-seed thin lines
    seeds={}
    for d in sorted(glob.glob(f"{RD}/gatestab_{CKPT}_{task}_*_647_*")):
        pts=series(d,"score/score")
        if pts: ax.plot([s/1000 for s,_ in pts],[v for _,v in pts],color=GREEN,lw=0.8,alpha=0.3)
    ax.plot(xs,mean,color=GREEN,lw=2.6,label="GT success (mean over seeds)")
    ax.fill_between(xs,np.clip(mean-std,0,1),np.clip(mean+std,0,1),color=GREEN,alpha=0.18,label="±1 std across seeds")
    ax.set_title(f"{tlabel} — success rate",fontsize=12,color=NAVY)
    ax.set_xlabel("training steps (k)"); ax.set_ylim(0,1); ax.grid(alpha=0.25)
    if col==0: ax.set_ylabel("GT success rate (20-ep eval)")
    ax.legend(fontsize=8,loc="upper left")
    # ---- Row 2: faithfulness — train GT vs VLM reward collected ----
    ax2=axes[1,col]
    xg,mg,sg,_=mean_band(task,"score/train_score")
    xr,mr,sr,_=mean_band(task,"score/train_return")
    ax2.plot(xg,mg,color=NAVY,lw=2.4,marker="o",ms=3,label="GT success (training rollouts)")
    ax2.fill_between(xg,np.clip(mg-sg,0,1),np.clip(mg+sg,0,1),color=NAVY,alpha=0.15)
    ax2.plot(xr,mr,color=AMBER,lw=2.4,marker="s",ms=3,label="VLM reward collected")
    ax2.fill_between(xr,np.clip(mr-sr,0,1),np.clip(mr+sr,0,1),color=AMBER,alpha=0.15)
    ax2.set_title(f"{tlabel} — reward vs true success",fontsize=12,color=NAVY)
    ax2.set_xlabel("training steps (k)"); ax2.set_ylim(0,1); ax2.grid(alpha=0.25)
    if col==0: ax2.set_ylabel("per-episode (mean of seeds)")
    ax2.legend(fontsize=8,loc="upper left")
fig.suptitle("≥1 demo: gate + run1 (model A)  ·  6 seeds", fontsize=12.5)
fig.tight_layout()
out="/shared/home/PKA4388/Master-Thesis/reward-model-study/deck/fig_versionA_run1gate.png"
fig.savefig(out,dpi=140,bbox_inches="tight"); print("saved",out)
# also print the numbers behind it
for task in ("coffeepush","boxclose"):
    xs,m,s,_=mean_band(task,"score/score")
    print(f"\n{task} eval-GT mean±std @steps:")
    for a,b,c in zip(xs,m,s): print(f"  {int(a)}k: {b:.2f} ± {c:.2f}")
