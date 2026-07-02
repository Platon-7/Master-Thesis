"""score/score curves for the deck, from train.log (wandb not required).
All gate curves below are ICL-OFF (no demonstration given at inference) — model A is
the ICL-TRAINED model, but in job 647 it was run WITHOUT ICL. The A·ICL-on curve is
added automatically (dashed) once the scheduled ICL-on runs (iclon_*) produce evals."""
import glob, re, statistics
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE="/shared/home/PKA4388/vlm_ibrl_runs"
def traj(d):
    pts=[]
    for line in open(d+"/train.log", errors="ignore"):
        m=re.match(r"^(\d+): score/score\s*:\s*([0-9.]+)", line)
        if m: pts.append((int(m.group(1)), float(m.group(2))))
    return pts
def mean_curve(dirs):
    curves=[traj(d) for d in dirs]; curves=[c for c in curves if c]
    if not curves: return None,None
    L=min(len(c) for c in curves)
    xs=[curves[0][i][0]/1000 for i in range(L)]
    M=[statistics.mean(c[i][1] for c in curves) for i in range(L)]
    lo=[min(c[i][1] for c in curves) for i in range(L)]
    hi=[max(c[i][1] for c in curves) for i in range(L)]
    return xs,(M,lo,hi)

models={"A · ICL-trained (run ICL-off)":("run1_icl_ours_step4000","#A66000"),
        "B · asymmetric loss":("run2_noicl_ours_step4000","#1E6B2E"),
        "C · standard loss":("run3_noicl_standard_step5000","#B01E2E")}
fig,axes=plt.subplots(1,2,figsize=(7.6,3.6),sharey=True)
for ax,task,tlabel in [(axes[0],"coffeepush","Coffee-Push"),(axes[1],"boxclose","Box-Close")]:
    for ml,(ck,col) in models.items():
        xs,band=mean_curve(sorted(glob.glob(f"{BASE}/gatestab_{ck}_{task}_*_647_*")))
        if not xs: continue
        M,lo,hi=band
        ax.plot(xs,M,color=col,lw=2.0,label=ml,marker="o",ms=2.5)
        ax.fill_between(xs,lo,hi,color=col,alpha=0.12)
    # ICL-ON (model A), gated — drawn ONLY once there is real data: >=3 seeds, each
    # with >=4 evals (excludes short smokes). Otherwise stays pending.
    icl_dirs=[d for d in sorted(glob.glob(f"{BASE}/iclon_run1_icl_ours_step4000_{task}_gate_*"))
              if len(traj(d))>=4]
    if len(icl_dirs)>=3:
        xs,band=mean_curve(icl_dirs)
        ax.plot(xs,band[0],color="#A66000",lw=2.0,ls="--",marker="s",ms=2.5,label="A · ICL-on")
        ax.fill_between(xs,band[1],band[2],color="#A66000",alpha=0.08)
    ax.set_title(tlabel,fontsize=11); ax.set_xlabel("training steps (k)"); ax.set_ylim(0,1.0); ax.grid(alpha=0.25)
axes[0].set_ylabel("true success rate (20-ep eval)")
axes[0].legend(fontsize=7.5,loc="upper left")
has_iclon=any(len([d for d in glob.glob(f"{BASE}/iclon_run1_icl_ours_step4000_{t}_gate_*") if len(traj(d))>=4])>=3
              for t in ("coffeepush","boxclose"))
sub = "solid = ICL-off · dashed = ICL-on (model A)" if has_iclon else "ICL-off (gate)"
fig.suptitle(f"Success rate vs steps, mean over seeds (band = min–max) · {sub}",fontsize=10.5)
fig.tight_layout()
out="/shared/home/PKA4388/Master-Thesis/reward-model-study/deck/fig_gate_curves.png"
fig.savefig(out,dpi=150,bbox_inches="tight"); print("saved",out,"| has ICL-on curve:",has_iclon)
