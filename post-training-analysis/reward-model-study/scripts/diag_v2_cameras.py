"""Check 2: disentangle CAMERA/zoom from v2 RENDER ENGINE.
Roll out v2 CoffeePush (scripted policy -> successes; noisy -> failures), render
each state from multiple cameras at 224, score each camera with Robometer-FT.
  - if a non-zoomed camera (corner3/corner/topview/...) recovers AUC ~0.9 -> CAMERA/zoom (fixable in v2)
  - if ALL v2 cameras stay ~0.5 -> the v2 RENDER ENGINE (only then is v3 on the table)
"""
import os, sys, numpy as np
sys.path.insert(0,"/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl")
sys.path.insert(0,"/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer")
from env.metaworld_wrapper import MetaWorldEnv
from env.robometer_utils import get_robometer_4b
from PIL import Image
import scipy.stats as st
TASK="Push a mug under a coffee machine."
MODEL="/scratch-shared/pkarageorgis1/Robometer_FT_consolidated/run1_icl_ours_step3000"
CAMERAS=["corner2","corner3","corner","topview","behindGripper","gripperPOV"]
N_SUCC=25; N_FAIL=25; MAXT=110; SUB=16
def auc(l,s):
    l=np.asarray(l);s=np.asarray(s);P=(l==1).sum();N=(l==0).sum()
    if P==0 or N==0:return float('nan')
    r=st.rankdata(s);return float((r[l==1].sum()-P*(P+1)/2)/(P*N))
def rollout(env, noisy):
    # returns dict cam->list-of-16-frames, and success bool
    buf={c:[] for c in CAMERAS}; env.reset(); succ=0
    for t in range(MAXT):
        a=env.get_heuristic_action()
        if noisy: a=(a+np.random.uniform(-1,1,size=a.shape)*0.8).clip(-1,1)
        obs,r,done,info=env.step(a)
        if int(info.get("success",0))==1: succ=1
        for c in CAMERAS:
            buf[c].append(env.render(camera_name=c,width=224,height=224))
        if done: break
    # subsample to 16 per camera
    idx=np.linspace(0,len(buf[CAMERAS[0]])-1,SUB).round().astype(int)
    return {c:[Image.fromarray(buf[c][i]) for i in idx] for c in CAMERAS}, succ
def main():
    np.random.seed(0)
    env=MetaWorldEnv("CoffeePush",camera_name="corner2",width=224,height=224)
    sc=get_robometer_4b(model_path=MODEL)
    data={c:[] for c in CAMERAS}; labels=[]
    ns=nf=0
    tries=0
    while (ns<N_SUCC or nf<N_FAIL) and tries<200:
        tries+=1; noisy = nf<N_FAIL and (ns>=N_SUCC or np.random.rand()<0.5)
        frames,succ=rollout(env,noisy)
        if succ and ns>=N_SUCC: continue
        if (not succ) and nf>=N_FAIL: continue
        labels.append(succ); ns+=succ; nf+=(1-succ)
        for c in CAMERAS: data[c].append(sc(frames[c],task=TASK,icl_frames=None)["success_prob"])
        if len(labels)%10==0: print(f"  collected {len(labels)} ({ns}s/{nf}f)…",flush=True)
    L=np.array(labels)
    print(f"\n==== v2 CoffeePush, {len(L)} eps ({int(L.sum())}s/{int((L==0).sum())}f) — AUC + success_prob means per camera ====")
    for c in CAMERAS:
        v=np.array(data[c]); a=auc(L,v)
        zoom=" (ZOOMED, what IBRL feeds)" if c=="corner2" else ""
        print(f"  {c:13s}: AUC={a:.3f}  succ_mean={v[L==1].mean():.3f}  fail_mean={v[L==0].mean():.3f}{zoom}")
    print("\nref: v3 curated AUC=0.97 (succ 0.77); v2 zoomed-corner2 live AUC=0.53 (succ 0.25)")
if __name__=="__main__": main()
