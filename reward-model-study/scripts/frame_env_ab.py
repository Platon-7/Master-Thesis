"""Controlled A/B in ONE env, ONE model load: score the SAME metaworld
trajectories via (a) npz frames and (b) decord-decoded MP4 frames. Isolates
frame-source from the env confound. If npz==mp4 here, the earlier 0.47-vs-0.98
gap was the ENV (qwen35 transformers), not the frame format."""
import json, os, sys
import numpy as np
sys.path.insert(0,"/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl")
sys.path.insert(0,"/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer")
from env.robometer_utils import get_robometer_4b
from datasets import load_from_disk
import decord

CACHE="/scratch-shared/pkarageorgis1/robometer_frames_hf_full_step2"
BAK="/projects/prjs1958/robometer_frames_hf_full/eval_metaworld_raw/robometer_frames_eval_metaworld.bak_pre_drop_metaworld_success_labels"

def within_task(score,lab,tasks):
    accs=[]
    for t in np.unique(tasks):
        s1=score[(tasks==t)&(lab==1)]; s0=score[(tasks==t)&(lab==0)]
        if len(s1) and len(s0):
            accs.append(((s1[:,None]>s0[None,:]).sum()+0.5*(s1[:,None]==s0[None,:]).sum())/(len(s1)*len(s0)))
    return float(np.mean(accs)) if accs else float('nan')

cdir=os.path.join(CACHE,"_projects_prjs1958_robometer_frames_hf_full_eval_metaworld_raw_robometer_frames_eval_metaworld")
ds=load_from_disk(os.path.join(cdir,"processed_dataset"))
raw=load_from_disk("/projects/prjs1958/robometer_frames_hf_full/eval_metaworld_raw/robometer_frames_eval_metaworld")
id2mp4={r["id"]: os.path.join(BAK, r["frames"].split("/",1)[1]) for r in raw}
scorer=get_robometer_4b(model_path=sys.argv[1])
print("max_frames",getattr(scorer,'max_frames','?'),flush=True)
N=int(sys.argv[2]) if len(sys.argv)>2 else 200
rows=ds.select(range(min(N,len(ds))))
pn,pm,lab,tk=[],[],[],[]
for i,r in enumerate(rows):
    with np.load(os.path.join(cdir,"frames",f"trajectory_{r['id']}.npz")) as d:
        npf=[np.asarray(x) for x in d["frames"]]
    vr=decord.VideoReader(id2mp4[r["id"]],num_threads=1)
    mpf=[np.asarray(x) for x in vr.get_batch(list(range(len(vr)))).asnumpy()]
    sn=scorer(npf,task=r["task"],detailed=True)["progress_reward"]
    sm=scorer(mpf,task=r["task"],detailed=True)["progress_reward"]
    pn.append(float(sn)); pm.append(float(sm)); lab.append(1 if r["quality_label"]=="successful" else 0); tk.append(r["task"])
    if (i+1)%80==0:
        print(f"  {i+1}/{len(rows)} npz_wt={within_task(np.array(pn),np.array(lab),np.array(tk)):.3f} mp4_wt={within_task(np.array(pm),np.array(lab),np.array(tk)):.3f}",flush=True)
pn,pm,lab,tk=map(np.array,(pn,pm,lab,tk))
print(f"\nDONE n={len(pn)}  (SAME env robometer_gpu_fa2, SAME model)")
print(f"  PROGRESS within-task  NPZ frames = {within_task(pn,lab,tk):.4f}")
print(f"  PROGRESS within-task  MP4 frames = {within_task(pm,lab,tk):.4f}")
print(f"  mean |npz_prog - mp4_prog| per traj = {np.abs(pn-pm).mean():.4f}")
