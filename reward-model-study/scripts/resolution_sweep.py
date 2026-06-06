"""Probe the user's format hypothesis directly: take the TRAINING npz frames
(240x240, score ~0.91) and resize to other resolutions BEFORE the scorer.
Everything else identical. If the model is format/resolution-sensitive, the
within-task ranking drops as we move off 240 (esp. to IBRL's 224)."""
import os, sys
import numpy as np
sys.path.insert(0,"/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl")
sys.path.insert(0,"/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer")
from env.robometer_utils import get_robometer_4b
from datasets import load_from_disk
from PIL import Image

CACHE="/scratch-shared/pkarageorgis1/robometer_frames_hf_full_step2"
RES=[240,224,200,168,128,84]

def resize(frames, r):
    if r==240: return frames
    return [np.asarray(Image.fromarray(f).resize((r,r), Image.BILINEAR)) for f in frames]

def within_task(score,lab,tasks):
    accs=[]
    for t in np.unique(tasks):
        s1=score[(tasks==t)&(lab==1)]; s0=score[(tasks==t)&(lab==0)]
        if len(s1) and len(s0):
            accs.append(((s1[:,None]>s0[None,:]).sum()+0.5*(s1[:,None]==s0[None,:]).sum())/(len(s1)*len(s0)))
    return float(np.mean(accs)) if accs else float('nan')

cdir=os.path.join(CACHE,"_projects_prjs1958_robometer_frames_hf_full_eval_metaworld_raw_robometer_frames_eval_metaworld")
ds=load_from_disk(os.path.join(cdir,"processed_dataset"))
N=int(sys.argv[2]) if len(sys.argv)>2 else 200
rows=ds.select(range(min(N,len(ds))))
scorer=get_robometer_4b(model_path=sys.argv[1])
print("max_frames",getattr(scorer,'max_frames','?'),"N",len(rows),flush=True)
# preload npz frames
data=[]
for r in rows:
    with np.load(os.path.join(cdir,"frames",f"trajectory_{r['id']}.npz")) as d:
        data.append(([np.asarray(x) for x in d["frames"]], r["task"], 1 if r["quality_label"]=="successful" else 0))
for res in RES:
    pg,lab,tk=[],[],[]
    for frames,task,y in data:
        fr=resize(frames,res)
        pg.append(float(scorer(fr,task=task,detailed=True)["progress_reward"])); lab.append(y); tk.append(task)
    wt=within_task(np.array(pg),np.array(lab),np.array(tk))
    print(f"  RES {res:3d}x{res:<3d}  progress within-task = {wt:.4f}",flush=True)
