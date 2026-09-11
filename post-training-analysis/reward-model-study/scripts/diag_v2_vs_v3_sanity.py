"""Validate the diagnostic + isolate v2-vs-v3. Score, through ONE identical code
path (get_robometer_4b, correct task string, no ICL):
  - v3 CURATED CoffeePush eval frames (must reproduce offline AUC ~1.0)
  - v2 LIVE GT-policy CoffeePush frames (the collapse, ~0.5)
If v3->1.0 and v2->0.5, the diagnostic is sound and the v2 collapse is real.
"""
import os, sys, numpy as np
from pathlib import Path
sys.path.insert(0,"/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl")
sys.path.insert(0,"/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer")
from env.robometer_utils import get_robometer_4b
from datasets import load_from_disk
import imageio.v3 as iio
from PIL import Image
import scipy.stats as st
TASK="Push a mug under a coffee machine."
MODEL="/scratch-shared/pkarageorgis1/Robometer_FT_consolidated/run1_icl_ours_step3000"
def auc(l,s):
    l=np.asarray(l);s=np.asarray(s);P=(l==1).sum();N=(l==0).sum()
    if P==0 or N==0:return float('nan')
    r=st.rankdata(s);return float((r[l==1].sum()-P*(P+1)/2)/(P*N))
def main():
    sc=get_robometer_4b(model_path=MODEL)
    # v3 curated
    ds=load_from_disk("/projects/prjs1958/robometer_frames_hf_full/eval_metaworld_raw/robometer_frames_eval_metaworld")
    BAK="/projects/prjs1958/robometer_frames_hf_full/eval_metaworld_raw/robometer_frames_eval_metaworld.bak_pre_drop_metaworld_success_labels"
    v3l,v3s=[],[]
    for r in ds:
        if "coffee_push" not in r["id"].lower(): continue
        mp4=os.path.join(BAK,r["frames"].split("/",1)[1])
        if not os.path.isfile(mp4): continue
        fr=[Image.fromarray(np.asarray(x)) for x in iio.imiter(mp4,plugin="pyav")]
        v3s.append(float(sc(fr,task=TASK,icl_frames=None)["success_prob"]))
        v3l.append(1 if r["quality_label"]=="successful" else 0)
    print(f"v3 CURATED CoffeePush: n={len(v3l)} ({sum(v3l)}s/{len(v3l)-sum(v3l)}f)  AUC={auc(v3l,v3s):.3f}  (expect ~1.0)",flush=True)
    print(f"   succ_mean={np.mean([s for s,l in zip(v3s,v3l) if l==1]):.3f}  fail_mean={np.mean([s for s,l in zip(v3s,v3l) if l==0]):.3f}",flush=True)
    # v2 live
    v2l,v2s=[],[]
    for p in sorted(Path("/scratch-shared/pkarageorgis1/sp_dist_eval/GTpolicy_frames").glob("ep*.npz")):
        d=np.load(p); fr=[Image.fromarray(d["frames"][k].astype(np.uint8)) for k in range(d["frames"].shape[0])]
        v2s.append(float(sc(fr,task=TASK,icl_frames=None)["success_prob"])); v2l.append(int(d["label"]))
    print(f"v2 LIVE CoffeePush:    n={len(v2l)} ({sum(v2l)}s/{len(v2l)-sum(v2l)}f)  AUC={auc(v2l,v2s):.3f}  (was ~0.5)",flush=True)
    print(f"   succ_mean={np.mean([s for s,l in zip(v2s,v2l) if l==1]):.3f}  fail_mean={np.mean([s for s,l in zip(v2s,v2l) if l==0]):.3f}",flush=True)
if __name__=="__main__": main()
