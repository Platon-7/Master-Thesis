"""Isolate WHY the on-policy CoffeePush AUC collapses to ~0.55: is it the frame
content, or a pipeline-adaptation bug (wrong task string / ICL mismatch)?

Re-score the SAME saved GT-policy rollout frames under controlled conditions:
  A) correct task string, NO ICL   <- matches the offline eval that gave AUC 1.0
  B) wrong task string,   NO ICL   <- reproduces my contaminated re-score (~0.55)
  C) correct task string, single-demo ICL <- reproduces the GT-policy CSV (~0.55)

If A recovers to ~0.9, the collapse was a pipeline artifact, not content shift.
"""
import json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl")
sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer")
from env.robometer_utils import get_robometer_4b
from PIL import Image
import imageio.v3 as iio

CORRECT = "Push a mug under a coffee machine."
WRONG   = "push the coffee cup"
FRAMES_DIR = "/scratch-shared/pkarageorgis1/sp_dist_eval/GTpolicy_frames"
MODEL = "/scratch-shared/pkarageorgis1/Robometer_FT_consolidated/run1_icl_ours_step3000"
ICL_DEMO = "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl/release/data/metaworld/CoffeePush_frame_stack_1_224x224_end_on_success/demonstrations/mw-coffee-push/frames"

def auc(l, s):
    l = np.asarray(l); s = np.asarray(s); P = (l == 1).sum(); N = (l == 0).sum()
    if P == 0 or N == 0: return float("nan")
    import scipy.stats as st
    r = st.rankdata(s); return float((r[l == 1].sum() - P * (P + 1) / 2) / (P * N))

def load_icl_frames(n=16):
    files = sorted(Path(ICL_DEMO).glob("*"))[:200]
    if not files: return None
    imgs = [np.asarray(Image.open(f).convert("RGB")) for f in files]
    idx = np.linspace(0, len(imgs) - 1, n).round().astype(int)
    return [Image.fromarray(imgs[i]) for i in idx]

def main():
    eps = sorted(Path(FRAMES_DIR).glob("ep*.npz"))
    print(f"{len(eps)} episodes; loading scorer…", flush=True)
    scorer = get_robometer_4b(model_path=MODEL)
    icl = load_icl_frames()
    print(f"icl demo frames loaded: {None if icl is None else len(icl)}", flush=True)
    recs = []
    for i, p in enumerate(eps):
        d = np.load(p); fr = [Image.fromarray(d["frames"][k].astype(np.uint8)) for k in range(d["frames"].shape[0])]
        lab = int(d["label"])
        A = float(scorer(fr, task=CORRECT, icl_frames=None)["success_prob"])
        B = float(scorer(fr, task=WRONG,   icl_frames=None)["success_prob"])
        C = float(scorer(fr, task=CORRECT, icl_frames=icl)["success_prob"])
        recs.append((lab, A, B, C))
        if (i + 1) % 40 == 0:
            L = [r[0] for r in recs]
            print(f"  [{i+1}/{len(eps)}] A={auc(L,[r[1] for r in recs]):.3f} "
                  f"B={auc(L,[r[2] for r in recs]):.3f} C={auc(L,[r[3] for r in recs]):.3f}", flush=True)
    L = [r[0] for r in recs]
    print("\n==== FINAL (n=%d, %d succ / %d fail) ====" % (len(recs), sum(L), len(L)-sum(L)))
    print(f"  A) correct string, NO ICL   : AUC = {auc(L,[r[1] for r in recs]):.3f}   <- offline match (was 1.0 offline)")
    print(f"  B) WRONG string,   NO ICL   : AUC = {auc(L,[r[2] for r in recs]):.3f}   <- my contaminated re-score")
    print(f"  C) correct string, +ICL     : AUC = {auc(L,[r[3] for r in recs]):.3f}   <- the GT-policy CSV setting")

if __name__ == "__main__":
    main()
