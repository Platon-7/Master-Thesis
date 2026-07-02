"""Confirm the demo-anchor bug: score the SAME demo frames as PIL vs numpy.
If PIL gives a sane (high) success_prob on a real-success demo and numpy gives ~0,
the bug was the frame type (numpy), not OOD."""
import os, sys, glob
import numpy as np
from PIL import Image
_REPO = os.environ.get("MT_REPO", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for sub in ("vlm_ibrl_v3", "Robometer"):
    sys.path.insert(0, os.path.join(_REPO, sub))
from env.robometer_utils import get_robometer_4b

CKPT = os.environ["ROBOMETER_FT_PATH"]
DEMO = os.environ["DEMO_DIR"]
TASK = os.environ.get("TASK_DESC", "Push a mug under a coffee machine.")

sc = get_robometer_4b(model_path=CKPT)
print(f"[probe] ckpt={CKPT}\n[probe] demo={DEMO}", flush=True)

# demo indices present
import re
files = [os.path.basename(p) for p in glob.glob(os.path.join(DEMO, "*.png"))]
idxs = sorted({f.split("_")[0] for f in files if f.split("_")[0].isdigit()}, key=int)
print(f"[probe] demos present: {idxs[:5]}... ({len(idxs)} total)", flush=True)

for di in idxs[:3]:
    frs = sorted(glob.glob(os.path.join(DEMO, f"{di}_*.png")))
    picks = np.linspace(0, len(frs) - 1, 16).round().astype(int)
    pil_win = [Image.open(frs[i]).convert("RGB") for i in picks]
    np_win = [np.asarray(im, dtype=np.uint8) for im in pil_win]
    sp_pil = float(sc(pil_win, task=TASK, icl_frames=None)["success_prob"])
    sp_np = float(sc(np_win, task=TASK, icl_frames=None)["success_prob"])
    print(f"  demo {di}: PIL success_prob={sp_pil:.4f}   numpy success_prob={sp_np:.4f}   (#frames={len(frs)})", flush=True)
print("\n[probe] VERDICT:", flush=True)
print("  if PIL >> numpy -> the demo-anchor bug was frame type (FIXED by passing PIL).", flush=True)
