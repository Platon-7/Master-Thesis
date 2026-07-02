"""Probe whether Qwen3.5-FT produces weird outputs on SMALL frame buffers — the
regime IBRL hits at the start of every episode (buffer grows 2,3,4,... frames)
but offline scoring never exercised (MINWIN=6). Uses STATIC demo frames so it
runs in the scoring-only vlm_ibrl_qwen35 env (no gymnasium/metaworld needed).
Feeds growing prefixes of a real CoffeePush demo, same scorer interface as IBRL.
"""
import os, sys, glob
import numpy as np

_REPO = os.environ.get("MT_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for sub in ("vlm_ibrl_v3", "Robometer"):
    sys.path.insert(0, os.path.join(_REPO, sub))

from env.robometer_utils import get_robometer_4b
from PIL import Image

CKPT = os.environ["QWEN35_FT_PATH"]
TASK = "Push a mug under a coffee machine."
FRAMES_DIR = ("/shared/home/PKA4388/Master-Thesis/vlm_ibrl/release/data/metaworld/"
              "CoffeePush_frame_stack_1_224x224_end_on_success/demonstrations/mw-coffee-push/frames")

print(f"[probe] loading {CKPT}", flush=True)
sc = get_robometer_4b(model_path=CKPT)
print(f"[probe] max_frames={sc.max_frames}", flush=True)

files = sorted(glob.glob(os.path.join(FRAMES_DIR, "0_*.png")))
frames = [Image.open(f).convert("RGB") for f in files]
print(f"[probe] loaded {len(frames)} static demo frames\n", flush=True)

print(f"{'n_frames':>9} | {'success_prob':>13} | {'progress':>9} | flag")
for n in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32]:
    if n > len(frames): break
    try:
        out = sc(frames[:n], task=TASK, icl_frames=None)
        sp, pg = float(out["success_prob"]), float(out["progress_reward"])
        flag = ""
        if np.isnan(sp) or np.isnan(pg): flag = "NaN!!"
        elif not (0.0 <= sp <= 1.0): flag = "OUT-OF-RANGE!!"
        elif sp in (0.0, 1.0): flag = "saturated"
        print(f"{n:>9} | {sp:>13.5f} | {pg:>9.5f} | {flag}", flush=True)
    except Exception as e:
        print(f"{n:>9} | EXCEPTION: {type(e).__name__}: {e}", flush=True)

print("\n[probe] reference: a real success should read HIGH success_prob; early "
      "frames (incomplete task) should read LOW but VALID (0-1, not NaN/constant). "
      "If small n is NaN/out-of-range/constant while large n is sane -> tiny-buffer bug.", flush=True)
