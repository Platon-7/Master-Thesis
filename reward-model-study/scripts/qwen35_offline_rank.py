"""Offline ranking sanity check for Qwen3.5-FT, UNRELATED to IBRL: score each
task's success clip against every task prompt (4x4). Uses the SAME scorer as
eval_dump (get_robometer_4b). A working model -> diagonal (matched task) HIGH,
off-diagonal (wrong task) LOW. Flat everywhere -> the success head is broken in
this setup generally (not IBRL-specific). Uses only local demo PNGs.
"""
import os, sys, glob
import numpy as np

_REPO = os.environ.get("MT_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for sub in ("vlm_ibrl_v3", "Robometer"):
    sys.path.insert(0, os.path.join(_REPO, sub))
from env.robometer_utils import get_robometer_4b
from env.vlm_prompts import METAWORLD_TASK_DESCRIPTIONS as TASKSTR
from PIL import Image

CKPT = os.environ["QWEN35_FT_PATH"]
DEMO_BASE = ("/shared/home/PKA4388/Master-Thesis/vlm_ibrl/release/data/metaworld/"
             "{T}_frame_stack_1_224x224_end_on_success/demonstrations/{mw}/frames")
TASKS = {  # task -> (dataset-dir-name, mw-subdir, success-frame-index t_end of demo 0)
    "CoffeePush": ("CoffeePush", "mw-coffee-push", 28),
    "Assembly":   ("Assembly",   "mw-assembly",    49),
    "BoxClose":   ("BoxClose",   "mw-box-close",   68),
    "StickPull":  ("StickPull",  "mw-stick-pull",  52),
}

def success_clip(task):
    dname, mw, tend = TASKS[task]
    d = DEMO_BASE.format(T=dname, mw=mw)
    files = sorted(glob.glob(os.path.join(d, "0_*.png")))[:tend + 1]  # up to success frame
    return [Image.open(f).convert("RGB") for f in files]

print(f"[rank] loading {CKPT}", flush=True)
sc = get_robometer_4b(model_path=CKPT)
print(f"[rank] max_frames={sc.max_frames}\n", flush=True)

clips = {t: success_clip(t) for t in TASKS}
for t, c in clips.items(): print(f"  {t}: success clip = {len(c)} frames", flush=True)

names = list(TASKS)
print("\n=== success_prob matrix: rows = clip (true success), cols = prompt fed ===")
print(f"{'clip \\ prompt':<14}" + "".join(f"{n:>12}" for n in names))
diag, offdiag = [], []
for ct in names:
    row = []
    for pt in names:
        out = sc(clips[ct], task=TASKSTR[pt], icl_frames=None)
        sp = float(out["success_prob"]); row.append(sp)
        (diag if ct == pt else offdiag).append(sp)
    star = "  <- matched-task should be highest in this row"
    print(f"{ct:<14}" + "".join(f"{v:>12.4f}" for v in row) + (star if False else ""))
print(f"\n[rank] matched (diagonal) mean = {np.mean(diag):.4f}   "
      f"mismatched (off-diagonal) mean = {np.mean(offdiag):.4f}")
print(f"[rank] VERDICT: {'DISCRIMINATES (matched>mismatched, success head works offline)' if np.mean(diag) > np.mean(offdiag) + 0.1 else 'FLAT / NO DISCRIMINATION (success head not responding even offline)'}", flush=True)
