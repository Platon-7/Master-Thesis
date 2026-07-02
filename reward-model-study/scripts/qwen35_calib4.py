"""Calibrate Qwen3.5 run4 on in-distribution eval keyframes for the 4 IBRL tasks.
Scores every fetched success/failure clip per task via the SAME get_robometer_4b
path IBRL uses, then reports per-task success_prob (and progress) distributions and
the best operating threshold (max success-TPR at failure-FPR <= 0.15). This is the
GO/NO-GO gate: a task with no usable offline threshold cannot work in autonomous RL.

This is a best-case (in-distribution) calibration — IBRL on-policy frames will be
no easier. Needs only scoring (no gymnasium/mujoco)."""
import os, sys, json, tarfile, io, re
import numpy as np

_REPO = os.environ.get("MT_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for sub in ("vlm_ibrl_v3", "Robometer"):
    sys.path.insert(0, os.path.join(_REPO, sub))
from env.robometer_utils import get_robometer_4b
from PIL import Image

CKPT = os.environ["QWEN35_FT_PATH"]
KF = "/shared/home/PKA4388/eval_keyframes"
idx = json.load(open(os.path.join(KF, "calib4_index.json")))
# human-readable task strings exactly as stored in the records
TASK_ORDER = ["coffee_push", "box_close", "stick_pull", "assembly"]

def load_episode(tar_rel, member):
    tarp = os.path.join(KF, tar_rel)
    frames = {}
    with tarfile.open(tarp) as tf:
        for m in tf.getmembers():
            if m.name.startswith(member + "/frame_") and m.name.endswith(".jpg"):
                idxn = int(re.search(r"frame_(\d+)", m.name).group(1))
                frames[idxn] = Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB")
    return [frames[k] for k in sorted(frames)]

def best_threshold(succ, fail, fpr_cap=0.15):
    """max TPR among thresholds with FPR<=cap. Returns (thr,tpr,fpr) or None."""
    if not succ or not fail: return None
    cand = sorted(set([round(x, 3) for x in (succ + fail)] + [0.5, 0.7, 0.8, 0.85, 0.9]))
    best = None
    for thr in cand:
        tpr = np.mean([s >= thr for s in succ]); fpr = np.mean([f >= thr for f in fail])
        if fpr <= fpr_cap:
            if best is None or tpr > best[1]: best = (thr, tpr, fpr)
    return best

print(f"[calib4] loading {CKPT}", flush=True)
sc = get_robometer_4b(model_path=CKPT)
print(f"[calib4] max_frames={sc.max_frames}\n", flush=True)

results = {}
for t in TASK_ORDER:
    eps = [e for e in idx["tasks"][t] if os.path.exists(os.path.join(KF, e["tar_rel"]))]
    succ, fail = [], []
    print(f"==== {t} ({len(eps)} eps) ====", flush=True)
    for e in eps:
        try:
            frames = load_episode(e["tar_rel"], e["member"])
            out = sc(frames, task=e["task"], icl_frames=None)
            sp, pg = float(out["success_prob"]), float(out["progress_reward"])
            (succ if e["label"] == "success" else fail).append(sp)
            print(f"  {e['label']:<8} sp={sp:.4f} pg={pg:.4f}  {e['episode_id'][:46]}", flush=True)
        except Exception as ex:
            print(f"  ERR {type(ex).__name__}: {ex}  {e['episode_id'][:40]}", flush=True)
    bt = best_threshold(succ, fail)
    sm = float(np.mean(succ)) if succ else float("nan")
    fm = float(np.mean(fail)) if fail else float("nan")
    smax = float(np.max(succ)) if succ else float("nan")
    results[t] = {"n_succ": len(succ), "n_fail": len(fail), "succ_mean": sm, "succ_max": smax,
                  "fail_mean": fm, "succ": succ, "fail": fail,
                  "op": ({"thr": bt[0], "tpr": bt[1], "fpr": bt[2]} if bt else None)}
    op = (f"thr={bt[0]:.2f} TPR={bt[1]:.2f} FPR={bt[2]:.2f}" if bt else "NO usable thr (FPR<=0.15)")
    print(f"  -> succ_mean={sm:.3f} (max {smax:.3f})  fail_mean={fm:.3f}  | {op}\n", flush=True)

json.dump(results, open(os.path.join(KF, "calib4_run4_results.json"), "w"), indent=1)
print("\n==================== SUMMARY (run4, in-distribution) ====================")
print(f"{'task':<13}{'succ_mean':>10}{'succ_max':>10}{'fail_mean':>10}   operating point")
for t in TASK_ORDER:
    r = results[t]; op = r["op"]
    ops = (f"thr {op['thr']:.2f}  TPR {op['tpr']:.2f}  FPR {op['fpr']:.2f}" if op else "NONE (no FPR<=0.15 thr)")
    print(f"{t:<13}{r['succ_mean']:>10.3f}{r['succ_max']:>10.3f}{r['fail_mean']:>10.3f}   {ops}")
print("\nVERDICT per task: GO if an operating point exists with TPR>=0.5; else NO-GO for IBRL.")
for t in TASK_ORDER:
    op = results[t]["op"]
    go = op and op["tpr"] >= 0.5
    print(f"  {t:<13} {'GO  (thr=%.2f)'%op['thr'] if go else 'NO-GO'}")
