"""Cheap test of the 'task description is the problem' hypothesis.

Re-rank StickPull + Assembly eval clips with the Robometer-FT success head under
SEVERAL candidate task descriptions, and report success-vs-failure separation per
description. If a richer description sharpens separation, the wording is a real lever;
if not, the description isn't the bottleneck. Pure scoring (no RL).

Model path via ROBOMETER_FT_PATH. Clips from the cached eval_keyframes index."""
import os, sys, json, tarfile, io, re
import numpy as np
from PIL import Image

_REPO = os.environ.get("MT_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for sub in ("vlm_ibrl_v3", "Robometer"):
    sys.path.insert(0, os.path.join(_REPO, sub))
from env.robometer_utils import get_robometer_4b

CKPT = os.environ["ROBOMETER_FT_PATH"]
KF = "/shared/home/PKA4388/eval_keyframes"
idx = json.load(open(os.path.join(KF, "calib4_index.json")))

# Candidate descriptions per task. D0 = current/training prompt (the control).
DESCRIPTIONS = {
    "stick_pull": [
        ("D0_train",  "Grasp a stick and pull a box with the stick."),
        ("D1_rich",   "The robot grasps the red stick and uses it to drag the brown box toward the green target sphere; success is the box reaching the target."),
        ("D2_goal",   "Move the box to the green goal using the stick."),
        ("D3_simple", "Use the stick to pull the object to the goal location."),
    ],
    "assembly": [
        ("D0_train",  "Pick up a nut and place it onto a peg."),
        ("D1_rich",   "The robot picks up the gray ring-shaped nut and places it onto the vertical peg until the nut is fully seated on the peg."),
        ("D2_goal",   "Put the nut onto the peg."),
        ("D3_simple", "Place the round nut over the peg so it rests on the base."),
    ],
}

def load_episode(tar_rel, member):
    tarp = os.path.join(KF, tar_rel); frames = {}
    with tarfile.open(tarp) as tf:
        for m in tf.getmembers():
            if m.name.startswith(member + "/frame_") and m.name.endswith(".jpg"):
                i = int(re.search(r"frame_(\d+)", m.name).group(1))
                frames[i] = Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB")
    return [frames[k] for k in sorted(frames)]

def best_threshold(succ, fail, cap=0.15):
    if not succ or not fail: return None
    cand = sorted(set([round(x,3) for x in succ+fail] + [0.5,0.7,0.8,0.85,0.9]))
    best = None
    for thr in cand:
        tpr = np.mean([s>=thr for s in succ]); fpr = np.mean([f>=thr for f in fail])
        if fpr <= cap and (best is None or tpr > best[1]): best = (thr,tpr,fpr)
    return best

print(f"[desc] loading {CKPT}", flush=True)
sc = get_robometer_4b(model_path=CKPT)
print(f"[desc] max_frames={sc.max_frames}\n", flush=True)

# pre-load clips once per task (reuse across descriptions)
clips = {}
for t in DESCRIPTIONS:
    clips[t] = []
    for e in idx["tasks"][t]:
        if not os.path.exists(os.path.join(KF, e["tar_rel"])): continue
        try:
            clips[t].append((e["label"], load_episode(e["tar_rel"], e["member"])))
        except Exception as ex:
            print(f"  load ERR {e['episode_id'][:40]}: {ex}", flush=True)
    nl = sum(1 for l,_ in clips[t] if l=="success"); print(f"[desc] {t}: {len(clips[t])} clips ({nl} success)", flush=True)

results = {}
for t, descs in DESCRIPTIONS.items():
    print(f"\n================= {t} =================", flush=True)
    results[t] = {}
    for tag, desc in descs:
        succ, fail = [], []
        for label, frames in clips[t]:
            try:
                out = sc(frames, task=desc, icl_frames=None)
                sp = float(out["success_prob"])
                (succ if label=="success" else fail).append(sp)
            except Exception as ex:
                print(f"  score ERR ({tag}): {ex}", flush=True)
        bt = best_threshold(succ, fail)
        sm = float(np.mean(succ)) if succ else float("nan")
        fm = float(np.mean(fail)) if fail else float("nan")
        sep = sm - fm
        results[t][tag] = {"succ_mean":sm,"fail_mean":fm,"sep":sep,
                           "op":({"thr":bt[0],"tpr":bt[1],"fpr":bt[2]} if bt else None),"desc":desc}
        op = (f"thr{bt[0]:.2f} TPR{bt[1]:.2f} FPR{bt[2]:.2f}" if bt else "no usable thr")
        print(f"  {tag:10s} succ={sm:.3f} fail={fm:.3f} sep={sep:+.3f} | {op}", flush=True)

json.dump(results, open(os.path.join(KF, "desc_rerank_results.json"),"w"), indent=1)
print("\n==================== SUMMARY (separation = succ_mean - fail_mean; higher=better) ====================")
for t in DESCRIPTIONS:
    base = results[t]["D0_train"]["sep"]
    print(f"\n{t} (D0 baseline sep={base:+.3f}):")
    for tag, r in sorted(results[t].items(), key=lambda kv:-kv[1]["sep"]):
        op = r["op"]; ops = (f"thr{op['thr']:.2f} TPR{op['tpr']:.2f} FPR{op['fpr']:.2f}" if op else "no usable thr")
        flag = "  <== best" if r["sep"]==max(x["sep"] for x in results[t].values()) else ""
        print(f"  {tag:10s} sep={r['sep']:+.3f} (vs D0 {r['sep']-base:+.3f})  {ops}{flag}")
