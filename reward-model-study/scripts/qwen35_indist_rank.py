"""DECISIVE test: score Qwen3.5 on its OWN in-distribution eval frames (the
robometer eval keyframes that produced the 'good offline' metrics), via the same
get_robometer_4b scoring path used by IBRL. If success_prob discriminates
success vs failure here, the scoring PATH is fine and the IBRL weirdness is a
frame-distribution issue (my earlier BC-demo probe was out-of-distribution). If
it's flat here too, the scoring path is genuinely broken.
"""
import os, sys, json, tarfile, io, re
import numpy as np
from PIL import Image

_REPO = os.environ.get("MT_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for sub in ("vlm_ibrl_v3", "Robometer"):
    sys.path.insert(0, os.path.join(_REPO, sub))
from env.robometer_utils import get_robometer_4b

CKPT = os.environ["QWEN35_FT_PATH"]
KF = "/shared/home/PKA4388/eval_keyframes"
recs = json.load(open("/shared/home/PKA4388/eval_keyframes/eval_sample.json"))

def load_episode(frames_path):
    tar_rel, member_prefix = frames_path.split("::")
    tarp = os.path.join(KF, tar_rel)
    frames = {}
    with tarfile.open(tarp) as tf:
        for m in tf.getmembers():
            if m.name.startswith(member_prefix + "/frame_") and m.name.endswith(".jpg"):
                idx = int(re.search(r"frame_(\d+)_", m.name).group(1))
                frames[idx] = Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB")
    return [frames[k] for k in sorted(frames)]

print(f"[indist] loading {CKPT}", flush=True)
sc = get_robometer_4b(model_path=CKPT)
print(f"[indist] max_frames={sc.max_frames}\n", flush=True)

print(f"{'label':<8}{'success_prob':>13}{'progress':>10}  episode / task")
by = {"success": [], "failure": []}
for r in recs:
    try:
        frames = load_episode(r["frames_path"])
        out = sc(frames, task=r["task"], icl_frames=None)
        sp, pg = float(out["success_prob"]), float(out["progress_reward"])
        by[r["label"]].append(sp)
        print(f"{r['label']:<8}{sp:>13.4f}{pg:>10.4f}  {r['episode_id'][:40]} | {r['task'][:28]}", flush=True)
    except Exception as e:
        print(f"{r['label']:<8}  ERROR {type(e).__name__}: {e}  ({r['episode_id'][:40]})", flush=True)

ms = np.mean(by["success"]) if by["success"] else float("nan")
mf = np.mean(by["failure"]) if by["failure"] else float("nan")
print(f"\n[indist] success-label mean = {ms:.4f}   failure-label mean = {mf:.4f}", flush=True)
print(f"[indist] VERDICT: {'DISCRIMINATES on in-distribution frames -> scoring path is FINE; IBRL weirdness was frame-distribution (OOD), not a code bug' if (ms == ms and ms > mf + 0.1) else 'FLAT even on in-distribution frames -> scoring path genuinely broken'}", flush=True)
