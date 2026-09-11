"""Score every growing prefix of the close-lid episodes with one reward model.

Mirrors vlm_ibrl/jobs/diag_causal_calib.py: at step t the model is handed frames[0:t+1]
(subsampled by the scorer to its own max_frames) and we record the success-head
probability at the last frame. Output: one JSON with the per-step series per episode.
"""
import os, sys, json, glob, time, argparse
import numpy as np
sys.path.insert(0, os.path.expanduser("~/Master-Thesis/vlm_ibrl"))
from env.robometer_utils import get_robometer_4b

CKPT = {"roboref": "/projects/prjs1958/Robometer_FT_consolidated/run2_noicl_ours_step4000",
        "robometer": "/home/pkarageorgis1/.cache/huggingface/hub/models--robometer--Robometer-4B/snapshots/beef63bc914c5c189329d49c6d712d96d632aa34"}  # local snapshot of robometer/Robometer-4B
# native frame budgets: every call is subsampled (linspace) or right-padded (repeat of the
# last real frame) to exactly this many frames before it reaches the model
MAXF = {"roboref": 16, "robometer": 8, "roboref_icl": 16}
CKPT["roboref_icl"] = "/projects/prjs1958/Robometer_FT_consolidated/run1_icl_ours_step4000"
ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["roboref", "robometer", "roboref_icl"], required=True)
ap.add_argument("--task", required=True)
ap.add_argument("--frames", default="/scratch-shared/pkarageorgis1/real_world_calib/frames")
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--out", required=True)
ap.add_argument("--icl_demo", default="", help="npz name of the success episode used as the in-context demo")
ap.add_argument("--icl_frames", type=int, default=16, help="demo frames, linspace over the episode (sim default 16)")
a = ap.parse_args()

scorer = get_robometer_4b(model_path=CKPT[a.model], max_frames=MAXF[a.model])
assert scorer.max_frames == MAXF[a.model], (scorer.max_frames, MAXF[a.model])
# show the fixed-length input the model actually receives for a short and a long prefix
from robometer.evals.eval_utils import raw_dict_to_sample
_z = np.load(sorted(glob.glob(f"{a.frames}/*.npz"))[0])["frames"]
for _n in (3, len(_z)):
    _s = raw_dict_to_sample(dict(frames=_z[:_n], task=a.task, id=0, metadata=dict(subsequence_length=_n),
                                 video_embeddings=None, text_embedding=None), max_frames=scorer.max_frames)
    _f = _s.trajectory.frames
    print(f"[check] prefix of {_n} frames -> model input {tuple(_f.shape)}; "
          f"last {max(0, scorer.max_frames - _n)} positions repeat the last real frame: "
          f"{bool((_f[-1] == _z[_n-1]).all()) if _n < scorer.max_frames else 'n/a (subsampled)'}", flush=True)
print(f"[score] model={a.model} ckpt={CKPT[a.model]} max_frames={scorer.max_frames} task={a.task!r}", flush=True)
icl = None
if a.icl_demo:
    _d = np.load(f"{a.frames}/{a.icl_demo}.npz")["frames"]
    icl = list(_d[np.linspace(0, len(_d) - 1, a.icl_frames).round().astype(int)])
    print(f"[icl] demo={a.icl_demo} ({len(_d)} frames) -> {len(icl)} context frames", flush=True)
eps = []
for f in sorted(glob.glob(f"{a.frames}/*.npz")):
    z = np.load(f); fr = z["frames"]; lab = int(z["label"]); T = len(fr)
    t0 = time.time(); steps, sps, prog = [], [], []
    for t in range(1, T, a.stride):
        out = scorer(list(fr[: t + 1]), task=a.task, icl_frames=icl)
        steps.append(t); sps.append(float(out["success_prob"])); prog.append(float(out["progress_reward"]))
    if steps[-1] != T - 1:                      # always include the full episode
        out = scorer(list(fr), task=a.task, icl_frames=icl); steps.append(T - 1)
        sps.append(float(out["success_prob"])); prog.append(float(out["progress_reward"]))
    name = os.path.basename(f)[:-4]
    eps.append(dict(name=name, label=lab, T=T, steps=steps, success_prob=sps, progress=prog))
    print(f"  {name}: GT={'S' if lab else 'F'} T={T} max_sp={max(sps):.3f} final_sp={sps[-1]:.3f} ({time.time()-t0:.0f}s)", flush=True)
json.dump(dict(model=a.model, ckpt=CKPT[a.model], max_frames=scorer.max_frames, task=a.task,
               stride=a.stride, icl_demo=a.icl_demo, icl_frames=a.icl_frames if icl else 0, episodes=eps), open(a.out, "w"))
print("wrote", a.out)
