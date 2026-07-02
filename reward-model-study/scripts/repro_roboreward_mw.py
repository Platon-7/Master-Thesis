"""Reproduce the Robometer-paper MetaWorld eval of RoboReward-4B (target ~0.746)
using OUR wrapper on the SAME eval clips (16x240x240 npz frames + the eval's own
task strings). If our wrapper discriminates success here, the wrapper is correct
and our RL-harness corner2 render is the mismatch. If not, the wrapper is wrong."""
import os, sys, re
import numpy as np

_REPO = os.environ.get("MT_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for sub in ("MetaWorld/metaworld_repo", "MetaWorld", "vlm_ibrl_v3", "Robometer"):
    sys.path.insert(0, os.path.join(_REPO, sub))

from env.roboreward_utils import get_roboreward_4b, prompt_roboreward
from env.vlm_prompts import roboreward_prompt
from datasets import load_from_disk
from PIL import Image
import scipy.stats as st

DS = ("/shared/home/PKA4388/robometer_frames_hf_full_step2/"
      "_fsx_PKA4388_robometer_frames_hf_full_eval_metaworld_raw_robometer_frames_eval_metaworld/processed_dataset")
TASK_KEY = os.environ.get("RR_TASK_KEY", "coffee_push")   # substring of id to select
MAXN = int(os.environ.get("RR_MAXN", "40"))
_PARSE = re.compile(r"ANSWER:\s*([1-5])")


def auc(l, s):
    l=np.asarray(l); s=np.asarray(s); P=(l==1).sum(); N=(l==0).sum()
    if P==0 or N==0: return float("nan")
    r=st.rankdata(s); return float((r[l==1].sum()-P*(P+1)/2)/(P*N))


def load_frames(ex):
    p = ex["frames"]
    if isinstance(p, str) and p.endswith(".npz") and os.path.exists(p):
        return list(np.load(p)["frames"])
    # fallback: reconstruct from frames dir by id
    return None


def score(model, processor, task, frames):
    imgs = [Image.fromarray(np.asarray(f).astype(np.uint8)) for f in frames]
    content = [
        {"type": "video", "video": imgs, "sample_fps": 20,
         "video_metadata": {"duration": len(imgs) / 20.0}},
        {"type": "text", "text": f"{roboreward_prompt}\n\nTask: {task}"},
    ]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": roboreward_prompt}]},
        {"role": "user", "content": content},
    ]
    kw = dict(max_new_tokens=16, do_sample=False, top_p=1.0, top_k=0, temperature=0)
    raw = prompt_roboreward(model=model, processor=processor, messages=messages,
                            debug=False, prompt_kwargs=kw)
    m = _PARSE.search(raw or "")
    return int(m.group(1)) if m else None


def main():
    print(f"[repro] loading eval dataset + RoboReward-4B (task_key={TASK_KEY}) ...", flush=True)
    ds = load_from_disk(DS)
    idx = [i for i in range(len(ds))
           if ds[i]["data_source"] == "metaworld" and TASK_KEY in ds[i]["id"].lower()]
    print(f"[repro] {len(idx)} '{TASK_KEY}' metaworld examples", flush=True)
    model, processor = get_roboreward_4b()
    scores, labels = [], []
    n_done = 0
    for i in idx:
        if n_done >= MAXN: break
        ex = ds[i]; fr = load_frames(ex)
        if fr is None:
            print(f"  [skip] {ex['id']} no frames on disk ({ex['frames'][:60]})", flush=True); continue
        sc = score(model, processor, ex["task"], fr)
        if sc is None:
            print(f"  [parse-fail] {ex['id']}", flush=True); continue
        lab = 1 if ex["quality_label"] == "successful" else 0
        scores.append((sc - 1) / 4.0); labels.append(lab)
        n_done += 1
        if n_done <= 12:
            print(f"  {ex['id'][:50]:50s} q={ex['quality_label']:10s} rr={sc} view={'gripperPOV' if 'gripperPOV' in ex['id'] or 'gripperPOV' in ex['frames'] else 'corner'}", flush=True)
    scores = np.array(scores); labels = np.array(labels)
    a = auc(labels, scores)
    print(f"\n[repro] n={len(scores)} succ={int((labels==1).sum())} fail={int((labels==0).sum())}", flush=True)
    print(f"[repro] succ_mean={scores[labels==1].mean() if (labels==1).any() else float('nan'):.3f} "
          f"fail_mean={scores[labels==0].mean() if (labels==0).any() else float('nan'):.3f}", flush=True)
    print(f"[repro] AUC={a:.3f}  (paper RoboReward-4B MetaWorld = 0.746)", flush=True)


if __name__ == "__main__":
    main()
