#!/usr/bin/env python3
"""Do the DINO video_embeddings the buffer passes break the success head?

Training logs success_prob ~0.0003 where offline calibration on the same task
measured ~0.83. The one structural difference: RobometerReplayBuffer passes
video_embeddings (DINOv2 features) and text_embedding, while the validated
RobometerScorer (vlm_ibrl/env/robometer_utils.py) passes None for both.

Scores identical frames both ways.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch

MODEL, TASK = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "PullCube-v1")
from robometer.evals.eval_utils import raw_dict_to_sample
from robometer.evals.eval_server import process_batch_helper
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator
from robometer_policy_learning.utils.robometer_utils import extract_success_probs_from_output
from transformers import AutoImageProcessor, AutoModel

CAL = os.path.join(os.environ["MS_ASSET_DIR"], "calibration")
z = np.load(os.path.join(CAL, f"{TASK}_trajs.npz"), allow_pickle=True)
frames, succ_step = z["frames"], z["succ_step"]
lengths = z["lengths"] if "lengths" in z.files else np.full(frames.shape[0], frames.shape[1])
instr = str(z["instruction"])

dev = torch.device("cuda")
cfg, tok, proc, model = load_model_from_hf(model_path=MODEL, device=dev)
if not getattr(cfg.data, "use_multi_image", False):
    cfg.data.use_multi_image = True
coll = setup_batch_collator(proc, tok, cfg, is_eval=True)
mf = int(getattr(cfg.data, "max_frames", 16))
plt_ = str(cfg.loss.progress_loss_type).lower()
disc = plt_ == "discrete" or "c51" in plt_

dname = "facebook/dinov2-base"
dmodel = AutoModel.from_pretrained(dname).to(dev).eval()
dproc = AutoImageProcessor.from_pretrained(dname)

def dino(fr):
    with torch.no_grad():
        x = dproc(images=list(fr), return_tensors="pt")
        x = {k: v.to(dev) for k, v in x.items()}
        return dmodel(**x).pooler_output.float().cpu().numpy()

def score(fr, emb):
    if len(fr) < mf:
        idx = np.linspace(0, len(fr) - 1, mf).round().astype(int)
        fr = fr[idx]
        if emb is not None: emb = emb[idx]
    raw = dict(frames=fr, task=instr, id=0, metadata=dict(subsequence_length=len(fr)),
               video_embeddings=emb, text_embedding=None)
    s = raw_dict_to_sample(raw_data=raw, max_frames=mf, sample_type="progress")
    out = process_batch_helper(model_type=cfg.model.model_type, model=model, tokenizer=tok,
                               batch_collator=coll, device=model.device,
                               batch_data=[s.model_dump()], job_id=0,
                               is_discrete_mode=disc, num_bins=cfg.loss.progress_discrete_bins)
    return float(extract_success_probs_from_output(out)[0])

succ = [i for i, s in enumerate(succ_step) if s >= 0][:6]
print(f"model={os.path.basename(MODEL.rstrip('/'))} max_frames={mf}")
print(f"{'ep':>4s} {'t':>4s} {'sp(emb=None)':>13s} {'sp(emb=DINO)':>13s}")
a, b = [], []
for e in succ:
    t = min(int(succ_step[e]) + 3, int(lengths[e]) - 1)
    fr = frames[e][: t + 1]
    emb = dino(fr)
    s0, s1 = score(fr, None), score(fr, emb)
    a.append(s0); b.append(s1)
    print(f"{e:4d} {t:4d} {s0:13.4f} {s1:13.4f}")
print(f"MEAN  emb=None {np.mean(a):.4f}   emb=DINO {np.mean(b):.4f}")
