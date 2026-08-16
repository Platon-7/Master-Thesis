#!/usr/bin/env python3
"""Is the low success_prob real, or a numerics/loading pathology?

On MetaWorld the fine-tuned model scores ~0.77 on successes and ~0.13 on failures
(reward-model-study/scripts/calibrate_threshold.py). On ManiSkill this repo measures
0.09/0.036 for run2 and 0.026/0.010 for run3 -- both classes squashed toward zero,
which is the signature the RobometerScorer notes warn about:

    "Some Robometer-family checkpoints consistently fall into a cuDNN bf16
     fast-path that produces ALL-NaN success_logits."
    (vlm_ibrl/env/robometer_utils.py)

Scores identical frames under three numeric configurations and prints the success
probabilities side by side. If cudnn-deterministic or fp32 lifts the values, the
compression is a numerics bug, not the model's opinion.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch

MODE = sys.argv[1] if len(sys.argv) > 1 else "default"   # default|cudnn|fp32|res480|res336|pad16|pad16res480
MODEL = sys.argv[2]
TASK = sys.argv[3] if len(sys.argv) > 3 else "PullCube-v1"

if MODE in ("cudnn", "fp32"):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

from robometer.evals.eval_utils import raw_dict_to_sample
from robometer.evals.eval_server import process_batch_helper
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator
from robometer_policy_learning.utils.robometer_utils import (
    extract_success_probs_from_output, extract_rewards_from_output)

CAL = os.path.join(os.environ["MS_ASSET_DIR"], "calibration")
z = np.load(os.path.join(CAL, f"{TASK}_trajs.npz"), allow_pickle=True)
frames, succ_step = z["frames"], z["succ_step"]
instruction = str(z["instruction"])

dev = torch.device("cuda")
cfg, tok, proc, model = load_model_from_hf(model_path=MODEL, device=dev)
if MODE == "fp32":
    model = model.float()
    for m in model.modules():
        for p in m.parameters(recurse=False):
            p.data = p.data.float()
if not getattr(cfg.data, "use_multi_image", False):
    cfg.data.use_multi_image = True
coll = setup_batch_collator(proc, tok, cfg, is_eval=True)
plt_ = cfg.loss.progress_loss_type.lower()
disc = plt_ == "discrete" or "c51" in plt_

# 8 clear successes (score at the success step) and 8 clear failures (final frame)
succ_eps = [i for i, s in enumerate(succ_step) if s >= 0][:8]
fail_eps = [i for i, s in enumerate(succ_step) if s < 0][:8]
sp_s, sp_f, pr_s = [], [], []
for e in succ_eps + fail_eps:
    t = int(succ_step[e]) if succ_step[e] >= 0 else frames.shape[1] - 1
    fr_in = frames[e][: t + 1]
    if "pad16" in MODE:
        # Feed at least max_frames frames. ManiSkill episodes succeed at step ~7, so a
        # growing prefix supplies 7-11 frames -- fine for base (max_frames=8) but well
        # short for the fine-tuned checkpoints (max_frames=16). calibrate_threshold.py
        # never scores below 16 frames (`for t in range(SUB, ...)`, SUB=16), and the
        # FT diagnostic calls the failure a "pipeline mismatch (224x224 + 5 frames)".
        # Pad at the FRONT by repeating the first frame, preserving temporal order and
        # the final frame that carries the success evidence.
        need = 16 - len(fr_in)
        if need > 0:
            fr_in = np.concatenate([np.repeat(fr_in[:1], need, axis=0), fr_in], axis=0)
    if MODE.startswith("res"):
        # Training data was 480x480 JPEG; we render ManiSkill at 224. The Qwen3-VL
        # processor maps image size directly to visual-token count, so a 224 input
        # gives the model far fewer tokens than it ever saw in training -- the
        # "flat output" pipeline mismatch documented in
        # vlm_ibrl/tools/diag_ft_on_training_frames.py.
        from PIL import Image as _I
        R = int(MODE[3:])
        fr_in = np.stack([np.asarray(_I.fromarray(x).resize((R, R), _I.BICUBIC)) for x in fr_in])
    raw = dict(frames=fr_in, task=instruction, id=e,
               metadata=dict(subsequence_length=len(fr_in)),
               video_embeddings=None, text_embedding=None)
    s = raw_dict_to_sample(raw_data=raw, max_frames=getattr(cfg.data, "max_frames", 16),
                           sample_type="progress")
    out = process_batch_helper(model_type=cfg.model.model_type, model=model, tokenizer=tok,
                               batch_collator=coll, device=model.device,
                               batch_data=[s.model_dump()], job_id=0,
                               is_discrete_mode=disc, num_bins=cfg.loss.progress_discrete_bins)
    sp = float(extract_success_probs_from_output(out)[0])
    pr = float(extract_rewards_from_output(out)[0])
    (sp_s if succ_step[e] >= 0 else sp_f).append(sp)
    if succ_step[e] >= 0:
        pr_s.append(pr)

name = os.path.basename(MODEL.rstrip("/"))
print(f"RESULT mode={MODE:8s} model={name:32s} "
      f"success_prob: succ={np.mean(sp_s):.4f} fail={np.mean(sp_f):.4f} "
      f"| progress@succ={np.mean(pr_s):.4f} "
      f"| nan={int(np.isnan(sp_s+sp_f).sum())}")
print(f"   succ values: {np.round(sp_s,4).tolist()}")
print(f"   fail values: {np.round(sp_f,4).tolist()}")
