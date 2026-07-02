"""Smoke test: can vlm_ibrl_qwen35 run robomimic IBRL with a Qwen3.5 reward model?
Verifies the full chain on a GPU node: robosuite+cv2 import, PixelRobosuite Can step,
metaworld still imports (don't-break check), and the Qwen3.5 run4 scorer loads + scores.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch
from pathlib import Path
from PIL import Image

print("1. import robosuite / cv2 / metaworld ...", flush=True)
import robosuite  # noqa
import cv2  # noqa
import metaworld  # noqa  (the env we must NOT break)
print("   OK (metaworld imports — not broken)", flush=True)

print("2. build PixelRobosuite Can + step ...", flush=True)
import train_bc
from env.robosuite_wrapper import PixelRobosuite
from common_utils import ibrl_utils as utils
policy, _, ep = train_bc.load_model("release/model/robomimic/can/model0.pt", "cuda")
ep = dict(ep); pol = list(ep["rl_cameras"])
ep["camera_names"] = list(dict.fromkeys(pol + ["agentview"])); ep["rl_cameras"] = pol
ep["episode_length"] = 60; ep["end_on_success"] = 0
env = PixelRobosuite(**ep)
obs, hi = env.reset()
with torch.no_grad(), utils.eval_mode(policy):
    for t in range(5):
        a = policy.act(obs, eval_mode=True)
        obs, r, term, succ, hi = env.step(a)
print("   OK (stepped 5 in PixelRobosuite Can)", flush=True)

print("3. load Qwen3.5 run4 scorer + score (with ICL) ...", flush=True)
from env.robometer_utils import get_robometer_4b
from env.robosuite_vlm_env import tensor_to_pil
sc = get_robometer_4b(model_path="/shared/home/PKA4388/checkpoints/Qwen35_FT_phase1_consolidated/run4_step6500")
icl_p = "/shared/home/PKA4388/vlm_ibrl_runs/icl_demo_can_median"
avail = sorted(q for q in Path(icl_p).iterdir() if q.name.startswith("0_") and q.suffix == ".png")
picks = np.linspace(0, len(avail) - 1, 16).round().astype(int)
icl = [np.asarray(Image.open(avail[i]).convert("RGB"), dtype=np.uint8) for i in picks]
frames = [tensor_to_pil(hi["agentview"])]
task = ("A can is placed in a bin in front of a single robot arm. There are four containers "
        "next to the bin. The robot must place the can into its corresponding container.")
out = sc(frames, task=task, icl_frames=icl)
print(f"   OK Qwen3.5 run4 scorer: success_prob={float(out['success_prob']):.4f} "
      f"progress={float(out.get('progress_reward', -1)):.4f}", flush=True)
print("ALL GREEN — vlm_ibrl_qwen35 can run robomimic IBRL with Qwen3.5", flush=True)
