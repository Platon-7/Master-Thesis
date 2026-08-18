#!/usr/bin/env python3
"""Evaluate a saved SAC actor on GT success. No reward model involved.

Written to settle whether run2's 12% eval at 150k was a property of the POLICY or an
artifact: the resumed run scored 0/80 in its first 80 episodes, which execute before
learning_starts and therefore reflect the restored policy with no gradient updates.
"""
import os, sys, glob
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch

CKPT, TASK, N = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 50
from robometer_policy_learning.envs.maniskill_utils import get_task_spec
from robometer_policy_learning.utils.env_utils import make_env
from robometer_policy_learning.rollouts.rollout_worker import extract_info_for_env
from robometer_policy_learning.utils.gpu_utils import convert_to_tensor, move_to_device
from transformers import AutoImageProcessor, AutoModel

spec = get_task_spec(TASK)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dino = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
env, _ = make_env(env_name=f"maniskill/{TASK}", num_envs=1, chunk_size=None,
                  max_episode_steps=spec.max_episode_steps, use_full_state=False,
                  dinov2_model=dino, dinov2_processor=dproc, device=str(dev),
                  terminate_on_success=False,
                  env_kwargs={"sim_backend": "physx_cpu", "image_size": 224,
                              "control_mode": spec.control_mode,
                              "reward_mode": "normalized_dense"})
actor = torch.load(os.path.join(CKPT, "actor.pt"), map_location=dev, weights_only=False)
actor.eval()
succ = 0
for e in range(N):
    obs, _ = env.reset(seed=90000 + e)
    hit = False
    for t in range(spec.max_episode_steps):
        with torch.no_grad():
            a, _ = actor.act(move_to_device(convert_to_tensor(obs), dev),
                             actor_state=None, deterministic=True)
        obs, _r, term, trunc, infos = env.step(a.detach().cpu().numpy())
        if bool(extract_info_for_env(infos, 0, 1).get("success", False)):
            hit = True
        if bool(term[0]) or bool(trunc[0]):
            break
    succ += int(hit)
env.close()
print(f"RESULT ckpt={CKPT.rstrip('/').split('/')[-2]}/{CKPT.rstrip('/').split('/')[-1]} "
      f"episodes={N} GT_success={succ} ({100*succ/N:.1f}%)")
