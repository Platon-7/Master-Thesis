#!/usr/bin/env python3
"""Diagnose why a known-good policy scores 0% through the GPU-parallel stack.

ManiSkill's trained PPO checkpoint scores 60% on PullCube through the CPU stack
and 0/64 through the GPU one, so the GPU adapter is losing something. The three
candidates this separates:

  1. success hidden by auto-reset -- gymnasium moves the terminal step's info to
     `final_info`, so reading info["success"] at an episode boundary returns the
     FRESH episode's flag (False) and success is never observed;
  2. the state vector differs on GPU (right width, wrong content/order), so the
     policy acts on garbage;
  3. actions are mishandled (dtype/device) by the batched env.

Prints info keys at reset, during early steps, and at the first episode boundary,
plus the state's actual values so it can be eyeballed against the CPU path.
"""
import os
import sys
import glob

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from scripts.verify_with_maniskill_ppo import build_actor
from robometer_policy_learning.envs.maniskill_utils import get_task_spec
from robometer_policy_learning.utils.env_utils import make_env

T = sys.argv[1] if len(sys.argv) > 1 else "PullCube-v1"
spec = get_task_spec(T)
ck = glob.glob(os.path.join(os.environ["MS_ASSET_DIR"], "demos", T, "**",
                            f"ppo_{spec.control_mode}_ckpt.pt"), recursive=True)
sd = torch.load(ck[0], map_location="cpu", weights_only=False)
actor = build_actor(sd, sd["actor_mean.0.weight"].shape[1], sd["actor_mean.6.weight"].shape[0])

env, _ = make_env(env_name=f"maniskill/{T}", num_envs=8, chunk_size=None,
                  max_episode_steps=spec.max_episode_steps, use_full_state=True,
                  env_kwargs={"sim_backend": "physx_cuda", "image_size": 224,
                              "control_mode": spec.control_mode,
                              "reward_mode": "normalized_dense"})
obs, info = env.reset(seed=0)
st = np.asarray(obs["state"])
print(f"reset: info keys={sorted(info.keys())[:12]}")
print(f"state shape={st.shape} dtype={st.dtype}")
print(f"state[0][:8]={np.round(st[0][:8], 4)}")
print(f"action_space={env.single_action_space}")

for t in range(spec.max_episode_steps + 3):
    a = actor(torch.as_tensor(np.asarray(obs["state"]), dtype=torch.float32)).detach().numpy()
    if t == 0:
        print(f"action shape={a.shape} dtype={a.dtype} range=({a.min():.3f},{a.max():.3f})")
    obs, r, term, trunc, info = env.step(a)
    ended = bool(np.any(term) | np.any(trunc))
    if t < 3 or ended:
        print(f"\nt={t} term={int(np.asarray(term).sum())} trunc={int(np.asarray(trunc).sum())} "
              f"reward[:3]={np.round(np.asarray(r)[:3], 3)}")
        print(f"   info keys={sorted(info.keys())[:12]}")
        for k in ("success", "is_success", "final_info", "final_observation", "_final_info"):
            if k in info:
                v = info[k]
                if isinstance(v, dict):
                    print(f"   {k}: dict keys={sorted(v.keys())[:8]}")
                    if "success" in v:
                        print(f"      final_info['success'] anyTrue={bool(np.asarray(v['success']).any())}")
                else:
                    arr = np.asarray(v)
                    print(f"   {k}: shape={arr.shape} anyTrue={bool(arr.any())}")
    if ended:
        break
env.close()
print("\nDIAG DONE")
