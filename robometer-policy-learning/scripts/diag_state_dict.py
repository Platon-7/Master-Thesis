#!/usr/bin/env python3
"""Name the state fields that differ between CPU and GPU physics.

The flat `state` vector differs at object-related indices with a repeated
0.003 -> 0.089 offset across two tasks, which looks like a z-height rather than a
different random draw. obs_mode="state_dict" gives the same quantities with names,
so the differing field can be identified instead of inferred from an index.
"""
import os, sys, json
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, gymnasium as gym, torch
import mani_skill.envs  # noqa

task, backend, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robometer_policy_learning.envs.maniskill_utils import get_task_spec
spec = get_task_spec(task)

env = gym.make(task, num_envs=1, obs_mode="state_dict", control_mode=spec.control_mode,
               render_mode="rgb_array", reward_mode="normalized_dense",
               sim_backend=backend, max_episode_steps=spec.max_episode_steps)
obs, _ = env.reset(seed=123)

def flat(d, p=""):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict):
            o.update(flat(v, p + k + "."))
        else:
            a = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
            o[p + k] = np.round(a.reshape(-1).astype(float), 5).tolist()
    return o

fields = flat(obs)
env.close()
json.dump(fields, open(out, "w"), indent=1)
print(f"{backend}: wrote {len(fields)} fields")
