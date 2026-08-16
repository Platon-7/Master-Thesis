#!/usr/bin/env python3
"""Compare the state vector CPU vs GPU physics produce for the same task/seed.

ManiSkill's PPO checkpoint scores 58% through our GPU stack on PullCube (CPU: 60%,
so it matches) but 2% on PokeCube (CPU: 30%). A policy that transfers on one task
and not another points at the observation, not the plumbing: same width, different
content or ordering, would leave the matmul valid but the policy blind.

Dumps state[0] at reset and after a few fixed actions. Run once per backend (PhysX
cannot switch inside one process), then diff the two files.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from robometer_policy_learning.envs.maniskill_utils import get_task_spec
from robometer_policy_learning.utils.env_utils import make_env

task, backend, out = sys.argv[1], sys.argv[2], sys.argv[3]
spec = get_task_spec(task)
env, _ = make_env(env_name=f"maniskill/{task}", num_envs=1, chunk_size=None,
                  max_episode_steps=spec.max_episode_steps, use_full_state=True,
                  env_kwargs={"sim_backend": backend, "image_size": 224,
                              "control_mode": spec.control_mode,
                              "reward_mode": "normalized_dense"})
obs, _ = env.reset(seed=123)
states = [np.asarray(obs["state"]).reshape(-1).copy()]
rng = np.random.RandomState(0)
acts = rng.uniform(-0.3, 0.3, size=(5, env.single_action_space.shape[0])).astype(np.float32)
for a in acts:
    obs, r, term, trunc, info = env.step(np.stack([a]))
    states.append(np.asarray(obs["state"]).reshape(-1).copy())
env.close()
np.savez(out, states=np.stack(states), backend=backend, task=task)
print(f"{backend:11s} {task:18s} state_dim={states[0].shape[0]}  reset[:8]={np.round(states[0][:8],4)}")
