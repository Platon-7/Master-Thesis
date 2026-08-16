#!/usr/bin/env python3
"""Measure transitions/second: CPU physics vs GPU physics at several env counts.

The number that matters is transitions/s, not iterations/s -- N parallel envs
yield N transitions per step, and it is transitions that drive both SAC's data
budget and the reward model's batch size.
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

from robometer_policy_learning.utils.env_utils import make_env


def bench(task, backend, num_envs, iters=100, render=True):
    env, eval_env = make_env(
        env_name=f"maniskill/{task}", num_envs=num_envs, chunk_size=None,
        use_full_state=True,
        env_kwargs={"sim_backend": backend, "image_size": 224,
                    "reward_mode": "normalized_dense",
                    **({"render_every_step": render} if backend != "physx_cpu" else {})},
    )
    obs, _ = env.reset(seed=0)
    a_space = env.single_action_space
    acts = np.stack([a_space.sample() for _ in range(num_envs)])
    for _ in range(5):                      # warmup
        env.step(acts)
    t0 = time.time()
    for _ in range(iters):
        env.step(acts)
    dt = time.time() - t0
    shapes = {k: tuple(np.asarray(v).shape) for k, v in obs.items()}
    env.close()
    if eval_env is not None and eval_env is not env:
        eval_env.close()
    return num_envs * iters / dt, shapes


if __name__ == "__main__":
    # ONE config per process, on purpose. SAPIEN raises
    #   "GPU PhysX can only be enabled once before any other code involving PhysX"
    # if a CPU env is created first in the same interpreter, so a single process
    # cannot benchmark both backends. The job script loops and invokes this once
    # per configuration.
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PullCube-v1")
    ap.add_argument("--backend", default="physx_cpu")
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    try:
        tps, shapes = bench(a.task, a.backend, a.envs, iters=a.iters, render=not a.no_render)
        print(f"RESULT {a.backend:12s} envs={a.envs:<5} render={not a.no_render!s:<5} "
              f"transitions/s={tps:10.1f}  {shapes}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"RESULT {a.backend:12s} envs={a.envs:<5} render={not a.no_render!s:<5} "
              f"FAILED  {type(e).__name__}: {str(e)[:110]}", flush=True)
