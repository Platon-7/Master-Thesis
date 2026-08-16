#!/usr/bin/env python3
"""Run ManiSkill's own trained PPO policy through our stack.

Why this and not the demo-action replay: replaying recorded actions open-loop
compares two simulators bit-for-bit, and ManiSkill's demos were recorded on
``physx_cuda`` while we run ``physx_cpu`` -- so replay fails even on a raw env,
which makes it useless as a test of our code. A *trained closed-loop policy*
does not care: it observes whatever state it is given and reacts. If it scores
well on the raw env but poorly through our wrapper, the wrapper is at fault.

It also answers a question no ManiSkill doc does. Their SAC baseline results are
marked WIP, and the tasks we are allowed to use (PullCube, PokeCube -- the
benchmarked PushCube/PickCube are in the RoboRef training corpus via FailSafe)
have no published RL numbers. This checkpoint is the only published evidence of
what success rate is achievable on this task in this control mode.

The checkpoint is a plain MLP (35 -> 256 -> 256 -> 256 -> 4) with Tanh
activations, shipped alongside the demonstrations.

    python scripts/verify_with_maniskill_ppo.py --task PullCube-v1 --episodes 20
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"


def build_actor(state_dict, obs_dim: int, act_dim: int):
    """Rebuild ManiSkill's PPO actor_mean: Linear/Tanh MLP, 3 hidden layers."""
    import torch.nn as nn

    actor = nn.Sequential(
        nn.Linear(obs_dim, 256), nn.Tanh(),
        nn.Linear(256, 256), nn.Tanh(),
        nn.Linear(256, 256), nn.Tanh(),
        nn.Linear(256, act_dim),
    )
    mapped = {}
    for k, v in state_dict.items():
        if k.startswith("actor_mean."):
            mapped[k[len("actor_mean."):]] = v
    actor.load_state_dict(mapped)
    actor.eval()
    return actor


def run_raw(task, actor, control_mode, episodes, max_steps):
    import gymnasium as gym
    import numpy as np
    import torch
    import mani_skill.envs  # noqa: F401

    env = gym.make(task, num_envs=1, obs_mode="state", control_mode=control_mode,
                   render_mode="rgb_array", reward_mode="normalized_dense",
                   sim_backend="physx_cpu", max_episode_steps=max_steps)
    n = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        for _ in range(max_steps):
            with torch.no_grad():
                a = actor(torch.as_tensor(np.asarray(obs), dtype=torch.float32).reshape(1, -1))
            obs, _r, term, trunc, info = env.step(a.numpy())
            s = info.get("success")
            if bool(np.asarray(s).reshape(-1)[0]):
                n += 1
                break
    env.close()
    return n


def run_ours(task, actor, control_mode, episodes, max_steps):
    import numpy as np
    import torch

    from robometer_policy_learning.utils.env_utils import make_env
    from robometer_policy_learning.rollouts.rollout_worker import extract_info_for_env

    # use_full_state=True is required, not incidental: their PPO actor takes the
    # privileged state vector (35 dims on PullCube, 54 on PokeCube). The default
    # (False) yields the 9-dim robot qpos, and the policy simply cannot be
    # evaluated against it -- "mat1 and mat2 shapes cannot be multiplied
    # (1x9 and 54x256)".
    env, eval_env = make_env(
        env_name=f"maniskill/{task}", num_envs=1, chunk_size=None,
        max_episode_steps=max_steps, use_full_state=True,
        env_kwargs={"sim_backend": "physx_cpu", "image_size": 224,
                    "control_mode": control_mode, "reward_mode": "normalized_dense"},
    )
    n = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=1000 + ep)
        for _ in range(max_steps):
            state = np.asarray(obs["state"]).reshape(1, -1)
            with torch.no_grad():
                a = actor(torch.as_tensor(state, dtype=torch.float32))
            obs, _r, term, trunc, infos = env.step(a.numpy())
            info_i = extract_info_for_env(infos, 0, 1)
            if bool(info_i.get("success", False)):
                n += 1
                break
    env.close()
    if eval_env is not None and eval_env is not env:
        eval_env.close()
    return n


def run_ours_gpu(task, actor, control_mode, episodes, max_steps, num_envs=16,
                 backend="physx_cuda"):
    """Same policy through the GPU-parallel stack.

    Episodes are not seed-matched to the CPU run (the batched env auto-resets on
    its own schedule), so this compares success RATES over a comparable number of
    episodes rather than episode-by-episode. That is enough to catch an adapter
    that mangles observations, actions, or success reporting -- the failure modes
    that would make GPU results untrustworthy.
    """
    import numpy as np
    import torch

    from robometer_policy_learning.utils.env_utils import make_env
    from robometer_policy_learning.rollouts.rollout_worker import extract_info_for_env

    env, eval_env = make_env(
        env_name=f"maniskill/{task}", num_envs=num_envs, chunk_size=None,
        max_episode_steps=max_steps, use_full_state=True,
        env_kwargs={"sim_backend": backend, "image_size": 224,
                    "control_mode": control_mode, "reward_mode": "normalized_dense"},
    )
    obs, _ = env.reset(seed=0)
    done_eps = 0
    succ_eps = 0
    hit = np.zeros(num_envs, dtype=bool)     # success seen in the current episode
    steps = np.zeros(num_envs, dtype=int)
    while done_eps < episodes:
        with torch.no_grad():
            a = actor(torch.as_tensor(np.asarray(obs["state"]), dtype=torch.float32)).numpy()
        obs, _r, term, trunc, infos = env.step(a)
        steps += 1
        for i in range(num_envs):
            info_i = extract_info_for_env(infos, i, num_envs)
            if bool(info_i.get("success", False)):
                hit[i] = True
            if term[i] or trunc[i] or steps[i] >= max_steps:
                done_eps += 1
                succ_eps += int(hit[i])
                hit[i] = False
                steps[i] = 0
                if done_eps >= episodes:
                    break
    env.close()
    if eval_env is not None and eval_env is not env:
        eval_env.close()
    return succ_eps, done_eps


def main() -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PullCube-v1")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--gpu", action="store_true", help="also score through the GPU stack")
    args = ap.parse_args()

    from robometer_policy_learning.envs.maniskill_utils import get_task_spec

    spec = get_task_spec(args.task)
    asset_dir = os.environ.get("MS_ASSET_DIR", os.path.expanduser("~/.maniskill"))
    ckpts = glob.glob(os.path.join(asset_dir, "demos", args.task, "**",
                                   f"ppo_{spec.control_mode}_ckpt.pt"), recursive=True)
    if not ckpts:
        print(f"{RED}no PPO checkpoint for {args.task} in control mode {spec.control_mode}{RESET}")
        print(f"  looked under {os.path.join(asset_dir, 'demos', args.task)}")
        return 2

    sd = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    obs_dim = sd["actor_mean.0.weight"].shape[1]
    act_dim = sd["actor_mean.6.weight"].shape[0]
    actor = build_actor(sd, obs_dim, act_dim)

    print("=" * 72)
    print(f"ManiSkill PPO checkpoint through our stack: {args.task}")
    print(f"  ckpt={os.path.basename(ckpts[0])}  obs_dim={obs_dim}  act_dim={act_dim}")
    print(f"  control_mode={spec.control_mode}  max_episode_steps={spec.max_episode_steps}")
    print("=" * 72, flush=True)

    raw = run_raw(args.task, actor, spec.control_mode, args.episodes, spec.max_episode_steps)
    print(f"  raw env   : {raw}/{args.episodes} = {100*raw/args.episodes:.0f}% success", flush=True)
    ours = run_ours(args.task, actor, spec.control_mode, args.episodes, spec.max_episode_steps)
    print(f"  our stack : {ours}/{args.episodes} = {100*ours/args.episodes:.0f}% success", flush=True)

    if args.gpu:
        g_succ, g_tot = run_ours_gpu(args.task, actor, spec.control_mode,
                                     max(args.episodes, 64), spec.max_episode_steps)
        print(f"  GPU stack : {g_succ}/{g_tot} = {100*g_succ/max(1,g_tot):.0f}% success", flush=True)

    print("=" * 72)
    if raw == 0:
        print(f"{RED}Checkpoint fails even on the raw env -- test inconclusive "
              f"(obs convention or normalization mismatch).{RESET}")
        return 2
    gap = raw - ours
    if gap <= max(2, 0.15 * raw):
        print(f"{GREEN}Adapter is faithful{RESET}: our stack matches the raw env "
              f"({ours} vs {raw}). The task IS solvable at ~{100*raw/args.episodes:.0f}% "
              f"in this control mode -- so a 0% SAC run is a learning problem, not plumbing.")
        return 0
    print(f"{RED}Our stack loses {gap}/{args.episodes} episodes vs the raw env{RESET} "
          f"-- the wrapper degrades a known-good policy. Bug is in the adapter.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
