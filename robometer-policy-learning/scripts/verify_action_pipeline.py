#!/usr/bin/env python3
"""Is the ManiSkill adapter faithful? Drive it with known-good actions.

Context: the same SAC/MLP code, same observation style (proprio + DINO), same
gamma/batch, and *fewer* steps reaches eval success 1.0 on LIBERO. On ManiSkill
the ground-truth dense arm sits at 0%. The RL code is therefore not the suspect
-- the adapter is, since it is the only part LIBERO does not share.

A learning curve cannot distinguish "hard task" from "adapter corrupts actions
or never reports success". Replaying ManiSkill's own demonstration actions can:
they are known to solve the task, so if they succeed on a raw env but fail
through our wrapper stack, the adapter is at fault, and the exact stage where
success disappears localises the bug.

Three stages, each adding one layer:
  [1] raw gym.make env, demo actions            -> demos themselves are good
  [2] + ManiSkillSingleEnvWrapper               -> our obs/action/info conversion
  [3] + SyncVectorEnv + make_env() full stack   -> what training actually drives

    python scripts/verify_action_pipeline.py --task PullCube-v1
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import List, Optional

os.environ.setdefault("MUJOCO_GL", "egl")

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"


def _ok(m): print(f"{GREEN}  PASS{RESET}  {m}", flush=True)
def _fail(m): print(f"{RED}  FAIL{RESET}  {m}", flush=True)
def _info(m): print(f"        {m}", flush=True)


def load_demo_actions(task: str, max_episodes: int = 3):
    """Return (list of (actions, episode_seed), control_mode) from ManiSkill's demo h5.

    The per-episode seed is essential: it determines the initial cube and goal
    placement. Replaying an open-loop action sequence from a different initial
    state is meaningless -- it was the reason an earlier version of this script
    reported 0/3 on the RAW env and briefly looked like a finding.
    """
    import glob
    import json

    import h5py
    import numpy as np

    from mani_skill.utils.io_utils import load_json  # noqa: F401  (validates install)

    asset_dir = os.environ.get("MS_ASSET_DIR", os.path.expanduser("~/.maniskill"))
    pattern = os.path.join(asset_dir, "demos", task, "**", "*.h5")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(
            f"no demonstration h5 under {os.path.join(asset_dir, 'demos', task)}\n"
            f"        download: python -m mani_skill.utils.download_demo {task}"
        )

    h5_path = files[0]
    json_path = h5_path.replace(".h5", ".json")
    control_mode = None
    if os.path.exists(json_path):
        with open(json_path) as fh:
            meta = json.load(fh)
        control_mode = meta.get("env_info", {}).get("env_kwargs", {}).get("control_mode")

    seeds = {}
    if os.path.exists(json_path):
        for e in meta.get("episodes", []):
            seeds[f"traj_{e['episode_id']}"] = e.get("episode_seed")

    episodes = []
    with h5py.File(h5_path, "r") as fh:
        for key in list(fh.keys())[:max_episodes]:
            if "actions" in fh[key]:
                episodes.append((np.asarray(fh[key]["actions"]), seeds.get(key)))
    if not episodes:
        raise RuntimeError(f"no 'actions' datasets in {h5_path}")
    _info(f"demos: {os.path.basename(h5_path)}  control_mode={control_mode}  episodes={len(episodes)}")
    return episodes, control_mode


def stage1_raw(task: str, episodes, control_mode: str, max_steps: int) -> bool:
    print(f"\n[1] raw gym.make env + demo actions")
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    n_success = 0
    for ep, seed in episodes:
        env = gym.make(
            task, num_envs=1, obs_mode="state", control_mode=control_mode,
            render_mode="rgb_array", reward_mode="normalized_dense",
            sim_backend="physx_cpu", max_episode_steps=max(max_steps, len(ep) + 1),
        )
        env.reset(seed=seed)
        succeeded = False
        for a in ep:
            _o, _r, term, trunc, info = env.step(a[None, :])
            s = info.get("success")
            succeeded = succeeded or bool(s.reshape(-1)[0] if hasattr(s, "reshape") else s)
            if succeeded:
                break
        env.close()
        n_success += int(succeeded)
    (_ok if n_success else _fail)(f"{n_success}/{len(episodes)} demo episodes reported success on the raw env")
    return n_success > 0


def stage2_wrapper(task: str, episodes, control_mode: str, max_steps: int) -> bool:
    print(f"\n[2] + ManiSkillSingleEnvWrapper (our obs/action/info conversion)")
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    from robometer_policy_learning.envs.maniskill_wrapper import ManiSkillSingleEnvWrapper

    n_success = 0
    for ep, seed in episodes:
        env = ManiSkillSingleEnvWrapper(
            gym.make(
                task, num_envs=1, obs_mode="state", control_mode=control_mode,
                render_mode="rgb_array", reward_mode="normalized_dense",
                sim_backend="physx_cpu", max_episode_steps=max(max_steps, len(ep) + 1),
            ),
            image_size=224,
        )
        env.reset(seed=seed)
        succeeded = False
        for a in ep:
            _o, _r, term, trunc, info = env.step(a)
            succeeded = succeeded or bool(info.get("success", False))
            if succeeded:
                break
        env.close()
        n_success += int(succeeded)
    (_ok if n_success else _fail)(f"{n_success}/{len(episodes)} demo episodes reported success through the wrapper")
    return n_success > 0


def stage3_full(task: str, episodes, control_mode: str) -> bool:
    print(f"\n[3] + full make_env() stack (SyncVectorEnv, as training drives it)")
    import numpy as np

    from robometer_policy_learning.utils.env_utils import make_env
    from robometer_policy_learning.rollouts.rollout_worker import extract_info_for_env

    n_success = 0
    for ep, seed in episodes:
        env, eval_env = make_env(
            env_name=f"maniskill/{task}",
            num_envs=1,
            chunk_size=None,
            max_episode_steps=max(len(ep) + 1, 50),
            env_kwargs={
                "sim_backend": "physx_cpu",
                "image_size": 224,
                "control_mode": control_mode,
                "reward_mode": "normalized_dense",
            },
        )
        env.reset(seed=seed)
        succeeded = False
        for a in ep:
            _o, _r, term, trunc, infos = env.step(np.asarray(a)[None, :])
            info_i = extract_info_for_env(infos, 0, 1)
            succeeded = succeeded or bool(info_i.get("success", False))
            if succeeded:
                break
        env.close()
        if eval_env is not None and eval_env is not env:
            eval_env.close()
        n_success += int(succeeded)
    (_ok if n_success else _fail)(f"{n_success}/{len(episodes)} demo episodes reported success through the full stack")
    return n_success > 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PullCube-v1")
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()

    from robometer_policy_learning.envs.maniskill_utils import get_task_spec

    spec = get_task_spec(args.task)
    print("=" * 72)
    print(f"Action-pipeline verification: {args.task}")
    print(f"configured control_mode={spec.control_mode}  max_episode_steps={spec.max_episode_steps}")
    print("=" * 72)

    try:
        episodes, demo_control_mode = load_demo_actions(args.task, args.episodes)
    except Exception as exc:
        _fail(f"could not load demonstrations: {exc}")
        return 2

    cm = demo_control_mode or spec.control_mode
    if demo_control_mode and demo_control_mode != spec.control_mode:
        print(
            f"{YELLOW}  NOTE{RESET}  demos are {demo_control_mode!r} but the task spec trains in "
            f"{spec.control_mode!r};\n        replaying in the demos' own mode -- action replay is only "
            f"meaningful there."
        )

    results = {}
    try:
        results["raw env"] = stage1_raw(args.task, episodes, cm, spec.max_episode_steps)
        results["wrapper"] = stage2_wrapper(args.task, episodes, cm, spec.max_episode_steps)
        results["full stack"] = stage3_full(args.task, episodes, cm)
    except Exception:
        traceback.print_exc()
        return 1

    print("\n" + "=" * 72)
    for k, v in results.items():
        print(f"  {(GREEN+'PASS'+RESET) if v else (RED+'FAIL'+RESET)}  {k}")
    print("=" * 72)
    if all(results.values()):
        print("Adapter is faithful: known-good actions succeed end to end.")
        print("=> the 0% is a learning problem (control mode / horizon / task difficulty),")
        print("   not a plumbing problem.")
        return 0
    first_bad = next(k for k, v in results.items() if not v)
    print(f"Success disappears at stage: {first_bad}  <- localise the bug there.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
