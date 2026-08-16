#!/usr/bin/env python3
"""Build a bank of in-context demonstration frames for ManiSkill tasks.

RoboRef-ICL (run1) conditions on a *successful demonstration of the same task*
prepended to the trajectory it is scoring. This script produces that
demonstration bank from ManiSkill's official demo datasets.

Method
------
ManiSkill ships demonstrations with ``obs_mode="none"`` -- actions and
environment states, but no pixels -- so frames have to be re-rendered. We
replay each episode and capture frames through
``ManiSkillSingleEnvWrapper``, i.e. the *same* wrapper the RL rollouts use.
That is deliberate: the demonstration must be visually identical in camera,
resolution and preprocessing to the query it is paired with, or the reward
model is comparing across a domain gap of our own making.

Replay uses two strategies, in order:
  1. **Actions** -- exact when the demo's control mode matches the env's.
  2. **Environment states** -- fallback that sets simulator state directly,
     immune to physics divergence (the demos were generated on the CUDA
     backend; we replay on CPU, which can drift).
Only episodes whose replay actually ends in success are kept, so a silent
divergence cannot leak a failed trajectory into the bank.

Prerequisites:
    export MS_ASSET_DIR=/scratch-shared/$USER/maniskill_assets
    python -m mani_skill.utils.download_demo PullCube-v1

Usage:
    python scripts/generate_maniskill_icl_demos.py --task PullCube-v1 --num-demos 32
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List, Optional

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _find_demo_files(task: str, asset_dir: str) -> List[str]:
    """Locate downloaded trajectory .h5 files for ``task``."""
    patterns = [
        os.path.join(asset_dir, "demos", task, "**", "*.h5"),
        os.path.join(asset_dir, "demos", "rigid_body", task, "**", "*.h5"),
    ]
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    return sorted(set(files))


def _pick_demo_file(files: List[str], control_mode: str) -> str:
    """Prefer a demo file recorded in our control mode (exact action replay)."""
    for path in files:
        if control_mode in os.path.basename(path):
            return path
    return files[0]


def _subsample(frames: np.ndarray, n: int) -> np.ndarray:
    """Take ``n`` evenly spaced frames, matching how RoboRef samples a clip."""
    if len(frames) == n:
        return frames
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return frames[idx]


def _set_env_state(wrapper, state) -> bool:
    """Best-effort direct state restore on the underlying ManiSkill env."""
    try:
        wrapper.env.unwrapped.set_state_dict(state)
        return True
    except Exception:
        return False


def generate(
    task: str,
    num_demos: int,
    num_frames: int,
    image_size: int,
    out_path: str,
    asset_dir: str,
    max_attempts: Optional[int] = None,
) -> int:
    import gymnasium as gym
    import h5py
    import mani_skill.envs  # noqa: F401  (registers env ids)

    from robometer_policy_learning.envs.maniskill_utils import assert_task_allowed, get_task_spec
    from robometer_policy_learning.envs.maniskill_wrapper import ManiSkillSingleEnvWrapper

    assert_task_allowed(task)
    spec = get_task_spec(task)

    files = _find_demo_files(task, asset_dir)
    if not files:
        print(f"ERROR: no demo .h5 found for {task} under {asset_dir}.")
        print(f"  Run: python -m mani_skill.utils.download_demo {task}")
        return 1
    h5_path = _pick_demo_file(files, spec.control_mode)
    json_path = os.path.splitext(h5_path)[0] + ".json"
    print(f"demo file : {h5_path}")

    meta = json.load(open(json_path)) if os.path.exists(json_path) else {}
    episodes = meta.get("episodes", [])
    demo_control_mode = meta.get("env_info", {}).get("env_kwargs", {}).get("control_mode")

    # Replay in the DEMO's control mode, not the RL policy's. An in-context
    # demonstration is only pixels -- the actuation used to produce them is
    # irrelevant to the reward model, and matching it makes action replay exact
    # instead of divergent. Camera and rendering are untouched, so the frames
    # stay visually consistent with the query frames they will be paired with.
    replay_control_mode = demo_control_mode or spec.control_mode
    if replay_control_mode != spec.control_mode:
        print(f"demo control_mode={demo_control_mode!r} != env control_mode={spec.control_mode!r}; "
              f"replaying in the demo's mode (frames are what matter, not actions)")
    else:
        print(f"demo control_mode={demo_control_mode!r} matches env -> action replay is exact")

    env = ManiSkillSingleEnvWrapper(
        gym.make(
            task,
            num_envs=1,
            obs_mode="state",
            control_mode=replay_control_mode,
            render_mode="rgb_array",
            reward_mode="sparse",
            sim_backend="physx_cpu",
            max_episode_steps=spec.max_episode_steps,
        ),
        image_size=image_size,
    )

    h5 = h5py.File(h5_path, "r")
    # Keep episode metadata aligned with the h5 group it describes.
    candidates = []
    for ep in episodes:
        key = f"traj_{ep['episode_id']}"
        if ep.get("success", False) and key in h5:
            candidates.append((key, ep))
    print(f"successful episodes available: {len(candidates)}")

    kept_frames: List[np.ndarray] = []
    kept_seeds: List[int] = []
    attempts = 0
    limit = max_attempts if max_attempts is not None else max(num_demos * 6, 60)

    for key, ep in candidates:
        if len(kept_frames) >= num_demos or attempts >= limit:
            break
        attempts += 1
        traj = h5[key]
        actions = np.asarray(traj["actions"])
        seed = int(ep.get("episode_seed", 0))

        try:
            env.reset(seed=seed, options=ep.get("reset_kwargs") or None)
        except TypeError:
            env.reset(seed=seed)

        frames = [env._render_image()]
        success = False
        for t in range(len(actions)):
            _obs, _r, terminated, truncated, info = env.step(actions[t])
            frames.append(env._render_image())
            success = bool(info.get("success", False)) or success
            if terminated or truncated:
                break

        # Fallback: drive the simulator by recorded state instead of actions.
        if not success and "env_states" in traj:
            try:
                from mani_skill.trajectory.utils import dict_to_list_of_dicts

                states = dict_to_list_of_dicts(traj["env_states"])
                env.reset(seed=seed)
                frames, success = [], False
                for state in states:
                    if not _set_env_state(env, state):
                        break
                    frames.append(env._render_image())
                if frames:
                    ev = env.env.unwrapped.evaluate()
                    success = bool(np.asarray(
                        ev["success"].cpu() if hasattr(ev["success"], "cpu") else ev["success"]
                    ).reshape(-1)[0])
            except Exception:
                pass

        if success and len(frames) >= 2:
            kept_frames.append(_subsample(np.stack(frames), num_frames))
            kept_seeds.append(seed)
            print(f"  kept {len(kept_frames):3d}/{num_demos}  (episode {key}, {len(frames)} raw frames)")

    h5.close()
    env.close()

    if not kept_frames:
        print("ERROR: no episodes replayed to a verified success -- nothing written.")
        return 1

    frames_arr = np.stack(kept_frames).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(
        out_path,
        frames=frames_arr,
        instruction=spec.instruction,
        task=task,
        seeds=np.asarray(kept_seeds),
    )
    mb = frames_arr.nbytes / 1e6
    print(f"\nwrote {frames_arr.shape} ({mb:.1f} MB raw) -> {out_path}")
    print(f"instruction: {spec.instruction!r}")
    print(f"verified-success demos: {len(kept_frames)} (from {attempts} attempts)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="PullCube-v1")
    ap.add_argument("--num-demos", type=int, default=32)
    ap.add_argument("--num-frames", type=int, default=16, help="RoboRef reads 16 frames per trajectory")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--out", default=None)
    ap.add_argument("--asset-dir", default=os.environ.get("MS_ASSET_DIR", os.path.expanduser("~/.maniskill")))
    ap.add_argument("--max-attempts", type=int, default=None)
    args = ap.parse_args()

    out = args.out or os.path.join(
        os.environ.get("MS_ICL_DIR", os.path.join(args.asset_dir, "icl_demos")), f"{args.task}.npz"
    )
    return generate(
        task=args.task,
        num_demos=args.num_demos,
        num_frames=args.num_frames,
        image_size=args.image_size,
        out_path=out,
        asset_dir=args.asset_dir,
        max_attempts=args.max_attempts,
    )


if __name__ == "__main__":
    sys.exit(main())
