#!/usr/bin/env python3
"""Standalone smoke test for the ManiSkill env adapter.

Deliberately imports as little as possible: only gymnasium, numpy, mani_skill
and this repo's ManiSkill wrapper. It does NOT import the RL stack (which pulls
in LIBERO / openpi / unsloth / metaworld), so it can be run in a lean
ManiSkill-only environment to prove the environment plumbing works before
committing to a full `uv sync`.

Checks, in order:
  1. The forbidden-task guard actually fires (PickCube/PushCube/StackCube).
  2. A single wrapped env produces unbatched numpy observations of the right
     dtype/shape -- this is the thing most likely to silently break, because
     ManiSkill returns batched torch tensors natively.
  3. ``info["success"]`` is a real python bool, not a 1-element tensor (the
     rollout workers do ``info.get("success", False)``, which would be
     truthy-but-wrong for a tensor).
  4. A SyncVectorEnv of N envs steps correctly.
  5. Rendered frames are saved so the camera view can be eyeballed -- an
     upside-down or badly-cropped frame silently degrades the VLM reward model.

Usage:
    python scripts/smoke_test_maniskill.py --task PullCube-v1
    python scripts/smoke_test_maniskill.py --task PickSingleYCB-v1 --save-frames out/
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np

# Offscreen rendering must be configured before any renderer import.
os.environ.setdefault("MUJOCO_GL", "egl")

# Make the repo importable when run directly from a checkout.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"


def _ok(msg: str) -> None:
    print(f"{GREEN}  PASS{RESET}  {msg}")


def _fail(msg: str) -> None:
    print(f"{RED}  FAIL{RESET}  {msg}")


def _info(msg: str) -> None:
    print(f"{YELLOW}  ..  {RESET}  {msg}")


def test_forbidden_guard() -> bool:
    """The FailSafe overlap guard must reject PickCube/PushCube/StackCube."""
    print("\n[1] Forbidden-task guard (FailSafe overlap)")
    from robometer_policy_learning.envs.maniskill_utils import assert_task_allowed

    passed = True
    for bad in ("PickCube-v1", "PushCube-v1", "StackCube-v1"):
        try:
            assert_task_allowed(bad)
            _fail(f"{bad} was allowed but is in-distribution via FailSafe")
            passed = False
        except ValueError:
            _ok(f"{bad} correctly rejected")
    for good in ("PullCube-v1", "PokeCube-v1", "PickSingleYCB-v1"):
        try:
            assert_task_allowed(good)
            _ok(f"{good} correctly allowed")
        except ValueError as exc:
            _fail(f"{good} was rejected: {exc}")
            passed = False
    return passed


def test_single_env(task: str, image_size: int, save_frames: str | None) -> bool:
    """One wrapped env must look like a plain unbatched numpy gym env."""
    print(f"\n[2] Single wrapped env: {task}")
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  (registers env ids)

    from robometer_policy_learning.envs.maniskill_utils import get_task_spec
    from robometer_policy_learning.envs.maniskill_wrapper import ManiSkillSingleEnvWrapper

    spec = get_task_spec(task)
    _info(f"instruction: {spec.instruction!r}")
    _info(f"control_mode={spec.control_mode}  max_episode_steps={spec.max_episode_steps}")

    raw = gym.make(
        task,
        num_envs=1,
        obs_mode="state",
        control_mode=spec.control_mode,
        render_mode="rgb_array",
        reward_mode="sparse",
        sim_backend="physx_cpu",
        max_episode_steps=spec.max_episode_steps,
    )
    env = ManiSkillSingleEnvWrapper(raw, image_size=image_size)

    passed = True
    obs, info = env.reset(seed=0)

    # observation structure
    if set(obs.keys()) != {"state", "image"}:
        _fail(f"obs keys are {sorted(obs.keys())}, expected ['image', 'state']")
        passed = False
    else:
        _ok("obs keys = ['image', 'state']")

    state, image = obs["state"], obs["image"]
    if not isinstance(state, np.ndarray) or state.ndim != 1:
        _fail(f"state must be a 1-D numpy array, got {type(state)} shape={getattr(state, 'shape', None)}")
        passed = False
    else:
        _ok(f"state is unbatched numpy, shape={state.shape}, dtype={state.dtype}")

    if not isinstance(image, np.ndarray) or image.shape != (image_size, image_size, 3):
        _fail(f"image must be ({image_size},{image_size},3) numpy, got {getattr(image, 'shape', type(image))}")
        passed = False
    elif image.dtype != np.uint8:
        _fail(f"image dtype must be uint8, got {image.dtype}")
        passed = False
    else:
        _ok(f"image is HWC uint8 {image.shape}")

    # A frame that is entirely one colour usually means the renderer produced
    # nothing useful (bad EGL/driver setup) -- worth catching loudly.
    if isinstance(image, np.ndarray) and image.size and image.std() < 1.0:
        _fail(f"rendered frame is nearly uniform (std={image.std():.3f}) -- check offscreen rendering")
        passed = False
    elif isinstance(image, np.ndarray) and image.size:
        _ok(f"rendered frame has real content (std={image.std():.1f})")

    # action space must be unbatched so SyncVectorEnv can drive it
    if len(env.action_space.shape) != 1:
        _fail(f"action_space should be 1-D for SyncVectorEnv, got shape {env.action_space.shape}")
        passed = False
    else:
        _ok(f"action_space is unbatched, shape={env.action_space.shape}")

    # stepping
    try:
        for _ in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        _ok("step() ran 3 times")
    except Exception as exc:
        _fail(f"step() raised: {exc}")
        traceback.print_exc()
        return False

    for name, value, want in (
        ("reward", reward, float),
        ("terminated", terminated, bool),
        ("truncated", truncated, bool),
    ):
        if not isinstance(value, want):
            _fail(f"{name} must be a python {want.__name__}, got {type(value).__name__}")
            passed = False
        else:
            _ok(f"{name} is a python {want.__name__} ({value})")

    # This is the one that silently corrupts success metrics if wrong.
    if "success" in info:
        if isinstance(info["success"], bool):
            _ok(f"info['success'] is a python bool ({info['success']})")
        else:
            _fail(f"info['success'] is {type(info['success']).__name__}, must be bool")
            passed = False
    else:
        _info("info has no 'success' key at this step (may appear only on termination)")

    if save_frames:
        os.makedirs(save_frames, exist_ok=True)
        path = os.path.join(save_frames, f"{task}_frame.png")
        try:
            from PIL import Image

            Image.fromarray(obs["image"]).save(path)
            _info(f"saved frame -> {path}  (check it is right-side up and the scene is visible)")
        except Exception as exc:
            _info(f"could not save frame: {exc}")

    env.close()
    return passed


def test_vector_env(task: str, num_envs: int, image_size: int) -> bool:
    """N envs under SyncVectorEnv must batch cleanly."""
    print(f"\n[3] SyncVectorEnv with num_envs={num_envs}")
    from robometer_policy_learning.envs.maniskill_wrapper import make_maniskill_env

    try:
        env, instruction = make_maniskill_env(
            env_id=task,
            num_envs=num_envs,
            image_size=image_size,
            seed=0,
            sentence_model=None,  # skip language: avoids importing robometer/sentence-transformers
        )
    except Exception as exc:
        _fail(f"make_maniskill_env raised: {exc}")
        traceback.print_exc()
        return False

    passed = True
    obs, _ = env.reset(seed=0)
    if obs["image"].shape != (num_envs, image_size, image_size, 3):
        _fail(f"batched image shape is {obs['image'].shape}, expected {(num_envs, image_size, image_size, 3)}")
        passed = False
    else:
        _ok(f"batched image shape {obs['image'].shape}")

    if obs["state"].shape[0] != num_envs:
        _fail(f"batched state leading dim is {obs['state'].shape[0]}, expected {num_envs}")
        passed = False
    else:
        _ok(f"batched state shape {obs['state'].shape}")

    try:
        actions = np.stack([env.single_action_space.sample() for _ in range(num_envs)])
        obs, rewards, term, trunc, _ = env.step(actions)
        _ok(f"vector step() ok; rewards shape={np.asarray(rewards).shape}")
    except Exception as exc:
        _fail(f"vector step() raised: {exc}")
        traceback.print_exc()
        passed = False

    env.close()
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="PullCube-v1", help="ManiSkill task id")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--save-frames", default=None, help="directory to write a sample frame to")
    args = parser.parse_args()

    print("=" * 72)
    print(f"ManiSkill adapter smoke test  --  task={args.task}")
    print("=" * 72)

    results = {}
    results["forbidden guard"] = test_forbidden_guard()

    try:
        import mani_skill  # noqa: F401
    except ImportError:
        print(f"\n{RED}mani_skill is not installed{RESET} -- only the guard test could run.")
        print("Install with:  pip install 'mani-skill>=3.0.0'")
        return 1

    results["single env"] = test_single_env(args.task, args.image_size, args.save_frames)
    results["vector env"] = test_vector_env(args.task, args.num_envs, args.image_size)

    print("\n" + "=" * 72)
    for name, ok in results.items():
        print(f"  {GREEN + 'PASS' + RESET if ok else RED + 'FAIL' + RESET}  {name}")
    print("=" * 72)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
