#!/usr/bin/env python3
"""One command that answers: is the ManiSkill + RoboRef environment ready?

Runs every check that must pass before a training arm is launched, in the same
import conditions the training job uses, and prints a single verdict. Written
because these failures are individually silent -- a shadowed package, a missing
demo bank, or a stale `robometer` all produce confusing errors deep inside a
run rather than at startup.

Checks:
  1. Core packages import (mani_skill, gymnasium, torch, transformers).
  2. `robometer` resolves to the RoboRef FORK, not the stale in-repo directory.
     The repo contains `robometer/` with no __init__.py, which Python treats as
     a namespace package and which shadows the real install whenever the repo
     root lands on sys.path. Caught here rather than mid-rollout.
  3. `ProgressSample.context_trajectory` exists -> in-context demos (run1) can
     actually be attached.
  4. `robometer_policy_learning` imports (so train.py needs no path hacks).
  5. ManiSkill env constructs, renders a non-degenerate frame, and reports
     python-typed step results.
  6. ICL demo banks are present and well-formed, per task.

Usage:
    python scripts/verify_maniskill_env.py
    python scripts/verify_maniskill_env.py --task PullCube-v1 --require-icl
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"
_results: dict[str, bool] = {}


def _ok(msg): print(f"{GREEN}  PASS{RESET}  {msg}")
def _fail(msg): print(f"{RED}  FAIL{RESET}  {msg}")
def _warn(msg): print(f"{YELLOW}  WARN{RESET}  {msg}")


def check_core() -> bool:
    print("\n[1] Core packages")
    ok = True
    # moviepy is in this list because the logger only demands it at the first
    # eval interval -- minutes into a run, after the reward model has loaded --
    # and then kills the job outright rather than degrading to no video.
    for mod in ("mani_skill", "gymnasium", "torch", "numpy", "transformers", "moviepy"):
        try:
            m = __import__(mod)
            _ok(f"{mod} {getattr(m, '__version__', '?')}")
        except Exception as exc:
            _fail(f"{mod}: {exc}")
            ok = False
    return ok


def check_robometer_not_shadowed() -> bool:
    """The single most dangerous failure: the stale in-repo dir winning."""
    print("\n[2] `robometer` resolves to the RoboRef fork (not the stale in-repo dir)")
    try:
        import robometer
    except Exception as exc:
        _fail(f"cannot import robometer: {exc}")
        return False

    path = getattr(robometer, "__file__", None)
    if path is None:
        _fail(
            "robometer imported as a NAMESPACE package (__file__ is None). The stale "
            "in-repo `robometer/` directory is shadowing the real install.\n"
            "        Fix: install the policy-learning package so the repo root is not needed\n"
            "        on sys.path:  pip install -e . --no-deps"
        )
        return False

    try:
        from robometer.data.dataset_types import ProgressSample  # noqa: F401
    except Exception as exc:
        _fail(f"robometer.data unavailable ({exc}) -- likely the stale in-repo copy at {path}")
        return False

    stale_marker = os.path.join("robometer-policy-learning", "robometer")
    if stale_marker in os.path.abspath(path):
        _fail(f"robometer resolved to the STALE in-repo copy: {path}")
        return False

    _ok(f"robometer -> {path}")
    return True


def check_icl_supported() -> bool:
    print("\n[3] In-context demonstrations supported (run1 / RoboRef-ICL)")
    try:
        from robometer.data.dataset_types import ProgressSample

        fields = getattr(ProgressSample, "model_fields", {})
        if "context_trajectory" in fields:
            _ok("ProgressSample.context_trajectory present -> ICL available")
            return True
        _fail(
            "ProgressSample has no `context_trajectory`: this robometer predates ICL "
            "(the upstream submodule). Install the fork:\n"
            "        pip install -e /path/to/Master-Thesis/Robometer --no-deps"
        )
        return False
    except Exception as exc:
        _fail(f"{exc}")
        return False


def check_policy_learning() -> bool:
    print("\n[4] robometer_policy_learning importable")
    try:
        import robometer_policy_learning

        _ok(f"robometer_policy_learning -> {robometer_policy_learning.__file__}")
        return True
    except Exception as exc:
        _fail(f"{exc}  (fix: pip install -e . --no-deps)")
        return False


def check_env(task: str) -> bool:
    print(f"\n[5] ManiSkill env end-to-end: {task}")
    try:
        import numpy as np
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401

        from robometer_policy_learning.envs.maniskill_utils import get_task_spec
        from robometer_policy_learning.envs.maniskill_wrapper import ManiSkillSingleEnvWrapper

        spec = get_task_spec(task)
        env = ManiSkillSingleEnvWrapper(
            gym.make(
                task, num_envs=1, obs_mode="state", control_mode=spec.control_mode,
                render_mode="rgb_array", reward_mode="sparse", sim_backend="physx_cpu",
                max_episode_steps=spec.max_episode_steps,
            ),
            image_size=224,
        )
        obs, _ = env.reset(seed=0)
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        checks = [
            (obs["image"].shape == (224, 224, 3), f"image {obs['image'].shape}"),
            (obs["image"].std() > 1.0, f"frame non-degenerate (std={obs['image'].std():.1f})"),
            (isinstance(reward, float), f"reward is float ({reward})"),
            (isinstance(info.get("success", False), bool), "info['success'] is bool"),
        ]
        env.close()
        ok = True
        for passed, msg in checks:
            (_ok if passed else _fail)(msg)
            ok = ok and passed
        return ok
    except Exception as exc:
        _fail(f"{exc}")
        traceback.print_exc()
        return False


def check_vectorized_rollout(task: str) -> bool:
    """Step the VECTORIZED env the way RolloutWorker does.

    check_env() builds a bare single env, which is not the object training sees.
    Two failures lived in exactly that gap and only surfaced after the reward
    model had loaded, minutes into a job:

      * SyncVectorEnv returns infos as a dict of batched arrays, which the worker
        handed to every sub-env unsliced -> "truth value of an array ... is
        ambiguous" on the first episode end.
      * RobometerRolloutWorker calls env.get_language_instruction(), which
        nothing attached when no sentence model was configured.

    So this steps the real vectorized env through the real helpers.
    """
    print(f"\n[7] Vectorized env as RolloutWorker consumes it: {task}")
    try:
        import numpy as np

        from robometer_policy_learning.utils.env_utils import make_env
        from robometer_policy_learning.rollouts.rollout_worker import extract_info_for_env, EpisodeTracker

        env, eval_env = make_env(
            env_name=f"maniskill/{task}",
            num_envs=1,
            chunk_size=None,
            env_kwargs={"sim_backend": "physx_cpu", "image_size": 224},
        )
        ok = True

        instr = env.get_language_instruction()
        (_ok if isinstance(instr, str) and instr else _fail)(f"get_language_instruction() -> {instr!r}")
        ok = ok and isinstance(instr, str) and bool(instr)

        env.reset(seed=0)
        tracker = EpisodeTracker(1)
        for _ in range(5):
            _, _, term, trunc, infos = env.step(np.stack([env.single_action_space.sample()]))
            info_i = extract_info_for_env(infos, 0, 1)
            try:
                succ = tracker.is_success(info_i)
            except Exception as exc:
                _fail(f"is_success on sliced info raised: {exc}")
                ok = False
                break
            if not isinstance(succ, (bool, np.bool_)):
                _fail(f"is_success returned {type(succ).__name__}, expected bool")
                ok = False
                break
        else:
            _ok("5 vector steps; per-env info slices to scalars, is_success returns bool")
        env.close()
        if eval_env is not None and eval_env is not env:
            eval_env.close()
        return ok
    except Exception as exc:
        _fail(f"{exc}")
        traceback.print_exc()
        return False


def check_icl_banks(asset_dir: str, task: str, require: bool) -> bool:
    print("\n[6] ICL demonstration banks")
    import numpy as np

    d = os.path.join(asset_dir, "icl_demos")
    banks = sorted(glob.glob(os.path.join(d, "*.npz")))
    if not banks:
        (_fail if require else _warn)(f"no banks in {d} (build: scripts/generate_maniskill_icl_demos.py)")
        return not require

    ok = True
    for p in banks:
        try:
            z = np.load(p, allow_pickle=True)
            f = z["frames"]
            good = f.ndim == 5 and f.dtype == np.uint8 and f.std() > 1.0
            (_ok if good else _fail)(f"{os.path.basename(p):24s} {f.shape} std={f.std():.1f}")
            ok = ok and good
        except Exception as exc:
            _fail(f"{os.path.basename(p)}: {exc}")
            ok = False

    if require and not os.path.exists(os.path.join(d, f"{task}.npz")):
        _fail(f"no bank for the requested task {task}")
        ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="PullCube-v1")
    ap.add_argument("--require-icl", action="store_true", help="fail if the ICL bank for --task is missing")
    ap.add_argument("--asset-dir", default=os.environ.get("MS_ASSET_DIR", os.path.expanduser("~/.maniskill")))
    ap.add_argument("--skip-env", action="store_true", help="skip the slow simulator check")
    args = ap.parse_args()

    print("=" * 72)
    print(f"ManiSkill + RoboRef environment verification   (task={args.task})")
    print(f"python: {sys.executable}")
    print("=" * 72)

    _results["core packages"] = check_core()
    _results["robometer not shadowed"] = check_robometer_not_shadowed()
    _results["ICL supported"] = check_icl_supported()
    _results["policy learning importable"] = check_policy_learning()
    if not args.skip_env:
        _results["maniskill env"] = check_env(args.task)
        _results["vectorized rollout path"] = check_vectorized_rollout(args.task)
    _results["ICL banks"] = check_icl_banks(args.asset_dir, args.task, args.require_icl)

    print("\n" + "=" * 72)
    for name, ok in _results.items():
        print(f"  {GREEN + 'PASS' + RESET if ok else RED + 'FAIL' + RESET}  {name}")
    all_ok = all(_results.values())
    print("=" * 72)
    print(f"{GREEN}ENVIRONMENT READY{RESET}" if all_ok else f"{RED}ENVIRONMENT NOT READY{RESET}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
