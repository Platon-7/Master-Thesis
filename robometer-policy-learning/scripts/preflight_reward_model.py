#!/usr/bin/env python3
"""Preflight: can we compose the arm configs and actually load a RoboRef checkpoint?

The six ManiSkill arms previously died ~4 minutes in, after a full (passing)
environment verification, on two things this script front-loads:

  1. Hydra override parsing for the `gt` arm (`reward_model=null` is not a valid
     config-group override).
  2. `load_model_from_hf` replaying the checkpoint's baked-in `use_unsloth: true`
     in an environment with no unsloth.

Both are cheap to test and expensive to discover inside a training job, so this
runs first. Exercises the real code paths -- the same Hydra composition and the
same loader train.py uses -- not stand-ins.

    python scripts/preflight_reward_model.py --ckpt /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")

GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
_results: dict[str, bool] = {}


def _ok(m): print(f"{GREEN}  PASS{RESET}  {m}", flush=True)
def _fail(m): print(f"{RED}  FAIL{RESET}  {m}", flush=True)


def check_hydra_arms(task: str) -> bool:
    """Compose every arm's overrides exactly as the job script passes them."""
    print("\n[1] Hydra composition of each arm's overrides", flush=True)
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    cfg_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "robometer_policy_learning",
        "configs",
    )
    # Exactly the common overrides the job script passes -- nothing extra. An
    # override for a key the config does not define is itself a composition
    # error, so inventing one here would test the harness, not the job.
    common = [f"env.env_name=maniskill/{task}"]
    arms = {
        "gt": ["env.use_gt_rewards=true", "env.env_kwargs.reward_mode=normalized_dense"],
        "succ": [
            "reward_model=robometer",
            "reward_model.use_success_detection=true",
            "reward_model.add_estimated_reward=false",
            "env.use_gt_rewards=false",
            "env.env_kwargs.reward_mode=sparse",
        ],
        "prog": [
            "reward_model=robometer",
            "reward_model.use_success_detection=false",
            "reward_model.add_estimated_reward=true",
            "env.use_gt_rewards=false",
            "env.env_kwargs.reward_mode=sparse",
        ],
        "icl": [
            "reward_model=robometer",
            "reward_model.use_success_detection=false",
            "reward_model.add_estimated_reward=true",
            "reward_model.icl_demo_path=/dev/null/bank.npz",
            "env.use_gt_rewards=false",
            "env.env_kwargs.reward_mode=sparse",
        ],
    }

    ok = True
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        for arm, overrides in arms.items():
            try:
                cfg = compose(config_name="maniskill_online_rl", overrides=common + overrides)
                rm = OmegaConf.select(cfg, "reward_model")
                expect_rm = arm != "gt"
                got_rm = rm is not None
                if got_rm != expect_rm:
                    _fail(f"{arm}: reward_model present={got_rm}, expected {expect_rm}")
                    ok = False
                    continue
                # icl must actually carry the demo path through to the buffer
                if arm == "icl" and OmegaConf.select(cfg, "reward_model.icl_demo_path") is None:
                    _fail("icl: reward_model.icl_demo_path did not survive composition")
                    ok = False
                    continue
                _ok(f"{arm}: composed (reward_model={'set' if got_rm else 'null'})")
            except Exception as exc:
                _fail(f"{arm}: {type(exc).__name__}: {exc}")
                ok = False
    return ok


def check_reward_model_loads(ckpt: str) -> bool:
    """Load the checkpoint through the exact function training_utils calls."""
    print(f"\n[2] load_model_from_hf on {ckpt}", flush=True)
    try:
        import torch
        from robometer.utils.save import load_model_from_hf

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t = time.time()
        exp_cfg, tokenizer, processor, model = load_model_from_hf(model_path=ckpt, device=device)
        _ok(f"loaded in {time.time() - t:.0f}s on {device}")

        if model is None:
            _fail("loader returned model=None")
            return False
        n_params = sum(p.numel() for p in model.parameters())
        if n_params == 0:
            _fail("model has zero parameters")
            return False
        _ok(f"{n_params / 1e9:.2f}B parameters")

        max_frames = getattr(getattr(exp_cfg, "data", None), "max_frames", None)
        _ok(f"checkpoint max_frames={max_frames}")
        for name in ("progress", "success"):
            _ok(f"head '{name}': {'present' if hasattr(model, f'{name}_head') else 'ABSENT'}")
        return True
    except Exception as exc:
        _fail(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PullCube-v1")
    ap.add_argument("--ckpt", default=None, help="checkpoint to load; skip stage 2 if omitted")
    args = ap.parse_args()

    print("=" * 72)
    print(f"ManiSkill arm preflight   (task={args.task})")
    print(f"python: {sys.executable}")
    print(f"ROBOMETER_DISABLE_UNSLOTH={os.environ.get('ROBOMETER_DISABLE_UNSLOTH', '<unset>')}")
    print("=" * 72, flush=True)

    _results["hydra arm composition"] = check_hydra_arms(args.task)
    if args.ckpt:
        _results["reward model loads"] = check_reward_model_loads(args.ckpt)

    print("\n" + "=" * 72)
    for name, ok in _results.items():
        print(f"  {(GREEN + 'PASS' + RESET) if ok else (RED + 'FAIL' + RESET)}  {name}")
    all_ok = all(_results.values())
    print("=" * 72)
    print(f"{GREEN}PREFLIGHT OK{RESET}" if all_ok else f"{RED}PREFLIGHT FAILED{RESET}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
