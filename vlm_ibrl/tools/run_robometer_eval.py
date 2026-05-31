"""Wrapper around ``robometer.evals.run_baseline_eval`` that disables
Unsloth + Flash-Attention via monkey-patch, then re-executes the
original Robometer script via ``runpy.run_path`` so Hydra's
``config_path="../configs"`` resolves correctly relative to
``run_baseline_eval.py``'s on-disk location.

Hydra args are forwarded verbatim via ``sys.argv``::

    python tools/run_robometer_eval.py \\
        reward_model=rbm \\
        model_path=robometer/Robometer-4B \\
        custom_eval.eval_types=[policy_ranking] \\
        custom_eval.policy_ranking=[rbm-1m-ood] \\
        max_frames=8 ...
"""

from __future__ import annotations

import os
import sys


def _install_shims() -> None:
    """Hide flash_attn; force use_unsloth=False at model-load time.

    Same shim as ``env/robometer_utils.py``. Robometer-4B's released
    ``config.yaml`` ships ``model.use_unsloth=True``, Unsloth requires
    torch 2.8 + cu128 which conflicts with this env's torch 2.4 + cu121.
    The non-Unsloth branch of ``setup_model_and_processor`` works fine
    once we flip the flag before the model is built.
    """
    sys.modules["flash_attn"] = None

    from robometer.models.rbm import RBM
    from robometer.utils import setup_utils

    if not hasattr(RBM, "all_tied_weights_keys"):
        RBM.all_tied_weights_keys = {}

    _orig_setup = setup_utils.setup_model_and_processor

    def _setup_patched(model_cfg, *args, **kwargs):
        model_cfg.use_unsloth = False
        return _orig_setup(model_cfg, *args, **kwargs)

    setup_utils.setup_model_and_processor = _setup_patched


def _resolve_run_baseline_eval_path() -> str:
    """Find Robometer's run_baseline_eval.py on disk via the package import."""
    import importlib.util

    spec = importlib.util.find_spec("robometer.evals.run_baseline_eval")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Cannot locate robometer.evals.run_baseline_eval — is Robometer on PYTHONPATH?"
        )
    return spec.origin


def main() -> int:
    _install_shims()

    script_path = _resolve_run_baseline_eval_path()
    print(f"[wrapper] shims installed; executing {script_path}", flush=True)

    # Replace our script name with the actual one so Hydra sees the right
    # location and ``config_path="../configs"`` resolves to
    # ``Robometer/robometer/configs/`` as the @hydra.main decorator expects.
    sys.argv[0] = script_path
    # Also cd into Robometer/ root so any relative outputs land where the
    # original recipe puts them.
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(script_path))))

    import runpy
    runpy.run_path(script_path, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
