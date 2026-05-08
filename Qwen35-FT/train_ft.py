"""Qwen35-FT entrypoint with the tar-shard data loader (HF + MP4 stack bypassed).

Mirrors the pattern in Robometer-FT/train_ft.py: monkey-patch
`setup_utils.setup_dataset` to dispatch to TarKeyframeRBMDataset BEFORE
the upstream `train` module is imported, so that the @hydra.main-decorated
train.main() (and its accelerate-launched worker copies) sees the patched
function for both train and eval datasets.

Why a separate wrapper instead of editing train.py:
  * train.py's @hydra.main runs at module import; we need the patch active
    before that, otherwise eval-dataset construction (line ~239) misses it.
  * Keeps the vendored copy (Qwen35-FT/robometer/) untouched so we don't
    diverge it further from upstream Robometer/.

The Hydra config_path resolves to "robometer/configs" relative to *this file*'s
location (Qwen35-FT/train_ft.py → Qwen35-FT/robometer/configs), which is
itself a symlink to upstream Robometer/robometer/configs. Same upstream
schema as Robometer-FT.
"""
from __future__ import annotations

import os
import sys

# Ensure Qwen35-FT/ is importable so `import robometer_ft_data` resolves
# (the directory containing this file).
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


# ---------------------------------------------------------------------------
# Patch setup_dataset BEFORE importing the upstream `train` module.
# ---------------------------------------------------------------------------

from robometer.utils import setup_utils as _setup_utils
from robometer.data.datasets.repeated_dataset import RepeatedDataset

from robometer_ft_data.tar_dataset import TarKeyframeRBMDataset


def _setup_dataset_tar_aware(cfg, is_eval=False, sampler_kwargs=None, **kwargs):
    """Always builds TarKeyframeRBMDataset. Source paths come from env vars
    (TAR_DATASET_ROOT etc) because upstream's DataConfig is a strict dataclass
    that rejects unknown fields. The whole point of train_ft.py is to use the
    tar loader — there's no fall-through."""
    if sampler_kwargs is None:
        sampler_kwargs = {}
    sampler_kwargs["random_seed"] = cfg.seed
    kwargs["sampler_kwargs"] = sampler_kwargs

    print(
        f"[train_ft] tar-loader: is_eval={is_eval}, "
        f"TAR_DATASET_ROOT={os.environ.get('TAR_DATASET_ROOT', '<default>')}"
    )
    dataset = TarKeyframeRBMDataset(config=cfg, is_evaluation=is_eval, **kwargs)
    if not is_eval:
        dataset = RepeatedDataset(dataset)
    return dataset


_setup_utils.setup_dataset = _setup_dataset_tar_aware


# ---------------------------------------------------------------------------
# Replicate upstream train.main() body, but with our own @hydra.main so that
# config_path resolves relative to THIS file (Qwen35-FT/train_ft.py).
# ---------------------------------------------------------------------------

import hydra
from omegaconf import DictConfig
from rich import print as rprint
from rich.panel import Panel

# Importing `train` triggers ConfigStore.store(...) registrations at module
# top-level for base_config / model_config / etc. Lazy-importing inside @hydra.main
# would cause "Could not load 'base_config'" at decoration time.
import train as _upstream_train_mod  # noqa: F401

from robometer.configs.experiment_configs import ExperimentConfig
from robometer.utils.config_utils import convert_hydra_to_dataclass
from robometer.utils.distributed import is_rank_0


def banner(msg: str) -> None:
    if is_rank_0():
        rprint(Panel.fit(msg, style="bold green"))


@hydra.main(version_base=None, config_path="robometer/configs", config_name="config")
def main(cfg: DictConfig) -> None:
    banner("Starting Qwen35-FT Training (tar-loader)")
    exp_cfg = convert_hydra_to_dataclass(cfg, ExperimentConfig)

    if exp_cfg.mode in ("train", "evaluate", "debug_corn"):
        _upstream_train_mod.train(exp_cfg)
    else:
        raise ValueError(f"Unknown mode: {exp_cfg.mode}")


if __name__ == "__main__":
    main()
