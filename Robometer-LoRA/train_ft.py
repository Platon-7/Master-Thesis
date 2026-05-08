"""Robometer-LoRA entrypoint with the tar-shard data loader (HF + MP4 stack
bypassed). Mirrors Robometer-FT/train_ft.py and Qwen35-FT/train_ft.py.

Why this is here even though Robometer-LoRA's bakeoff_run.job historically
launched upstream's train.py directly: we patch setup_utils.setup_dataset
BEFORE upstream's @hydra.main fires, so accelerate-launched workers see
the patched function for both train and eval datasets. The wrapper also
declares its own @hydra.main with config_path="upstream_configs" (a symlink
to Robometer/robometer/configs/) so Hydra's relative resolution works without
having to cd into upstream Robometer/ at job launch.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOMETER_DIR = os.environ.get(
    "ROBOMETER_DIR", os.path.join(os.path.dirname(HERE), "Robometer")
)
for p in (HERE, ROBOMETER_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Patch setup_dataset BEFORE importing upstream `train`.
# ---------------------------------------------------------------------------

from robometer.utils import setup_utils as _setup_utils
from robometer.data.datasets.repeated_dataset import RepeatedDataset

from robometer_ft_data.tar_dataset import TarKeyframeRBMDataset


def _setup_dataset_tar_aware(cfg, is_eval=False, sampler_kwargs=None, **kwargs):
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
# Replicate upstream `main()` body, with our own @hydra.main pointing at the
# upstream_configs symlink so config_path resolves relative to THIS file.
# ---------------------------------------------------------------------------

import hydra
from omegaconf import DictConfig
from rich import print as rprint
from rich.panel import Panel

import train as _upstream_train_mod  # noqa: F401  (registers ConfigStore)

from robometer.configs.experiment_configs import ExperimentConfig
from robometer.utils.config_utils import convert_hydra_to_dataclass
from robometer.utils.distributed import is_rank_0


def banner(msg: str) -> None:
    if is_rank_0():
        rprint(Panel.fit(msg, style="bold green"))


@hydra.main(version_base=None, config_path="upstream_configs", config_name="config")
def main(cfg: DictConfig) -> None:
    banner("Starting Robometer-LoRA Training (tar-loader)")
    exp_cfg = convert_hydra_to_dataclass(cfg, ExperimentConfig)

    if exp_cfg.mode in ("train", "evaluate", "debug_corn"):
        _upstream_train_mod.train(exp_cfg)
    else:
        raise ValueError(f"Unknown mode: {exp_cfg.mode}")


if __name__ == "__main__":
    main()
