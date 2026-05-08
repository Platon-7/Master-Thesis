"""Robometer-FT entrypoint — full FT of Robometer-4B with the tar-shard
data loader (HF + MP4 stack bypassed).

This file replicates the few-line body of upstream `Robometer/train.py:main()`
verbatim, but mounts its own `@hydra.main` against a local symlink to upstream
configs (`Robometer-FT/upstream_configs/`). This is necessary because importing
upstream `train.py` as a module from a wrapper makes Hydra resolve
`config_path="robometer/configs"` against THIS file's location instead of
upstream's, leading to "Primary config module 'robometer.configs' not found".
The local symlink + filesystem-style config_path sidesteps Hydra's namespace-
package-vs-init.py confusion entirely.

Before Hydra runs we monkey-patch `setup_dataset` to dispatch to
`TarKeyframeRBMDataset` when `data.use_tar_loader` is True. The patch is
applied at module top-level so worker processes (under accelerate launch)
see it before `train()` reads its dataset.
"""
from __future__ import annotations

import os
import sys

# Ensure both paths are importable when invoked under accelerate launch.
HERE = os.path.dirname(os.path.abspath(__file__))
ROBOMETER_DIR = os.environ.get(
    "ROBOMETER_DIR", os.path.join(os.path.dirname(HERE), "Robometer")
)
for p in (HERE, ROBOMETER_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Patch setup_dataset BEFORE upstream train.py imports it.
# ---------------------------------------------------------------------------

from robometer.utils import setup_utils as _setup_utils
from robometer.data.datasets.repeated_dataset import RepeatedDataset

from robometer_ft_data.tar_dataset import TarKeyframeRBMDataset

_original_setup_dataset = _setup_utils.setup_dataset


def _setup_dataset_tar_aware(cfg, is_eval=False, sampler_kwargs=None, **kwargs):
    """Always builds TarKeyframeRBMDataset. Source paths come from env vars
    (TAR_DATASET_ROOT etc) because upstream's DataConfig is a strict dataclass
    that rejects unknown fields. The whole point of train_ft.py is to use the
    tar loader — there's no fall-through."""
    if sampler_kwargs is None:
        sampler_kwargs = {}
    sampler_kwargs["random_seed"] = cfg.seed
    kwargs["sampler_kwargs"] = sampler_kwargs

    import os as _os
    print(
        f"[train_ft] tar-loader: is_eval={is_eval}, "
        f"TAR_DATASET_ROOT={_os.environ.get('TAR_DATASET_ROOT', '<default>')}"
    )
    dataset = TarKeyframeRBMDataset(config=cfg, is_evaluation=is_eval, **kwargs)
    if not is_eval:
        dataset = RepeatedDataset(dataset)
    return dataset


_setup_utils.setup_dataset = _setup_dataset_tar_aware


# ---------------------------------------------------------------------------
# Replicate upstream `main()` body, but with our own @hydra.main.
# ---------------------------------------------------------------------------

import hydra
from omegaconf import DictConfig
from rich import print as rprint
from rich.panel import Panel

# Import upstream `train` module at MODULE LOAD time. Its top-level body
# (lines 51-59) registers the structured-config schemas with Hydra's
# ConfigStore — `base_config`, `model_config`, etc — which Hydra needs in
# order to resolve `defaults: - base_config` from upstream's config.yaml.
# Lazy-importing this inside @hydra.main causes "Could not load 'base_config'".
import train as _upstream_train_mod  # noqa: F401  (side-effect: ConfigStore.register)

# These all live in upstream Robometer/ — safe to import after `train`.
from robometer.configs.experiment_configs import ExperimentConfig
from robometer.utils.config_utils import convert_hydra_to_dataclass
from robometer.utils.distributed import is_rank_0


def banner(msg: str) -> None:
    if is_rank_0():
        rprint(Panel.fit(msg, style="bold green"))


@hydra.main(version_base=None, config_path="upstream_configs", config_name="config")
def main(cfg: DictConfig) -> None:
    banner("Starting Robometer-FT Training (tar-loader)")
    exp_cfg = convert_hydra_to_dataclass(cfg, ExperimentConfig)

    if exp_cfg.mode in ("train", "evaluate"):
        _upstream_train_mod.train(exp_cfg)
    else:
        raise ValueError(f"Unknown mode: {exp_cfg.mode}")


if __name__ == "__main__":
    main()
