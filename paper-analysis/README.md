# Paper analysis

Scripts that turn experiment outputs into the paper's figures and tables. They read data from `/projects/prjs1958/` and `/scratch-shared/`, which is not in git, and write figure sources for the Overleaf project.

| Folder | Produces |
|---|---|
| `reward-hacking/` | The on-policy false-positive analysis on ManiSkill (Table 3): per-step AUROC and FP rate from the training episodes. |
| `maniskill-figure/` | The dense-reward ManiSkill figure: stitches the Robo-Dopamine runs and draws the curves. |
| `metaworld-figure/` | Converts the MetaWorld bands from standard deviation to standard error. |
| `robomimic-figure/` | The Robomimic curves. `build_robomimic_panel.py` writes RoboRef's mean and standard-error band over the chosen seeds into the sparse-reward figure. |
| `frame-budget/` | The 8-versus-16-frame ablation (Table 7): pulls runs from W&B and computes the metrics. |
| `real-world-calibration/` | Success-threshold calibration on recorded real-robot episodes: scores every prefix with a reward model, then sweeps thresholds. |
