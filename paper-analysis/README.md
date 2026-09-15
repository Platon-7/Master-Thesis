# Paper analysis

Scripts that turn experiment outputs into the paper's figures and tables. They read data from `/projects/prjs1958/` and `/scratch-shared/`, which is not in git, and write figure sources for the Overleaf project.

| Folder | Produces |
|---|---|
| `reward-hacking/` | On-policy reward hacking on ManiSkill. `goodhart_steps.py` reproduces Table 3's FP column and draws it over training; `goodhart_curves.py` is the per-episode-peak version plus the per-seed scatter; `onpolicy_fp.py` pools per-episode peaks. Set `ROBOREF_RUNS` to the extracted runs. |
| `maniskill-figure/` | The dense-reward ManiSkill figure: stitches the Robo-Dopamine runs and draws the curves. |
| `metaworld-figure/` | Converts the MetaWorld bands from standard deviation to standard error. |
| `robomimic-figure/` | The Robomimic curves. `build_robomimic_panel.py` writes RoboRef's mean and standard-error band over the chosen seeds into the sparse-reward figure. |
| `frame-budget/` | The 8-versus-16-frame ablation (Table 7): pulls runs from W&B and computes the metrics. |
| `real-world-calibration/` | Success-threshold calibration on recorded real-robot episodes: scores every prefix with a reward model, then sweeps thresholds. |
