# Master Thesis: failure-aware VLM reward models for robot RL

Research code behind **RoboRef**, a fine-tune of Robometer trained on densely labeled robot failures with an asymmetric loss, and its evaluation offline and in downstream RL.

## Repository map

| Folder | What it contains | Conda env |
|---|---|---|
| [`Robometer/`](Robometer/) | The core reward-model library: upstream Robometer plus our changes. RoboRef's asymmetric loss lives in `robometer/trainers/rbm_heads_trainer.py`, its setting in `robometer/configs/experiment_configs.py`. Everything else imports this. | `robometer_gpu_fa2` |
| [`Robometer-FT/`](Robometer-FT/) | Configs and SLURM jobs that train inside `Robometer/`. `run2_noicl_ours` is RoboRef, `run3_noicl_standard` the symmetric control, `run2_noicl_ours_8f` the 8-frame ablation, `run1_icl_ours` the in-context variant. | `robometer_gpu_fa2` |
| [`Qwen35-FT/`](Qwen35-FT/) | A self-contained Robometer fork for Qwen3.5 backbones. | `robometer_qwen35_gpu` |
| [`simulated-data-generation/`](simulated-data-generation/) | Scripted failure generators with labels read from simulator state: `MetaWorld/` and `Failsafe/` (ManiSkill). | `metaworld`, `failsafe` |
| [`real-world-data-generation/`](real-world-data-generation/) | The two-stage VLM+LLM labeling pipeline (Qwen3.5-35B-A3B describes frames, Qwen3-32B grades them): `Droid-Failures/` for DROID, `Robometer/` for relabeling RBM-1M failures. | `qwen_host` |
| [`RoboReward/`](RoboReward/) | Work built on RoboReward: `FPS/` (length-bias study), `Robo-Reward/` (RoboReward dataset relabeling), `roborewardbench/` (RoboRewardBench evaluation). | `roboreward` |
| [`vlm_ibrl/`](vlm_ibrl/) | Image-based IBRL with VLM rewards: MetaWorld and Robomimic sparse-reward RL. See [`README_METAWORLD.md`](vlm_ibrl/README_METAWORLD.md). | `demo2reward` (+ `source set_env.sh`) |
| [`robometer-policy-learning/`](robometer-policy-learning/) | SAC with dense VLM rewards on ManiSkill, plus real-robot (DROID) tooling. | `maniskill_rl` |
| [`post-training-analysis/`](post-training-analysis/) | `Robometer-LoRA/` (loss study, and `scripts/build_val_splits.py`, which builds the offline evaluation splits) and `reward-model-study/` (success-threshold calibration, diagnostics). | `robometer_gpu` |
| [`jobs/`](jobs/) | Repo-wide utility jobs. | |

Each project has its own README with details. Retired one-off scripts sit in an `archive/` folder inside their project.

## Where the paper results come from

| Result | Code |
|---|---|
| Simulated failures and their labels | `simulated-data-generation/` |
| Real-world failure labels (VLM+LLM pipeline) | `real-world-data-generation/` |
| Offline evaluation splits | `post-training-analysis/Robometer-LoRA/scripts/build_val_splits.py` |
| Training RoboRef and the symmetric control | `Robometer-FT/` jobs, running inside `Robometer/` |
| Frame-budget ablation | `Robometer-FT/configs/run2_noicl_ours_8f.yaml` |
| RoboRewardBench | `RoboReward/roborewardbench/` |
| Dense-reward RL on ManiSkill | `robometer-policy-learning/jobs/snellius_maniskill_sac.job` |
| Sparse-reward RL on MetaWorld and Robomimic | `vlm_ibrl/jobs/` |
| Success-threshold calibration | `post-training-analysis/reward-model-study/scripts/calibrate_threshold*.py`, `vlm_ibrl/jobs/diag_causal_calib.py` |
| Length-bias study | `RoboReward/FPS/` |

## Data and checkpoints

Not in git. On Snellius:

- Training corpus: `/projects/prjs1958/robometer_frame_dataset/`
- Evaluation splits: `/projects/prjs1958/robometer_frames_splits_full/`
- Checkpoints: `/projects/prjs1958/Robometer_FT_consolidated/` (`run2_noicl_ours_step4000` is RoboRef, `run3_noicl_standard_step5000` the symmetric control)
- Hugging Face cache used by the jobs: `/scratch-shared/$USER/hf_cache`

## Clusters

Jobs exist for Snellius (`snellius_*`), DAS-5 and the Toyota cluster. The latter two use absolute paths, so check paths and environment names before submitting on a different machine.
