# RoboMeter Failure Scoring Pipeline — Instructions

## Overview

This document describes a two-stage VLM+LLM pipeline for assigning fine-grained progress labels (1–4) to failure episodes in the RoboMeter dataset. The labeled failures will be used to train a reward model as part of a master's thesis project on robot learning.

The same pipeline has already been applied to ~5,500 DROID failure episodes. This run targets the failure episodes contained in RoboMeter's other constituent datasets (RACER, LIBERO, RoboArena, and others), totalling approximately **46,000 confirmed failures** plus an unknown number from additional archives.

---

## Pipeline Description

For each failure episode, the pipeline proceeds in two stages:

**Stage 1 — Frame description (VLM: Qwen3.5-35B-A3B)**
Eight evenly-spaced keyframes are extracted from the episode. Each frame is passed to the VLM with a structured prompt requesting: (1) the state of task-relevant objects, (2) the current state of the robot end-effector, and (3) task completion status.

**Stage 2 — Progress grading (LLM: Qwen3-32B)**
The eight frame descriptions are passed to the LLM, which assigns a cumulative 1–4 progress score to the episode.

**Score definitions:**
| Score | Meaning |
|-------|---------|
| 1 | No progress — robot never approached the task object |
| 2 | Approach only — robot moved toward the object but did not engage |
| 3 | Partial progress — robot grasped or partially executed the task |
| 4 | Major progress — robot completed >50% of the task before failing |

Both stages are **fully resumable**: if a job is interrupted, re-running the same command resumes from the last completed episode.

---

## Hardware Requirements

| Resource | Requirement |
|----------|-------------|
| GPUs | 2× with ≥24 GB VRAM (A100-40GB recommended; A6000, RTX 3090/4090 also work) |
| RAM | ≥80 GB |
| Scratch storage | ~50 GB (extracted keyframes + intermediate files) |
| Model cache | ~70 GB (Qwen3.5-35B-A3B + Qwen3-32B, downloaded once on first run) |

Source archives are streamed directly from HuggingFace and do not need to be stored locally.

---

## Environment Setup

```bash
conda create -n robometer python=3.10 -y
conda activate robometer

pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.4.3 transformers>=4.40.0 accelerate huggingface_hub Pillow numpy tqdm requests
```

**Required environment variables:**
```bash
export HF_TOKEN=hf_xxxxxxxxxxxx          # HuggingFace access token
export HF_HOME=/scratch/hf_cache         # Model cache directory (~70 GB)
export DATA_DIR=/scratch/robometer_out   # Output directory
```

---

## Running the Pipeline

Each archive is processed in two sequential steps.

**Step 1 — Keyframe extraction** (CPU-only, minutes per archive)

Streams the archive from HuggingFace, identifies failure episodes via embedded quality labels, and saves 8 JPEG keyframes per episode.

```bash
python selective_download.py \
    --archive <archive_name> \
    --data-dir $DATA_DIR
```

**Step 2 — VLM + LLM scoring** (GPU, see time estimates)

```bash
python score_robometer_failures.py \
    --archive <archive_name> \
    --data-dir $DATA_DIR
```

Both steps can be run together via:
```bash
bash run.sh <archive_name>
```

For SLURM environments, a representative job script:
```bash
#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=2
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

conda activate robometer
export HF_TOKEN=hf_xxxxxxxxxxxx
export DATA_DIR=/scratch/robometer_out

bash run.sh jesbu1_racer_rfm_racer_train
```

---

## Archive List

The following archives should be processed. Archives are grouped by confidence in failure count.

### Group A — Confirmed failures, single-file archives (~36,000 failures, ~61 GB total)

Each archive can be run as a separate SLURM job. RACER train (23k failures, ~8 GPU-hours) should be submitted as a standalone job.

| Archive | Confirmed failures | Stream size |
|---------|-------------------|-------------|
| `jesbu1_racer_rfm_racer_train` | 23,391 | 14.0 GB |
| `jesbu1_racer_rfm_racer_val` | 5,820 | 3.6 GB |
| `jesbu1_auto_eval_rfm_auto_eval_rfm` | 3,721 | 30.8 GB |
| `ykorkmaz_libero_failure_rfm_libero_90_failure` | 4,312 | 7.5 GB |
| `ykorkmaz_libero_failure_rfm_libero_10_failure` | 498 | 0.9 GB |
| `ykorkmaz_libero_failure_rfm_libero_object_failure` | 489 | 1.0 GB |
| `ykorkmaz_libero_failure_rfm_libero_spatial_failure` | 486 | 0.8 GB |
| `ykorkmaz_libero_failure_rfm_libero_goal_failure` | 456 | 0.8 GB |
| `jesbu1_rfm_new_mit_franka_rfm_rfm_new_mit_franka_rfm` | 80 | 0.9 GB |
| `jesbu1_usc_koch_p_ranking_rfm_usc_koch_p_ranking_all` | 50 | 0.3 GB |
| `jesbu1_rfm_new_mit_franka_rfm_nowrist_rfm_new_mit_franka_rfm_nowrist` | 40 | 0.5 GB |
| `aliangdw_utd_so101_policy_ranking_utd_so101_policy_ranking` | 20 | 0.2 GB |
| `aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking` | 12 | 0.1 GB |
| `jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top` | 10 | 0.1 GB |
| `jesbu1_utd_so101_clean_policy_ranking_wrist_utd_so101_clean_policy_ranking_wrist` | 10 | 0.1 GB |
| `aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking` | 8 | 0.1 GB |
| `ykorkmaz_usc_trossen_rfm_usc_trossen` | 6 | 0.1 GB |
| `jesbu1_mit_franka_p-rank_rfm_mit_franka_p-rank_rfm` | 5 | <0.1 GB |

### Group B — Confirmed failures, split-part archive (~6,800 failures, ~37 GB)

The extraction script handles split archives automatically — pass the base name without any `.part-*` suffix.

| Archive | Confirmed failures | Stream size | Parts |
|---------|-------------------|-------------|-------|
| `jesbu1_roboarena_eval_debug_nowrist_roboarena_eval_debug_nowrist` | 6,757 | 36.9 GB | 4 |

### Group C — Unknown failure count (~661 GB total)

These archives contain frame data and quality labels, but the failure count could not be determined without a full stream. The extraction script identifies and saves only failure episodes; if an archive contains zero failures, it exits immediately.

| Archive | Stream size | Parts | Notes |
|---------|-------------|-------|-------|
| `jesbu1_failsafe_rfm_failsafe` | 90.9 GB | 3 | Archive name suggests failure content |
| `anqil_rh20t_subset_rfm_rh20t_robot` | 92.6 GB | 10 | Multi-embodiment robot dataset |
| `jesbu1_soar_rfm_soar_rfm` | 57.4 GB | 6 | Unknown failure rate |
| `jesbu1_roboarena_0825_rfm_roboarena` | 54.9 GB | 2 | Real-robot evaluation data |
| `jesbu1_galaxea_rfm_galaxea_part1_r1_lite` | 61.0 GB | 2 | Humanoid robot (Galaxea R1) |
| `jesbu1_galaxea_rfm_galaxea_part2_r1_lite` | 71.6 GB | 2 | Humanoid robot (Galaxea R1) |
| `jesbu1_galaxea_rfm_galaxea_part3_r1_lite` | 77.7 GB | 2 | Humanoid robot (Galaxea R1) |
| `jesbu1_galaxea_rfm_galaxea_part4_r1_lite` | 59.8 GB | 2 | Humanoid robot (Galaxea R1) |
| `jesbu1_galaxea_rfm_galaxea_part5_r1_lite` | 45.6 GB | 2 | Humanoid robot (Galaxea R1) |
| `jesbu1_molmoact_rfm_molmoact_dataset_household` | 49.0 GB | 2 | Simulated household tasks |

> **Note on Galaxea (humanoid robot):** The VLM prompt references "gripper" which does not apply to the Galaxea R1's human-like hands. If failures are found in these archives, the scores will still be generated (the VLM adapts to what it observes) but the descriptions may be less precise than for standard robot arms.

---

## Archives to Exclude

The following archives should **not** be processed, for the reasons given.

| Archive(s) | Reason |
|-----------|--------|
| `aliangdw_metaworld_*` | Custom MetaWorld failures generated separately |
| `jesbu1_roboreward_rfm_*` | Handled via a separate pipeline |
| `jesbu1_oxe_rfm_oxe_droid`, `bridge_v2`, `bc_z`, `fractal`, `robo_set`, `language_table` | Embedding-only archives — no frame data, cannot be scored |
| All egocentric archives (EgoDex, EPIC Kitchens, RH20T human, USC Koch human, etc.) | VLM prompts are designed for third-person robot-view footage |
| `jesbu1_h2r_rfm_h2r`, `jesbu1_ph2d_rfm_ph2d` | Confirmed 0 failures |
| All `abraranwar_libero_rfm_libero256_*` | Confirmed 0 failures in quality index |
| `jesbu1_molmoact_rfm_molmoact_dataset_tabletop` | Confirmed 0 failures |
| All OXE eval archives (except `jesbu1_auto_eval_rfm_auto_eval_rfm`) | Confirmed 0 failures |
| `abraranwar_agibotworld_alpha_*` | Embedding-only archives |

---

## Compute Estimate

Based on a calibration run on 5,500 DROID failures (Qwen3.5-35B-A3B + Qwen3-32B, 2× A100-40GB):

| Metric | Value |
|--------|-------|
| Throughput | 118.6 episodes / hour per 2-GPU job |
| Time per episode | 3.9 s (VLM) + 26.4 s (LLM) = 30.3 s |
| RACER train alone (23k episodes) | ~8 GPU-hours |
| Group A + B combined (~43k episodes) | ~360 GPU-hours |
| Groups A + B + C combined (~55–60k episodes) | ~480–500 GPU-hours |

**Recommended submission plan:** one SLURM job per archive in Groups A and B (19 jobs), plus one per archive in Group C (10 jobs) = 29 jobs total, each with `--time=24:00:00 --gpus=2`. All 29 jobs can be submitted simultaneously.

**Estimated SBU cost:** ~120,000–160,000 SBUs (central estimate ~135,000).

---

## Output Format

All output is written to `$DATA_DIR/`:

```
$DATA_DIR/
  manifests/           ← per-episode metadata (episode_id, task, archive, n_frames)
  keyframes/           ← 8 JPEG frames per episode (can be deleted after scoring)
  vlm_descriptions/    ← Stage 1 output JSONL
  scores/              ← Final output — one JSONL per archive
```

Each line in `scores/<archive>_scored.jsonl` represents one scored episode:
```json
{
  "episode_id": "ep_000042_<uuid>",
  "archive": "jesbu1_racer_rfm_racer_train",
  "category": "real_robot",
  "task": "Pick the cup and place it on the plate.",
  "robometer_label": "failure",
  "score": [3, "Robot grasped the cup but dropped it before reaching the plate."],
  "frames": [ ... ]
}
```

**Required output:** please tar and transfer the `scores/` and `manifests/` directories only. The `keyframes/` and `vlm_descriptions/` directories can be deleted to recover disk space.

---

## Environment Variable Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | *(required)* | HuggingFace access token |
| `DATA_DIR` | `/data/robometer_failures` | Root output directory |
| `VLM_MODEL_ID` | `Qwen/Qwen3.5-35B-A3B` | Stage 1 vision-language model |
| `LLM_MODEL_ID` | `Qwen/Qwen3-32B` | Stage 2 language model |
| `VLM_BATCH_SIZE` | `32` | Reduce to 16 if GPU OOM occurs |
| `HF_HOME` | `/data/hf_cache` | HuggingFace model cache |

---

## Troubleshooting

| Symptom | Resolution |
|---------|-----------|
| GPU OOM during VLM stage | Set `VLM_BATCH_SIZE=16` |
| Job interrupted mid-run | Re-run the same command; pipeline resumes automatically |
| Split archive fails to extract | Confirm the base archive name is used (no `.tar.part-*` suffix) |
| `index_mappings.json not found` | Known issue with some compressed archives; the script falls back to streaming — if it fails, contact Platon |
