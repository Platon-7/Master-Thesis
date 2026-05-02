# RoboMeter Failure Scoring Pipeline — Instructions for Leonardo

## Overview

This document describes a two-stage VLM+LLM pipeline for assigning fine-grained progress labels (1–4) to failure episodes in the RoboMeter dataset.

**This run covers Group A (18 single-file archives, 36,543 failures) and Group B excluding Failsafe (3 split-part archives, 32,390 failures). Total: 68,933 failures + 26,045 successes = 94,978 episodes.**

**Leonardo: You will do Group A, ignore everything related to Group B**

The same pipeline has already been applied to ~5,500 DROID failure episodes.

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
| GPUs | 2× with ≥24 GB VRAM (A100-40GB recommended) |
| RAM | ≥80 GB |
| Scratch storage | ~50 GB per job (extracted keyframes + intermediate files; delete after scoring) |
| Model cache | ~70 GB one-time download (Qwen3.5-35B-A3B + Qwen3-32B) |

---

## Step 0 — Environment Setup

```bash
conda create -n robometer python=3.10 -y
conda activate robometer

pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.4.3 transformers>=4.40.0 accelerate huggingface_hub Pillow numpy tqdm requests
```

Clone the repo (once):
```bash
git clone <repo_url> ~/Master-Thesis
cd ~/Master-Thesis/Real-World-Failures/Robometer
```

---

## Step 1 — Get the archives

The 18 Group A archives (~62 GB total) and 3 Group B archives (~150 GB total) are available in **two ways** — use whichever applies to you:

### Option A — Direct local path (if you have read access to `/projects/prjs1958`)

Set this environment variable and skip the download entirely:

```bash
export LOCAL_ARCHIVE_DIR=/projects/prjs1958/robometer_full_dataset/raw_archives
```

The pipeline will read directly from there. Nothing to download.

### Option B — Download from Google Drive

Archives are shared at: https://drive.google.com/drive/folders/17YOWNSQBlaKYLq8gbpq2-rRxYWTKbO9P?usp=drive_link

Install `rclone` and configure a Google Drive remote (one-time setup):

**Leonardo: You don't have access probably so follow Option B**

```bash
rclone config
# → n (new remote) → name it anything, e.g. gdrive → storage type 22 (Google Drive)
# → leave client_id and client_secret blank → scope 1 (full access)
# → N (no browser on cluster) → run the printed rclone authorize command on your laptop
#   and paste the token back
```

Then download the archives (replace `gdrive` with whatever you named your remote):

```bash
GDRIVE_FOLDER_ID=17YOWNSQBlaKYLq8gbpq2-rRxYWTKbO9P
DEST=/scratch-shared/$USER/robometer_archives

mkdir -p $DEST/single $DEST/split

# Group A — single-file archives (~62 GB)
rclone copy "gdrive:single/" $DEST/single/ \
    --drive-root-folder-id $GDRIVE_FOLDER_ID --transfers 4 --progress

# Group B — split-part archives (~150 GB)
rclone copy "gdrive:split/" $DEST/split/ \
    --drive-root-folder-id $GDRIVE_FOLDER_ID --transfers 2 --progress

export LOCAL_ARCHIVE_DIR=$DEST
```

---

## Step 2 — Set output directory and model cache

```bash
export DATA_DIR=/scratch-shared/$USER/robometer_out
export HF_HOME=/scratch-shared/$USER/hf_cache   # ~70 GB, downloaded once on first run
export HF_TOKEN=hf_xxxxxxxxxxxx                  # needed only to download model weights
```

---

## Step 3 — Run the pipeline

Each archive is one command:

```bash
bash run.sh <archive_name>
```

Example:
```bash
bash run.sh jesbu1_racer_rfm_racer_train
```

The script auto-detects whether the archive is a single `.tar` or a split-part directory and calls the correct extractor. Both stages run sequentially; the job is fully resumable if interrupted.

---

## SLURM Submission (Snellius)

Submit one job per archive. A template:

```bash
#!/bin/bash
#SBATCH --job-name=robometer_<archive_short>
#SBATCH --output=/home/$USER/robometer_logs/robometer_<archive_short>_%j.out
#SBATCH --partition=gpu_a100
#SBATCH --gpus=2
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate robometer

export LOCAL_ARCHIVE_DIR=/projects/prjs1958/robometer_full_dataset/raw_archives
# OR if using Google Drive download:
# export LOCAL_ARCHIVE_DIR=/scratch-shared/$USER/robometer_archives

export DATA_DIR=/scratch-shared/$USER/robometer_out
export HF_HOME=/scratch-shared/$USER/hf_cache
export HF_TOKEN=hf_xxxxxxxxxxxx
export VLM_BATCH_SIZE=20
export PYTHONUNBUFFERED=1

SCRIPT_DIR=~/Master-Thesis/Real-World-Failures/Robometer

bash $SCRIPT_DIR/run.sh jesbu1_racer_rfm_racer_train
```

Create one copy per archive, changing the `--job-name`, `--output`, and the last `run.sh` argument.

---

## Archive List

### Group A — Single-file archives (submit one job per archive)

| Archive | Failures | Est. wall time (2× A100) |
|---------|----------|--------------------------|
| `jesbu1_racer_rfm_racer_train` | 23,391 | ~8 × 24h jobs (shard by episode range if needed) |
| `jesbu1_racer_rfm_racer_val` | 5,820 | ~49h (2–3 jobs) |
| `jesbu1_auto_eval_rfm_auto_eval_rfm` | 3,721 | ~31h (2 jobs) |
| `ykorkmaz_libero_failure_rfm_libero_90_failure` | 4,312 | ~36h (2 jobs) |
| `ykorkmaz_libero_failure_rfm_libero_10_failure` | 498 | <5h (1 job) |
| `ykorkmaz_libero_failure_rfm_libero_object_failure` | 489 | <5h (1 job) |
| `ykorkmaz_libero_failure_rfm_libero_spatial_failure` | 486 | <5h (1 job) |
| `ykorkmaz_libero_failure_rfm_libero_goal_failure` | 456 | <5h (1 job) |
| `jesbu1_rfm_new_mit_franka_rfm_rfm_new_mit_franka_rfm` | 80 | <1h (1 job) |
| `jesbu1_usc_koch_p_ranking_rfm_usc_koch_p_ranking_all` | 50 | <1h (1 job) |
| `jesbu1_rfm_new_mit_franka_rfm_nowrist_rfm_new_mit_franka_rfm_nowrist` | 40 | <1h (1 job) |
| `aliangdw_utd_so101_policy_ranking_utd_so101_policy_ranking` | 20 | <1h (1 job) |
| `aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking` | 12 | <1h (1 job) |
| `jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top` | 10 | <1h (1 job) |
| `jesbu1_utd_so101_clean_policy_ranking_wrist_utd_so101_clean_policy_ranking_wrist` | 10 | <1h (1 job) |
| `aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking` | 8 | <1h (1 job) |
| `ykorkmaz_usc_trossen_rfm_usc_trossen` | 6 | <1h (1 job) |
| `jesbu1_mit_franka_p-rank_rfm_mit_franka_p-rank_rfm` | 5 | <1h (1 job) |

**RACER train** is the largest archive (23,391 failures, ~197h sequential). The pipeline is resumable, so you can submit multiple 24h jobs pointing to the same `DATA_DIR` and it will continue where it left off each time. Submit 8–9 sequential jobs, or ask Platon about sharding.

### Group B — Split-part archives (submit one job per archive; parts assembled automatically)

| Archive | Failures | Parts | Est. wall time |
|---------|----------|-------|----------------|
| `jesbu1_soar_rfm_soar_rfm` | 12,009 | 6 | ~101h (5 jobs) |
| `jesbu1_roboarena_0825_rfm_roboarena` | 10,753 | 2 | ~91h (4 jobs) |
| `jesbu1_roboarena_eval_debug_nowrist_roboarena_eval_debug_nowrist` | 6,757 | 4 | ~57h (3 jobs) |

Pass the **base archive name** (no `.part-*` suffix) to `run.sh`. It finds and assembles the parts automatically.

---

## Full Robometer Dataset Breakdown

**Robometer = Group A failures + Group B failures + all successes = 94,978 episodes**

| Component | Episodes |
|-----------|----------|
| Group A failures (18 single-file archives) | 36,543 |
| Group B failures (3 split archives) | 32,390 |
| **Total failures** | **68,933** |
| Successes from failure archives (16 archives) | 19,859 |
| Successes from success-pool extras (8 archives) | 6,186 |
| **Total successes** | **26,045** |
| **Grand total** | **94,978** |

### Success breakdown by archive

**From failure archives (also contain failures):**

| Archive | Family | Successes |
|---------|--------|-----------|
| `jesbu1_racer_rfm_racer_train` | racer | 5,724 |
| `jesbu1_auto_eval_rfm_auto_eval_rfm` | auto_eval | 4,956 |
| `jesbu1_soar_rfm_soar_rfm` | soar | 4,803 |
| `jesbu1_roboarena_0825_rfm_roboarena` | roboarena | 1,626 |
| `jesbu1_racer_rfm_racer_val` | racer | 1,407 |
| `jesbu1_roboarena_eval_debug_nowrist_roboarena_eval_debug_nowrist` | roboarena | 1,009 |
| `jesbu1_rfm_new_mit_franka_rfm_rfm_new_mit_franka_rfm` | mit_franka | 138 |
| `jesbu1_rfm_new_mit_franka_rfm_nowrist_rfm_new_mit_franka_rfm_nowrist` | mit_franka | 69 |
| `jesbu1_usc_koch_p_ranking_rfm_usc_koch_p_ranking_all` | usc_koch | 50 |
| `aliangdw_utd_so101_policy_ranking_utd_so101_policy_ranking` | utd_so101 | 20 |
| `ykorkmaz_usc_trossen_rfm_usc_trossen` | usc_trossen | 15 |
| `aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking` | usc_xarm | 12 |
| `jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top` | utd_so101 | 10 |
| `jesbu1_utd_so101_clean_policy_ranking_wrist_utd_so101_clean_policy_ranking_wrist` | utd_so101 | 10 |
| `aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking` | usc_franka | 8 |
| `jesbu1_mit_franka_p-rank_rfm_mit_franka_p-rank_rfm` | mit_franka | 2 |
| **Subtotal** | | **19,859** |

**Success-pool extras (success-only archives):**

| Archive | Family | Successes |
|---------|--------|-----------|
| `abraranwar_libero_rfm_libero256_90` | libero | 3,950 |
| `abraranwar_libero_rfm_libero256_object` | libero | 456 |
| `abraranwar_libero_rfm_libero256_spatial` | libero | 433 |
| `abraranwar_libero_rfm_libero256_goal` | libero | 432 |
| `abraranwar_usc_koch_rewind_rfm_usc_koch_rewind` | usc_koch | 407 |
| `abraranwar_libero_rfm_libero256_10` | libero | 388 |
| `jesbu1_usc_koch_human_robot_paired_usc_koch_human_robot_paired_robot` | usc_koch | 100 |
| `aliangdw_utd_so101_human_utd_so101_human` | utd_so101 | 20 |
| **Subtotal** | | **6,186** |

All 26,045 successes belong to robot families that also have failures — none are orphaned.

---

## Full Dataset Audit (entire `/projects/prjs1958/robometer_full_dataset/`)

The downloaded Robometer archive collection contains **far more** than the 24 archives used for the operative Robometer subset. Full audit of every `index_mappings.json` in the dataset:

| Metric | Count |
|--------|-------|
| Archives present | 87 |
| Archives excluded (humanoid / hand / metaworld / droid / failsafe / usc_koch_human) | 8 |
| Archives scanned | 79 |
| Archives that errored (missing inner `index_mappings.json`) | 6 |
| **Total successes across dataset** | **1,295,674** |
| **Total failures across dataset** | **157,351** |
| **Total episodes across dataset** | **1,453,025** |

Of the 1,295,674 successes, only **26,045** belong to archives whose robot family also has failures → those are the ones extracted for Robometer. The remaining **1,255,963** successes come from archives with no matching failures anywhere in the dataset (orphan successes — not used for in-context pairing).

### Orphan successes by source

| Source / family | Archives | Successes |
|-----------------|----------|-----------|
| agibotworld (alpha + headcam) | 2 | 433,821 |
| egodex (parts 1–5 + test) | 6 | 313,109 |
| OXE (bc_z, fractal, bridge_v2, language_table, robo_set, furniture_bench, 17 others) | 23 | 309,765 |
| galaxea r1_lite (parts 1–5) | 5 | 108,118 |
| epic | 1 | 37,030 |
| rh20t (human + robot) | 2 | 29,969 |
| molmoact (household + tabletop) | 2 | 15,546 |
| h2r | 1 | 2,254 |
| motif | 1 | 83 |
| fino_net | 1 | 82 |
| **True orphans (not in Robometer subset)** | **44** | **1,249,777** |
| Archive-orphans whose family IS in Robometer (libero256, usc_koch_rewind, utd_so101_human, usc_koch_paired_robot) — already counted in the 26,045 | 8 | 6,186 |
| **Total orphan successes** | **52** | **1,255,963** |

### Excluded from audit (per project convention)

`jesbu1_oxe_rfm_oxe_droid`, `aliangdw_metaworld_metaworld_eval`, `aliangdw_metaworld_metaworld_train`, `jesbu1_hand_paired_rfm_hand_paired_human`, `jesbu1_hand_paired_rfm_hand_paired_robot`, `jesbu1_humanoid_everyday_rfm_humanoid_everyday_rfm`, `jesbu1_failsafe_rfm_failsafe`, `jesbu1_usc_koch_human_robot_paired_usc_koch_human_robot_paired_human`.

---

## Orphan-Success Frame Extraction

The 1,255,963 orphan successes are further categorized for the unified frame dataset (`/projects/prjs1958/robometer_frame_dataset/robometer/`). Run by `extract_orphan_successes.py`, output goes to a separate folder `keyframes_orphan_success/` (parallel to `keyframes_success/`), with manifests `manifests/<archive>_orphan_successes.jsonl`.

Categorization is grounded in the RBM-1M paper's own dataset taxonomy (Appendix: Individual RBM-1M Training Dataset Details).

### Excluded — humanoid robot data (7 archives, 541,939 successes)

Both AgiBot G1 and Galaxea R1 Lite are bimanual humanoid platforms (the paper explicitly calls Galaxea R1 Lite a "large-scale humanoid dataset"). Skipped to keep the orphan-success pool consistent with the robot-arm focus of the rest of Robometer.

| Archive | Successes |
|---------|-----------|
| `abraranwar_agibotworld_alpha_headcam_rfm_agibotworld` | 216,911 |
| `abraranwar_agibotworld_alpha_rfm_agibotworld` | 216,910 |
| `jesbu1_galaxea_rfm_galaxea_part1_r1_lite` | 22,110 |
| `jesbu1_galaxea_rfm_galaxea_part2_r1_lite` | 24,888 |
| `jesbu1_galaxea_rfm_galaxea_part3_r1_lite` | 24,741 |
| `jesbu1_galaxea_rfm_galaxea_part4_r1_lite` | 21,511 |
| `jesbu1_galaxea_rfm_galaxea_part5_r1_lite` | 14,868 |
| **Subtotal** | **541,939** |

### Excluded — human-only / human-hand data (9 archives, 366,618 successes)

EgoDex and Epic-Kitchens are the paper's "Human only" datasets; the `_human` half of RH20T is the human-demonstration side of that paired dataset. H2R is a data-augmentation pipeline that composites simulated robot arms into Ego4D / SSv2 egocentric human videos — the underlying frames are human-ego content, not real robot rollouts, so we treat it as human-hand data.

| Archive | Successes |
|---------|-----------|
| `jesbu1_egodex_rfm_egodex_part1` | 45,232 |
| `jesbu1_egodex_rfm_egodex_part2` | 94,488 |
| `jesbu1_egodex_rfm_egodex_part3` | 51,899 |
| `jesbu1_egodex_rfm_egodex_part4` | 43,199 |
| `jesbu1_egodex_rfm_egodex_part5` | 75,076 |
| `jesbu1_egodex_rfm_egodex_test` | 3,215 |
| `jesbu1_epic_rfm_epic` | 37,030 |
| `anqil_rh20t_subset_rfm_rh20t_human` | 14,225 |
| `jesbu1_h2r_rfm_h2r` | 2,254 |
| **Subtotal** | **366,618** |

### Excluded — already extracted by `extract_successes.py` (8 archives, 6,186 successes)

These archive-orphans belong to robot families that DO have failure archives (libero, usc_koch, utd_so101), so they are already in `keyframes_success/`. Skipped to avoid duplicate extraction.

| Archive | Successes |
|---------|-----------|
| `abraranwar_libero_rfm_libero256_90` | 3,950 |
| `abraranwar_libero_rfm_libero256_object` | 456 |
| `abraranwar_libero_rfm_libero256_spatial` | 433 |
| `abraranwar_libero_rfm_libero256_goal` | 432 |
| `abraranwar_usc_koch_rewind_rfm_usc_koch_rewind` | 407 |
| `abraranwar_libero_rfm_libero256_10` | 388 |
| `jesbu1_usc_koch_human_robot_paired_usc_koch_human_robot_paired_robot` | 100 |
| `aliangdw_utd_so101_human_utd_so101_human` | 20 |
| **Subtotal** | **6,186** |

### Target — extracted into `keyframes_orphan_success/` (28 archives, 341,220 successes)

OXE robot-arm subset (Franka, WidowX, xArm, Sawyer, etc.), MolmoACT (Franka), RH20T robot-half, MotIF, Fino-Net.

| Archive | Family | Successes |
|---------|--------|-----------|
| `jesbu1_oxe_rfm_oxe_fractal20220817_data` | oxe_fractal | 87,204 |
| `jesbu1_oxe_rfm_oxe_bridge_v2` | oxe_bridge_v2 | 72,930 |
| `jesbu1_oxe_rfm_oxe_language_table` | oxe_language_table | 50,000 |
| `jesbu1_oxe_rfm_oxe_bc_z` | oxe_bc_z | 39,347 |
| `jesbu1_oxe_rfm_oxe_robo_set` | oxe_robo_set | 36,480 |
| `anqil_rh20t_subset_rfm_rh20t_robot` | rh20t | 15,744 |
| `jesbu1_molmoact_rfm_molmoact_dataset_household` | molmoact | 11,872 |
| `jesbu1_oxe_rfm_eval_oxe_bridge_v2_eval` | oxe_bridge_v2 | 10,094 |
| `jesbu1_oxe_rfm_oxe_furniture_bench_dataset_converted_externally_to_rlds` | oxe_furniture_bench | 5,100 |
| `jesbu1_molmoact_rfm_molmoact_dataset_tabletop` | molmoact | 3,674 |
| `jesbu1_oxe_rfm_oxe_utaustin_mutex` | oxe_utaustin_mutex | 1,500 |
| `jesbu1_oxe_rfm_oxe_berkeley_cable_routing` | oxe_berkeley_cable_routing | 1,482 |
| `jesbu1_oxe_rfm_oxe_jaco_play` | oxe_jaco_play | 976 |
| `jesbu1_oxe_rfm_oxe_berkeley_rpt_converted_externally_to_rlds` | oxe_berkeley_rpt | 904 |
| `jesbu1_oxe_rfm_oxe_toto` | oxe_toto | 902 |
| `jesbu1_oxe_rfm_oxe_iamlab_cmu_pickup_insert_converted_externally_to_rlds` | oxe_iamlab_cmu | 631 |
| `jesbu1_oxe_rfm_oxe_stanford_hydra_dataset_converted_externally_to_rlds` | oxe_stanford_hydra | 570 |
| `jesbu1_oxe_rfm_oxe_berkeley_mvp_converted_externally_to_rlds` | oxe_berkeley_mvp | 480 |
| `jesbu1_oxe_rfm_oxe_berkeley_fanuc_manipulation` | oxe_berkeley_fanuc | 415 |
| `jesbu1_oxe_rfm_oxe_aloha_mobile` | oxe_aloha_mobile | 272 |
| `jesbu1_oxe_rfm_oxe_imperialcollege_sawyer_wrist_cam` | oxe_imperial_sawyer | 168 |
| `jesbu1_oxe_rfm_oxe_ucsd_kitchen_dataset_converted_externally_to_rlds` | oxe_ucsd_kitchen | 150 |
| `jesbu1_motif_rfm_motif_rfm` | motif | 83 |
| `jesbu1_fino_net_rfm_fino_net` | fino_net | 82 |
| `jesbu1_oxe_rfm_oxe_austin_buds_dataset_converted_externally_to_rlds` | oxe_austin_buds | 50 |
| `jesbu1_oxe_rfm_oxe_dlr_edan_shared_control_converted_externally_to_rlds` | oxe_dlr_edan | 48 |
| `jesbu1_oxe_rfm_oxe_tokyo_u_lsmo_converted_externally_to_rlds` | oxe_tokyo_lsmo | 48 |
| `jesbu1_oxe_rfm_oxe_nyu_rot_dataset_converted_externally_to_rlds` | oxe_nyu_rot | 14 |
| **Total** | | **341,220** |

Sanity: 541,939 + 366,618 + 6,186 + 341,220 = 1,255,963 ✓ (matches `audit_report.json → orphan_success_total`).

### Running the extraction

```bash
# count only (verifies registry against audit_report.json)
python extract_orphan_successes.py --count-only

# all 29 archives — resume-safe
sbatch jobs/extract_orphan_successes.job

# single archive
sbatch --export=ARCHIVE=jesbu1_oxe_rfm_oxe_bridge_v2 jobs/extract_orphan_successes.job
```

### Archives that errored during the audit (inner `index_mappings.json` not at expected path)

`jesbu1_oxe_rfm_eval_oxe_bc_z_eval`, `jesbu1_oxe_rfm_eval_oxe_berkeley_cable_routing_eval`, `jesbu1_oxe_rfm_eval_oxe_jaco_play_eval`, `jesbu1_oxe_rfm_eval_oxe_toto_eval`, `jesbu1_oxe_rfm_eval_oxe_viola_eval`, `jesbu1_ph2d_rfm_ph2d`. All six are in families with no failures anyway (orphan by construction) — not extracted.

### RoboReward archives (kept aside — handled separately)

The audit also found the following RoboReward archives in the dataset. They are **not** part of Leonardo's Group A/B — Platon processes these through a separate pipeline:

| Archive | Successes | Failures |
|---------|-----------|----------|
| `jesbu1_roboreward_rfm_roboreward_train` | 8,425 | 36,647 |
| `jesbu1_roboreward_rfm_roboreward_val` | 974 | 5,258 |
| `jesbu1_roboreward_rfm_roboreward_test` | 527 | 2,304 |
| `jesbu1_roboreward_rfm_high_res_roboreward_train` | 8,425 | 36,647 |
| `jesbu1_roboreward_rfm_high_res_roboreward_val` | 974 | 5,258 |
| `jesbu1_roboreward_rfm_high_res_roboreward_test` | 527 | 2,304 |
| **Subtotal (ignore for Group A/B)** | **19,852** | **88,418** |

Report source: `/projects/prjs1958/robometer_full_dataset/audit_report.json`.

---

## Compute Estimate

Based on calibration: **~110 episodes/hour per 2-GPU job** (includes model load time).

| Batch | Failures | GPU-hours | SBUs (~150/GPU-hr) |
|-------|----------|-----------|---------------------|
| Group A | 36,543 | ~666 | ~100,000 |
| Group B (3 archives) | ~32,400 | ~590 | ~89,000 |
| **Total** | **~69,000** | **~1,256** | **~188,000** |

---

## Output Format

All output goes to `$DATA_DIR/`:

```
$DATA_DIR/
  manifests/          ← per-episode metadata (one JSONL per archive)
  keyframes/          ← 8 JPEG frames per episode (delete after scoring to save space)
  vlm_descriptions/   ← Stage 1 output JSONL
  scores/             ← Final output — one JSONL per archive
```

Each line in `scores/<archive>_scored.jsonl`:
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

**When done: tar and transfer `scores/`, `manifests/`, and `keyframes/` back to me. Only `vlm_descriptions/` can be deleted to save space.**

---

## Environment Variable Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `LOCAL_ARCHIVE_DIR` | Yes (or `HF_TOKEN`) | Root dir with `single/` and `split/` subdirs |
| `DATA_DIR` | Yes | Root output directory |
| `HF_TOKEN` | Yes (model download) | HuggingFace token (for downloading model weights on first run) |
| `HF_HOME` | Recommended | HuggingFace model cache (~70 GB) |
| `VLM_MODEL_ID` | No | Stage 1 model (default: `Qwen/Qwen3.5-35B-A3B`) |
| `LLM_MODEL_ID` | No | Stage 2 model (default: `Qwen/Qwen3-32B`) |
| `VLM_BATCH_SIZE` | No | VLM batch size (default: 32; reduce to 16 if GPU OOM) |

---

## Troubleshooting

| Symptom | Resolution |
|---------|-----------|
| GPU OOM during VLM stage | Set `VLM_BATCH_SIZE=16` |
| Job interrupted mid-run | Re-run the same command — pipeline resumes automatically |
| `No part files found` error | Confirm `LOCAL_ARCHIVE_DIR` points to the dir with `single/` and `split/` subdirs |
| `index_mappings.json not found` | Script falls back to streaming; if it still fails, contact Platon |
| First job silent for 20–30 min | Normal — vllm is downloading Qwen3.5-35B-A3B + Qwen3-32B (~70 GB) into `HF_HOME` on first run. Subsequent jobs start immediately from cache. |
| Model download very slow | Ensure `HF_HOME` is on scratch storage, not home directory (home quota will overflow) |
