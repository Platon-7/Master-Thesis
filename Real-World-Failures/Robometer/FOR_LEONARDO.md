# RoboMeter Failure Scoring Pipeline — Instructions for Leonardo

## Overview

This document describes a two-stage VLM+LLM pipeline for assigning fine-grained progress labels (1–4) to failure episodes in the RoboMeter dataset.

**This run covers Group A (18 single-file archives, 36,543 failures) and Group B excluding Failsafe (3 split-part archives, ~32,400 failures). Total: ~69,000 failures.**

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
