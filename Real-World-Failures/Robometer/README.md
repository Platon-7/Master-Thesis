# RoboMeter Failure Scoring Pipeline

Scores failure trajectories from the [`robometer/processed_datasets`](https://huggingface.co/datasets/robometer/processed_datasets) HuggingFace dataset using a two-stage VLM+LLM pipeline:

- **Stage 1 (VLM):** Qwen3.5-35B-A3B describes each of 8 keyframes per episode
- **Stage 2 (LLM):** Qwen3-32B assigns a 1–4 progress score using the frame descriptions

This is the same pipeline used to score ~5,500 DROID failure videos (see `Robo-Reward-FPS/droid_failures/`).

---

## Hardware Requirements

- 2× GPU with ≥24 GB VRAM each (A100-40GB, A6000, RTX 3090/4090, etc.)
- ~80 GB RAM
- ~500 GB disk for data + model cache (models ~70 GB, data varies by dataset)

---

## Quick Start

### Option A — Docker (recommended for non-HPC machines)

```bash
# 1. Build the container (one-time, ~15 min)
docker build -t robometer-pipeline .

# 2. Run on a single dataset
docker run --gpus all --rm -it \
  -e HF_TOKEN=hf_xxxxxxxxxxxx \
  -v /your/data:/data \
  robometer-pipeline \
  bash /pipeline/run.sh jesbu1_oxe_rfm_oxe_bridge_v2

# 3. Run all single-file real-robot archives
docker run --gpus all --rm -it \
  -e HF_TOKEN=hf_xxxxxxxxxxxx \
  -v /your/data:/data \
  robometer-pipeline \
  bash /pipeline/run.sh all real_robot
```

### Option B — Conda environment (for HPC / Slurm clusters)

```bash
# 1. Create environment
conda create -n robometer python=3.11 -y
conda activate robometer

# 2. Install PyTorch + vLLM
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.4.3 transformers>=4.40.0 accelerate huggingface_hub Pillow numpy tqdm

# 3. Run
export HF_TOKEN=hf_xxxxxxxxxxxx
export DATA_DIR=/scratch/robometer_failures
bash run.sh jesbu1_oxe_rfm_oxe_bridge_v2
```

---

## Dataset Categories

Run `python extract_failures.py --list-datasets` to see all available archives.

| Category | Examples |
|----------|---------|
| `real_robot` | OXE (Bridge V2, DROID, BC-Z, RT-1…), RACER, MIT Franka, USC Koch, RoboArena |
| `simulated` | MetaWorld, LIBERO (10/90/Goal/Object/Spatial), MolmoAct |
| `human_egocentric` | EgoDex, EPIC Kitchens, RH20T human, Hand-Paired human |
| `eval_benchmark` | OXE Eval splits, MIT Franka Policy Ranking |

> **Note:** The `jesbu1_roboreward_rfm_*` archives (DROID subset) are marked `skip=True` in `datasets_catalog.py` — those were already scored and live in `Robo-Reward-FPS/droid_failures/`.

> **Large archives** (AgiBot World ~1 TB, EgoDex parts) are marked `split_parts=True` and are skipped by default. Contact Platon if you want to process these — they require manual `cat` reassembly before extraction.

---

## Output Structure

```
/data/robometer_failures/
  hf_cache/                          ← HuggingFace model + dataset cache
  keyframes/
    jesbu1_oxe_rfm_oxe_bridge_v2/
      ep_000003_<uuid>/
        frame_0_0.00s.jpg
        frame_1_0.50s.jpg
        ...
        frame_7_3.50s.jpg
      ep_000007_<uuid>/
        ...
  manifests/
    jesbu1_oxe_rfm_oxe_bridge_v2.jsonl   ← one line per failure episode
  vlm_descriptions/
    jesbu1_oxe_rfm_oxe_bridge_v2_descriptions.jsonl
  scores/
    jesbu1_oxe_rfm_oxe_bridge_v2_scored.jsonl
```

### Manifest JSONL format (per episode)

```json
{
  "episode_id": "ep_000003_7cd5a40c-...",
  "archive": "jesbu1_oxe_rfm_oxe_bridge_v2",
  "category": "real_robot",
  "task": "Pick up the banana and place it in the bowl.",
  "robometer_label": "failure",
  "n_source_frames": 32,
  "keyframes_dir": "keyframes/jesbu1_oxe_rfm_oxe_bridge_v2/ep_000003_..."
}
```

### Scored JSONL format (per episode)

```json
{
  "episode_id": "ep_000003_...",
  "archive": "jesbu1_oxe_rfm_oxe_bridge_v2",
  "category": "real_robot",
  "task": "Pick up the banana and place it in the bowl.",
  "robometer_label": "failure",
  "score": 2,
  "frames": [
    {"frame_idx": 0, "timestamp": 0.0, "description": "...", "score": 1},
    ...
    {"frame_idx": 7, "timestamp": 3.5, "description": "...", "score": 2}
  ]
}
```

**Score meaning (1–4):**
- **1** — No progress (robot at start position, task not begun)
- **2** — Approach only (robot moved toward object but did not complete the action)
- **3** — Partial progress (grasped / partially executed the task)
- **4** — Major progress (>50% task done, but ultimately failed)

---

## Running Individual Stages

If you need to split across job time limits:

```bash
# Stage 1 only — VLM frame descriptions (GPU-intensive, ~3-4 hrs per large dataset)
bash run.sh jesbu1_oxe_rfm_oxe_bridge_v2 "" --stage 1

# Stage 2 only — LLM grading from saved descriptions (~1 hr per large dataset)
bash run.sh jesbu1_oxe_rfm_oxe_bridge_v2 "" --stage 2
```

Stage 1 and 2 are both fully resumable — if interrupted, re-run the same command and it picks up where it left off.

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | *(required)* | HuggingFace token for dataset + model download |
| `DATA_DIR` | `/data/robometer_failures` | Root output directory |
| `VLM_MODEL_ID` | `Qwen/Qwen3.5-35B-A3B` | VLM for frame descriptions |
| `LLM_MODEL_ID` | `Qwen/Qwen3-32B` | LLM for progress grading |
| `VLM_BATCH_SIZE` | `32` | Frames per VLM batch (reduce if OOM) |
| `HF_HOME` | `/data/hf_cache` | HuggingFace cache directory |

---

## Troubleshooting

**OOM on VLM:** Reduce `VLM_BATCH_SIZE` to 16 or 8.

**OOM on LLM:** Set `VLM_MODEL_ID=Qwen/Qwen3-8B` (smaller, less accurate).

**Slow download:** The dataset archives range from ~1 GB to ~500 GB. Smaller single-file archives (`jesbu1_roboreward_rfm_roboreward_test`) are a good place to test first.

**"index_mappings.json not found":** Some archives may have a different internal structure. Please report the archive name to Platon.

---

## Contact

Questions: Platon Karageorgis — p.karageorgis@student.vu.nl
