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

## About the source data

Each RoboMeter archive is a sharded tar that bundles:

- `frames/` — one `trajectory_<uuid>.npz` per episode, containing an RGB video array
  (shape `(T, H, W, 3)`, typically 8 fps for real robot, T ≈ 20–200 steps)
- `embeddings/` — per-episode precomputed embedding `.pt` files (not used here)
- `index_mappings.json` — episode-level metadata:
  - `task_indices` — `{task_string: [idx, …]}`
  - `source_indices` — `{source_dataset: [idx, …]}`
  - `quality_indices` — `{success: [...], failure: [...], partial_success: [...]}`

**Both successes and failures are in every archive.** Our pipeline filters down to the
`quality_indices.failure` list (optionally + `partial_success` with `--include-partial`)
and never writes the successful episodes to disk.

Tasks vary per archive:

- OXE archives → "pick up the banana and place it in the bowl", "wipe the table", etc.
  (many tasks, free-form natural language)
- LIBERO archives → procedurally-named benchmark tasks ("stack the red blocks on the
  blue block", "pick up the plate and place it in the microwave", etc.)
- FailSafe (`jesbu1_failsafe_rfm_failsafe`, 3 split parts) → manipulation trajectories
  produced by the FailSafe failure-generation framework.
  **58,153 failures out of 71,614 total episodes (≈81% failure rate)** — the remaining
  ~13,461 are successes. Source archive already on-cluster at
  `/projects/prjs1958/robometer_full_dataset/raw_archives/split/jesbu1_failsafe_rfm_failsafe/`
  (no HuggingFace download needed). Tasks are in `index_mappings.json.task_indices`.

### Task / failure count overview

Per-archive counts (failures vs. total) are tracked in `failure_counts.json`.
It is populated by `find_split_failures.py` (which streams the last N MB of each tar to
read `index_mappings.json` without downloading the whole archive). To refresh/extend
it with new archives:

```bash
HF_TOKEN=hf_xxx python find_split_failures.py > new_counts.txt
```

As of the last scan: **19,864** failures across all single-file archives, and split
archives are tracked in `/projects/prjs1958/robometer_full_dataset/split_failure_counts.json`
(e.g. FailSafe: 58,153 failures / 71,614 total).

### Manual review workflow

If you want to eyeball the FailSafe data before committing to the full VLM+LLM
scoring run:

```bash
HF_TOKEN=hf_xxx sbatch jobs/sample_failsafe.job
```

This downloads ~10 failure+partial episodes from the archive and writes 8 keyframes per
episode under `$DATA_DIR/keyframes/jesbu1_failsafe_rfm_failsafe/ep_*/`. Open a few
to confirm the episodes look like what we'd want to score.

---

## In-Context Learning Pairing

Every failure is paired with a success trajectory that acts as an expert
demonstration at training time (the failure is the query under review; the
success is the context). The pairing manifests live at
`/projects/prjs1958/robometer_full_dataset/pairs/` — one
`<archive>_pairs.jsonl` per failure archive plus a consolidated `report.json`.

### Scope and exclusions

Failures are drawn from the 21 Group A + Group B archives listed in
`FOR_LEONARDO.md`. Successes are pooled from **every** RoboMeter archive in the
same robot family — including success-only archives (e.g. LIBERO successes come
from `abraranwar_libero_rfm_libero256_*`, not from `ykorkmaz_libero_failure_*`).

Excluded from both sides:
- `jesbu1_oxe_rfm_oxe_droid` — already paired in `Real-World-Failures/Droid-Failures/`
- `aliangdw_metaworld_*` — simulator handled separately
- `jesbu1_hand_paired_*`, `jesbu1_usc_koch_human_robot_paired_..._human` — human video
- `jesbu1_humanoid_everyday_*` — humanoid robot
- `jesbu1_failsafe_*` — handled in its own pipeline

### Match tiers (higher = closer match)

Same greedy two-pass logic as DROID, adapted for cross-archive success pools:

| Tier | Meaning |
|---|---|
| **1 — same_task_fresh** | Same task string, same archive, unused success |
| **2 — same_task_family_fresh** | Same task, different archive in same family, unused |
| **3 — same_task_reused** | Same task (in-archive or family), reused |
| **4 — same_family_other_task_fresh** | Same family, different task, unused success |
| **5 — same_family_other_task_reused** | Same family, different task, reused |
| **6 — unpaired** | No same-family success anywhere (marked for manual review) |

Pair records carry `origin_archive`, `family`, and `tier` so downstream training
can filter out tier 3+ (or any other threshold) without re-running the matcher.

### Global tier distribution

68,933 failures paired; **zero unpaired**.

| Tier | Count | % |
|---|---:|---:|
| 1 — same task, in-archive, fresh | 14,076 | 20.4% |
| 2 — same task, cross-archive, fresh | 6,932 | 10.1% |
| 3 — same task, reused | 36,377 | 52.8% |
| 4 — same family, other task, fresh | 2,293 | 3.3% |
| 5 — same family, other task, reused | 9,255 | 13.4% |
| 6 — unpaired | 0 | 0.0% |

**Same-task share (tiers 1 + 2 + 3): 83.3%.** The remaining 16.7% fall back to
other tasks within the same robot family — these are flagged in the report so
they can be dropped or down-weighted later.

### Per-family breakdown

| Family | Failures | Successes | Tier-1+2 fresh | Tier-3 reused | Tier-4+5 other-task |
|---|---:|---:|---:|---:|---:|
| racer        | 29,211 | 7,131 | 24.4% | 75.6% | 0% |
| roboarena    | 17,510 | 2,635 | 2.3%  | 32.1% | 65.7% |
| soar         | 12,009 | 4,803 | 35.9% | 64.1% | 0% |
| libero       |  6,241 | — *   | 88.1% | 11.1% | 0.8% |
| auto_eval    |  3,721 | 4,956 | 93.6% |  6.4% | 0% |
| mit_franka   |    125 |   209 | 64.0% | 34.4% | 1.6% |
| usc_koch     |     50 |    50 | 100%  | 0%    | 0% |
| utd_so101    |     40 |    40 | 100%  | 0%    | 0% |
| usc_xarm     |     12 |    12 | 100%  | 0%    | 0% |
| usc_franka   |      8 |     8 | 100%  | 0%    | 0% |
| usc_trossen  |      6 |    15 | 100%  | 0%    | 0% |

\* LIBERO successes come from `abraranwar_libero_rfm_libero256_{10,90,goal,object,spatial}`
(success-only archives contributing to the `libero` family pool).

### Notable outliers

- **RACER (75.6% tier-3 reused)**: 29k failures vs. 7k successes — dataset
  imbalance, not a matching bug. Every reuse is still same-task.
- **SOAR (64.1% tier-3)**: same story, 12k failures vs. 4.8k successes.
- **roboarena (65.7% tier-4/5)**: many failure tasks (e.g. *"put the strawberry
  in the pink bowl"*, *"erase the board"*, *"put marker in the jar"*) have
  zero successes anywhere in the roboarena family. These pairs fall back to a
  different task — flagged in `report.json.per_archive[...].fail_tasks_missing_success`.
- **LIBERO — 50 orphan failures**: task *"pick up the butter and put it in the
  basket"* has no matching success in the libero256 pool. Fell to tier-4.
- **Clean archives (100% tier-1)**: `usc_koch`, `utd_so101`, `usc_xarm`,
  `usc_franka`, `usc_trossen`, plus `auto_eval` at 93.6%. Gold-standard ICL pairs.

### Reproduction

```bash
# Builds all pair manifests + report.json (staging partition, ~5 min)
sbatch jobs/pair_robometer.job
```

Deterministic: `random.Random(42)` seed, archives processed in dict insertion
order. Re-running overwrites existing pair files.

---

## Contact

Questions: Platon Karageorgis — p.karageorgis@student.vu.nl
