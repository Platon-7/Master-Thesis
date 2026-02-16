# RL-VLM-F with Qwen3-VL

This is a modified version of [RL-VLM-F](https://github.com/yuqingd/rl-vlm-f) adapted to use **Qwen3-VL-8B** as the vision-language model instead of the Gemini API. The original framework learns reward functions from VLM feedback for reinforcement learning tasks.

## Main Idea

The original RL-VLM-F uses Gemini API calls to query a VLM for preference labels between two trajectory segments. We replace these cloud API calls with a **locally-hosted Qwen3-VL-8B model** running on the same machine. This provides:

- **No API costs** – Qwen3-VL is open-source
- **Offline capability** – Works on HPC clusters without internet access
- **Lower latency** – Local inference avoids network round-trips
- **Reproducibility** – Fixed model weights ensure consistent results

The VLM receives pairs of trajectory images and returns preference labels (0, 1, or -1 for no preference) that train a reward model for the RL agent.

---

## Environment Setup

We use a **Singularity/Apptainer container** instead of a conda environment due to EGL rendering incompatibilities with the MuJoCo version on our HPC cluster.

To build the container:

```bash
apptainer build rl_vlm_container.sif complete_env.def
```

The definition file `complete_env.def` contains all dependencies including:
- PyTorch with CUDA support
- Qwen3-VL-8B model dependencies
- MuJoCo and mujoco-py
- MetaWorld environments

---

## Major Changes

### 1. Modified VLM Inference (`vlms/gemini_infer.py`)

We replaced the Gemini API calls with HTTP requests to our local Qwen server. **The filename is kept as `gemini_infer.py` to minimize changes across the codebase.**

The original file is preserved as `vlms/gemini_infer-backup.py`.

#### How it works:

The file provides two main functions that the reward model calls:

**`gemini_query_1(query_list, temperature=0)`** — Direct VLM Query
- Receives a mixed list of text strings and images (PIL/numpy) from the RL codebase
- Converts images to base64-encoded PNGs using `_encode_image()`
- Builds a structured payload via `_build_payload()` that formats content as `[{"type": "text", "text": "..."}, {"type": "image", "image": "base64..."}]`
- Appends "(Answer concisely in 2 sentences.)" to prevent Qwen from generating verbose responses
- Sends POST request to the local server's `/generate` endpoint
- Strips Qwen3's `<think>...</think>` reasoning tags from the response (Qwen3 uses chain-of-thought by default)
- Returns the cleaned text response

**`gemini_query_2(query_list, summary_prompt, temperature=0)`** — Two-Step Preference Query
- First calls `gemini_query_1()` to get a vision description of the trajectory comparison
- Then makes a second text-only call with the `summary_prompt` to extract a preference label
- Uses strict `max_tokens=16` limit since we only need a single digit (0, 1, or -1)
- Implements robust regex parsing to extract the label:
  - Searches for explicit `-1` first (to avoid matching the `1` in `-1`)
  - Then searches for `0` or `1` as word boundaries
  - Falls back to keyword heuristics ("uncertain", "can't tell", "equal" → `-1`)
- Returns the extracted label as a string

#### System Architecture:

```
┌─────────────────────────┐         ┌─────────────────────────┐
│     GPU 0 (RL Agent)    │         │  GPU 1 (VLM Server)     │
│                         │         │                         │
│  ┌───────────────────┐  │         │  ┌───────────────────┐  │
│  │ Trajectory Images │──┼──HTTP───┼─▶│   Qwen3-VL-8B     │  │
│  └───────────────────┘  │  POST   │  └─────────┬─────────┘  │
│           │             │         │            │            │
│           ▼             │         │            ▼            │
│  ┌───────────────────┐  │         │  ┌───────────────────┐  │
│  │   Reward Model    │◀─┼─────────┼──│ Label (0/1/-1)    │  │
│  └─────────┬─────────┘  │  JSON   │  └───────────────────┘  │
│            │            │         │                         │
│            ▼            │         │                         │
│  ┌───────────────────┐  │         │                         │
│  │   SAC Agent       │  │         │                         │
│  │   (Updates)       │  │         │                         │
│  └───────────────────┘  │         │                         │
└─────────────────────────┘         └─────────────────────────┘
```

#### Key design decisions:
- **Server URL from environment**: `SERVER_HOST` and `SERVER_PORT` allow multiple concurrent training runs on different ports
- **Temperature=0**: Ensures deterministic, reproducible VLM outputs for stable reward learning
- **Concise prompting**: Added suffix to prevent Qwen from writing 300+ token explanations

### 2. Local VLM Server (`local_server.py`)

A FastAPI server that hosts Qwen3-VL-8B-Instruct for inference. This decouples the VLM from the RL training process, allowing them to run on separate GPUs.

```python
# Start the server (runs on GPU 1)
python local_server.py
```

#### Architecture:

**Model Loading (runs once at startup):**
- Loads Qwen3-VL-8B-Instruct with **4-bit NF4 quantization** via BitsAndBytes
- Uses `bfloat16` compute dtype for efficient inference
- `device_map="auto"` handles multi-GPU placement automatically
- `trust_remote_code=True` required for Qwen's custom model architecture
- Processor configured with `min_pixels=128*28*28`, `max_pixels=1280*28*28` for image preprocessing (128 and 1280 patches of 28×28 pixels respectively, this is a standard configuration for Qwen-VL to balance performance (details) vs. memory (VRAM).)

**API Endpoint (`POST /generate`):**

Accepts JSON with the Qwen chat format:
```json
{
  "messages": [{"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image", "image": "data:image/png;base64,..."}]}],
  "max_tokens": 512,
  "temperature": 0
}
```

Processing pipeline:
1. Converts Pydantic models to Qwen's expected dictionary format
2. Applies chat template via `processor.apply_chat_template()`
3. Extracts image/video inputs using `process_vision_info()` from `qwen_vl_utils`
4. Tokenizes text and preprocesses images together
5. Generates with configurable decoding:
   - `temperature=0` → Greedy decoding (`do_sample=False`) for deterministic outputs
   - `temperature>0` → Sampling with `top_p=0.9`
6. Decodes and returns the generated text

#### Key design decisions:
- **4-bit quantization**: Reduces VRAM from ~16GB to ~8GB, fitting on a single TitanRTX
- **Greedy decoding for training**: Same input always produces same output, critical for stable reward learning
- **Verbose logging**: Prints full Qwen output for debugging VLM behavior
- **Environment variable configuration**: `LOCAL_MODEL_PATH` allows using a cached local snapshot for offline clusters

> **Hardware Note**: This 2-GPU split is designed for workstation GPUs (e.g., 2× TitanRTX 24GB). If using a large data-center card (A100 80GB, H100), you can run both processes on `device=0` by setting `CUDA_VISIBLE_DEVICES=0` for both and adjusting memory—the combined footprint is ~12GB (8GB VLM + 4GB RL).

### 3. New MetaWorld Tasks

We added support for four new MetaWorld environments: **Assembly**, **BoxClose**, **CoffeePush**, and **StickPull**. Task descriptions are taken from the [MetaWorld benchmark page](https://metaworld.farama.org/benchmark/task_descriptions/) to stay consistent with other baselines (DITTO, RoboReward).

| Task | Environment ID | Task Description |
|------|---------------|-----------------|
| Assembly | `metaworld_assembly-v2` | pick up a nut and place it onto a peg |
| BoxClose | `metaworld_box-close-v2` | grasp the cover and close the box with it |
| CoffeePush | `metaworld_coffee-push-v2` | push a mug under a coffee machine |
| StickPull | `metaworld_stick-pull-v2` | grasp a stick and pull a box with the stick |

**Settings for the new tasks** (differ from the original sweep/soccer tasks):

| Setting | Value | Rationale |
|---------|-------|-----------|
| Camera | `corner2` | Consistent with other baselines |
| Episode length | 100 steps | No action repeat |
| Training termination | After max steps only | Episodes do NOT end on task success during training |
| Eval termination | On task success | Episodes end early when the task is solved during evaluation |
| Eval frequency | Every 5,000 steps | More frequent evaluation |
| Eval episodes | 20 | Random initializations for robust success rate estimates |

#### Changes made

**`utils.py`** — Gym API compatibility and configurable environments:
- `GymV5ToV4Compat` wrapper: MetaWorld v2 returns 5 values `(obs, reward, terminated, truncated, info)` but old gym's `TimeLimit` expects 4. This wrapper sits between `NormalizedBoxEnv` and `TimeLimit` to bridge the API gap. It also strips the `mode` argument from `render()` calls that old gym injects.
- `make_metaworld_env(cfg)`: Creates MetaWorld environments with configurable camera (`metaworld_camera`), episode length (`max_episode_steps`), and random seed.

**`prompt.py`** — Task descriptions for VLM queries:
- Added entries in `clip_env_prompts` and `goal_env_prompts` for all four new tasks.

**`train_PEBBLE.py`** — Eval-time success termination:
- During evaluation, if `eval_terminate_on_success=true`, episodes end when `info['success']` is true. Training episodes are unaffected.

**`config/train_PEBBLE.yaml`** — Three new config parameters with backward-compatible defaults:
- `metaworld_camera: null` — uses env-specific camera when null
- `max_episode_steps: 0` — uses `env.max_path_length` when 0
- `eval_terminate_on_success: false` — no early termination when false

### 4. SLURM Job Script (`jobs/run_qwen_container.job`)

The job script orchestrates training on a SLURM cluster with 2 GPUs:
- **GPU 0**: RL training (SAC agent + reward model)
- **GPU 1**: Qwen3-VL server

Key features:
- Starts the Qwen server first, waits for it to load (~5-10 min)
- Configures MuJoCo with Xvfb for headless rendering
- Supports concurrent runs with different `SERVER_PORT` values
- Artifact storage on scratch filesystem for large checkpoints
- All paths are user-agnostic (uses `$USER` and `$SCRATCH_DIR`)

---

## Minor Changes

All modified files contain a `# Change:` comment above each modification for easy identification.

### `reward_model.py`

1. **Null path validation** (line ~248): Handles `None`, `"None"`, `"null"`, and empty strings for `cached_label_path` to prevent path formatting errors.

2. **Improved query logging** (line ~726): Modified to log ALL VLM queries including `-1` (no preference) labels for analysis, while preserving original training behavior that filters out `-1` labels.

3. **VLM query CSV logging** (line ~751): Added logging to `vlm_query_log.csv` with columns:
   - `ground_truth_score_1`, `ground_truth_score_2`: True rewards for comparison
   - `vlm_predicted_label`: VLM's preference (0, 1, or -1)
   - `timestamp`, `train_iter`: For temporal analysis

### `train_PEBBLE.py`

1. **Added imports** (line ~8): `shutil`, `glob` for checkpoint management.

2. **Removed unused imports** (line ~23): Commented out `blip_infer_2` and `clip_infer` imports that caused errors and weren't used with Qwen.

3. **Artifact root support** (line ~39): `ARTIFACT_ROOT` environment variable redirects heavy outputs (checkpoints, eval images) to scratch storage.

4. **Checkpoint sync function** (line ~159): `_sync_latest_models()` copies current checkpoint to home while keeping full history on scratch.

5. **Evaluation artifacts** (line ~195): Evaluation outputs go to scratch if `ARTIFACT_ROOT` is set.

6. **Scratch model directory** (line ~360): Full checkpoint history stored on scratch, only latest synced to home.

7. **Lazy VLM imports** (lines ~563, ~571): BLIP2 and CLIP imports moved inside their respective branches to avoid network downloads at module load time.

8. **Checkpoint saving** (lines ~615, ~626): Modified to save to scratch with sync to home.

9. **Eval success termination** (line ~255): When `eval_terminate_on_success=true`, evaluation episodes end early on `info['success']`. Training episodes are unaffected.

### `utils.py`

1. **SoftGym fallback** (line ~21): Added try/except for SoftGym imports with empty placeholders (`env_arg_dict = {}`, `SOFTGYM_ENVS = []`) so MetaWorld tasks can run without SoftGym installed.

2. **`GymV5ToV4Compat` wrapper**: Bridges MetaWorld v2's new gymnasium step API (5 return values) with old gym's `TimeLimit` wrapper (4 return values). Also overrides `render()` to strip the positional `mode` argument that old gym injects.

3. **`make_metaworld_env(cfg)`**: Creates MetaWorld environments with support for configurable camera name (`cfg.metaworld_camera`), episode length (`cfg.max_episode_steps`), and random seed. Wrapping order: `TimeLimit(GymV5ToV4Compat(NormalizedBoxEnv(env)))`.

### `prompt.py`

1. **New task descriptions**: Added `clip_env_prompts` and `goal_env_prompts` entries for Assembly, BoxClose, CoffeePush, and StickPull environments.

### `config/train_PEBBLE.yaml`

1. **New config parameters**: `metaworld_camera`, `max_episode_steps`, and `eval_terminate_on_success` with backward-compatible defaults (`null`, `0`, `false`).

---

## Visualization

After training completes, use the provided scripts to generate figures:

### Learning Curves (Figure 4)

Plots success rate vs. environment steps from the evaluation log:

```bash
python scripts/plot_figure4.py <path_to_eval.csv> --output figure4.png --title "Sweep Into – Success Rate"
```

**Input**: `eval.csv` generated during training (found in the experiment's log directory)  
**Columns required**: `step`, `success_rate`


### VLM Accuracy vs. Difficulty (Figure 6)

Plots VLM prediction accuracy binned by query difficulty (reward difference):

```bash
python scripts/plot_figure6.py <path_to_vlm_query_log.csv> --output figure6.png --title "VLM Accuracy"
```

**Input**: `vlm_query_log.csv` generated during training  
**Columns required**: `ground_truth_score_1`, `ground_truth_score_2`, `vlm_predicted_label`


The plot shows a stacked histogram with:
- **Correct**: VLM chose the trajectory with higher ground truth reward
- **Incorrect**: VLM chose the lower-reward trajectory
- **No Preference**: VLM returned -1

---

## Quick Start

1. **Build the container** (one-time):
   ```bash
   apptainer build /var/scratch/$USER/rl_vlm_container_v4.sif complete_env.def
   ```

2. **Download Qwen3-VL weights** (one-time, requires internet):
   ```bash
   # Run interactively to cache the model
   python -c "from transformers import AutoProcessor, AutoModelForVision2Seq; AutoProcessor.from_pretrained('Qwen/Qwen3-VL-8B-Instruct', trust_remote_code=True); AutoModelForVision2Seq.from_pretrained('Qwen/Qwen3-VL-8B-Instruct', trust_remote_code=True)"
   ```

3. **Submit a training job**:

   Original tasks (use default camera and episode length):
   ```bash
   sbatch --export=TASK_ENV=metaworld_sweep-into-v2,SERVER_PORT=8001 jobs/run_qwen_container.job
   sbatch --export=TASK_ENV=metaworld_soccer-v2,SERVER_PORT=8002 jobs/run_qwen_container.job
   ```

   New tasks (corner2 camera, 100-step episodes, 20 eval episodes, eval terminates on success, eval every 5k steps):
   ```bash
   sbatch --export=TASK_ENV=metaworld_assembly-v2,SERVER_PORT=8001,METAWORLD_CAMERA=corner2,MAX_EPISODE_STEPS=100,NUM_EVAL_EPISODES=20,EVAL_TERM_SUCCESS=true,EVAL_FREQUENCY=5000 jobs/run_qwen_container.job
   sbatch --export=TASK_ENV=metaworld_box-close-v2,SERVER_PORT=8002,METAWORLD_CAMERA=corner2,MAX_EPISODE_STEPS=100,NUM_EVAL_EPISODES=20,EVAL_TERM_SUCCESS=true,EVAL_FREQUENCY=5000 jobs/run_qwen_container.job
   sbatch --export=TASK_ENV=metaworld_coffee-push-v2,SERVER_PORT=8003,METAWORLD_CAMERA=corner2,MAX_EPISODE_STEPS=100,NUM_EVAL_EPISODES=20,EVAL_TERM_SUCCESS=true,EVAL_FREQUENCY=5000 jobs/run_qwen_container.job
   sbatch --export=TASK_ENV=metaworld_stick-pull-v2,SERVER_PORT=8004,METAWORLD_CAMERA=corner2,MAX_EPISODE_STEPS=100,NUM_EVAL_EPISODES=20,EVAL_TERM_SUCCESS=true,EVAL_FREQUENCY=5000 jobs/run_qwen_container.job
   ```

   **Environment variables** (all optional, with defaults):
   | Variable | Default | Description |
   |----------|---------|-------------|
   | `TASK_ENV` | `metaworld_sweep-into-v2` | MetaWorld environment ID |
   | `SERVER_PORT` | `8001` | Port for Qwen3-VL server (use different ports for concurrent runs) |
   | `SCRATCH_DIR` | `/var/scratch/$USER` | Scratch directory for large files (container, model weights, artifacts) |
   | `CONTAINER` | `$SCRATCH_DIR/rl_vlm_container_v4.sif` | Path to the Apptainer container |
   | `METAWORLD_CAMERA` | env-specific | Camera name (e.g., `corner2`) |
   | `MAX_EPISODE_STEPS` | env default | Max steps per episode (e.g., `100`) |
   | `NUM_EVAL_EPISODES` | `1` | Number of evaluation episodes |
   | `EVAL_TERM_SUCCESS` | `false` | End eval episodes on task success |
   | `EVAL_FREQUENCY` | `10000` | Steps between evaluations |

4. **Generate figures** after training:
   ```bash
   python scripts/plot_figure4.py exp/<exp_name>/eval.csv --output results/figure4.png
   python scripts/plot_figure6.py exp/<exp_name>/vlm_query_log.csv --output results/figure6.png
   ```

---

## Acknowledgments

This work builds upon:
- [RL-VLM-F](https://github.com/yufeiwang63/RL-VLM-F) by Yufei Wang et al.
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) by Alibaba
- [MetaWorld](https://github.com/Farama-Foundation/Metaworld) benchmark
