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

### 3. SLURM Job Script (`jobs/run_qwen_container.job`)

The job script orchestrates training on a SLURM cluster with 2 GPUs:
- **GPU 0**: RL training (SAC agent + reward model)
- **GPU 1**: Qwen3-VL server

Key features:
- Starts the Qwen server first, waits for it to load (~5-10 min)
- Configures MuJoCo with Xvfb for headless rendering
- Supports concurrent runs with different `SERVER_PORT` values
- Artifact storage on scratch filesystem for large checkpoints

Example submission:
```bash
sbatch --export=TASK_ENV=metaworld_sweep-into-v2,SERVER_PORT=8001 jobs/run_qwen_container.job
```

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

### `utils.py`

1. **SoftGym fallback** (line ~21): Added try/except for SoftGym imports with empty placeholders (`env_arg_dict = {}`, `SOFTGYM_ENVS = []`) so MetaWorld tasks can run without SoftGym installed.

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
   apptainer build /var/scratch/$USER/rl_vlm_container.sif complete_env.def
   ```

2. **Download Qwen3-VL weights** (one-time, requires internet):
   ```bash
   # Run interactively to cache the model
   python -c "from transformers import AutoProcessor, AutoModelForVision2Seq; AutoProcessor.from_pretrained('Qwen/Qwen3-VL-8B-Instruct', trust_remote_code=True); AutoModelForVision2Seq.from_pretrained('Qwen/Qwen3-VL-8B-Instruct', trust_remote_code=True)"
   ```

3. **Submit a training job**:
   ```bash
   sbatch --export=TASK_ENV=metaworld_sweep-into-v2,SERVER_PORT=8001 jobs/run_qwen_container.job
   ```

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
