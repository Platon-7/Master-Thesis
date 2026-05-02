# Qwen3.5-4B Full Fine-Tune Ablation

Drop-in codebase for full fine-tuning **vanilla `Qwen/Qwen3.5-4B`** on the per-frame
progress task — no Robometer pretrain, no LoRA. Goal: isolate Robometer-4B's
pretraining contribution against the LoRA bake-off in `Robometer-LoRA/`.

The codebase is **bulletproof and recipe-ready**. The user finalizes the recipe on
the Robometer-LoRA bake-off (loss choice, ICL settings, ICL dropout, learning rate,
…), then drops back here, sets a few env vars, and `sbatch`-launches.

## "Go" — typical launch (Loss 2 example)

```bash
cd Qwen35-FT
export WANDB_API_KEY="$YOUR_KEY"

# Optional but recommended: 30-second pre-flight check before sbatch.
# Catches missing deps / dataset paths / WANDB key / SLURM account issues early.
bash scripts/preflight.sh

# Just go with the defaults baked into configs/train_base.yaml + configs/loss2_c51.yaml:
sbatch jobs/train_loss2.job

# Or override anything via env vars at submit time:
EXTRA="++training.learning_rate=5e-6 ++data.icl_prob=0.3 ++data.icl_task_dropout=0.2" \
MAX_STEPS=15000 \
sbatch jobs/train_loss2.job
```

For Loss 1 (CORN), swap to `jobs/train_loss1.job`. Both jobs are byte-identical
except for the loss config they sed-flatten on top of `train_base.yaml`.

## What the launch knobs are

Every recipe knob from the Robometer-LoRA bake-off is exposed as a Hydra `++key=value`
override. Pass via `EXTRA="..."` env var. Common ones:

| Knob | Override | Default |
|---|---|---|
| Loss family | (already chosen by which `train_loss{1,2}.job` you sbatch) | Loss 2 |
| ICL on/off | `++data.use_icl=false` | true |
| ICL probability | `++data.icl_prob=0.5` | 0.5 |
| ICL task-instruction dropout | `++data.icl_task_dropout=0.2` | (not set) |
| Learning rate | `++training.learning_rate=5e-6` | 2e-5 |
| Warmup ratio | `++training.warmup_ratio=0.05` | 0.1 |
| Weight decay | `++training.weight_decay=0.0` | 0.01 |
| Max steps | `MAX_STEPS=15000` (env var, top-level) | 7500 |
| Loss-1 CORN bias init | `++loss.corn_bias_init_priors=[0.602,0.399,0.209,0.076]` | (not set) |
| Loss-2 asymmetric λ | `++loss.asymmetric_lambda=0.5` | 0.3 |

Anything in `Robometer/robometer/configs/experiment_configs.py` is reachable.

## Relationship to `Robometer-LoRA/`

Same task, same datasets, same losses, same eval splits. Differences:

| | Robometer-LoRA | Qwen35-FT (this dir) |
|---|---|---|
| Base model | `Qwen/Qwen3-VL-4B-Instruct` | `Qwen/Qwen3.5-4B` |
| Loaded checkpoint | `robometer/Robometer-4B` | _none_ (vanilla) |
| Training mode | LoRA (PEFT) | Full FT (`use_peft: false`) |
| Launcher | direct `python train.py` | `accelerate launch --config_file configs/distributed/fsdp_qwen35.yaml` |
| GPUs | 1×H100 | 4×H100 (FSDP) |
| Conda env | `robometer_gpu` (transformers 4.57.2) | `robometer_qwen35_gpu` (transformers 5.7) |
| WandB project | `Robometer_LoRA` | `Qwen35_FT` |
| Weights dir | `/projects/prjs1958/LoRA_weights/` | `/projects/prjs1958/Qwen35_FT_weights/` |

`Robometer/robometer/` source is **never modified**. This codebase vendors a parallel
`robometer/` Python package — mostly symlinks back to the original; only `models/rbm.py`
and `utils/setup_utils.py` are real-file copies (Qwen3.5 dispatch + lazy unsloth/bnb
imports added). Verify:

    md5sum Robometer/robometer/models/rbm.py Robometer/robometer/utils/setup_utils.py
    git status Robometer/

## Layout

```
Qwen35-FT/
├── README.md                 # this file
├── MODIFICATIONS.md          # log of vendored changes vs original Robometer/robometer/
├── requirements.txt          # transformers>=5.7,<6 + matching deps
├── train.py                  # symlink → Robometer/train.py (Hydra entry point)
├── robometer/                # vendored package — symlinks except 2 real files
│   ├── models/rbm.py         # REAL — Qwen3.5 dispatch added
│   └── utils/setup_utils.py  # REAL — Qwen3.5 dispatch + lazy unsloth/bnb imports
├── configs/
│   ├── train_base.yaml       # base full-FT overrides
│   ├── loss1_corn.yaml       # CORN asymmetric (Loss 1)
│   ├── loss2_c51.yaml        # C51 + BCE asymmetric (Loss 2; default first run)
│   └── distributed/
│       └── fsdp_qwen35.yaml  # accelerate FSDP config — wraps Qwen3_5VisionBlock + Qwen3_5DecoderLayer
├── jobs/
│   ├── train_loss1.job       # Loss 1 launcher (4×H100 FSDP)
│   ├── train_loss2.job       # Loss 2 launcher (4×H100 FSDP) — DEFAULT
│   ├── smoke_tests.job       # CPU-node smoke runner
│   └── smoke_5_7.job         # GPU-node smoke runner (5+7 only)
├── scripts/
│   ├── run_all_smoke_tests.sh
│   └── smoke_test_{00..07}_*.py    # 8 progressive checks
└── logs/                     # SLURM stdout/stderr (gitignored)
```

## Setup

The conda env is built fresh from scratch — the existing `robometer_gpu` env is
**never** modified or cloned (per user instruction).

```bash
module load 2025 Anaconda3/2025.06-1
conda create -n robometer_qwen35_gpu -c conda-forge --override-channels python=3.10 -y
conda activate robometer_qwen35_gpu

# CUDA 12.4 torch (required for flash-attn / flash-linear-attention)
pip install --index-url https://download.pytorch.org/whl/cu124 'torch>=2.4' torchvision

# Core deps
pip install -r requirements.txt

# Flash-linear-attention (Triton-based, no compile needed).
pip install --no-deps flash-linear-attention fla-core einops
```

`flash-attn` and `causal-conv1d` need `nvcc` (not present on Snellius login nodes).
Install them on a GPU compute node via SLURM:

```bash
sbatch jobs/install_flash_attn.job   # 1×H100, ~20 min compile
```

If you skip the flash-attn install, training still works — transformers falls back
to SDPA attention (still fast, slightly slower than FlashAttention 2).

The build is automated — kicked off as a background install:
`logs/install_flash.log` — when it ends with `FLASH_DONE`, the env is ready.

**If the env ends up broken** (peft / torchvision import errors after install — known issue
because flash-linear-attention 0.5.0's PyPI deps yank torch up to 2.11+cu13 and break the
cu124-pinned wheels), one-command recovery:

```bash
bash scripts/repair_env.sh   # pins torch 2.6.0+cu124, fla --no-deps, rebuilds flash-attn
bash scripts/preflight.sh    # confirm
```

> Note: `unsloth` and `bitsandbytes` are intentionally **not** installed. The
> vendored `setup_utils.py` lazy-imports `FastVisionModel` only when
> `cfg.use_unsloth=true` (we never set that for full FT). bitsandbytes was already
> dead code at module level (the `bnb` parameter shadows it). Drop-clean.

## Running smoke tests

Smoke tests verify the Qwen3.5 dispatch + architecture quirks incrementally.
All 8 are confirmed passing as of 2026-05-01.

```bash
cd Qwen35-FT
bash scripts/run_all_smoke_tests.sh
# or, if the login node is too small for the 4B model load:
sbatch jobs/smoke_tests.job
```

Smoke 04 wrote `configs/_discovered_modules.json` with the actual layer naming
(MLP uses Qwen2.5-style `gate_proj/up_proj/down_proj`, NOT Qwen3-VL's `linear_fc1/2`).

## Comparison to Robometer-LoRA

After both runs converge, compare on the four `policy_ranking` eval splits:
`robometer_frames_eval_droid`, `eval_robometer`, `eval_metaworld`, `eval_failsafe`.
Both projects on WandB under `nlp-squad`.
