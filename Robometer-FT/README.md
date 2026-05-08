# Robometer-FT — Full fine-tune of Robometer-4B

Sister directory to `Robometer-LoRA/` (preliminary LoRA bake-off) and `Qwen35-FT/`
(full FT of vanilla Qwen3.5-4B). This directory does **full fine-tuning of
Robometer-4B**: same dataset and ICL setup as the LoRA bake-off, but every
weight trains and we initialize from the released Robometer-4B checkpoint.

## Layout

| Path                                  | Purpose                                                     |
| ------------------------------------- | ----------------------------------------------------------- |
| `configs/train_base.yaml`             | Hyperparams, dataset, ICL, eval — shared by both losses     |
| `configs/loss1_corn.yaml`             | Loss 1 preset (asymmetric CORN, fresh 4-logit head)         |
| `configs/loss2_c51.yaml`              | Loss 2 preset (asymmetric C51 + asymmetric BCE)             |
| `configs/distributed/fsdp.yaml`       | Accelerate FSDP config — Qwen3-VL layer class names         |
| `jobs/train_loss1.job`                | SLURM launcher for Loss 1 (multi-GPU FSDP)                  |
| `jobs/train_loss2.job`                | SLURM launcher for Loss 2 (multi-GPU FSDP)                  |

No `train.py` is vendored — the jobs `cd ../Robometer` and run upstream's
`train.py`. Same conda env as the LoRA bake-off (`robometer_gpu`) — full FT
disables `model.use_unsloth` (LoRA-only accelerator) and `model.use_peft`
(otherwise PEFT and unsloth are no-ops in this env).

## Differences from the sister directories

| Axis                          | Robometer-LoRA            | Robometer-FT (this)             | Qwen35-FT                           |
| ----------------------------- | ------------------------- | ------------------------------- | ----------------------------------- |
| Base model                    | Qwen3-VL-4B               | Qwen3-VL-4B                     | Qwen3.5-4B (different architecture) |
| `use_peft`                    | true                      | **false**                       | false                               |
| `use_unsloth`                 | true (default)            | **false**                       | false                               |
| `load_from_checkpoint`        | `robometer/Robometer-4B`  | `robometer/Robometer-4B`        | (none — vanilla Qwen)               |
| FSDP layer class              | (single GPU, no FSDP)     | `Qwen3VLVisionBlock,...`        | `Qwen3_5VisionBlock,...`            |
| `gradient_accumulation_steps` | 2                         | **1**                           | 1                                   |
| Dataset / eval splits         | Same                      | Same                            | Same                                |
| ICL pairs                     | Same                      | Same                            | Same                                |
| WandB project                 | `Robometer_LoRA`          | `Robometer_FT`                  | `Qwen35_FT`                         |

The two FT setups (this one and Qwen35-FT) are deliberately the cleanest possible
ablation pair: same data, same loss presets, same multi-GPU recipe. The only
difference is the starting point — Robometer-4B (with pretrained progress/success
heads) vs vanilla Qwen3.5-4B (heads init fresh). That isolates the value of
Robometer's pretraining.

## Launch

Single node, 4×H100 (default):

```bash
sbatch jobs/train_loss1.job
sbatch jobs/train_loss2.job
```

Multi-node (e.g. 2 nodes × 4 H100 = 8 GPUs total):

```bash
sbatch --nodes=2 jobs/train_loss1.job
```

Override hyperparameters at submit time without editing the YAML:

```bash
sbatch --export=ALL,MAX_STEPS=4000,EXTRA="++training.learning_rate=1e-5 ++data.icl_prob=0.3" \
       jobs/train_loss2.job
```

Resume from a partial run (full state — optimizer, scheduler, step counter):

```bash
sbatch --export=ALL,CKPT_RESUME=/projects/prjs1958/Robometer_FT_weights/loss2_<JID>/checkpoint-1000 \
       jobs/train_loss2.job
```

## Picking the loss

Once the LoRA bake-off (`Robometer-LoRA/`) declares a winner, run the matching
loss here. If both losses look similar in the LoRA arena, run both losses' full
FT and let the §Winning-criterion metric (Success AUC + ranking accuracy +
per-source slicing, from `losses.md`) decide. The FT runs are expensive
(~13–19h pure training × 4 GPUs at 7500 steps), so prefer running the LoRA
winner only.

## Things to revisit once a LoRA winner is declared

1. **LR** — full FT typically wants a smaller LR than LoRA. Default here is
   `2e-5` (matches upstream's full-FT recipe doc). If the LoRA winner used
   something different and converges fast, consider scaling down to `5e-6`.
2. **`max_steps`** — currently 7500 to match LoRA. Full FT may converge in
   fewer steps because every weight moves; keep an eye on validation loss
   plateau.
3. **`gradient_accumulation_steps`** — 1 by default (effective batch =
   `per_device * num_gpus` = 8 × 4 = 32 on single node). LoRA bake-off used
   effective batch 16; if results look noisy, drop `per_device_train_batch_size`
   to 4.
