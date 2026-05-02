# LoRA bake-off — May-1 ablation plan

Seven runs that span the L1 (CORN ordinal) vs L2 (asymmetric C51 + BCE) choice, the
asymmetry-strength sweep within each loss, and the ICL task-instruction dropout effect.

## The runs

| Run | Loss | Hyperparam      | ICL dropout | LR     | max_steps | Tests                                |
|----:|:-----|:----------------|:-----------:|:------:|:---------:|:-------------------------------------|
| 1   | L1   | vanilla (c=0)   | off         | 4e-5   | 4000      | symmetric baseline                   |
| 2   | L1   | c=0.5           | off         | 4e-5   | 4000      | mild asymmetry                       |
| 3   | L1   | c=1.5           | off         | 4e-5   | 4000      | strong asymmetry                     |
| 4   | L1   | c=0.5           | **on**      | 4e-5   | 4000      | corruption effect on L1 (vs Run 2)   |
| 5   | L2   | λ=0.3           | off         | 2e-5   | 4000      | original L2                          |
| 6   | L2   | λ=0.5           | off         | 2e-5   | 4000      | mild λ                               |
| 7   | L2   | λ=0.3           | **on**      | 2e-5   | 4000      | corruption effect on L2 (vs Run 5)   |

Notes:
- **L1 uses lr=4e-5** (2× the bake-off base); L2 stays at lr=2e-5.
- **All runs cap at 4000 steps** (down from base's 7500 — past runs plateaued well before
  step 7500, so cutting saves ~45% of compute).
- **ICL is always on** (icl_prob=0.5 from base, 50% of training samples get a demo). The
  "ICL dropout" column toggles `data.icl_task_dropout` — Chris's per-call task-instruction
  corruption that fires only on ICL-attached samples (and only at training time, never eval).
- **eval every 500 steps**, **warmup_steps=0**, dual-ICL eval enabled — all inherited
  unchanged from base + preset.
- **No CORN head pretrain** — the May-1 head pretrain attempts (with and without ICL)
  both produced a frame-index-shortcut head (σ_5 trajectories numerically identical
  between success and failure queries, max-diff 2e-4). All 7 runs ship with a fresh head
  per the user's pre-committed fallback ("if it doesn't work then we go with the freshly
  initialized head").

## Hypotheses being tested

- **L1 asymmetry sweep (Runs 1 → 2 → 3)**: does up-weighting the negative-side BCE term
  (raising c) improve calibration on the head's rare upper thresholds, or does it push
  the head into "predict negative" regardless of input?
- **L1 vs L2 head start (Run 1 vs Run 5)**: which loss is easier to optimize from a fresh
  head — symmetric ordinal CORN, or asymmetric distributional C51?
- **Asymmetry-strength sensitivity (Run 5 vs Run 6)**: is the original λ=0.3 too aggressive?
- **ICL task-corruption effect (Run 4 vs Run 2; Run 7 vs Run 5)**: does corrupting the
  natural-language task instruction force the model to lean on the demo trajectory rather
  than memorizing the exact wording? Hypothesis: improves robustness on tasks whose
  language doesn't match training-time phrasings.

## How to launch

Each run is its own SLURM job. The launcher reads three configs per run:
`train_lora_base.yaml` → `<loss_preset>.yaml` → `<run_overlay>.yaml` (last-wins precedence).

```bash
cd /gpfs/home3/pkarageorgis1/Master-Thesis/Robometer-LoRA

# L1 runs (loss1_corn preset)
RUN=run1_l1_c0           LOSS_PRESET=loss1_corn  sbatch jobs/bakeoff_run.job
RUN=run2_l1_c05          LOSS_PRESET=loss1_corn  sbatch jobs/bakeoff_run.job
RUN=run3_l1_c15          LOSS_PRESET=loss1_corn  sbatch jobs/bakeoff_run.job
RUN=run4_l1_c05_dropout  LOSS_PRESET=loss1_corn  sbatch jobs/bakeoff_run.job

# L2 runs (loss2_c51 preset)
RUN=run5_l2_lambda03         LOSS_PRESET=loss2_c51  sbatch jobs/bakeoff_run.job
RUN=run6_l2_lambda05         LOSS_PRESET=loss2_c51  sbatch jobs/bakeoff_run.job
RUN=run7_l2_lambda03_dropout LOSS_PRESET=loss2_c51  sbatch jobs/bakeoff_run.job
```

All 7 can launch in parallel (each gets its own H100). Outputs land at
`/projects/prjs1958/LoRA_weights/<RUN>_<JOBID>/`.

## Files

- `configs/run1_l1_c0.yaml` ... `configs/run7_l2_lambda03_dropout.yaml` — per-run overlays
  (only what differs from `train_lora_base.yaml + <preset>.yaml`).
- `configs/loss1_corn.yaml`, `configs/loss2_c51.yaml` — loss-family presets (head shape,
  loss type, head-training flags).
- `configs/train_lora_base.yaml` — shared base (model id, datasets, ICL, training
  hyperparams, eval cadence).
- `jobs/bakeoff_run.job` — generic launcher; takes `RUN` + `LOSS_PRESET` env vars.

## Verification trail

The override resolution was verified before submission. For each run, the merged Hydra
overrides were dumped via the same pipeline the job uses, and the last occurrence of
every duplicate key was manually checked to match the table.

**Important plumbing fix (May-1)**: the existing job scripts (pretrain_corn_head.job,
eval_test_set.job, etc.) use `grep -h FILE1 FILE2 ...` to merge YAMLs. On this cluster's
filesystem, `grep -h` REVERSES multi-file output order — the LAST file argument's content
emits first. The 2-file pretrain configs worked by accident (preset emits last → wins
under Hydra's last-wins `++` semantics). Our 3-file pattern would have broken silently
because base would emit last and override the run overlay. `bakeoff_run.job` uses
`cat FILE1 FILE2 FILE3 | grep ...` instead, which guarantees order.

| Run | resolved hyperparam | resolved lr | resolved max_steps | resolved icl_task_dropout |
|----:|--------------------:|------------:|-------------------:|--------------------------:|
| 1   | c=0.0               | 4.0e-5      | 4000               | off (absent)              |
| 2   | c=0.5               | 4.0e-5      | 4000               | off (absent)              |
| 3   | c=1.5               | 4.0e-5      | 4000               | off (absent)              |
| 4   | c=0.5               | 4.0e-5      | 4000               | true                      |
| 5   | λ=0.3               | 2.0e-5      | 4000               | off (absent)              |
| 6   | λ=0.5               | 2.0e-5      | 4000               | off (absent)              |
| 7   | λ=0.3               | 2.0e-5      | 4000               | true                      |

## What we are NOT changing across runs

Anything not listed in the table is held constant via `train_lora_base.yaml` and the loss
preset:

- Base model: `Qwen/Qwen3-VL-4B-Instruct` (loaded from `robometer/Robometer-4B`).
- LoRA enabled (`model.use_peft: true`).
- Dataset: `robometer_frames_train` (~18k trajectories).
- ICL on with `icl_prob=0.5`, pairs index `pairs_index_train.jsonl`.
- 7500 steps, eval every 500, save every 1000.
- Dual-ICL eval (each round runs once with `icl_prob=1.0` and once with `0.0`).
- Per-run wandb experiment name = `bakeoff_runN_<descriptor>`.
