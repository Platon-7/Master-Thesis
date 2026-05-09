# LoRA bake-off — May-3 ablation plan (v2 cache relaunch)

Nine runs that span the L1 (CORN ordinal) vs L2 (asymmetric C51 + BCE) choice, the
asymmetry-strength sweep within each loss, the ICL task-instruction dropout effect, and
(runs 8–9) a 3-point failure-KL anchor weight sweep for resisting drift on the failure
distribution.

## The runs

| Run | Loss | Hyperparam      | ICL dropout | KL anchor   | LR     | max_steps | Tests                                |
|----:|:-----|:----------------|:-----------:|:-----------:|:------:|:---------:|:-------------------------------------|
| 1   | L1   | vanilla (c=0)   | off         | —           | 4e-5   | 7500      | symmetric baseline                   |
| 2   | L1   | c=0.5           | off         | —           | 4e-5   | 7500      | mild asymmetry                       |
| 3   | L1   | c=1.5           | off         | —           | 4e-5   | 7500      | strong asymmetry                     |
| 4   | L1   | c=0.5           | **on**      | —           | 4e-5   | 7500      | corruption effect on L1 (vs Run 2)   |
| 5   | L2   | λ=0.3           | off         | —           | 2e-5   | 7500      | original L2                          |
| 6   | L2   | λ=0.5           | off         | —           | 2e-5   | 7500      | mild λ                               |
| 7   | L2   | λ=0.3           | **on**      | —           | 2e-5   | 7500      | corruption effect on L2 (vs Run 5)   |
| 8   | L2   | λ=0.3           | off         | **w=0.1**   | 2e-5   | 7500      | failure-KL pilot weight (vs Run 5)   |
| 9   | L2   | λ=0.3           | off         | **w=0.3**   | 2e-5   | 7500      | failure-KL max-recipe weight         |

Notes:
- **L1 uses lr=4e-5** (2× the bake-off base); L2 stays at lr=2e-5.
- **All runs train to 7500 steps** (May-3 update: bumped from 4000 — earlier May-1
  attempt hit the 24h walltime cap near step 2000; with the v2 cache fix in place we
  want the full presentation-equivalent budget).
- **Walltime is 72h** (`#SBATCH --time=72:00:00` in `jobs/bakeoff_run.job`); pure training
  is ~13–19h at ~6–9 sec/step, so 72h leaves ample headroom for eval rounds + slowdowns.
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
- **Failure-KL anchor (Run 8 vs Run 5)**: does maintaining a FIFO buffer of past failure
  logits and adding `λ_KL · KL(P_old ‖ P_new)` on each success step prevent the model
  from quietly forgetting what failures look like during success-heavy phases? Hypothesis:
  improves long-horizon failure detection without sacrificing success calibration.

## Reproducible HF cache build (v2)

The bake-off trains and evals against `/projects/prjs1958/robometer_frames_hf_v2/` —
a clean rebuild done on May-3 from `pairs_index_<split>.jsonl` files in
`/scratch-shared/$USER/robometer_frames_splits/`. Old `/projects/prjs1958/robometer_frames_hf/`
cache is preserved untouched as evidence/police-work for the regression investigation.

What's in v2 vs the original cache:
- `frame_labels` column populated for failures (the orig schema dropped it silently,
  causing failures to be trained against `t/T → 1.0` linear targets identical to
  successes — see source-rearrangement notes below).
- `episode_dir`-first manifest lookup in the loader → 100% frame_labels coverage on
  failsafe + metaworld eval (was 0%/54% on the orig cache).
- droid family added to default loader families (was silently skipped).
- All 7 splits materialised: train (18k), warmup (1.5k), test (3.5k), eval_droid (1k),
  eval_robometer (2.5k), eval_metaworld (573), eval_failsafe (552).

Build command (one job per split, all 7 in parallel — staging partition, CPU-only):

```bash
cd /gpfs/home3/pkarageorgis1/Master-Thesis/Robometer-LoRA
mkdir -p /projects/prjs1958/robometer_frames_hf_v2

for SPLIT in train warmup test eval_droid eval_robometer eval_metaworld eval_failsafe; do
    sbatch --export=ALL,SPLIT=$SPLIT jobs/preprocess_split_v2.job
done
```

The job (`jobs/preprocess_split_v2.job`) sets
`ROBOMETER_PROCESSED_DATASETS_PATH=/projects/prjs1958/robometer_frames_hf_v2` and runs:

1. `dataset_upload.generate_hf_dataset` — reads frames from
   `/projects/prjs1958/robometer_frame_dataset/`, writes the HF dataset to
   `<HF_OUT_DIR>/<SPLIT>_raw/robometer_frames_<SPLIT>/`.
2. `robometer.data.scripts.preprocess_datasets` — builds the training-ready cache
   (path-encoded subdir like `_projects_prjs1958_robometer_frames_hf_v2_<SPLIT>_raw_robometer_frames_<SPLIT>/`).
3. Creates the short-name symlink `robometer_frames_<SPLIT>/` → path-encoded dir, which
   is what the trainer's `_get_available_datasets` resolves against.

Wall-clock: ~2-5 min per small split, ~12 min for train (18k trajectories). Total
disk: ~20 GB.

**Source-data prerequisites** (one-time rearrangement done before v2):
- `roboreward/manifests/*_failures.jsonl` — `frame_labels` field merged in from
  `roboreward/scores/*_scored.jsonl` (the field existed only under `scores/` previously).
- `robometer/manifests/*_failures.jsonl` — 21 new files synthesised from
  `robometer/scores/*_scored.jsonl` (robometer's `manifests/` was previously orphan-success
  only; the failures with `frame_labels` lived under `scores/`).
- `droid/manifests/droid_failures.jsonl` — 1 new file synthesised from
  `droid/metadata/scored_full_droid_shard*.jsonl`'s per-frame `score` field.
- See `MODIFICATIONS.md` for the full source-rearrangement record.

Backup of source dataset before rearrangement:
`/scratch-shared/pkarageorgis1/robometer_frame_dataset_bak_20260503_134115/` (rsync,
224 GB).

## How to launch

Each run is its own SLURM job. The launcher reads three configs per run:
`train_lora_base.yaml` → `<loss_preset>.yaml` → `<run_overlay>.yaml` (last-wins precedence).

The `ROBOMETER_PROCESSED_DATASETS_PATH` env var **must** point at the v2 cache —
otherwise the trainer falls back to the broken-rebuilt original cache.

```bash
cd /gpfs/home3/pkarageorgis1/Master-Thesis/Robometer-LoRA
HF_V2=/projects/prjs1958/robometer_frames_hf_v2

# L1 runs (loss1_corn preset)
sbatch --export=ALL,RUN=run1_l1_c0,LOSS_PRESET=loss1_corn,ROBOMETER_PROCESSED_DATASETS_PATH=$HF_V2          jobs/bakeoff_run.job
sbatch --export=ALL,RUN=run2_l1_c05,LOSS_PRESET=loss1_corn,ROBOMETER_PROCESSED_DATASETS_PATH=$HF_V2         jobs/bakeoff_run.job
sbatch --export=ALL,RUN=run3_l1_c15,LOSS_PRESET=loss1_corn,ROBOMETER_PROCESSED_DATASETS_PATH=$HF_V2         jobs/bakeoff_run.job
sbatch --export=ALL,RUN=run4_l1_c05_dropout,LOSS_PRESET=loss1_corn,ROBOMETER_PROCESSED_DATASETS_PATH=$HF_V2 jobs/bakeoff_run.job

# L2 runs (loss2_c51 preset)
sbatch --export=ALL,RUN=run5_l2_lambda03,LOSS_PRESET=loss2_c51,ROBOMETER_PROCESSED_DATASETS_PATH=$HF_V2          jobs/bakeoff_run.job
sbatch --export=ALL,RUN=run6_l2_lambda05,LOSS_PRESET=loss2_c51,ROBOMETER_PROCESSED_DATASETS_PATH=$HF_V2          jobs/bakeoff_run.job
sbatch --export=ALL,RUN=run7_l2_lambda03_dropout,LOSS_PRESET=loss2_c51,ROBOMETER_PROCESSED_DATASETS_PATH=$HF_V2  jobs/bakeoff_run.job
sbatch --export=ALL,RUN=run8_l2_lambda03_kl,LOSS_PRESET=loss2_c51,ROBOMETER_PROCESSED_DATASETS_PATH=$HF_V2       jobs/bakeoff_run.job
```

All 8 can launch in parallel (each gets its own H100). Outputs land at
`/projects/prjs1958/LoRA_weights/<RUN>_<JOBID>/`.

## Files

- `configs/run1_l1_c0.yaml` ... `configs/run7_l2_lambda03_dropout.yaml`,
  `configs/run8_l2_lambda03_kl.yaml` — per-run overlays (only what differs from
  `train_lora_base.yaml + <preset>.yaml`).
- `configs/loss1_corn.yaml`, `configs/loss2_c51.yaml` — loss-family presets (head shape,
  loss type, head-training flags). `loss2_c51.yaml` includes `failure_label_smoothing: linear`
  (May-3) to match the pretrained C51 head's expected target shape.
- `configs/train_lora_base.yaml` — shared base (model id, datasets, ICL, training
  hyperparams, eval cadence).
- `jobs/bakeoff_run.job` — generic launcher; takes `RUN` + `LOSS_PRESET` +
  `ROBOMETER_PROCESSED_DATASETS_PATH` env vars. Walltime: 72h (May-3).
- `jobs/preprocess_split_v2.job` — HF cache build, one invocation per split.

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

| Run | resolved hyperparam | resolved lr | resolved max_steps | resolved icl_task_dropout | resolved failure_kl |
|----:|--------------------:|------------:|-------------------:|--------------------------:|--------------------:|
| 1   | c=0.0               | 4.0e-5      | 7500               | off (absent)              | off                 |
| 2   | c=0.5               | 4.0e-5      | 7500               | off (absent)              | off                 |
| 3   | c=1.5               | 4.0e-5      | 7500               | off (absent)              | off                 |
| 4   | c=0.5               | 4.0e-5      | 7500               | true                      | off                 |
| 5   | λ=0.3               | 2.0e-5      | 7500               | off (absent)              | off                 |
| 6   | λ=0.5               | 2.0e-5      | 7500               | off (absent)              | off                 |
| 7   | λ=0.3               | 2.0e-5      | 7500               | true                      | off                 |
| 8   | λ=0.3               | 2.0e-5      | 7500               | off (absent)              | weight=1.0          |

## What we are NOT changing across runs

Anything not listed in the table is held constant via `train_lora_base.yaml` and the loss
preset:

- Base model: `Qwen/Qwen3-VL-4B-Instruct` (loaded from `robometer/Robometer-4B`).
- LoRA enabled (`model.use_peft: true`).
- Dataset: `robometer_frames_train` (~18k trajectories) from the v2 HF cache.
- ICL on with `icl_prob=0.5`, pairs index `pairs_index_train.jsonl`.
- 7500 steps, eval every 500, save every 1000.
- Dual-ICL eval (each round runs once with `icl_prob=1.0` and once with `0.0`).
- Per-run wandb experiment name = `bakeoff_runN_<descriptor>`.

## May-3 relaunch — what changed since the May-1 attempt

The May-1 bake-off (job IDs `6rbgwy6s`, `efv1j9ve`, `3cauk8nh`, `6fnmmncs`, `zjo5w8u3`,
`s4u9bhpb`, `wvcelmbv`) was killed at the 24h walltime cap near step 2000, evaluated
against a corrupted rebuild of the original HF cache, and silently trained against
failure trajectories that had no per-frame supervision (`frame_labels` was being dropped
at the HF schema layer). All three issues are addressed in the May-3 relaunch:

1. **Walltime**: 24h → 72h. Earlier runs hit timeout near step 2000; new budget covers
   7500 steps with margin.
2. **HF cache**: original `/projects/prjs1958/robometer_frames_hf/` (broken rebuild) →
   v2 at `/projects/prjs1958/robometer_frames_hf_v2/`. The original cache is preserved
   untouched as police-work evidence; v2 has `frame_labels` correctly populated on every
   split's failures (100% coverage on train, warmup, test, all 4 eval splits).
3. **Loss config**: `loss2_c51.yaml` now sets `data.failure_label_smoothing: linear` so
   L2 failures get a piecewise-linear ramp interpolated between rubric anchor frames
   (matches the pretrained C51 head's expected target shape). L1 keeps `none` since L1
   bucketises both classes onto the same 5-quantile grid.
4. **Loader fixes**: `_build_manifest_index` adds `droid` to default families;
   `manifest_index.get(episode_dir) or manifest_index.get(episode_id)` recovers
   failsafe/metaworld eval coverage from 0%/54% → 100%.
5. **New run**: Run 8 (L2 + failure-KL anchor) added to test KL-anchored failure
   distribution stability.

## May-4 relaunch — HF cache cleanup + bake-off resume

The May-3 launch crashed at step 764 on a `frame_labels` length mismatch in a
roboreward-sourced droid trajectory (16 frame_labels, 6 frames). Root cause: roboreward's
copy of the droid family had inconsistent labels-vs-frames counts on a small subset, and
roboreward was never supposed to be in the LoRA training set in the first place.

### What changed in the cache (canonical = `/projects/prjs1958/robometer_frames_hf_v2/`)

Only **two splits** were rebuilt; the other five remain May-3 builds:

| Split            | Build  | Source                                                        |
|:-----------------|:-------|:--------------------------------------------------------------|
| `train`          | May-4  | rebuilt without roboreward (18,000 → 17,315 trajectories)     |
| `eval_droid`     | May-4  | rebuilt without roboreward (1,041 → 962 trajectories)         |
| `test`           | May-3  | unchanged                                                     |
| `warmup`         | May-3  | unchanged                                                     |
| `eval_robometer` | May-3  | unchanged                                                     |
| `eval_metaworld` | May-3  | unchanged                                                     |
| `eval_failsafe`  | May-3  | unchanged                                                     |

The `pairs_index_train.jsonl` and `pairs_index_eval_droid.jsonl` under
`/scratch-shared/$USER/robometer_frames_splits/` were filtered to drop roboreward rows
(backups: `*.bak_pre_roboreward_filter`); the May-4 rebuild was driven from those filtered
indices.

Pre-May-4 versions of the rebuilt splits are preserved as backups:
`{train,eval_droid}_raw.bak_pre_no_roboreward/` and the matching path-encoded preprocessed
dirs `*_raw_robometer_frames_*.bak_pre_no_roboreward/` inside the canonical dir. They are
NOT live data — only kept for rollback/forensics.

### Why the `_gpu_a100*/` dirs still exist (the "v2_a100_v2 mess")

The May-4 rebuild was first attempted on staging then raced across cbuild / gpu_h100 /
gpu_a100 partitions to beat the queue. Each partition wrote to its own scratch dir:

- `/projects/prjs1958/robometer_frames_hf_v2_gpu_a100/`     — eval_droid build
- `/projects/prjs1958/robometer_frames_hf_v2_gpu_a100_v2/`  — train build (backup attempt)

When the train + eval_droid builds completed, their dirs were `mv`'d into canonical.
**However**, the HF preprocessed dataset bakes **absolute NPZ paths** into its arrow rows
at build time. After the move, the arrow data still points at the original `_gpu_a100*/`
build paths. Rather than rewrite the arrow (slow, brittle), we left the parent dirs in
place and put **symlinks back to canonical**:

```
/projects/prjs1958/robometer_frames_hf_v2_gpu_a100/
├── eval_droid_raw                                                        → canonical/eval_droid_raw
└── _projects_prjs1958_robometer_frames_hf_v2_gpu_a100_eval_droid_raw_*/  → canonical/_projects_..._eval_droid_raw_*/

/projects/prjs1958/robometer_frames_hf_v2_gpu_a100_v2/
├── train_raw                                                             → canonical/train_raw
└── _projects_prjs1958_robometer_frames_hf_v2_gpu_a100_v2_train_raw_*/    → canonical/_projects_..._train_raw_*/
```

Result: every absolute NPZ path inside the canonical arrow data still resolves correctly
through the symlinks. Downside: the canonical dataset is no longer relocatable on its own.
Acceptable since the user has decided to drop the HF-cache approach in future iterations.

### Resume mechanism (May-4 jobs)

The May-3 jobs all reached step ≥500 before crashing, so the per-run `SaveBestCallback`
had stamped a `ckpt-avg-5metrics=0.0000_step=500/` checkpoint with full optimizer +
scheduler + step counter state. The May-4 launcher (`jobs/bakeoff_run.job` with the
`CKPT_RESUME` env var) passes this as `++training.resume_from_checkpoint=<path>`, so the
new jobs pick up at step 500 with continuity in optimizer momentum, LR schedule, and
WandB curves.

WandB resumption uses the per-run `WANDB_RUN_ID=<id> WANDB_RESUME=allow` env vars so
the May-3 metric history continues into the same run. Net loss vs an uninterrupted
training: 264 wasted steps (504→764 of the May-3 attempt) that get retraced. No eval
results from May-3 are lost — they live on the same WandB run.

### May-4 job IDs

The first relaunch attempt (jobs 22458449–22458464) failed at launch on a Hydra parse
error: the resume checkpoint path contains `=` characters (`ckpt-avg-5metrics=0.0000_step=500`),
and the unquoted `++training.resume_from_checkpoint=$CKPT_RESUME` override produced
multiple `=` signs that Hydra's grammar rejected. `bakeoff_run.job` was patched to wrap
the value in single quotes before the second relaunch.

| Run                          | New JID    | Old JID    | WandB Run ID | Preset       |
|:-----------------------------|:-----------|:-----------|:-------------|:-------------|
| `run1_l1_c0`                 | 22458562   | 22443128   | pmj014oq     | loss1_corn   |
| `run2_l1_c05`                | 22458563   | 22443129   | rt0572ds     | loss1_corn   |
| `run3_l1_c15`                | 22458564   | 22443130   | ibo7d7hm     | loss1_corn   |
| `run4_l1_c05_dropout`        | 22458565   | 22443131   | x2fvnbr0     | loss1_corn   |
| `run5_l2_lambda03`           | 22458566   | 22443132   | kdr9jbd0     | loss2_c51    |
| `run6_l2_lambda05`           | 22458567   | 22443133   | gu2lnixa     | loss2_c51    |
| `run7_l2_lambda03_dropout`   | 22458568   | 22443135   | sxb6rc8g     | loss2_c51    |
| `run8_l2_lambda03_kl`        | 22458569   | 22443273   | eejpy0ud     | loss2_c51    |
| `run9_l2_lambda03_kl_w03`    | 22458570   | 22443312   | 98osea5v     | loss2_c51    |

Pre-launch validation (also May-4): the OLD-step7500 ckpt (`loss2_22244009/.../checkpoint-7500`)
was probed against the post-merge canonical cache via `jobs/eval_test_set.job`
(`OUTPUT_TAG=oldckpt_postmerge_v2_symlinked`, job 22457408). Test split ranking_acc was
identical to the pre-merge baseline (iclon=0.7574, icloff=0.7083) and eval_droid loaded
+ scored cleanly on the new no-roboreward build (iclon=0.5969, icloff=0.5314).
