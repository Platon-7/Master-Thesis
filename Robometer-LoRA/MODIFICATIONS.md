# Robometer LoRA Loss-Function Sweep — Setup Plan

Goal: evaluate progress-head loss functions on the preprocessed keyframe dataset at
`/projects/prjs1958/robometer_frame_dataset/` before committing to a full LoRA run.

## Upstream state (cloned at `../Robometer/`)

- Base model: `Qwen/Qwen3-VL-4B-Instruct` (+ optional init from `robometer/Robometer-4B`).
- Three heads: `progress`, `preference`, `success` (`robometer/models/heads.py`).
- LoRA is first-class: `model.use_peft=true` in `robometer/configs/config.yaml`.
- Progress loss is **already configurable** via `loss.progress_loss_type ∈ {l1, l2, discrete}`
  with `loss.progress_discrete_bins` (`_compute_progress_loss_helper` in
  `robometer/trainers/rbm_heads_trainer.py:2086`).
- Per-frame progress for successes is **auto-derived** from frame indices + the
  `dataset_success_cutoff.txt` entry (`helpers.py:497-578`). No manual success labels.
- Per-frame progress for failures is **masked out by default** (`should_compute_progress`
  returns 0 for `quality_label ∈ {suboptimal, failure, failed}`); we must inject our
  curated `frame_labels` to use them.
- Built-in augmentations work on frame indices — compatible with our 16 fixed keyframes
  (no 64-frame requirement). Enabled via `preference_strategy_ratio` and
  `progress_strategy_ratio` vectors in `config.yaml:92-94`.
- **No built-in ICL (query + success-demo context) mode.** `use_multi_image=True` exists
  but does not prepend a context trajectory. A collator patch is required.

## Local dataset recap (`/projects/prjs1958/robometer_frame_dataset/`)

| family    | size  | manifests                        | keyframes                                |
| --------- | ----- | -------------------------------- | ---------------------------------------- |
| failsafe  | 2.9 G | `{pick,push,stack}_{f,s}.jsonl`  | tar shards per archive                   |
| metaworld | 42 G  | `<task>_{failures,successes}.jsonl` | tar shards per archive                |
| droid     | 15 G  | `<ds>_{f,s}.jsonl`               | tar shards per archive                   |
| robometer | 138 G | `*.jsonl` (Group A/B)            | tar shards (paired + orphan successes)   |
| roboreward| 12 G  | `*.jsonl`                        | tar shards                               |

Per-episode shape: 16 keyframes (`frame_NN_<time>s.jpg`) + `meta.json` + manifest row with
`frame_labels: List[int in {1,2,3,4}]` (1 = no progress, 4 = terminal success).
Failures are curated by hand (non-monotonic allowed); successes are
auto-derived by Robometer from frame index + cutoff.

### The partner-pair index

`/projects/prjs1958/robometer_frame_dataset/pairs_unified.jsonl` (598,511 rows) — the
single source of truth for "which episodes have a paired partner available as ICL
context." Fields: `episode_id`, `partner_episode_id`, `tier`, `frames_path`,
`partner_archive`, `partner_task`.

Per-source counts:

| source                    | pairs   |
| ------------------------- | ------- |
| metaworld                 | 29,528  |
| failsafe                  | 1,376   |
| droid (curated)           | 11,366  |
| robometer (Group A/B)     | 65,443  |
| robometer_orphan_success  | 490,798 |

Additional per-archive orphan-pair files under
`robometer/pairs_orphan/*_orphan_pairs.jsonl` (the oxe_droid one is our success source).

## Training set composition (~18k datapoints)

**Failures (~9k, all have partners):**
- A small slice of `robometer` Group A (the curated Group A from `pairs_unified.jsonl`).
- A small slice of the `droid` family (curated — **not** `jesbu1_oxe_rfm_oxe_droid`
  orphans).
- Mix ratio to be tuned so the total is ~9k and the split is ~balanced.

**Successes (~9k, all orphans with partners):**
- Sampled from `robometer/keyframe_orphan_success/` via
  `robometer/pairs_orphan/*_orphan_pairs.jsonl`.
- **~4.5k must come from `jesbu1_oxe_rfm_oxe_droid_orphan_pairs.jsonl`** so the success
  origin distribution approximately matches the failure origin distribution (which
  includes curated droid data). The remaining ~4.5k spreads across the other orphan
  archives.

**ICL coin flip (50/50 per batch element):**
- Heads → query alone.
- Tails → query + success-demo context prepended. Context = the `partner_episode_id`
  trajectory from `pairs_unified.jsonl` (or the per-archive orphan pair file).
- Because every sampled datapoint was filtered on "has partner", the context always
  exists and there is no fall-back logic.

## Evaluation set

Held-out, sampled before training begins:

| bucket            | size    | source                                   |
| ----------------- | ------- | ---------------------------------------- |
| droid (curated)   | ~1.1k   | 10% of our droid failures + their pairs  |
| robometer (A/B)   | ~6.5k   | 10% of Group A/B                         |
| metaworld         | ~500    | random, trusted labels                   |
| failsafe          | ~500    | random, trusted labels                   |

Metaworld + failsafe weighted higher per-example to reflect label trust. Eval also runs
with the same 50/50 ICL coin flip so the train/eval distribution matches.

## Progress-label convention

| trajectory label | `target_progress` source                                                   |
| ---------------- | -------------------------------------------------------------------------- |
| `successful`     | Robometer default — linear t/T from frame index + cutoff (sim=1.0, real=0.95) |
| `failure`        | **Our curated `frame_labels ∈ {1..4}`**, mapped to [0,1] or bin indices via `frame_labels_to_progress.py` |

This requires the sampler patch (§A.5). Without it, failure labels are dropped.

## Augmentation budget for the first sweep

| augmentation                    | weight | note                                           |
| ------------------------------- | ------ | ---------------------------------------------- |
| `forward`                       | 1      | baseline                                       |
| `reverse`                       | 1      | teaches direction                              |
| `rewind`                        | 1      | teaches non-monotonic handling (redundant with our failure labels — worth ablating on/off separately once baseline lands) |
| `DIFFERENT_TASK_INSTRUCTION`    | 1      | task grounding                                 |
| `SUBOPTIMAL` (preference-only)  | 1      | natural fit: we already have failure+success pairs |

Config via `preference_strategy_ratio` and `progress_strategy_ratio` — no code change.

## Required modifications

### A. Upstream `Robometer/` edits — 2 new, 5 edited

1. **New loader** `dataset_upload/dataset_loaders/robometer_frames_loader.py` (~100 LOC).
   Driven by `pairs_unified.jsonl`. Reads manifest JSONL + extracts 16 JPGs from the tar
   shard on demand. Carries `frame_labels` (per-frame list) and
   `partner_episode_id`/`partner_frames_path` for ICL.
2. **Router entry** in `dataset_upload/generate_hf_dataset.py` main() — one `elif` branch
   (~8 LOC) routing `"robometer_frames" in dataset_name` → our loader.
3. **Data-gen config** `dataset_upload/configs/data_gen_configs/robometer_frames.yaml` —
   path + per-family filter + train/eval split sizes.
4. **Cutoff file append** — add to `robometer/data/dataset_success_cutoff.txt`:
   `robometer_frames_metaworld,1.0`, `robometer_frames_failsafe,1.0`,
   `robometer_frames_droid,0.95`, `robometer_frames_robometer,0.95`.
5. **Sampler patch** `robometer/data/samplers/progress.py` (~30 LOC). When a trajectory
   has a per-frame `frame_labels` list, short-circuit the default linear-progress
   computation and emit `target_progress` from our mapping. Only applies to failures —
   successes still flow through the stock path.
6. **Collator patch** `robometer/data/collators/rbm_heads.py` (~80 LOC). Accept
   `context_trajectory` from the sample. When present, prepend the context frames +
   instruction to the prompt; target labels still reference the query trajectory only.
   Gated by a new `data.use_icl: bool` flag + `data.icl_prob: float` coin flip.

### B. New `Robometer-LoRA/` workspace — this folder

```
Robometer-LoRA/
├── MODIFICATIONS.md                 ← this file
├── README.md                        ← run instructions
├── configs/
│   ├── dataset_frames.yaml          ← data_gen config per family
│   ├── train_lora_base.yaml         ← Hydra overrides shared across loss variants
│   └── loss_sweep.yaml              ← matrix: {l1, l2, discrete@4, discrete@10}
├── data_adapters/
│   ├── robometer_frames_loader.py   ← authoritative copy; symlinked into Robometer
│   ├── frame_labels_to_progress.py  ← {1,2,3,4} → bins / continuous
│   └── build_splits.py              ← sample 9k/9k + eval splits from pairs_unified.jsonl
├── scripts/
│   ├── preprocess.sh                ← uv run python -m dataset_upload.generate_hf_dataset …
│   ├── train_one.sh                 ← one run with a given loss override
│   └── launch_sweep.py              ← reads loss_sweep.yaml → submits sbatch array
├── jobs/
│   ├── preprocess.job               ← SLURM, no GPU
│   ├── train_lora.job               ← SLURM, 1× GPU, parameterised by LOSS_ID
│   └── loss_sweep.job               ← SLURM array (one task per loss variant)
├── logs/                            ← stdout/stderr (gitignored)
└── results/                         ← eval CSVs & plots (gitignored)
```

No code duplication: `data_adapters/robometer_frames_loader.py` is the source of truth;
the upstream `Robometer/dataset_upload/dataset_loaders/` copy is a symlink so edits
round-trip.

## Loss variants in the first sweep

| id          | `progress_loss_type` | `progress_discrete_bins` | notes                            |
| ----------- | -------------------- | ------------------------ | -------------------------------- |
| `l1`        | l1                   | —                        | regression baseline              |
| `l2`        | l2                   | —                        | regression baseline              |
| `disc4`     | discrete             | 4                        | matches `frame_labels ∈ {1..4}` directly |
| `disc10`    | discrete             | 10                        | Robometer default                |

**Deferred** — `disc4_mono` (monotonic penalty on successes only, ~20 LOC). Adds a knob
only worth pulling if the baseline sweep shows `disc4` under-performing `l2` on success
ordering.

## Modification count summary

| where                            | files new | files edited | LOC   |
| -------------------------------- | --------- | ------------ | ----- |
| upstream `Robometer/`            | 2         | 5            | ~220  |
| `Robometer-LoRA/` (this folder)  | 12        | 0            | ~500  |
| **total**                        | **14**    | **5**        | **~720** |

Excluding the ICL collator patch (if deferred), upstream drops to ~140 LOC / 4 edited
files. ICL is the single biggest upstream change; the rest is boilerplate.

## Open questions / decisions still to make

1. ICL coin-flip probability (default 0.5 — reasonable? or 0.3 to reduce prompt length
   cost, or 0.7 to push harder on context use?).
2. Turn `rewind` augmentation off? It teaches non-monotonic patterns which our failure
   labels already supply. Running both with and without would be one of the more
   informative ablations once the loss sweep lands.
3. Train budget per variant — current `train_lora_base.yaml` has `max_steps=400`,
   `per_device_train_batch_size=4`, `gradient_accumulation_steps=2` → ~3.2k examples
   seen per variant, roughly one pass over a ~9k-aligned sample. Bump to 800 for a
   proper two-pass screen?
