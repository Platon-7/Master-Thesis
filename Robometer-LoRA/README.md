# Robometer-LoRA

Loss-function bake-off for LoRA-finetuning Robometer-4B on the preprocessed keyframe dataset
at `/projects/prjs1958/robometer_frame_dataset/`. Two candidate losses (`losses.md`):

- **Loss 1** — asymmetric CORN ordinal loss, fresh 4-logit head, progress-only.
- **Loss 2** — Chris' asymmetric C51 + asymmetric BCE, pretrained head weights kept,
  progress + success heads.

Phase-1 ICL plumbing and Phase-2 label-harmonization (failure rubric, success bucketization
+ noise, endpoint anchors) are shared across both variants. See `MODIFICATIONS.md` for the
detailed design.

## Layout

```
configs/          Hydra overrides — train_lora_base.yaml + per-loss presets
data_adapters/    keyframe-tar → Robometer trajectory dict
scripts/          build_splits.py + four CPU smoke tests
jobs/             SLURM job files (preprocess_split / train_loss1 / train_loss2)
logs/ results/    run artefacts (gitignored)
```

Upstream dependency: sibling clone at `../Robometer/`.

## Conda environments

Two envs (CPU smoke vs CUDA training):

```bash
# CPU env — used to run smoke tests on the login node.
conda create -n robometer_train -c conda-forge --override-channels python=3.10
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt

# GPU env — used by the SLURM training jobs.
conda create -n robometer_gpu  -c conda-forge --override-channels python=3.10
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision
# Then: full Robometer install (transformers, peft, accelerate, qwen-vl-utils,
# unsloth, xformers==0.0.32.post2, bitsandbytes, etc. — see Robometer/pyproject.toml).
```

## Smoke tests (CPU, no checkpoint required)

```bash
python scripts/smoke_test_icl.py                   # ICL collator branch
python scripts/smoke_test_labels.py                # rubric / smoothing / unmasking
python scripts/smoke_test_asymmetric_losses.py     # c51_asymmetric + bce_asymmetric (Loss 2)
python scripts/smoke_test_corn_loss.py             # corn_asymmetric (Loss 1)
python scripts/smoke_test_stratified_warmup.py     # warmup-only → exact 50/50 phase switch
```

All four should print "all checks passed."

## End-to-end training pipeline

```bash
# 1. Build train + warmup + eval splits (~30 s, login node).
python scripts/build_splits.py \
    --output-dir /scratch-shared/$USER/robometer_frames_splits

# 2. Materialise HF datasets per split — six SLURM jobs, runnable in parallel.
for SPLIT in train warmup eval_droid eval_robometer eval_metaworld eval_failsafe; do
    SPLIT=$SPLIT sbatch jobs/preprocess_split.job
done

# 3. Smoke train (10 steps each, catches pipeline errors before a real run).
MAX_STEPS=10 sbatch jobs/train_loss1.job
MAX_STEPS=10 sbatch jobs/train_loss2.job

# 4. Real bake-off runs in parallel.
sbatch jobs/train_loss1.job
sbatch jobs/train_loss2.job
```

Outputs land in `Robometer-LoRA/logs/` (text logs) and on Weights & Biases under project
`robometer_loss_bakeoff` (override via `WANDB_PROJECT`).

## Wandb metrics — what to look at

Robometer already logs per-step `prog_loss`, `spearman_corr`, `success_loss`, `success_auprc`,
balanced accuracy, etc. The bake-off-specific additions (P7) are:

- `train|eval/prog_ordinal_mae` — average ordinal-units error per frame (off-by-one severity
  that binary accuracy misses). In bin units for `c51_asymmetric`, in {1..5}-level units for
  `corn_asymmetric`.
- `train|eval/success_fp_rate` — false-positive rate per batch on the success head. The
  primary "reward hacking" diagnostic from `losses.md`.
- `train|eval/success_fn_rate` — false-negative rate per batch.

Trajectory-level `success_auc` and `ECE` (per `losses.md` §Evaluation) require accumulating
predictions across the full eval set; deferred until we see the per-step signals on real
runs and decide it's worth the eval-loop plumbing.

## Configuration knobs you'll most likely tune

| Knob | Default | Purpose |
|---|---|---|
| `data.icl_prob` | 0.5 | Per-sample probability of prepending a successful-demo trajectory as in-context example. |
| `data.failure_label_smoothing` | `none` | Loss 2 sets to `linear` to ramp stair-step rubric labels. Loss 1 keeps `none`. |
| `data.success_label_noise_std` | 0.0 (Loss 2), 0.1 (Loss 1) | Per-frame Gaussian noise on success labels (interior frames only — endpoints anchored). |
| `loss.asymmetric_lambda` | 0.3 | Loss 2 — damps the false-negative side of both heads. |
| `loss.corn_asymmetric_c` | 1.5 | Loss 1 — `α_k = 1 + c·(k-2)` schedule for cumulative thresholds. |
| `model.progress_head_mode` | `c51` | Set to `corn` for Loss 1 (4-logit head, random init). |
| `data.stratified_batch_balance` | `false` | When true, `StratifiedWarmupBatchSampler` enforces exact 50/50 fail/success per batch in the main phase. Both Loss 1 and Loss 2 presets enable this. |
| `data.warmup_steps` | 0 (base), 2000 (Loss 1), 1000 (Loss 2) | First N training steps draw batches only from the warmup pool (failures only). After step N the sampler switches to stratified 50/50. |

## When something breaks

- `smoke_test_*.py` failure → P1/P2/P3/P4 regression in the upstream Robometer files (head,
  sampler, collator, trainer). Bisect by checking `git diff Robometer/`.
- Preprocess job crash on `preprocess_datasets` step → likely an OmegaConf interpolation
  miss (`SPLIT` or `HF_DIR` env var not exported). The job sets both; running it manually
  outside SLURM means re-exporting them.
- Train job OOM → GPU 80 GB is enough for 4B + LoRA + ICL. If short, drop
  `data.icl_prob` to 0.0 first (skips the demo-frame doubling).
- `unsloth` / `xformers` import error on training start → the `robometer_gpu` env's torch
  version drifted. Re-pin `torch==2.8.0` and `xformers==0.0.32.post2`.

## Outstanding open items

1. Inference-side LoRA-adapter loading bug from `robometer/robometer#23` — applies when we
   go to use the trained checkpoint in `example_inference.py` / `eval_server.py`. Surfaces
   only at evaluation, not training.
2. Trajectory-level eval metrics (Success AUC, ECE) — deferred per above.
3. `loss_sweep.yaml` is stale (refers to the old l1/l2/disc4/disc10 sweep that pre-dates the
   actual bake-off). Safe to ignore; the per-loss presets are the source of truth now.
4. **Preprocess MP4 round-trip — must be removed before any full-dataset run.**
   Robometer's `dataset_upload/helpers.py:create_trajectory_video_optimized` re-encodes
   our already-extracted JPGs into per-episode H.264 MP4 files at preprocess time. The
   `output.use_video=false` flag in `dataset_frames.yaml` does not actually skip this
   (the function ignores its `use_video` parameter — verified in
   `dataset_upload/helpers.py:333-384` and `generate_hf_dataset.py:182-225`). For the
   ~22k-episode bake-off this is bounded waste (~22k MP4 inodes, ~30 GB extra disk, plus
   ffmpeg CPU); accepted for now to unblock training. **For the hypothetical full
   ~600k-episode run this becomes ~600k inodes / ~800 GB and must be fixed first.**
   Three viable fixes (smallest-to-biggest patch):
     - Patch `create_hf_trajectory` to actually honor `use_video=false` (skip MP4, embed
       JPG bytes via `Sequence(Image())`). ~30 LOC in `Robometer/dataset_upload/helpers.py`.
     - Custom branch for `robometer_frames_*` inside `generate_hf_dataset.py::main` that
       bypasses MP4 encoding entirely. ~50 LOC, no upstream-shared change.
     - Tar-direct dataset at training time, no preprocess output at all. ~80 LOC, zero
       inodes / disk for the cache; slower per-batch I/O.
