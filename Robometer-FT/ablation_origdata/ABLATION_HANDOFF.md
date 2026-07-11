# Loss-vs-Data Ablation — Handoff (2026-07-11)

This document hands off the **loss-vs-data ablation** to an agent on a
different cluster. Everything below was built and partially tested on the AWS
research-partner-cluster. **All absolute paths in this doc and in the reference
job script are AWS-cluster paths and WILL differ on your cluster. All Slurm
directives (partition, account, GPU type) and the wandb entity/project are
cluster-specific too. Adapt them; do not copy them blindly.**

## 1. What this experiment is and why

Our thesis fine-tune ("run2": Robometer-4B + our re-curated failure data + our
asymmetric loss) beats the released Robometer-4B on offline metrics. That
result confounds two contributions: the **data** (dense-labeled failure data)
and the **loss** (asymmetric reweighting). This ablation disentangles them by
training Robometer-4B **on its own original data** (no failure additions of
ours) in two arms:

- **asym arm**: upstream data + our asymmetric progress loss + our asymmetric
  success loss (lambda = 0.3)
- **std arm**: upstream data + upstream losses, otherwise identical
  (this twin also prices the effect of merely continuing training)

Reading the 2x2 afterwards (against run2 and the released baseline, both of
which we already have):

- abl-asym close to run2  => the **loss** carries the gains
- abl-asym close to baseline => the **data** carries the gains
- abl-std vs baseline => how much "just more training" is worth

## 2. Training recipe (must match run2 for a fair read)

Continue-training from the released Robometer-4B checkpoint
(HF: the released 4B reward model; on AWS it sits in
`/shared/home/PKA4388/checkpoints/Robometer-4B`, a Qwen3-VL-4B with 3 reward
heads: C51 10-bin progress, BCE success, preference):

| knob | value | why |
|---|---|---|
| progress_pred_type | `absolute_first_frame` | matches run2; do NOT use `awt` (it interacts badly with the asymmetric head and would confound the read) |
| max_frames | 16 | matches run2 |
| max_steps | 5000 | matches run2 (run2's selected checkpoint is s4000; save every 500 so we can pick per-step) |
| save_steps | 500 | checkpoint selection later |
| learning_rate | 1e-5 | matches run2 |
| per_device_train_batch_size | 8, on 8 GPUs | matches run2's effective batch |
| bf16 + gradient_checkpointing | true | fits A100-40GB / H100 |
| train_preference_head | true (upstream default) | upstream consumes its archives' failures ONLY via preference pairs; without the preference head the original failure data would be unused |
| sample_type_ratio | `[1,0,0]` (upstream default) | keep upstream's sampling |
| asym arm extra | `++loss.progress_loss_type=c51_asymmetric ++loss.success_loss_type=bce_asymmetric ++loss.asymmetric_lambda=0.3` | our loss |
| std arm | none of the three overrides | upstream loss |

## 3. Code: fresh upstream clone + one patch

Do NOT use our Robometer fork. The ablation must run on the **authors'
unmodified pipeline** so the data treatment is theirs. Steps:

1. `git clone https://github.com/robometer/robometer Robometer-temp`
2. `git apply robometer_asym_loss.patch` (file in this directory; 207 lines,
   touches exactly 2 files):
   - `robometer/trainers/rbm_heads_trainer.py` — aliases `c51_asymmetric`
     into the discrete-mode checks and adds the asymmetric reweighting after
     the CE line (detached p_hat vs p_t, `torch.where`, lambda from config);
     wraps the success loss so `bce_asymmetric` computes `L_neg +
     lambda*L_pos` with empty-class guards, `else` falls through to the
     original balanced code.
   - `robometer/configs/experiment_configs.py` — adds `success_loss_type`
     and `asymmetric_lambda` fields to the loss config.
   With the three overrides absent, every added branch is dormant — the std
   arm runs byte-identical upstream code.
3. Python env: needs the upstream repo's deps (transformers with Qwen3-VL,
   accelerate, flash-attn, datasets, hydra). On AWS we reused our FT conda env
   (`robometer_gpu_fa2`). The warning `You are using a model of type qwen3_vl
   to instantiate a model of type qwen2_5_vl` at load is benign — it appears
   in every healthy run of ours.

## 4. Data: the "origdata" cache

Upstream loads preprocessed (step-2) caches via env var
`ROBOMETER_PROCESSED_DATASETS_PATH` and resolves each entry of
`data.train_datasets` to a subdirectory named by
`dataset_path.replace("/", "_")`. Each per-dataset dir must contain
`dataset_info.json`, `processed_dataset/` (HF datasets on disk), and
`index_mappings.json`.

`build_origdata_cache.py` (in this directory) builds the ablation's train
cache from our merged `train_no_extras` step-2 cache. What it does:

- drops **metaworld** (28,956 rows) and **failsafe** (1,618) entirely — our
  additions, not upstream's;
- drops **droid AND failure-ish** rows (6,037) — our relabeled failures; the
  original OXE droid successes stay;
- keeps everything else **including the original archives' failures** (82,632
  rows) — upstream consumes these only as `quality_label` for preference
  sampling; it never reads our `frame_labels` column, so its presence is
  contamination-proof;
- remaps `index_mappings.json` indices and prunes empty groups.

Result on AWS: **609,416 rows** written to
`robometer_frames_origdata_step2/robometer_origdata_train` (the dir name has
no slash so it mangles to itself). At training time the loader applies its own
filtering and reports **597,359 usable trajectories (70,577 failure /
526,782 successful)** — both numbers are expected, do not chase the gap.

**Input requirement on your cluster:** the merged step-2 cache
(`train_no_extras`) must exist there. On AWS it lives at
`robometer_frames_hf_full_step2/_fsx_PKA4388_robometer_frames_hf_full_train_no_extras_raw_robometer_frames_train_no_extras`
(the long name IS the path-mangling). Point `SRC`/`DST_ROOT` in the script to
your cluster's copies. If your cluster only has the step-1/raw data, the
step-2 preprocessing has to run first (upstream `robometer` repo, step-2
script) — ask the user rather than guessing.

- Eval set: upstream's own eval split. On AWS we symlinked our copy into the
  cache root as
  `_fsx_PKA4388_robometer_frames_hf_full_eval_robometer_raw_robometer_frames_eval_robometer`
  and passed `++data.eval_datasets=[/fsx/PKA4388/robometer_frames_hf_full/eval_robometer_raw/robometer_frames_eval_robometer]`
  (the override value is the UNMANGLED path; the loader mangles it to the
  symlink name). Reproduce the same trick with your paths.

## 5. Job script

`ablation_origdata_reference.job` (this directory) is the exact AWS Slurm
script, kept as a **reference**, not for direct submission. Anatomy:

- single node, 8 GPUs, `accelerate launch` with an FSDP config
  (`Robometer-FT/configs/distributed/fsdp.yaml` in this repo), 96 CPUs,
  exclusive;
- `ARM=asym|std` env var switches the loss overrides;
- output dir is **stable per arm** (`abl_origdata_asym` / `abl_origdata_std`,
  no job-id suffix) so the built-in auto-resume block (scan for latest
  `checkpoint-*`, pass `++training.resume_from_checkpoint=`) works across
  resubmissions — keep this property on your cluster;
- `++logging.wandb_project=Robometer_FT ++logging.wandb_entity=nlp-squad` —
  **replace with an entity/project your wandb key can write to**; upstream's
  default project (`rbm-model`) does not exist for our key and crashes
  `wandb.init` with `CommError: project not found` right after dataset load
  (we hit exactly this). `WANDB_MODE=offline` is the safe fallback.
- all paths in the script are absolute AWS paths — rewrite them.

Launch = two jobs, same script, `ARM=asym` and `ARM=std`.

## 6. What has actually been tested (and what has NOT)

Tested and verified on AWS:

1. **CPU smoke test (login node, stubbed GPU deps)**: hydra config composes
   with all our overrides; the cache loads through the unmodified upstream
   loader; batches mix sample kinds; the asym code paths execute. Note: the
   smoke printed `rejected trajectories with nonzero progress targets: 14/24`
   — this is EXPECTED upstream behavior (negatives generated from successes
   carry ramp targets), not a bug.
2. **Full 8-GPU FSDP launch on a spot A100 node (job 1595)**: dataset loaded
   (597,359 trajectories, counts above), preference sampler initialized
   ("Has suboptimal: True"), model shards loaded — then crashed at
   `wandb.init` (the CommError above, since fixed by the overrides in §5).
3. The patch applies cleanly to upstream HEAD as of 2026-07-10 and the std
   arm's code path is dormant-identical (verified by reading the diff — all
   changes are behind the two new config values).

NOT yet tested — nobody has seen these run:

- an actual **training step** (loss goes down, no NaN, throughput);
- **checkpoint saving** at step 500 and the **auto-resume** block;
- the asym loss numerics at scale (it mirrors our fork's implementation,
  which trained run2 successfully, but this re-implementation inside upstream
  has never taken a gradient step).

So: after launching, watch the first ~100 steps of BOTH arms (losses finite
and decreasing, `grad_norm` sane, ~step timing consistent with 5000 steps
fitting your wall-time), and confirm a `checkpoint-500` appears.

## 7. After training

1. Consolidate the FSDP checkpoints to HF format if needed (our run2 workflow;
   watch for the mixed-dtype gotcha: consolidated checkpoints can carry stray
   fp32 params — unify dtypes at load).
2. Offline evals for the 2x2 read: the OOD held-out-embodiment set is the fair
   comparison for all four models; the in-distribution eval is the sharpest
   probe. (Eval harness lives in the thesis repo, `reward-model-study/`.)
3. Compare: abl-asym vs run2 vs released-4B vs abl-std, per §1.

## 8. Files in this directory

- `ABLATION_HANDOFF.md` — this file
- `robometer_asym_loss.patch` — the 2-file loss patch for the fresh upstream clone
- `build_origdata_cache.py` — builds the origdata train cache (edit SRC/DST first)
- `ablation_origdata_reference.job` — the AWS job script, reference only
