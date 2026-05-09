# Full FT data-loading handoff: bypass HF wrap + MP4 entirely

## What this is

Instructions for the agent implementing data loading for the Robometer full
fine-tune ("big run"). The current LoRA bake-off goes through a fragile
HuggingFace dataset cache + ffmpeg MP4 intermediate. That stack has caused
multiple wasted compute days. The big run should skip both.

**Goal**: source JPG tar shards → trainer frame array, in one direct path. No
MP4 encoding, no HF `Dataset.save_to_disk`, no per-trajectory NPZ extraction.

## Why we're doing this

Catalog of bugs the HF + MP4 stack produced during the LoRA bake-off:

1. **`frame_labels` silently dropped from HF schema**. The
   `dataset_upload/generate_hf_dataset.py` `BASE_FEATURES` dict was missing
   `frame_labels`, so failure trajectories trained against a `t/T` linear
   target identical to successes. Failures were unsupervised for ~all of
   May-1 bake-off. Cost: 1 wasted bake-off attempt.

2. **Non-deterministic ffmpeg encoding**. Multi-threaded x264 produces
   different byte-level outputs across rebuilds. Ran `-threads 1` builds (v3
   lossless) → AUC dropped 0.7439 → 0.7165 because pixel-format drift
   (yuv420p ↔ yuv444p) made decoded pixels off-distribution from training.
   Cost: 2 days of debugging "regression" that was actually cache content drift.

3. **Absolute NPZ paths baked into arrow data**. HF preprocessing writes
   per-trajectory NPZ files at the build-time absolute path and stores that
   path in the arrow `frames` column. After we `mv`'d the dataset to a
   canonical location, every trajectory load failed with `NPZ not found`
   until we created cross-directory symlinks at the original build paths.
   Now the canonical dataset is non-relocatable without rewriting arrow data.
   See `/projects/prjs1958/robometer_frames_hf_v2_gpu_a100*/` for the
   symlink workaround that's still in place at handoff time.

4. **`n_keyframes` manifest claim ≠ actual JPGs in tar**. At least one droid
   trajectory (`REAL_2023-07-13_Thu_Jul_13_14-10-53_2023`) has manifest
   claiming 16 keyframes + 16 frame_labels but the keyframes/ tar has only 6
   JPGs. The HF preprocessing extracts 6 NPZs, ships an arrow row with
   `num_frames=6, frame_labels=[16 entries]`, and the trainer crashes at
   the labels/frames length check in `samplers/base.py:1019`. We've patched
   this with `SampleSkipRequest` (skip-and-retry) but the underlying tar
   inconsistency lives in source data. An audit job
   (`Robometer-LoRA/jobs/audit_keyframe_counts.job`) writes every flagged
   episode to `/scratch-shared/$USER/dataset_audit_report.tsv` — read this
   first before deciding what to filter.

5. **HF preprocessing is two-stage**. First `dataset_upload/generate_hf_dataset.py`
   builds the raw HF dataset (~1GB per split, references MP4s). Then
   `robometer/data/scripts/preprocess_datasets.py` extracts NPZs and writes
   a path-encoded sibling directory (~30GB per split). Two stages = two
   places to fingerprint-mismatch. Three sources of drift: source tar →
   MP4 → NPZ.

## Recommended approach

**Read JPGs directly from tar shards into the trainer.** Skip MP4 entirely.
Skip HF Dataset entirely. Use the manifests as the index.

### Source data layout (do not modify)

```
/projects/prjs1958/robometer_frame_dataset/
├── droid/
│   ├── manifests/*.jsonl                  ← episode metadata + frame_labels
│   ├── keyframes/shards/shard-NNNNN.tar   ← default-view JPGs (used by trainer)
│   ├── keyframes_ext2/shards/...          ← secondary view (optional)
│   └── keyframes_wrist/shards/...         ← wrist view (optional)
├── failsafe/
├── metaworld/
├── robometer/         (585k episodes — biggest family)
└── roboreward/        (NOT used in LoRA bake-off; user decides for full FT)
```

Manifest JSONL row schema (verified):
```json
{
  "episode_id": "REAL_2023-07-13_Thu_Jul_13_14-10-53_2023",
  "archive": "REAL", "family": "droid",
  "task": "Use object to pick up something.",
  "label": "failure",                       // success | failure | partial
  "terminal_reward": 3,
  "n_source_frames": null,
  "n_keyframes": 16,                        // CLAIM — may not match tar contents
  "frame_labels": [1,1,1,...,3],           // length should match actual JPG count
  "fps": null
}
```

JPG path inside tar: `<episode_id>__<task_words>/frame_NN_<seconds>s.jpg`

### Pair index (which trajectory pairs with which for ICL)

`/scratch-shared/$USER/robometer_frames_splits/pairs_index_<split>.jsonl`

Per-row: `{episode_id, partner_episode_id, label, partner_label, source, family, task, ...}`

For full FT, decide upfront:
- If using ICL: keep this index, use it to fetch demo trajectories at sample time.
- If dropping ICL: ignore.

### What to keep from existing code

`Robometer/robometer/`:
- `trainers/rbm_heads_trainer.py` — keep entirely. Trainer logic is fine.
- `data/samplers/{base,progress,pref}.py` — keep all sampler logic. The
  `_get_traj_from_data`, `_create_progress_sample`, etc. are the right
  abstraction. They expect a dict like `{id, frames, frame_labels,
  data_source, quality_label, task, ...}` where `frames` is either a numpy
  array (T,H,W,C) OR a string path to load. **Pass numpy arrays directly**
  to skip the path indirection.
- `data/samplers/base.py:SampleSkipRequest` — keep, useful for any
  unrecoverable per-sample issue.
- `data/datasets/rbm_data.py:RBMDataset` — the skip-and-retry `__getitem__`
  works for any `_generate_sample_from_item` source. Keep it; just wire
  a different upstream dataset object.
- All loss code (`rbm_heads_trainer.py` loss functions, `losses.md` for
  the recipe) — keep entirely.

### What to replace

`Robometer/robometer/data/datasets/base.py:BaseDataset` and the HF-loading
code paths. The current flow is:
```
HF Dataset.load_from_disk(...) → row['frames'] is NPZ path → load_frames_from_npz
```
Replace with:
```
ManifestIndex(jsonl path) → on __getitem__: open tar shard, decode JPGs to
  np.ndarray (T,H,W,C), return dict with frames as ndarray + frame_labels +
  metadata
```

Concrete shape:
```python
class TarKeyframeDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_paths: list[str], frame_dataset_root: str,
                 view: str = "keyframes"):
        self.rows = []  # list of (episode_id, family, archive, task, label,
                        #          frame_labels, tar_path, jpg_member_prefix)
        for mpath in manifest_paths:
            for row in jsonl(mpath):
                # resolve tar_path: <root>/<family>/<view>/shards/shard-NNNNN.tar
                # need a shard mapping — either build at init by listing all tar
                # contents, or store shard hint in manifest
                ...

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        # Open tar, find members matching r.jpg_member_prefix, load each as PIL,
        # stack into (T,H,W,C) uint8 ndarray. Optionally cache the tar handle
        # per worker (NOT across workers — fork() invalidates).
        frames = load_jpgs_from_tar(r.tar_path, r.episode_id)  # np.ndarray
        # Skip-and-retry handled by RBMDataset wrapper — raise SampleSkipRequest
        # if frames count != len(frame_labels), or if frames count == 0, etc.
        return {
            "id": r.episode_id,
            "data_source": f"robometer_frames_{r.family}",
            "quality_label": "successful" if r.label == "success" else "failure",
            "task": r.task,
            "frames": frames,                     # ndarray, NOT a path
            "frame_labels": r.frame_labels,
            "partial_success": (r.label == "partial"),
        }
```

Then wrap with the existing `RBMDataset(config, ..., dataset=TarKeyframeDataset(...))`
or write a thin adapter that satisfies the current samplers' expected interface.

### What to AVOID — explicit list

Do NOT use any of these:
- `Robometer/dataset_upload/generate_hf_dataset.py` — this is the MP4 builder.
- `Robometer/robometer/data/scripts/preprocess_datasets.py` — NPZ extractor.
- Any `Dataset.save_to_disk` or `load_from_disk` against
  `/projects/prjs1958/robometer_frames_hf*/`.
- ffmpeg or any video-encoding subprocess in the data path.
- The `.npz` files in `_projects_..._<split>_raw_robometer_frames_<split>/frames/`
  cached preprocessing — those are the bug surface.

The current LoRA bake-off configs (`Robometer-LoRA/configs/train_lora_base.yaml`,
`run*.yaml`, `loss*.yaml`) reference `data.train_datasets=[robometer_frames_train]`
which the trainer resolves via `ROBOMETER_PROCESSED_DATASETS_PATH` to the HF cache.
For the full FT, replace this resolution: introduce a `data.tar_dataset_root` config
and `data.manifest_paths` config, build the dataset directly.

### Episode-id duplicates across failure/success splits

A small number of trajectories share an episode_id across the failure and success
splits within the SAME archive — meaning the same eid has on-disk JPGs in both
`keyframes/<arch>/` (one recording, labeled failure) and `keyframes_success/<arch>/`
(a separate recording of the same task with the same eid, labeled success). The
two recordings have different frame timestamps and different file sizes, so they
ARE different trajectories — just sharing an identifier.

Confirmed examples in droid (7 cases, 5/5/2026 audit):
- `REAL_2023-12-14_Thu_Dec_14_13-57-01_2023`
- `RAD_2023-12-16_Sat_Dec_16_22-33-57_2023`
- (5 more)

`pairs_unified.jsonl` resolves the duplicate by keeping ONE row per eid (typically
labeled failure) and referencing the success-side counterpart implicitly through
the row's `partner_episode_id` field. Our normalized per-archive manifests mirror
disk reality: the eid appears in BOTH `<fam>_<arch>_failures.jsonl` AND
`<fam>_<arch>_successes.jsonl` because both recordings exist physically.

Implication for the loader: **don't key on raw `episode_id` alone.** Use
`(manifest_path, episode_id)` or `(label, episode_id)` so the failure-side and
success-side recordings of the same eid are addressed separately. Otherwise the
loader will silently shadow one recording with the other.

May happen in other families too — we only verified droid; other families likely
have similar small counts.

### Filtering bad episodes

Before launching, run the audit job (already exists, may have completed by handoff
time): `Robometer-LoRA/jobs/audit_keyframe_counts.job`. Read the report at
`/scratch-shared/$USER/dataset_audit_report.tsv`. It flags episodes where:
- `claim_eq_actual=0`: manifest n_keyframes ≠ actual JPGs in tar
- `labels_eq_actual=0`: len(frame_labels) ≠ actual JPGs
- `actual_short_lt16=1`: actual JPG count < 16
- `no_jpgs_found=1`: manifest has the row but tar has nothing

Recommendation: filter episodes where `claim_eq_actual=0` OR `no_jpgs_found=1`
out of training before the run starts (rather than rely on `SampleSkipRequest`
runtime skipping). The runtime skip works but burns a sample slot per epoch.

### Loss + hyperparam recommendations from the bake-off

(For context — pick what's relevant; not authoritative for full FT.)

From the 9-run LoRA bake-off (BAKEOFF_RUNS.md has full details):

- **Best loss**: L2 (asymmetric C51 + asymmetric BCE), λ=0.5 (run6).
- **KL anchor**: pilot value 0.1 was promising at step 2000 but declined at 2500.
  Worth re-testing at full FT scale where the success/failure mix is different.
- **ICL**: weak signal at LoRA scale (capacity-limited). User's hypothesis: full
  FT will give the model enough capacity to actually use demos. Keep
  `data.icl_prob=0.5`.
- **ICL task dropout**: HURT in LoRA. Skip in full FT unless you have a specific
  reason.
- **LR**: LoRA used 2e-5 (L2) / 4e-5 (L1) and showed late-training decline (over-
  shooting). Full FT should bias lower; sweep `{1e-5, 1.5e-5, 2e-5}`.
- **max_steps**: LoRA plateaued by step 2000 of 7500. Full FT may extend the
  productive horizon, but plan for `max_steps=2500-3500` initially and extend
  if the curve is still climbing at the end.
- **failure_label_smoothing**: keep `linear` for L2 (matches pretrained C51
  head's t/T expectation).

### Validation before launching

Before the long-running training:

1. **Smoke-test the dataset class on 100 trajectories**. Iterate, check that
   every row returns a valid dict with frames as `(T,H,W,C)` uint8, T>=5.
   Confirm `frame_labels` length matches `frames.shape[0]`.

2. **Run a 50-step training pilot** with the new dataset wired in. Confirm:
   - First batch loads (no path errors)
   - Loss decreases over the first 10 steps (sanity)
   - WandB metrics stream live (not blocked by partial-history filter — give
     it a fresh run id, don't try to resume the LoRA bake-off runs)

3. **Compare against an existing LoRA checkpoint** on the test split: load
   the v2-trained `loss2_22244009/checkpoint-7500` and run policy_ranking on
   the same data via the new dataset. Ranking_acc on test should be
   ~0.7574 (iclon) / ~0.7083 (icloff) — within 0.02 of those baselines.
   If far off, the new loader has a subtle bug.

### File and path conventions to follow

- Output checkpoints: `/projects/prjs1958/<full_ft_run_name>/`
- WandB project: pick a NEW one, don't reuse `Robometer_LoRA` (LoRA-bakeoff
  curves would clutter the dashboard).
- Job scripts: put under `Robometer-FT/jobs/` (separate from `Robometer-LoRA/jobs/`).
- Configs: `Robometer-FT/configs/` — fresh tree, don't extend the LoRA configs.

### Open questions for the user

These weren't fully resolved at handoff time — confirm before launching:

- **Full FT vs higher-rank LoRA?** User leans full FT if compute allows; LoRA
  r=64 or r=128 as fallback.
- **Roboreward in/out?** It's currently OUT of LoRA training. The audit will
  show how many roboreward episodes have data issues — decide based on that.
- **Multi-view augmentation?** ext2/wrist views are available for droid +
  metaworld + failsafe. Could 3× the data via random view selection per epoch.
- **Dataset success/failure ratio?** LoRA used 50/50. User said full FT will
  have "way more successes" — pin that exact ratio before launch.

## TL;DR for the agent

1. Skip MP4. Skip HF cache. Read JPGs from tar shards directly.
2. Reuse the trainer + samplers + losses from `Robometer/robometer/`.
3. Replace ONLY the dataset class — feed the existing samplers a dict with
   numpy frames + labels + metadata.
4. Run the audit, filter out the bad episodes pre-flight.
5. Fresh WandB run, fresh config tree, fresh output dir.
