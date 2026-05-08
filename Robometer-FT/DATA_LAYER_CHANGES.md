# DATA_LAYER_CHANGES.md

Changelog for the tar-shard direct-loader work — replaces the HF + MP4 + NPZ
pipeline that caused the LoRA "predictions are random" disaster.

## Status

**STARTED** 2026-05-05. **CODE COMPLETE; SHARD INDEX + SMOKE TEST NOT YET RUN.**

`Robometer/robometer/` originals are byte-identical to session start (verified
md5 — `rbm.py 24642d5d…`, `setup_utils.py 608b6fe6…`, `rbm_data.py bceb9828…`,
`base.py f4cbf2ca…`). All new code lives under `Robometer-FT/`.

## What changed

| File | New / modified | Purpose |
|---|---|---|
| `robometer_ft_data/__init__.py` | NEW | package surface |
| `robometer_ft_data/tar_index.py` | NEW | one-shot scan that maps episode_id → tar path. Cached as JSON. |
| `robometer_ft_data/tar_dataset.py` | NEW | `TarKeyframeIndex` (HF-Dataset-API mimic) + `TarKeyframeRBMDataset(RBMDataset)` that overrides `_load_all_datasets`. |
| `scripts/build_shard_index.py` | NEW | run once before first training (~3 min on 16 cores). |
| `scripts/smoke_test_tar_data.py` | NEW | loads 100 random samples; asserts frame_labels intact, length-matched, varied for failures. The disaster guard. |
| `train_ft.py` | NEW | wrapper that monkey-patches `setup_dataset` to dispatch to TarKeyframeRBMDataset, then invokes upstream train.py's Hydra entrypoint. |
| `configs/train_base.yaml` | EDITED | added `data.use_tar_loader=true` + `data.tar_dataset_root` + `data.tar_families` + `data.shard_index_path`. Old `data.train_datasets` kept as no-op placeholder. |
| `jobs/train_loss1.job` | EDITED | now invokes `train_ft.py` instead of `train.py`. PYTHONPATH includes Robometer-FT/ + Robometer/. |
| `jobs/train_loss2.job` | EDITED | same as above. |

## Cut-line: what does NOT change

Everything downstream of `_load_all_datasets`:
  - upstream `Robometer/robometer/data/datasets/{base,rbm_data}.py` — untouched
  - upstream `Robometer/robometer/data/samplers/*.py` — untouched
  - upstream `Robometer/robometer/trainers/rbm_heads_trainer.py` — untouched
  - upstream `Robometer/robometer/utils/setup_utils.py` — untouched
  - all loss code, all collators, all eval, all FSDP/multi-node — untouched

The only behavioral change is *where the dataset rows come from*. They flow
through the same samplers, with `frames` as ndarray (already supported by
`samplers/base.py:937` — `if isinstance(traj["frames"], str): load_frames_from_npz(...) else: use as-is`).

## What the new dataset returns

For every accessed row, `TarKeyframeIndex[idx]` produces:

```python
{
    "id":              str,                   # episode_id from manifest
    "data_source":     "robometer_frames_<family>",
    "quality_label":   "successful" | "failure",   # binary; partials filtered
    "task":            str,
    "is_robot":        True,
    "partial_success": False,                 # constant — partials dropped
    "frame_labels":    list[int],             # rubric values 1..4 for failure
    "frames":          np.ndarray (T,H,W,3) uint8,   # lazy-decoded from tar
    "embeddings_path": None,
    "lang_vector":     None,
    "text_embedding":  None,
}
```

The values that matter for training: `id`, `data_source`, `quality_label`,
`task`, `frames`, `frame_labels`. The other keys exist because some sampler
branches probe for them.

## What gets filtered out

At index build time:

  1. `label != "success"` and `label != "failure"` — drops "partial" rows entirely.
  2. `frame_labels` missing or empty — drops rows that can't be supervised.
  3. `episode_id not in shard_index` — drops rows whose data isn't on disk.

Counts logged at index build (sample output line):
> manifest rows: 665219 total | kept 597001 | dropped 22 (non-binary label) |
> 14 (missing frame_labels) | 68182 (no shard match)

## The disaster guard

`scripts/smoke_test_tar_data.py` enforces:
  * every sample has `len(frame_labels) == frames.shape[0]`
  * failure rows have rubric values in `{1,2,3,4}` (not the t/T pattern that broke LoRA)
  * `≥60%` of sampled failure trajectories have `≥2` distinct frame label values
    (sanity that supervision actually varies along the trajectory)

If any of these fail, the smoke test halts before any GPU compute is wasted.

## How to run (in order)

```bash
cd Robometer-FT

# 1. one-time prep — scans all tars, writes ~80 MB JSON cache.
/home/pkarageorgis1/.conda/envs/robometer_gpu/bin/python scripts/build_shard_index.py
# Takes ~3 min on 16 cores.

# 2. validate the data path before any GPU run.
/home/pkarageorgis1/.conda/envs/robometer_gpu/bin/python scripts/smoke_test_tar_data.py
# ~30 sec. Must end with "PASS: tar data path produces well-formed samples".

# 3. submit training as before (multi-node pattern unchanged).
sbatch --export=ALL,WANDB_API_KEY="$WANDB_API_KEY" jobs/train_loss2.job

# Or with a small pilot first:
sbatch --export=ALL,EXTRA="++training.max_steps=50 ++training.eval_steps=999999 ++training.save_strategy=no",WANDB_MODE=disabled \
       jobs/train_loss2.job
```

## Revert path

If anything goes wrong at training time, fall back to the old HF loader by
passing `++data.use_tar_loader=false` in EXTRA. `train_ft.py` will pass through
to upstream `setup_dataset` and the run is identical to pre-change behavior
(albeit with the known HF bug surface).

To completely undo this codebase: delete `Robometer-FT/robometer_ft_data/`,
`train_ft.py`, `scripts/{build_shard_index,smoke_test_tar_data}.py`. Revert
`configs/train_base.yaml` and the two `jobs/train_loss*.job` to the prior
versions (in the multi-node-validated state from `Qwen35-FT`'s
`MULTI_NODE_CHANGES.md` — the diff is one-shot).
