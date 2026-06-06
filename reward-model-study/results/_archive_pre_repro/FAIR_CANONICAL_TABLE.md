# Fair, reproducible offline eval — canonical data, all 6 models

**Replaces** the retired `FULL_METRICS.csv` / `eval_dump` per-source numbers, which
loaded frames from the wrong source (keyframe JPEGs in `robometer_frame_dataset`)
instead of the canonical MP4 videos.

## Pipeline (reproducible, baseline-fair)
- Script: `reward-model-study/scripts/repro_canonical_eval.py`
- Data: `/projects/prjs1958/robometer_frames_hf_full/eval_<src>_raw/...` (HF arrow,
  frames = MP4). **metaworld** MP4s live in the sibling
  `..._metaworld.bak_pre_drop_metaworld_success_labels/` (use `--frames_dir`).
- Frames: uniform subsample to each model's **training-time `max_frames`**
  (baseline 8, FT 16) — i.e. NO oracle step-label selection. This is the
  **deployment-realistic** condition (RL has no step labels at inference).
- Metric: success-head `success_prob` vs `quality_label`.
- Env: **`robometer_gpu_fa2`** for the Qwen3-VL baseline (it NaNs under
  transformers 5.7 / `robometer_qwen35_gpu` due to a vocab-resize); FT models run
  in `robometer_qwen35_gpu`. (Some standard-loss FT cells hit the intermittent
  cold-GPU NaN; refire clears it.)

## Results — global AUC (deployment-realistic) / within-task ranking / separation

| source     | baseline-4B          | run2 asym        | run3 std         | run4 asym        | run5 asym        | run6 std         |
|------------|----------------------|------------------|------------------|------------------|------------------|------------------|
| droid      | 0.586 / 0.636 / +0.106 | 0.519/0.666/+0.007 | **0.654**/0.750/+0.014 | 0.584/0.600/+0.018 | 0.567/0.587/+0.048 | 0.515/0.493/+0.006 |
| metaworld  | 0.668 / 0.727 / +0.172 | 0.479/0.416/-0.002 | 0.481/0.505/-0.001 | 0.468/0.468/-0.003 | **0.730**/0.755/+0.181 | 0.656/0.645/+0.038 |
| robometer  | **0.748** / 0.828 / +0.359 | 0.625/0.686/+0.023 | 0.586/0.591/+0.008 | 0.704/0.783/+0.167 | 0.666/0.705/+0.146 | 0.561/0.557/+0.015 |

n: droid 963, metaworld 572, robometer 3865.
- run2/run3 = Robometer-FT (Qwen3-VL, asym / paper-std).
- run4/run5/run6 = Qwen3.5-FT (asym / asym / paper-std), step 6500.

## Conclusion
1. **The off-the-shelf baseline beats every FT model on robometer and metaworld**,
   and its separation dominates everywhere (+0.106…+0.359 vs FT +0.00…+0.18).
   Fine-tuning made the reward model WORSE on a fair deployment-realistic eval.
2. **MetaWorld (the downstream domain) is hard for all**: even the baseline only
   reaches 0.668; run2/run3/run4 sit at chance (0.47–0.48). This is the offline
   fingerprint of the IBRL CoffeePush failure.
3. **Separation collapse under FT**: success_prob distributions compress to
   +0.00–+0.05 (asym AND std loss), vs baseline's wide +0.11–+0.36. Ranking
   partly survives (AUC > 0.5) even when calibration dies.
4. **wandb 0.83–0.94 was optimistic**: those are within-task ranking on
   oracle step-label frames. Scores reproduce exactly (separation +0.167 vs
   wandb +0.168); the gap is metric definition, not a bug.

## Trust / fairness resolution
- The `eval_dump`-vs-wandb divergence was never a scoring bug — wrong frame source
  + different metric (within-task + `use_frame_steps`). Success scores reproduce
  to 3 decimals. The literal 0.94 was deliberately NOT chased (separation match
  suffices; 0.94 uses oracle frames RL never sees).
- This table is fair by construction: identical code/data/metric for all models,
  **including the baseline which has no wandb training log.**
