# Scoring the reward model correctly (read this before debugging "the model is broken")

Written after a long debugging session on ManiSkill in which the success head appeared
catastrophically broken (0.05 on true successes) and was not. Every wrong turn below
was actually taken; they are recorded so nobody takes them again.

## 0. The canary

For a fine-tuned Robometer checkpoint, **per-episode max `success_prob` on a true
success should be ~0.7-0.9**. Measured references:

| setting | model | successes | failures |
|---|---|---|---|
| MetaWorld CoffeePush (`reward-model-study`) | FT | ~0.77 | ~0.13 |
| Robomimic Can (`diag_causal_calib.py`) | run2 | ~0.80 | — |
| ManiSkill PullCube (this repo, correct pipeline) | run2 | **0.81-0.85** | see calib |

If you are seeing 0.01-0.4, **the pipeline is wrong, not the model.** Do not calibrate
a threshold on those numbers, and do not conclude the head is poorly calibrated,
"hedging", or out-of-distribution. Fix the pipeline first. The published baseline
(`Robometer-4B`) is a useful control: it has `max_frames: 8` instead of 16, so it
tolerates the frame-count bug below and keeps scoring ~0.85 while the fine-tuned
models collapse. **Base looking fine does NOT mean the pipeline is fine.**

## 1. Always hand the collator exactly `max_frames` frames

Robometer's collator only ever REDUCES. From
`Robometer/robometer/data/datasets/helpers.py::linspace_subsample_frames`:

```python
if effective_total <= num_frames:
    # If we have fewer (or equal) frames than requested, return all frames
    indices = list(range(effective_total))
```

So a 7-frame clip stays 7 frames. The upsampling must happen at the CALL SITE. The
validated sampler is `sub16` in `reward-model-study/scripts/calibrate_threshold.py`:

```python
idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
frames = frames[idx]          # always exactly max_frames, repeats spread evenly
```

Why this never bit LIBERO/Robomimic/MetaWorld: their episodes run 200-400 steps, so a
growing prefix passes 16 frames within the first few steps -- their calibration can
simply start at `t=16`. **ManiSkill episodes are 50 steps and succeed around step 7**,
so the prefix never reaches 16 on its own.

Measured impact (PullCube, run2, at the GT success step):

| frames fed | success_prob (true successes) | (failures) |
|---|---|---|
| 7-11 (raw prefix) | 0.053 | 0.024 |
| padded to 16 | 0.369 | 0.024 |

Note failures did NOT move -- a real fix lifts only the positives.

Do NOT front-pad by repeating frame 0: it makes the clip look stuck at the start and
biases progress down (run2 progress@success 0.203 -> 0.188). Use linspace.

Implemented in `RobometerReplayBuffer._pad_to_max_frames` (training path) and in
`scripts/causal_calib_maniskill.py::score` (calibration).

## 2. The statistic is per-episode MAX, over a full episode

Padding alone is not enough. At the GT success step there is genuinely little evidence
and the model is right to be uncertain; the head peaks 2-3 steps LATER.

| how it is scored | run2 on true successes |
|---|---|
| one window at the GT success step | 0.369 |
| per-episode max over a full 50-step episode | **0.81-0.85** |

So: run the episode to the horizon (`terminate_on_success=false`), score the GROWING
video every step, and take the **max over the episode**.

A pointwise ROC over (step, label) pairs is the WRONG object. The detector fires ONCE
and ends the episode, so a pointwise FPR of 2% can still mean it fires early in most
episodes. Calibrate on per-episode max with Youden J, exactly as
`vlm_ibrl/jobs/diag_causal_calib.py` does.

## 3. Build the env the way training built it

Saved SAC actors were trained on proprio + **DINOv2-base** features. `make_env` only
attaches `VectorDinoEmbeddingWrapper` when you pass `dinov2_model`:

```python
from transformers import AutoImageProcessor, AutoModel
dino = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
env, _ = make_env(..., dinov2_model=dino,
                  dinov2_processor=AutoImageProcessor.from_pretrained("facebook/dinov2-base"),
                  device=str(dev))
```

Without it the actor receives 384-d proprio and dies with

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x384 and 768x512)
```

The wrapper ADDS `dino_embedding` and keeps `obs["image"]` (`new_obs = dict(obs)`),
despite a class docstring claiming it "replaces" the image -- so raw frames are still
available for scoring.

## 4. Episodes must stop at their own boundary

The vector env AUTO-RESETS on termination. A collection loop that runs a fixed number
of steps and ignores `terminated`/`truncated` will splice a fresh episode's frames onto
the current one -- and if you label everything after the success step as "positive",
the positive class is contaminated with new-episode frames. This dragged the baseline's
apparent separation from 0.86/0.32 down to 0.41/0.38 and made it look like chance.

## 5. What is NOT the problem (all tested, all ruled out)

Do not spend time on these again:

* **cuDNN / bf16 numerics.** `cudnn.benchmark=False, deterministic=True` and full fp32
  give byte-identical outputs. (The all-NaN pathology in `RobometerScorer` is real but
  distinct -- it shows as `nan`, not as small values.)
* **Input resolution.** Once the frame count is right, 224 and 480 give IDENTICAL
  results. The working Robomimic pipeline asserts 224 (`robosuite_wrapper.py`,
  `fetch_img`). Raising resolution appears to help ONLY because extra visual tokens
  partially compensate for missing frames -- it is a symptom, not the cause.
* **`extract_success_probs_from_output`.** The copies in
  `robometer.evals.eval_utils` and `robometer_policy_learning.utils.robometer_utils`
  are identical; no sigmoid is missing.
* **Sample construction.** `raw_dict_to_sample(raw_data=..., max_frames=...,
  sample_type="progress")` with `video_embeddings=None, text_embedding=None` matches
  `RobometerScorer` exactly.
* **Missing head weights.** `success_head.*` is present in the fine-tuned checkpoints
  and statistically near-identical to base (only `similarity_head` is absent, unused).

## 6. Loading gotchas that ARE real

From `vlm_ibrl/env/robometer_utils.py::RobometerScorer`:

* `setup_batch_collator(processor, tokenizer, cfg, is_eval=True)` -- argument ORDER.
  Passing cfg first raises `'Qwen3VLProcessor' object has no attribute 'data'`.
* `cfg.data.use_multi_image = True` before building the collator.
* `ROBOMETER_DISABLE_UNSLOTH=1` (checkpoints bake `use_unsloth: true`).
* Match any C51 variant for discrete progress, not the literal `"discrete"`:
  fine-tuned checkpoints use `c51_asymmetric`.
* Hide `flash_attn` for Qwen3-VL models (RBM does not support FA2). Not installed in
  the ManiSkill venv, so currently moot.
* `ROBOMETER_FORCE_FP32=1` exists for checkpoints whose bf16 kernels produce NaN.

## 7. Threshold and the reward-hacking guards

Calibrate with `scripts/causal_calib_maniskill.py` (port of `diag_causal_calib.py`):
rollouts from a LADDER of saved SAC checkpoints spanning the training arc -- the
deployment distribution, not oracle-plus-random -- scored streaming, thresholded on
per-episode max via Youden J, plus gate guidance.

Expect a threshold in **0.7-0.9**, consistent across tasks and simulators.

Two guards exist and are OFF by default. They are remedies for reward hacking, not
prophylactics -- enabling them from the start makes it impossible to tell whether the
threshold was right or the guards masked the problem:

| knob | default | what it does |
|---|---|---|
| `success_detection_duration` | 1 (no vote) | majority vote over a window before firing |
| `success_detection_min_ep_steps` | 0 (no gate) | ignore fires before N steps into the episode |

Set `min_ep_steps` from the calibration's gate window (between the latest fake fire and
the earliest real fire).

## 8. Reading a live run

Every detector fire logs one line from the buffer:

```
[DETECT] fired ep=.. step_in_ep=.. gt_success=0/1 n_fire=.. n_false=.. false_rate=..
```

* stream of `gt_success=0` at small `step_in_ep`, rising `false_rate` -> threshold too
  LOW; reward hacking. Apply the gate and/or raise the threshold.
* no `[DETECT]` lines at all, episodes always hitting the horizon -> threshold too
  HIGH; `TERMINATE=1` has silently degraded into `TERMINATE=0`.

`RPL_LOG_REWARD=1` prints `gt_in / vlm / final_train_reward` for the first 30 steps
(proves no GT reward leaks into a reward-model arm). `RPL_LOG_DISCRIM=1` logs
per-episode RM reward next to the GT label.

## 9. Where the validated code lives

| file | what it is authoritative for |
|---|---|
| `vlm_ibrl/jobs/diag_causal_calib.py` | the calibration protocol actually used (causal threshold + gate) |
| `reward-model-study/scripts/calibrate_threshold.py` | `sub16`, episode-level vs streaming statistics |
| `vlm_ibrl/env/robometer_utils.py` | correct model loading and scoring |
| `vlm_ibrl/env/robosuite_vlm_env.py` | detection semantics: consecutive fires, min_ep_steps gate |
| `vlm_ibrl/tools/diag_ft_on_training_frames.py` | names the failure: "pipeline mismatch (224x224 + 5 frames)" |

**When something looks broken, read these before writing anything new.** The parallel
implementation written from scratch during this session reproduced three bugs these
files had already solved.
