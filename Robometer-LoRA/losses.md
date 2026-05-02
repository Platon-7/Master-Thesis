# LoRA Loss Bake-off: Setup for Two Competing Reward-Model Losses

## Purpose of this file

This document specifies the LoRA fine-tuning experiment that compares two candidate loss functions for a robotic-manipulation reward model. A fresh chat session should be able to use this document to implement the full training and evaluation pipeline without prior conversation context.

The experiment is Phase 1 of a larger plan: we run two lightweight LoRA fine-tunes (one per candidate loss) on the same data and architecture, pick the winner on held-out metrics, and commit to that loss for Phase 2 full training with additional components (ICL 2-stage, failure-rehearsal KL anchor, etc. — not needed here).

## Context in two sentences

We are fine-tuning Robometer-4B (a VLM-based reward model for robotic manipulation, derived from Qwen3-VL-4B) to produce a progress/success signal that is *conservative* enough to avoid RL reward hacking. Robometer's original formulation supervises progress only on successful trajectories via `p_t = t/T`; our dataset adds dense per-frame ordinal labels (1–5) on failures too, and the bake-off tests whether a new ordinal/asymmetric loss outperforms a minimal asymmetric modification of Robometer's existing C51 loss.

## Architecture and base model

- **Backbone**: Robometer-4B, loaded from the publicly released pretrained weights (itself a fine-tune of Qwen3-VL-4B). HuggingFace repo id: `robometer/Robometer-4B`. Do not re-train the backbone; use released weights as the starting point.
- **Heads**: Robometer ships three heads — `progress` (C51, N=10 bins, confirmed from the checkpoint's own `config.yaml` where `exp_name` ends in `…_discrete_10_bins_part2`), `success` (binary logit), `preference`. Head usage per variant:
  - **Loss 1 (ours, ordinal)**: replace the progress head with a fresh 4-logit CORN head; keep/discard the other two at the author's discretion (not central to the comparison).
  - **Loss 2 (Chris')**: keep BOTH the existing C51 progress head AND the existing binary success head — weights loaded directly from the release. Only the *loss functions* applied to these heads change (see §Loss 2). Drop the preference head.
- **LoRA**: adapters on attention projection matrices (q_proj, k_proj, v_proj, o_proj) and MLP (gate, up, down). Rank 32, alpha 64, dropout 0.05 — matches the Robometer-4B release config. Freeze all other backbone weights. Apply LoRA to the backbone only; heads are trained fully.
- **Precision**: bf16 for backbone forward, fp32 for LoRA adapters and head.

## Dataset

Three labeled sources supply training queries and demos; a simulator source is reserved entirely for evaluation.

### Data sources

| Source | Role | Size | Notes |
|---|---|---|---|
| DROID failure pairs | Failure queries + their demos | 5,500 pairs | Per-frame ordinal labels 1–5 from VLM+LLM pipeline on the failure; paired success is the demo (ICL on). |
| Robometer Group A subset | Failure queries + their demos | 5,000 pairs | Same structure as DROID. |
| Robometer orphan successes | Success queries + their demos | ~9,450 sampled | Pulled from the ~500K orphan pool. Success-success (query, demo) pairs matched by `(task, lab)` — see prerequisite below. |
| Failsafe simulator | Evaluation only | ~500 episodes | Simulator ground-truth ordinal labels. |

### Splits

- **DROID**: 10% held out for eval → 550 eval pairs, 4,950 train pairs.
- **Robometer Group A subset**: 10% held out for eval → 500 eval pairs, 4,500 train pairs.
- **Orphan success pairs**: match and sample ~9,450 success→success pairs. Hold out 10% → ~945 eval, ~8,505 train. (See prerequisite below.)
- **Failsafe**: all ~500 episodes used for evaluation only (no training).

Split by trajectory ID, not frame, to avoid train/eval leakage.

### Prerequisite: orphan success–success matching

Before training can start, orphan successes must be paired into `(query_success, demo_success)` tuples. Procedure:

1. Extract metadata per orphan: `task_id` / task description, `lab_id`, robot/camera identifiers.
2. Group orphans by `(task_id, lab_id)` tuples.
3. Within each group, sample success-success pairs such that `query ≠ demo`. Aim for ~9,450 total pairs; drop groups too small to pair safely (require at least 2 successes per group).
4. Store pairs as JSONL: `{"query_traj_id": ..., "demo_traj_id": ..., "task_id": ..., "lab_id": ...}`.

Estimated time: ~1 day of data-engineering work. Do this before any LoRA training starts.

### Training example construction

Two example types, sampled together into each batch:

1. **Failure-query examples** (from DROID + Group A): 9,450 total.
   - `query` = failure trajectory, labels from VLM+LLM pipeline.
   - `demo` = the paired success trajectory from the same pair.

2. **Success-query examples** (from orphan pool): ~9,450 total.
   - `query` = one orphan success, labels from the monotone formula `y_t = min(ceil(5 · t / 16), 5)` for t = 1..16 → (1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5).
   - `demo` = a different orphan success matched by `(task_id, lab_id)` from the pre-computed matching.

**Total training examples**: ~18,900, naturally balanced 50/50 between failure and success queries.

**Eval-time construction**: same pattern on held-out splits. Failsafe eval uses the simulator's native success/failure trajectories paired into (demo, query) per episode.

### Distribution caveat

Orphan successes come from a broader task distribution than DROID/Group A's failures. If the orphan pool's task coverage skews far from the failure queries' tasks, there may be mild distribution shift during training. Check: confirm that the sampled orphan `task_id`s overlap meaningfully with DROID/Group A task coverage. If not, filter orphans to tasks represented in the failure set.

## Data format

Each training example is a single **(demo, query, query_labels)** triple:

```
demo:          tensor [16, H, W, C]  — success trajectory frames
query:         tensor [16, H, W, C]  — trajectory to be evaluated (failure or success)
query_labels:  tensor [16]           — per-frame ordinal labels ∈ {1..5}
is_success:    bool                   — whether query is a success trajectory
```

For **ICL-off** training (50% of batches), demo is omitted — the model sees only the query.
For **ICL-on** training (50% of batches), demo and query are concatenated in the model's context; self-attention runs across both; loss is computed only on query frames.

## Batch composition

- **Per-example ICL coin flip**: independently, with probability 0.5, each example is trained with ICL on (demo + query in context) or ICL off (query only). This exposes the model to both regimes.
- **Uniform sampling** over the ~18,900 constructed (demo, query) tuples. No stratified sampling or oversampling — the failure/success balance is already 50/50 by construction.
- Shuffle each epoch.
- Batch size: 8 examples per step (adjust based on GPU memory; ICL-on examples roughly double the token count).

## Loss 1 — Our asymmetric CORN loss (per-frame only)

### Head and outputs

Replace Robometer's three heads (C51 progress, binary success, preference) with a single CORN head:

- Per frame t, the head outputs 4 logits: `z_{t,k}` for k ∈ {2, 3, 4, 5}.
- `σ(z_{t,k}) = P(y_t ≥ k)`. These are cumulative-threshold probabilities.
- Ordinal label y_t ∈ {1..5} converts to threshold bits: `b_{t,k} = 1[y_t ≥ k]`.
  - y_t = 1 → (0, 0, 0, 0)
  - y_t = 2 → (1, 0, 0, 0)
  - y_t = 3 → (1, 1, 0, 0)
  - y_t = 4 → (1, 1, 1, 0)
  - y_t = 5 → (1, 1, 1, 1)

### Loss formula

```
L_frame = - (1/T) Σ_t Σ_{k=2..5} [
              β_k · b_{t,k}       · log σ(z_{t,k})
            + α_k · (1 - b_{t,k}) · log (1 - σ(z_{t,k}))
          ]
```

where:
- T = 16 (frames per query)
- β_k = 1 (uniform positive-class weight)
- α_k = 1 + c · (k − 2) for k ∈ {2, 3, 4, 5}, so α_2=1, α_3=1+c, α_4=1+2c, α_5=1+3c
- c ∈ {0.5, 1.5, 3.0} — sweep once, pick best on eval (small pilot at start of Phase 1).

No L_traj, no L_cal, no KL anchor in Phase 1. Just L_frame.

### Recovering P(success) at inference

`P(success at frame t) = σ(z_{t,5})`. A trajectory-level success score can be derived as `max_t σ(z_{t,5})` or `σ(z_{T,5})` depending on preference — both are available from the per-frame outputs.

## Loss 2 — Chris' asymmetric C51 + asymmetric BCE

Source: two slides ("FoundRewardModelLoss.backup.pptx", shared April 2026). The intent, in Chris' own words: *"fine-tune Robometer for predicting binary success and task completion % but strongly punish reward overestimation"*. The mechanism: keep both heads' architecture and symmetric-loss machinery from Robometer, wrap each in a per-element asymmetric weight that damps whichever direction would cause over-confident positive predictions.

### Heads active

- **Progress head**: Robometer's existing C51, N=10 bins. Unchanged weights load from `robometer/Robometer-4B` release.
- **Success head**: Robometer's existing binary logit head. Unchanged.
- **Preference head**: disabled (`model.train_preference_head: false`).

### Per-frame progress targets

Upstream of the loss, the sampler produces `target_progress ∈ [0, 1]` per frame (Phase-2 pipeline — already implemented):
- **Failures**: per-frame rubric decimals `{0.0, 0.25, 0.5, 0.75}` from the curated labels {1..4}, optionally piecewise-linearly ramped (`data.failure_label_smoothing: "linear"`) so the target shape is comparable to success `t/T`.
- **Successes**: `t/T` scalar, optionally perturbed by small Gaussian noise (`data.success_label_noise_std > 0`).

These scalars then get projected into Robometer's existing C51 soft-target distribution via the upstream helper `convert_continuous_to_discrete_bin_c51` (`robometer/data/datasets/helpers.py:55`). After projection, `target_progress` flows into the trainer already shaped `[B, T, N]` — exactly the input the existing symmetric-CE path already handles.

### Per-frame success targets

Reuse Robometer's existing `compute_success_labels` path untouched: binary target of 1 on frames where cumulative progress ≥ success cutoff (typically only the terminal frame of a success trajectory), 0 elsewhere. See `robometer/data/datasets/helpers.py`.

### Progress loss (replaces Robometer's symmetric C51 cross-entropy)

For a batch of predicted logits `[B, T, N]` and soft targets `[B, T, N]`:

```python
bin_centers = torch.linspace(0, 1, N)                                   # [N]

# Scalar expected prediction and scalar target. Detached on both sides — the asymmetric
# weight is a routing signal, not a gradient path, so predictions must not backprop
# through their own weighting.
p_hat = (F.softmax(logits, dim=-1) * bin_centers).sum(-1).detach()      # [B, T]
p_t   = (target_dist              * bin_centers).sum(-1).detach()       # [B, T]

# Per-frame asymmetric weight: overestimation → full, underestimation → lambda
weight = torch.where(p_hat > p_t, 1.0, lambda_val).detach()             # [B, T]

# Weighted CE against the soft C51 target distribution
ce = F.cross_entropy(logits.flatten(0, 1), target_dist.flatten(0, 1),
                     reduction='none').view(B, T)                        # [B, T]
L_prog = (weight * ce * mask).sum() / (mask.sum() + eps)
```

`mask` is Robometer's existing progress mask (`target_progress_mask` × `padding_mask`), so frames where progress should not be computed contribute zero as today. Implementation anchor: **new `elif loss_type == "c51_asymmetric"` branch** in `robometer/trainers/rbm_heads_trainer.py::_compute_progress_loss_helper` (current symmetric discrete path handles the 3-D `target_dist` case already — the asymmetric variant reuses it and only adds the pre-computed `weight` multiplier and the scalar `p_hat, p_t` derivation via `convert_bins_to_continuous`).

### Success loss (replaces Robometer's existing class-balanced BCE)

For per-frame binary logits `[B, T]` and targets `[B, T]`:

```python
bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # [B, T]
bce = bce * combined_mask                                                    # same mask Robometer applies today

mask_neg = (targets == 0) & combined_mask.bool()
mask_pos = (targets == 1) & combined_mask.bool()

L_neg = bce[mask_neg].mean()
if mask_pos.any():
    L_pos = bce[mask_pos].mean()
    L_succ = L_neg + lambda_val * L_pos
else:
    # All-failures batches have zero positive frames — guard against NaN from empty .mean().
    # The lambda·mean(positives) term is just zero in this case, so drop it.
    L_succ = L_neg
```

Implementation anchor: **replace the existing class-balanced weighting branch** in `robometer/trainers/rbm_heads_trainer.py::_compute_success_loss_helper` (around line 1910) when `loss.success_loss_type == "bce_asymmetric"`; leave the default class-balanced path available under the existing switch for regression purposes.

### Total loss

Straight sum, matching Robometer's existing aggregation (`rbm_heads_trainer.py::compute_loss` around line 1796):

```
L_total = L_prog + L_succ
```

No relative `α·L_prog + β·L_succ` weighting — Robometer's pre-patch total is unweighted and we follow suit.

### Hyperparameters

| Name | Value | Justification |
|---|---|---|
| `lambda_val` | `0.3` | Single shared scalar used in both the progress and success loss. Chris' slide uses one symbol `lambda_val` for both. A λ<1 damps the "underestimation / positive class" side. Start at 0.3; sweep {0.1, 0.3, 0.5} if first run is surprising. |
| `N` (progress bins) | `10` | Matches the pretrained Robometer-4B checkpoint exactly (verified via its own `config.yaml`). Any other value would force a random-init of the progress head. |

### Recovering P(success) at inference

- Preferred: `P(success at frame t) = σ(success_logits_t)`  — the success head's output is already calibrated for this purpose.
- Alternative (from progress head only, for trajectories without the success head): `P(bin_{N−1})` = probability mass on the terminal progress bin.
  - Or its expected-value proxy: `p̂_t > 0.9`.

### Open questions (parked for Chris)

1. **One vs two lambdas** — slide uses one symbol; we follow the slide unless he splits them.
2. **Success loss normalisation form** — slide is `mean(neg) + λ·mean(pos)` (class-level mean, imbalance-invariant). We use this verbatim. An alternative `(weight·bce).mean()` would be imbalance-sensitive; only swap if he asks.
3. **Relative weighting of progress vs success** — straight sum by default; Robometer does the same.

## Loss 3 — Failure-rehearsal KL anchor (augments Loss 1 OR Loss 2)

This is **not a third standalone loss** — it's an additive distillation term that augments
either Loss 1 or Loss 2. Implemented in `Robometer/robometer/trainers/failure_kl.py` and
wired into `_compute_progress_loss` in `rbm_heads_trainer.py`.

### Motivation

Even with `data.stratified_batch_balance=true` (50/50 success/failure batches), two
asymmetries push the model toward under-weighting failures:

1. **Label noise asymmetry** — success labels (`y_t = ⌈5t/16⌉` from the synthetic monotone
   formula) are clean by construction; failure labels come from the curated rubric and
   are noisier. Optimizers gravitate toward the cleaner side.
2. **Distributional asymmetry in full-FT runs** — when the orphan-success pool is much
   larger than the curated-failure pool (the regime we plan to enter for full FT), even
   stratified mixing pulls the model toward the success distribution because the success
   side has more independent gradients available across an epoch.

The result: the model gradually forgets failure-mode predictions as training progresses.
A same-input distillation anchor (in the spirit of EWC and LwF) keeps the model honest
on past failures.

### Theory

Let $M_t$ denote the model's parameters at training step $t$. We maintain a buffer of
recent failure-batch snapshots $(x^{\mathrm{fail}}_{t-n}, z_{t-n})$ where
$z_{t-n} = M_{t-n}(x^{\mathrm{fail}}_{t-n})$ are the progress logits the model produced
when the failure was originally processed.

On every **success** training step (where the natural loss $\mathcal L_{\mathrm{progress}}$
is computed on the success batch as usual), we additionally:

1. Pop a buffered failure snapshot $(x^{\mathrm{fail}}, z_{\mathrm{old}})$.
2. Re-forward $x^{\mathrm{fail}}$ through the **current** model: $z_{\mathrm{new}} = M_t(x^{\mathrm{fail}})$.
3. Add the KL anchor:

$$
\mathcal L_{\mathrm{kl}}(t) \;=\; \mathrm{KL}\!\left(P_{\mathrm{old}} \;\|\; P_{\mathrm{new}}\right)
$$

with $P_{\mathrm{old}} = \mathrm{softmax}(z_{\mathrm{old}})\,.\mathrm{detach}()$ and
$P_{\mathrm{new}} = \mathrm{softmax}(M_t(x^{\mathrm{fail}}))$. Gradient flows only through
$z_{\mathrm{new}}$. The total objective on a success step becomes

$$
\mathcal L_{\mathrm{total}}(t) \;=\; \mathcal L_{\mathrm{progress}}(t) \;+\; \lambda_{\mathrm{kl}}\,\mathcal L_{\mathrm{kl}}(t).
$$

#### Per-loss-family KL form

The progress head's output shape determines the KL:

* **Loss 1 (CORN)** — head emits 4 cumulative-threshold logits per frame; each is the
  parameter of an independent Bernoulli $P(y_t \ge k) = \sigma(z_{t,k})$. The per-frame
  KL is the **sum across thresholds** of Bernoulli KL:

  $$
  \mathrm{KL}_{\mathrm{CORN}}(t) \;=\; \sum_{k=2}^{5}\!\Big[
    p_{\mathrm{old},k}\,\log\frac{p_{\mathrm{old},k}}{p_{\mathrm{new},k}}
    + (1-p_{\mathrm{old},k})\,\log\frac{1-p_{\mathrm{old},k}}{1-p_{\mathrm{new},k}}
  \Big]
  $$

  computed numerically in log-space via `F.logsigmoid` for stability.

* **Loss 2 (asymmetric C51 + BCE)** — head emits 10 bin logits per frame, softmax-normalised.
  Per-frame KL is the standard categorical:

  $$
  \mathrm{KL}_{\mathrm{C51}}(t) \;=\; \sum_{b=1}^{10}
    p_{\mathrm{old},b}\,\log\frac{p_{\mathrm{old},b}}{p_{\mathrm{new},b}}.
  $$

  The success-head KL (binary) is intentionally NOT added — Loss 2's success head is
  already supervised on every success step via the asymmetric BCE; adding a KL anchor on
  it would over-determine its predictions.

#### Per-frame masking

Per-frame KL terms are aggregated using the stored mask
$M = M_{\mathrm{tp}} \cdot M_{\mathrm{plf}}$ (target_progress_mask × predict_last_frame_mask)
captured at push time, applied frame-wise:

$$
\mathcal L_{\mathrm{kl}} \;=\; \frac{\sum_{b,t} M_{b,t}\,\mathrm{KL}_{\mathrm{per-frame}}(b,t)}{\sum_{b,t} M_{b,t}}.
$$

This guarantees padded / unsupervised frames don't contribute, identical to the masking
the natural loss already applies.

### Buffer policy — peek-when-below-N FIFO

A pure pop-on-success FIFO would drain the buffer to empty in O(N) success steps under
50/50 stratification (insertion rate = consumption rate). To avoid that, we use a
**peek-when-below-N** retention rule:

```
on FAILURE batch:
    push(snapshot)                        # deque(maxlen=N) auto-evicts oldest if full

on SUCCESS batch (when KL applies):
    snapshot = head_of_buffer             # always anchor against the OLDEST entry
    if len(buffer) >= N:                  # peek-when-below-N: only pop at capacity
        buffer.pop_left()                 # consume; next-oldest becomes head
    # else: peek-only — same snapshot survives until a new failure arrives
```

Steady-state behavior under 50/50 stratification: buffer reaches size $N$ in $\sim\!2N$
wall-steps, then maintains exactly $N$. Each success step consumes the oldest entry
($\sim\!N$ failure-pushes back $\approx 2N$ wall-steps stale); each failure step refills
the tail. Staleness is bounded by $N$.

### Implementation

Three components:

#### 1. Pure-logic helpers — `Robometer/robometer/trainers/failure_kl.py`

* `FailureKLBuffer(maxlen=N)` — deque with the peek-when-below-N retention rule. Methods:
  `push`, `peek_head`, `consume_head`, `is_full`.
* `compute_failure_kl(z_old, z_new, mask=...)` — dispatches on `z.shape[-1]` (4 → CORN
  Bernoulli, otherwise → categorical). Always detaches `z_old` defensively.
* `build_buffer_entry(progress_inputs, progress_pred)` — packages a snapshot with all
  tensors detached and moved to CPU. Stores only the model-forward fields plus
  `target_progress_mask` and `predict_last_frame_mask` for KL masking.
* `move_entry_to_device(entry, device, dtype=...)` — symmetric move-back-to-GPU.
* `build_concat_batch(success_inputs, failure_inputs, pad_token_id=...)` — for the
  concat-batch fast path: stacks both batches along dim 0, padding sequence length to
  the longer side. Returns `(combined_dict, b_succ, b_fail)` so the caller can split the
  output back via `split_concat_logits(logits, b_succ)`.

#### 2. Trainer integration — `RBMHeadsTrainer._compute_progress_loss`

The buffer + apply gate live on the trainer instance. Behavior is fully gated on
`config.loss.failure_kl_weight > 0` — when zero (default), the trainer's `__init__` does
not even create the buffer, and the per-step path is bit-identical to the disabled run.

When enabled, on each step:

* **Detect quality** via `quality_labels` field on the batch.
* **Failure batch** — run the normal forward + main loss, then push a buffer entry.
* **Success batch** with KL gating satisfied (buffer non-empty; every-N counter; below-N
  retention) — two paths:
  * `failure_kl_concat_batch=True` (concat fast path) — pre-flight peek the buffer head,
    build a combined batch, run **one** forward, split outputs into the success-loss
    half and the KL half. Single forward + single backward covers both losses.
  * `failure_kl_concat_batch=False` (sequential, default) — run the success forward as
    normal, then re-forward the buffered failure input separately. Two forwards, one
    backward.

#### 3. Detach-backbone fast path — `_progress_head_only_forward`

When `failure_kl_detach_backbone=True`, the KL re-forward runs the backbone in `no_grad`
to capture the input the progress head WOULD see (via a `forward_pre_hook` on the head),
then re-runs only the progress head with autograd on the detached hidden states. KL
gradient flows only through head parameters — backbone (and any LoRA adapters on it)
receive no gradient from the KL term.

This is the right setting for **LoRA-only smoke tests**, where the backbone is mostly
frozen anyway; for **full FT** keep `failure_kl_detach_backbone=False` so the full model
gets anchored.

### Configuration

All knobs live on `LossConfig` and default to a fully-disabled state:

| Name | Default | Effect |
|---|---:|---|
| `failure_kl_weight` ($\lambda_{\mathrm{kl}}$) | `0.0` | $> 0$ activates the anchor. Pilot at 0.1; sweep $\{0.05, 0.1, 0.3\}$ if the first run is surprising. |
| `failure_kl_buffer_size` ($N$) | `10` | Buffer depth. Steady-state staleness $\le 2N$ wall-steps. |
| `failure_kl_apply_when_buffer_below_size` | `True` | Apply the KL anchor as soon as the buffer has $\ge 1$ entry. The peek-when-below-N retention rule prevents over-anchoring on a single entry. Set `False` to wait until the buffer is exactly full. |
| `failure_kl_detach_backbone` | `False` | Right default for full FT. Set `True` for LoRA smoke tests if speed matters more than anchor strength. |
| `failure_kl_concat_batch` | `False` | Set `True` for $\sim\!15\text{–}40\%$ wall-clock saving via single combined forward. Doubles peak forward-pass memory; safe on H100 80GB but check before enabling on smaller GPUs. |
| `failure_kl_apply_every_n_success` | `1` | Apply on every $N$-th success step. Increase to throttle the wall-clock cost at the price of fewer anchor updates per epoch. |

### Wall-clock cost

| Mode | Per success step | Per failure step | Avg overhead at 50/50 stratification |
|---|---|---|---|
| Sequential, no detach | $\sim\!2\times$ forward, $\sim\!1.5\times$ backward | unchanged | $\sim\!50\%$ |
| Sequential, detach backbone | $\sim\!1\times$ no_grad forward + $\sim\!1\times$ grad forward (head only) | unchanged | $\sim\!30\text{–}40\%$ |
| Concat batch, no detach | $\sim\!1\times$ bigger forward + $\sim\!1\times$ bigger backward | unchanged | $\sim\!17\%$ |
| Concat batch, detach backbone | not implemented (concat already gives most of the gain) | — | — |

For full FT (the eventual target), recommended setting is **`concat_batch=True`,
`detach_backbone=False`**: ~17% slowdown is a small price for the anchor and the model
gets the full gradient signal.

### Observability

Logged every training step when the anchor is enabled:

* `loss/failure_kl` — KL term value (or absent on steps where KL skipped).
* `loss/failure_kl_buffer_fill` — current buffer size $\in [0, N]$.
* `time/failure_kl_concat_forward` (concat path) or `time/failure_kl_reforward`
  (sequential path) — wall-clock cost of the extra forward.

### When to ship

Out of scope for the current 7-run bake-off (Runs 1–7). Plan: pick the top-1 or top-2
performers from the bake-off, then re-run those with `failure_kl_weight=0.1` enabled as
**Phase-1.5**. Subsequently, the chosen recipe carries forward into the **Phase-2 full
fine-tuning**, where the failure-rehearsal anchor is expected to matter most (no LoRA
freezing, larger orphan-success pool, longer training horizon).

### Smoke-test coverage

`Robometer-LoRA/scripts/smoke_test_failure_kl.py` (27 tests, all GPU-free, run in <2 s):

* Buffer push / peek / consume + the peek-when-below-N retention rule (with multi-step
  alternating-stream verification).
* `detect_batch_quality` for every input shape (lists, mixed, missing).
* CORN Bernoulli KL: zero-when-identical, positive-when-different, gradient flows through
  $z_{\mathrm{new}}$ only, matches a closed-form Bernoulli KL on a hand-checked case.
* Categorical KL: same correctness checks at $K=10$, cross-checked against `F.kl_div`.
* Mask handling, including the trailing-singleton `[B, T, 1]` form the trainer produces.
* Buffer-entry construction (CPU placement, detachment, irrelevant-key dropping).
* `build_concat_batch` (basic, padding, batch-size split round-trips, optional-field
  passthrough) and `split_concat_logits`.
* Detach-backbone fast path on a stub model: gradient lands ONLY on head params (not
  backbone params), and head-only re-forward produces numerically identical logits to a
  full forward.

## Training hyperparameters

Same for both loss runs.

- Optimizer: AdamW, lr 1e-4 (LoRA adapters), lr 5e-5 (head), weight decay 0.01
- LR schedule: linear warmup over first 5% of steps, then cosine decay to 10% of peak
- Total training steps: ~15,000 (roughly 6–8 epochs over 18,900 examples at batch size 8)
- Gradient clipping: 1.0
- Mixed precision: bf16
- LoRA rank 16, alpha 32, dropout 0.05
- Seed: fix to 42 for reproducibility; optionally run a second seed (1337) for both losses if compute allows

### Compute budget estimate

On 1× H100 (80GB) with LoRA on Robometer-4B and coin-flip ICL:
- Per-step time: ~2.0s (ICL-on) / ~1.2s (ICL-off), expected ~1.6s
- Total time per run: 15,000 steps × 1.6s ≈ 6.7 hours
- Both runs: ~13–14 hours total on 1 GPU, less on 2+.

## Evaluation protocol

Evaluate every 1,500 training steps on the full held-out set. Report final metrics at end of training.

### Primary metrics (all computed on held-out eval set)

1. **Success AUC**: ROC-AUC of trajectory-level success probability vs ground-truth success/failure label. Trajectory-level score = `max_t σ(z_{t,5})` for Loss 1, `max_t p̂_t` for Loss 2.

2. **Success-vs-Failure ranking accuracy**: for each (failure, success) pair in the eval set, check whether the model assigns higher success probability to the success. Report fraction correct.

3. **Expected Calibration Error (ECE)** on P(success): bin predictions into 10 equal-width buckets, compute |accuracy − confidence| per bucket, weighted average. Lower is better.

4. **Per-frame ordinal MAE**: for Loss 1, argmax over `P(y=k)` derived from CORN cumulative probabilities; for Loss 2, round `4·p̂_t + 1` to nearest integer. Compute MAE against ground-truth labels. Note: MAE is partially bounded by label noise (VLM+LLM labels on DROID are moderately noisy at intermediate levels; simulator labels on Failsafe are clean).

5. **False-positive rate on successes**: fraction of non-success frames (y < 5) with `P(success) > 0.5`. Critical for RL downstream use.

### Eval-set slicing

Report all metrics separately for each source:
- DROID eval (550 pairs, VLM-labeled)
- Group A eval (500 pairs, VLM-labeled)
- Failsafe eval (~500 episodes, simulator ground truth — **most trustworthy subset**)

The Failsafe metrics are the most reliable because labels are from the simulator, not a VLM pipeline. Weight this source heaviest when deciding the winner.

### Winning criterion

Pick the loss that wins on **Failsafe success AUC + Failsafe FP rate**, breaking ties with ranking accuracy on the full held-out set. Document the decision.

## Suggested file structure for the implementation

```
thesis_root/
├── losses.md                    (this file)
├── lora_bakeoff/
│   ├── train.py                 (main training loop)
│   ├── losses/
│   │   ├── corn_asymmetric.py   (Loss 1)
│   │   └── c51_asymmetric.py    (Loss 2)
│   ├── data/
│   │   ├── orphan_matching.py   (one-time pre-processing: pairs orphan successes by task/lab)
│   │   ├── dataset.py           (loads failure pairs + orphan success-success pairs)
│   │   ├── sampler.py           (uniform shuffle, per-example ICL coin flip)
│   │   └── failsafe_eval.py     (eval-only loader)
│   ├── model/
│   │   ├── heads.py             (CORN head + C51 head wrappers)
│   │   └── lora_wrap.py         (LoRA adapter setup)
│   ├── eval/
│   │   └── metrics.py           (AUC, ECE, MAE, FP rate)
│   └── configs/
│       ├── loss1_ours.yaml
│       └── loss2_chris.yaml
```

## Critical implementation notes

1. **Tokenization of video frames**: Robometer-4B uses Qwen3-VL's vision encoder. Keep the existing preprocessing (resize to native input size, normalize per Qwen3-VL's stats). Do not modify the vision encoder.

2. **ICL-on context layout**: `[<demo frames 1..16>] [<separator tokens>] [<query frames 1..16>]`. Loss is computed only on query frame positions. Exact separator tokens follow Qwen3-VL's multi-image prompt convention.

3. **Head replacement for Loss 1**: the original Robometer head outputs (C51 distribution, success logit, preference score). For Loss 1, replace with a fresh MLP: `Linear(hidden_dim → hidden_dim/2) → GELU → Linear(hidden_dim/2 → 4)` producing the 4 CORN logits per frame. Initialize fresh.

4. **Head preservation for Loss 2**: both the progress (C51, 10 bins) and success (binary) heads load from the pretrained Robometer-4B release unchanged. Only the *loss functions* applied to their outputs change — there is no head arch change, no new output dim, no random re-init. The preference head is dropped via `model.train_preference_head: false`.

5. **Label handling for success-only pairs**: when the query is a success, per-frame labels are derived from `y_t = ceil(5 · t / T) = ceil(5t/16)`, capped at 5 — a monotone ascending sequence from 1 to 5 across the 16 frames. This is cleaner than failure labels (which come from the VLM+LLM pipeline) and should be used directly without re-labeling.

6. **No trajectory-level loss**: only per-frame losses. Do not add L_traj, L_cal, KL anchor, or any regularizer beyond the ones specified.

7. **Determinism**: set torch seed, numpy seed, and Python's random seed; use `torch.use_deterministic_algorithms(True)` where feasible. Document if LoRA+bf16 introduces minor non-determinism.

## What is explicitly out of scope

- L_traj (trajectory-level loss) — dropped.
- L_cal (Brier calibration term) — handled post-hoc via temperature scaling if needed, not during training.
- KL rehearsal anchor / Mean Teacher — see §Loss 3 (Failure-rehearsal KL anchor). Out of
  scope for the 7-run bake-off; planned as a Phase-1.5 follow-up on the bake-off winner(s).
- Multi-view augmentation — not used in Phase 1.
- Chris' preference head modifications — not part of Chris' loss as defined here.
- Full (non-LoRA) fine-tuning — Phase 2.

## Deliverables from this experiment

1. Two trained LoRA checkpoints (one per loss).
2. Full eval metrics table (per-source breakdown).
3. A short decision memo (~200 words) stating which loss won and why, with specific numbers cited. This memo becomes input to Phase 2 planning.
