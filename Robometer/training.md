# Phase 2 Full Training — Design Decisions

## Scope

This document captures confirmed decisions for **Phase 2 full training** of the reward model. It is explicitly **not** the LoRA bake-off spec — that lives in `losses.md` and tests loss-function choice in isolation. Phase 2 begins after the bake-off picks a winner (the "best loss" referenced below as `L_best`).

Some items are still TBD (notably whether we use continued fine-tuning or explicit distillation for the ICL-off stage). Document is updated as decisions land.

## Confirmed training strategy

### Data composition

All labeled sources combined:
- DROID failure pairs (full ~5,500, with orphan-matched success queries added)
- Robometer Group A (full ~5,000)
- Robometer orphan successes (~9,450 as in LoRA, can be scaled up)
- MetaWorld / PlayWorld simulator pairs (~15K)
- Failsafe simulator (~2K held mainly for eval; small training slice possible)

Splits preserved from LoRA (10% held out per source, trajectory-ID level). Failsafe remains primarily an eval set.

### ICL handling

**Per-example Bernoulli(0.5) ICL coin flip**, same as LoRA bake-off. Each training example independently:
- With p=0.5: demo + query in context
- With p=0.5: query only

This replaces the earlier two-stage plan (stage 1 ICL on, stage 2 ICL off). Stochastic mixing in a single training run is simpler and natively teaches the model to operate in both regimes.

### Class balance

- Failures and successes appear in approximately equal proportions per batch.
- Failures may be repeated per epoch to match success volume (orphan success pool is much larger).
- No explicit stratification sampler needed if the data loader is constructed with balanced example counts upstream.

## KL rehearsal anchor (confirmed mechanism)

The anchor prevents the model's failure predictions from drifting during success-heavy training. Applied only on success steps.

### Mechanism

**Failure step:**
1. Compute standard loss: `L = L_best(failure_prediction)` against per-frame labels.
2. Store the current example in the replay buffer: append `(x_fail, z_current, context_flags)` where:
   - `x_fail`: the failure query input (video frames, and demo if ICL was on)
   - `z_current`: the per-frame logits the current model produced on this failure
   - `context_flags`: was ICL on, which demo was used (if any)
3. Evict oldest if buffer is full (FIFO, buffer size N = TBD, start with 128–256).

**Success step:**
1. Compute standard loss on the current success: `L_std = L_best(success_prediction)` against per-frame labels.
2. Sample **1** entry from the buffer: `(x_fail, z_old, context_flags)`.
3. Re-run the current model on `x_fail` under the same `context_flags` → `z_new`.
4. Compute **forward KL**: `L_KL = KL(softmax(z_old) || softmax(z_new))`, applied per frame and averaged over the 16 frames.
5. Total loss: `L_total = L_std + λ_KL · L_KL`.

Only 1 failure is drawn per success step. To offset the resulting noise in the KL signal, batch size is increased (see below).

### Buffer management

- **FIFO eviction** — oldest entries drop out when buffer fills.
- **Keep-on-sample** — drawing an entry for KL does not remove it. A failure can anchor multiple successes during its buffer lifetime.
- **Store under the context it was seen** — if failure was processed with ICL on, its re-run during KL also uses the same demo. If ICL was off, re-run is also query-only.

### Direction of KL

**Forward KL**: `KL(P_old || P_new)`.
- `P_old = softmax(z_old)`: the stored reference distribution (old model's predictions on the failure when first seen).
- `P_new = softmax(z_new)`: the current model's predictions on the same failure.
- Interpretation: "keep the current distribution covering the support of the historical distribution." Gradient penalizes the current model for dropping mass where the past model placed mass.

Confirmed in Rahaf's whiteboard drawing as forward, not reverse.

### Hyperparameters

| Name | Meaning | Starting value |
|---|---|---|
| `N` | Buffer size | 128–256 |
| `k` | Failures drawn per success step | 1 |
| `λ_KL` | KL anchor weight | Sweep {0.1, 0.5, 1.0} |

All to be tuned during Phase 2.

## Batch size considerations

- LoRA bake-off batch size: 8 examples/step.
- Phase 2 full training: **increase batch size** to amortize the extra forward pass introduced by the KL anchor on success steps and to parallelize the rehearsal computation across GPUs.
- Target: batch size 32–64 if GPU memory allows (depends on whether full fine-tuning or still adapter-based).
- Effective successes per step at batch=32 ≈ 16 → 16 additional forwards per step for KL. Acceptable if parallelized.

Exact batch size pinned down once we know:
- Phase 1 winner (determines head structure and memory footprint)
- Hardware availability (single H100 vs multi-GPU)
- Whether Phase 2 uses full fine-tune or LoRA at higher rank

## What's still TBD

- **Loss choice (`L_best`)** — determined by Phase 1 bake-off.
- **Full fine-tune vs LoRA** — full FT preferred if compute allows; LoRA (higher rank, e.g., 64) as fallback.
- **Training duration** — depends on convergence observed in LoRA runs, scaled up.
- **Whether to add an explicit distillation step** — Rahaf's drawing had mixed-training only, no separate distillation stage. Keep single-stage for now.
- **Multi-view augmentation** — the ×3 viewpoints available for DROID / MetaWorld / Failsafe. Likely useful in Phase 2; add as an ablation.
- **Temperature scaling for calibration** — apply post-hoc on a held-out split before reporting final metrics.

## Out of scope (explicitly)

- **L_traj** — dropped. Frame-trajectory consistency handled structurally via the backbone's self-attention; no separate trajectory loss.
- **L_cal (Brier term)** — dropped from the training loss. Calibration handled post-hoc via temperature scaling.
- **Two-stage training with distillation** — superseded by single-stage stochastic-ICL approach.
- **Mean Teacher / EMA-weights anchor** — superseded by the FIFO replay buffer approach above.
- **L_cons (frame-trajectory consistency)** — not needed; see L_traj reasoning.

## Implementation sketch

```python
buffer = FIFOBuffer(max_size=N)

for step, batch in enumerate(dataloader):
    for example in batch:
        if example.is_failure:
            # Standard failure step
            z = model(example.query, demo=example.demo)
            loss = L_best(z, example.labels)
            buffer.append(
                x_fail=example.query,
                z_old=z.detach(),
                demo=example.demo,
                icl_on=example.icl_on,
            )
        else:  # success step
            # Standard success loss
            z_success = model(example.query, demo=example.demo)
            loss_std = L_best(z_success, example.labels)

            # KL rehearsal anchor
            entry = buffer.sample(k=1)
            if entry is not None:
                z_new = model(entry.x_fail, demo=entry.demo)
                loss_kl = kl_forward(
                    softmax(entry.z_old),
                    softmax(z_new),
                )
                loss = loss_std + lambda_kl * loss_kl
            else:
                loss = loss_std

        loss.backward()
    optimizer.step()
```

Actual implementation will batch-parallelize the KL forward pass across the success-indexed examples in the batch.
