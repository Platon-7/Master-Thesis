# CORN Head Pretrain — Debug Plan

**Context**: The Apr-30 CORN head pretrain attempt collapsed to constant-output predictions
within 150 steps and never recovered. `avg_succ_fail_diff` went from 6.5e-5 (essentially
zero) at step 50 to bit-identical 0.0 at steps 150/200/250. Loss bounced in the 0.24-0.33
band without descent. Initial hypothesis: head-only training with a frozen Robometer-4B
backbone can't extract per-frame discrimination because the C51 head was carrying that
information during Robometer-4B's training.

This doc lays out the structured debug plan and reports outcomes.

---

## Step 1 — Gradient sanity (~30 min, GPU)

**Question**: Is the CORN loss producing nonzero gradients on the head's parameters?

A "constant output" pattern can come from a buggy loss whose gradient w.r.t. the head is
zero or numerically truncated. Before blaming the backbone or the data distribution,
verify the gradient signal exists.

**Procedure**:
1. Load Robometer-4B with the pretrain config (frozen backbone, fresh CORN head).
2. Build one minibatch (8 trajectories, 16 frames each).
3. Forward → loss → `loss.backward()`.
4. Inspect `head.weight.grad` for each Linear layer in the CORN head.

**Pass condition**: gradients are finite, nonzero, and have magnitude ≥ 1e-4 in fp32.
**Fail conditions**:
- All-zero grads → loss has a bug (mask zeroing out, detach, wrong dim reduction, etc.)
- Tiny grads (1e-8 or smaller) → bf16 precision starvation, switch head to fp32
- NaN grads → numerical instability

---

## Step 2 — Backbone-embedding per-frame discrimination (~15 min, GPU)

**Question**: Does the frozen Robometer-4B backbone produce per-frame hidden states that
*differ enough* across frames within a trajectory to support per-frame ordinal prediction?

If the backbone collapses per-frame info (because Robometer-4B's training shifted that
info into the C51 head), no amount of CORN head training can recover it from the frozen
backbone alone.

**Procedure**:
1. Sample 20 partnered trajectories (10 success + 10 failure, mix of sources).
2. Run each through the frozen backbone.
3. Extract the per-`<|prog_token|>` hidden states (the same vectors fed to the head).
4. For each trajectory, compute pairwise cosine similarity between its 16 frame embeddings.
5. Compute pairwise cosine similarity *across* trajectories (success-final vs failure-final).

**Healthy backbone**:
- Within-trajectory similarities: **0.80–0.95** (similar but not identical).
- Within-trajectory variance > 0.02.
- Success-final vs failure-final cross-cosine: **< 0.85** (distinguishable).

**Collapsed backbone**:
- Within-trajectory similarities: **> 0.99** (frames look identical to the head).
- Cross-trajectory similarities: **> 0.99** (success and failure look identical).

If collapsed, head training cannot fix it — we need to either unfreeze the top backbone
layers, distill from the C51 head's output, or accept that LoRA-from-scratch is the
only viable Loss-1 recipe.

**Bonus check**: also dump *exactly what the head receives* (after the trainer's per-frame
extraction) — confirm the trainer-level pooling/extraction isn't itself collapsing variation.

---

## Step 3 — Class-imbalance failure mode (~15 min, data-only) ✅ DONE

**Question**: Is the symmetric BCE preferring `σ=0` because the positive class is too rare?

For each threshold `k ∈ {2, 3, 4, 5}`, the CORN head trains a separate sigmoid `σ(z_k)` to
predict `P(y ≥ k)`. If `P(y ≥ k)` is small enough, the symmetric BCE has a near-trivial
minimum at "always predict negative" with small gradient toward escape.

**Result**:

| label y | failure-side count | failure-side fraction |
|---|---:|---:|
| 1 (no progress) | 31,627 | 50.5% |
| 2 (approach) | 17,175 | 27.4% |
| 3 (partial) | 12,361 | 19.8% |
| 4 (major) | 1,429 | 2.3% |
| 5 (success) | 0 | **0.0%** |

| threshold k | combined P(y ≥ k) ¹ | symmetric-BCE basin |
|---|---:|---|
| k=2 | 0.62 | balanced ✓ |
| k=3 | 0.39 | mild imbalance |
| k=4 | 0.20 | imbalanced — pos_weight helps |
| k=5 | 0.095 | severe — pos_weight required |

¹ combined = 0.5 × failure-side + 0.5 × success-side, where success-side labels are
synthesized at sample time as `y_t = ceil(5·t/16)` ∈ {1..5}.

**Verdict**: imbalance is real. Symmetric BCE without pos_weight has a strong "predict
zero" pull on thresholds 4 and 5, which propagates through the head's shared trunk and
collapses all four logits to constants.

**Fix**: per-threshold `pos_weight = N_neg / N_pos`:
- β_2 ≈ 0.6  (no boost — already balanced)
- β_3 ≈ 1.6
- β_4 ≈ 4.0
- β_5 ≈ 9.5

Plumbed into the symmetric CORN loss path via `BCEWithLogitsLoss(pos_weight=...)`.

---

## Step 4 — Tiny-data overfit (~30 min, GPU)

**Question**: With the pos_weight fix in, can the CORN head overfit a tiny dataset?

Standard ML debug: if a model can't memorize 50 examples in 200 steps with the same
recipe, the model has a structural problem (loss bug, capacity issue, optimization
pathology) — not a data-volume problem.

**Procedure**:
1. Subset the train pool to ~50 trajectories (mix of success and failure).
2. Run the CORN head pretrain with pos_weight enabled, 200 steps, same lr.
3. Watch train loss.

**Pass condition**: train loss descends to near-zero (< 0.05) by step 200, AND
`avg_succ_fail_diff` becomes meaningfully nonzero on the eval split (> 0.01).

**Fail condition**: loss bounces in a noisy band like the original run, and
`avg_succ_fail_diff` stays at machine-zero. This implicates a structural problem
beyond class imbalance.

---

## Parallelization plan

- **Step 3** (data-only): completed.
- **Steps 1 + 2** (single GPU job, single model load — the diagnostic script does both
  on the same forward pass): one SLURM job, ~15-30 min.
- **Step 4** (separate config + tiny-data setup): one SLURM job, ~30 min.

These two GPU jobs can run in parallel — they don't share any state.

---

## Outcome ledger

| step | status | result |
|---|---|---|
| 1 | done ✅ | gradients healthy on all head layers (final layer |grad|_mean = 4.4e-3, max = 5.6e-2) |
| 2 | done ✅ | within=0.837, across=0.756 — backbone IS discriminative per-frame and per-trajectory |
| 3 | done ✅ | confirmed: k=2 P=0.40, k=3 P=0.21, k=4 P=0.077, **k=5 P=0.000** at the dataloader |
| sanity | done ✅ | 10-step pos_weight run: chaotic loss, σ(z_5) range still flat 0.008 |
| 4 | skipped | "no third config" rule — see verdict |

### Step 1 — Gradient sanity (job 22394873)

All 6 head parameters showed healthy gradients, no NaN, no zero collapse:

| param | shape | abs_mean | abs_max |
|---|---|---:|---:|
| progress_head.0.weight | [1280, 2560] | 2.4e-4 | 1.2e-2 |
| progress_head.0.bias | [1280] | 1.0e-4 | 7.6e-4 |
| progress_head.1.weight | [1280] | 1.7e-4 | 2.4e-3 |
| progress_head.1.bias | [1280] | 1.9e-4 | 1.4e-3 |
| **progress_head.4.weight** | [4, 1280] | **4.4e-3** | **5.6e-2** |
| **progress_head.4.bias** | [4] | **9.9e-3** | **1.5e-2** |

The CORN final layer gets substantial gradient signal (largest in the network). Loss path is wired correctly.

### Step 2 — Backbone discrimination (same job)

| metric | value | healthy band | result |
|---|---:|---|---|
| within-trajectory cosine | **0.837** | 0.80–0.95 | ✅ |
| across-trajectory cosine | **0.756** | < 0.85 | ✅ |

Per-frame hidden states are similar but not identical within a trajectory (head can distinguish frames), and different trajectories have distinct embeddings. **Backbone is healthy** — it carries the per-frame info the head needs.

---

## Verdict

The pretrain failure is **not a fundamental incompatibility**. The backbone can support per-frame ordinal prediction, and the head receives real gradients. The problem is a **specific, narrow recipe issue**:

1. **k=5 is information-empty**: P(y=5) = 0% on the failure side and the only positives are the synthetic last-frame `y=5` from successes (~6% combined). At the dataloader, the rate is 0.00 — too thin for stable learning.
2. **k=2,3,4 are learnable** but the symmetric-BCE recipe with `lr=1e-3 × pos_weight∈[1.2, 3.5]` blew the optimizer apart in the 10-step sanity test (loss spike to 5.94 at step 6). The lr × pos_weight product is too aggressive.

A **second pos_weight + lower lr** config could likely work. But per the user's standing rule: **no third pretrain config this sprint**. We ship L2.

**Action items**:
- ✅ pos_weight plumbing stays in the codebase (unused but ready for a future sprint)
- ✅ ship L2 (asymmetric C51 + BCE) as the surviving Loss-1 candidate
- ✅ document this debug trail so a future sprint can pick up with: lower lr (1e-4 or 5e-5), use β_k=[1.2, 1.9, 3.5, ~] with k=5 either dropped (predict 4 thresholds → output for k=5 derived) or down-weighted heavily

---

## May-1 update — pretrain SUCCEEDED with corrected recipe

Three things changed after the verdict above:

1. **Debug script bug found**: my `y = (tp*4).round().clamp(1,5)` was missing the `+1` shift the trainer uses. The "P(y≥5)=0%" finding was actually P(y≥4) misidentified. Re-running the corrected debug (job 22395228) showed:

   | k | actual P(y≥k) | sqrt pos_weight |
   |---|---:|---:|
   | 2 | 0.602 | 0.81 |
   | 3 | 0.399 | 1.23 |
   | 4 | 0.209 | 1.95 |
   | 5 | **0.076** | 3.48 |

   k=5 has 7.6% positives, not zero — the basin IS escapable.

2. **Math derivation killed pos_weight**: with bias-init at σ=p_k, the expected gradient under pos_weight β_k is `-p(1-p)(β-1)`. Non-zero whenever β≠1. The 50-step sanity (job 22396654) confirmed this: σ(z_5) drifted from 0.076 to 0.197 in 50 steps. **pos_weight defeats bias-init.** The fix is bias-init alone.

3. **Final recipe** (configs/corn_head_pretrain.yaml):
   - bias-init final-layer bias to logit(priors) = [+0.41, -0.41, -1.33, -2.50]
   - β_k = 1 (no pos_weight)
   - lr = 1e-4 with 5% warmup
   - 500 steps, symmetric CORN (c=0)

**Result** (job 22397021, 500 steps):

| k | prior | σ_mean | %dev | range_width |
|---|---:|---:|---:|---:|
| 2 | 0.602 | 0.591 | -1.8% | **0.519** |
| 3 | 0.399 | 0.400 | +0.3% | **0.308** |
| 4 | 0.209 | 0.207 | -0.7% | **0.214** |
| 5 | 0.076 | 0.067 | -11.7% | **0.096** |

- σ(z_5) range 0.096 — nearly 2× the 0.05 gate.
- All means within ±12% of priors (no degenerate drift).
- Per-sample Spearman σ(z_5) vs target_progress: **0.553 mean** — head correctly orders frames within trajectories.

**Pretrain checkpoint**: `/projects/prjs1958/LoRA_weights/corn_head_pretrain_22397021/corn_head_pretrain/` — ready for loss1_corn LoRA bake-off.

**Lesson**: my "k=5 information-empty" verdict was wrong, downstream of a one-line bug in my own diagnostic. Always re-derive the data distribution from the actual loss code, not from a hand-rolled debug script.
