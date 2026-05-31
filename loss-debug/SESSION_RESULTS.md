# Session results log — 2026-05-21 evening

All numbers from CoffeePush release demos unless stated otherwise. Two-question framework:

> **Q1: How do we make the VLM get better at sparse-RL?**
> **Q2: Why is it not working (yet)?**

Each model section reports offline measurements (AUC, ECE, TPR@FPR=5%), online IBRL results, and how each datapoint speaks to Q1 / Q2.

---

## Cross-cutting findings (read first)

### Q2 — the obstacles we found and (partially) fixed tonight

| Bug | Effect | Status |
|---|---|---|
| `setup_utils.py` `ignore_mismatched_sizes=True` silently dropped a row of trained Robometer-4B embeddings | Every pre-fix 4B eval was wrong (1 row of random init). Fixed in commits `f0064ac` + `f048519`. | FIXED |
| `mm_token_type_ids` not passed to Qwen3.5 in eval pipeline | Qwen3.5-FT inference crashed in `compute_3d_position_ids` | FIXED (other agent) |
| `vlm_ibrl_qwen35` env did not exist (Qwen3.5 needs transformers 5.7 but vlm_ibrl deps live in transformers 4.57 / torch 2.4) | Couldn't run Qwen3.5-FT in IBRL at all | FIXED — built combined env, see Task #16 |
| Cold-GPU bf16 NaN trap on asymmetric-loss checkpoints (Robometer-FT 4000/5000, Qwen3.5-FT 4000/5000) | All-NaN success_prob on first forward; cuDNN fast-path picks a NaN-producing kernel on cold start | MITIGATED — multi-pass warm-up + cudnn deterministic + fp32 fallback in `RobometerScorer.__init__` |
| Per-frame vs trajectory-level GT labeling inconsistency in offline CM | Earlier "headline" numbers (TPR=0.711 in SUMMARY.md) used sticky labels; current code uses correct per-frame labels | FIXED — code uses non-sticky labels, all current numbers are with the right labeling |
| Single-shot offline CoffeePush eval is non-deterministic across runs (same model, same input → AUC swings ±0.30) | Cannot claim AUC/ECE as a model property at this compute budget | OPEN — would need `torch.use_deterministic_algorithms(True)` + multi-seed for publication |
| Flaky GPU node gcn83 produced silent NaN for two CM jobs | Confounded earlier results; added to `--exclude` list | WORKAROUND in place |
| Robometer-4B Kendall_last reproduction: paper claims 0.66, we get 0.058 | We cannot reproduce the published baseline metric | OPEN (Task #14) |

### Q1 — what the data says about how to make the VLM better

1. **Loss recipe matters but not the way the original LoRA plot suggested.** The asymmetric C51 + asymmetric BCE (λ=0.3) compresses BOTH success and failure predictions toward 0 (not just failures), reducing the absolute pos–neg separation by ~50% relative to paper-standard loss. Trajectory-level ECE is *worse* with asymmetric loss (0.27–0.40) than paper-standard (0.11–0.14). The LoRA "ECE → 0.05" finding was a per-frame at-goal-label artifact of base-rate matching, not real calibration improvement. (See `fig1_per_source.png`, `fig3_ece_and_separation.png`.)
2. **ICL helps Robometer-FT but hurts/flat for Qwen3.5-FT and 4B** — matches the user's training-time observation that ICL-on > ICL-off for the Robometer-FT family, while for Qwen3.5-FT ICL-off catches up at later checkpoints.
3. **Threshold calibration on the offline distribution does carry over to IBRL** for Robometer-FT step-3000 (validated tonight, Config A). This is the first clean methodological signal in our work — pick τ at offline FPR=5%, get a learning curve in IBRL.

---

## Robometer-FT

Base model: Robometer-4B (Qwen3-VL-4B-Instruct + Robometer pretraining).

### Run 1 (`run1_icl_ours`) — ICL + asymmetric loss (λ=0.3)

This is the recipe the user is presenting as "ours". Three checkpoints tested.

#### Checkpoint step-3000 (`/scratch-shared/.../Robometer_FT_consolidated/run1_icl_ours_step3000`)

**Offline CM on CoffeePush release demos (per-frame at-goal labels):**

| Variant | n | AUC [95% CI] | sparse ECE | TPR @ FPR=5% (τ) |
|---|---|---|---|---|
| no-ICL (60-clip) — run #1 | 60 | 0.565 [0.40, 0.71] | 0.299 | 0.208 (τ=?) |
| no-ICL (60-clip) — run #2 | 60 | 0.751 (run-to-run swing ±0.31) | 0.310 | — |
| no-ICL (300-clip) — run | 300 | 0.445 [0.38, 0.51] (below chance!) | 0.044 | 0.000 |
| no-ICL (300-clip) — re-run | 300 | 0.751 [0.70, 0.80] | 0.310 | 0.216 |
| **+ ICL (300-clip)** | 300 | **0.849 [0.79, 0.90]** | (high) | **0.59 at τ=0.0192** ⭐ |

ICL flips this checkpoint from below-chance to the best operating point we've measured all night.

**IBRL on CoffeePush, 60k env steps, seed=1:**

| Config | β / τ | Setup | Peak train_score (step) | num_success climb |
|---|---|---|---|---|
| A | 0.0 / **τ=0.0192** | +ICL (offline-calibrated 5% FPR threshold) | **0.12 at 45k** ⭐ | 12 → 24 |
| B | 0.0 / 0.0 | +ICL (continuous, rank-based) | 0.08 at 45k | 11 → 20 |
| C | 0.5 / 0.05 | +ICL (mixed progress+success) | 0.06 at 55k | 14 → 23 |

The calibrated-threshold config A wins by 50–100% over the alternatives. **Real learning curve — train_score climbs from 0.02 → 0.12 over training, num_success doubles.** Single seed; replication TODO.

> Q1 finding: offline FPR calibration transfers to IBRL — same threshold from offline analysis gives best IBRL signal.
> Q2 finding: 0.12 peak is 6–7× below Demo2Reward's published ~80%.

#### Checkpoint step-4000 (`run1_icl_ours_step4000`)

| Variant | AUC | TPR @ FPR=5% (τ) |
|---|---|---|
| no-ICL (300-clip) | 0.709 [0.65, 0.77] | 0.164 |
| + ICL (300-clip) | 0.816 [0.76, 0.88] | 0.32 at τ=0.131 |

IBRL not tested.

#### Checkpoint step-5000 (`run1_icl_ours_step5000`)

| Variant | AUC | TPR @ FPR=5% (τ) | Notes |
|---|---|---|---|
| no-ICL (300-clip) | 0.594 [0.53, 0.66] | 0.103 | |
| + ICL (300-clip, **fp32 forced**) | 0.744 [0.67, 0.81] | 0.25 at τ=0.188 | Required fp32 — bf16 deterministically NaN'd |

IBRL not tested.

### Run 2 (`run2_noicl_ours`) — asymmetric loss, no ICL

Used in the 2×2 loss-recipe ablation in `loss-debug/`. Saved checkpoints at steps 4000 and 5000 + ckpt-avg saves.

| Source | Asymmetric pos_mean / neg_mean | Standard (run3) pos_mean / neg_mean |
|---|---|---|
| droid    | 0.27 / 0.24 → sep 0.029 | 0.48 / 0.47 → sep 0.014 |
| robometer | 0.25 / 0.13 → sep 0.121 | 0.50 / 0.31 → sep 0.187 |
| metaworld | 0.22 / 0.11 → sep 0.108 | 0.54 / 0.31 → sep 0.231 |
| failsafe  | 0.18 / 0.08 → sep 0.097 | 0.45 / 0.20 → sep 0.250 |

Standard loss outperforms asymmetric on absolute separation in 7/8 (base × source) cells.

> Q1 finding: paper-standard loss separates classes better than our asymmetric recipe. The asymmetric loss compresses both classes toward 0.

### Run 3 (`run3_noicl_standard`) — paper-standard loss, no ICL

Comparison baseline for run 2. Same eval coverage. **Better separation and better trajectory-level calibration** (see fig3_ece_and_separation.png).

| Base | Loss | AUC | trajectory-level ECE | sep (pooled) |
|---|---|---|---|---|
| Robometer-4B | run2 asymmetric | 0.841 | **0.301** | +0.118 |
| Robometer-4B | run3 standard | 0.791 | **0.113** | +0.173 |

> Q1 finding: the asymmetric loss does NOT improve trajectory-level calibration over paper-standard. The LoRA "ECE drops" finding was on per-frame labels with ~5% base rate.

---

## Robometer-4B (baseline)

Public release: `robometer/Robometer-4B` from HuggingFace.

**Critical caveat: 4B's offline CM outputs are NON-DETERMINISTIC across runs.** Same model, same input, AUC varies 0.30 to 0.82, TPR @ FPR=5% varies 0.00 to 0.38. The compressed prediction range (always within ~0.07 wide for raw outputs) makes float noise routinely flip rank order.

**Offline CM dumps inventory (CoffeePush, no-ICL):**

| Job | n | sp range | AUC | TPR @ FPR≤5% (τ) | Notes |
|---|---|---|---|---|---|
| 22815198 | 60 | [0.31, 0.39] | **0.823** | 0.36 (τ=0.373) | Pre-fix sticky labels |
| 22869124 | 60 | [0.24, 0.28] | 0.748 | 0.31 (τ=0.270) | Pre-fix sticky labels |
| 22872215 | 60 | [0.23, 0.33] | 0.683 | 0.38 (τ=0.295) | Post-fix non-sticky |
| 22980746 | 60 | [0.25, 0.32] | 0.742 | 0.38 (τ=0.289) | Post-fix |
| 23012810 | 60 | [0.04, 0.59] | 0.513 | 0.08 (τ=0.459) | Outlier wide range — embed-tokens-pre-fix |
| 23023002 | 60 | [0.07, 0.10] | 0.718 | 0.17 (τ=0.094) | Post-fix, compressed range |
| 23023503 | 300 | [0.37, 0.49] | 0.660 | **0.38 (τ=0.434)** | Post-fix 300-clip — **the one chosen for the IBRL baseline** |
| 23023672 | 300 | [0.03, 0.07] | 0.653 | 0.06 (τ=0.063) | Post-fix 300-clip but compressed-range mode |

**With ICL** (300-clip): AUC 0.616, TPR @ FPR=5% = 0.04 at τ=0.404. **ICL hurts 4B**, consistent with the user's intuition that 4B wasn't trained with ICL.

**Kendall_last reproduction:** paper claims 0.66 on rbm-1m-ood; we get **+0.058** mean across the 6 OOD families. Separate bug, Task #14.

**IBRL — earlier 300k-step sweep (no ICL, β × τ ∈ {0,0.5,1} × {0,0.3,0.6}):** train_score=0 across all 18 configs.

**IBRL — tonight, head-to-head with FT-config-A:** Job 23025533 running now. Setup: no-ICL, β=0, **τ=0.434** (offline-calibrated), 60k env steps, seed=1. Result pending.

> Q1 framing once 23025533 lands: if 4B flatlines while FT step-3000 + ICL reaches 0.12, the comparison shows our method (FT + ICL + offline-calibrated τ) > baseline, even if both are below Demo2Reward.
> Q2 framing: 4B's TPR @ FPR=5% best-case is only 0.38 (vs FT's 0.59) — even at the same protocol the operating-point geometry doesn't support strong RL learning.

---

## Qwen3.5-FT

Base model: Qwen/Qwen3.5-4B (Qwen3.5-VL-4B). Different from Robometer-4B base.

### Run 4 (`run4_icl_ours_phase1`) — ICL + asymmetric loss

#### Checkpoint step-3000

| Variant | AUC | TPR @ FPR=5% (τ) |
|---|---|---|
| no-ICL (300-clip) | 0.740 [0.69, 0.79] | 0.233 |
| + ICL (300-clip) | 0.724 [0.65, 0.79] | 0.00 (predictions too compressed) |

IBRL not tested.

#### Checkpoint step-4000

| Variant | AUC | TPR @ FPR=5% (τ) |
|---|---|---|
| no-ICL (300-clip) | **0.848 [0.80, 0.89]** | **0.500** ⭐ (best non-ICL operating point of any model) |
| + ICL (300-clip) | 0.794 [0.73, 0.86] | 0.00 (compressed) |

IBRL not tested with calibrated τ. ICL hurts the operating point dramatically (crushes predictions below any useful threshold).

#### Checkpoint step-5000

| Variant | AUC | TPR @ FPR=5% (τ) |
|---|---|---|
| no-ICL (300-clip) | 0.780 [0.72, 0.83] | 0.078 |
| + ICL (300-clip) | 0.830 [0.76, 0.88] | 0.10 at τ=0.025 |

**IBRL — last night (no ICL, β=0, τ=0):** train_score peak **0.10 at 25–30k env steps, eval_score 0.25 at 30k**, num_success 15 → 39. The first positive RL result we ever obtained. Single seed.

> Q1 finding: Qwen3.5-FT step-5000 was the only IBRL learner in last night's runs. Reverse pattern from offline metrics (it had worst offline AUC) — TD-learner consumes rank-order, not absolute confidence.

### Run 5 (`run5_noicl_ours` = wandb w0otbkig) — asymmetric, no ICL

Used in the 2×2 loss ablation. See `loss-debug/fig1_per_source.png`. Compresses both pos and neg predictions like the Robometer-FT counterpart.

### Run 6 (`run6_noicl_standard` = wandb u9u7seky) — paper-standard, no ICL

Comparison baseline for run5. Wider separation between classes than run5.

---

## Summary table — best-case operating points

| Model + ICL state | Best AUC | Best TPR @ FPR=5% (τ) | IBRL peak (train_score, step) |
|---|---|---|---|
| Robometer-4B no-ICL | 0.66 (300-clip) | **0.38 at τ=0.434** | (pending) |
| Robometer-4B + ICL | 0.62 | 0.04 (ICL hurts) | n/a |
| Robometer-FT s3000 no-ICL | 0.75 | 0.22 | (untested, but had failed earlier with non-calibrated τ) |
| **Robometer-FT s3000 + ICL** | **0.85** | **0.59 at τ=0.0192** | **0.12 at 45k** ⭐ |
| Robometer-FT s4000 + ICL | 0.82 | 0.32 | (untested) |
| Robometer-FT s5000 + ICL (fp32) | 0.74 | 0.25 | (untested) |
| Qwen3.5-FT s3000 + ICL | 0.72 | 0.00 (compressed) | (untested) |
| Qwen3.5-FT s4000 no-ICL | **0.85** | **0.50** | (untested) |
| Qwen3.5-FT s5000 no-ICL | 0.78 | 0.08 | **eval_score 0.25 at 30k, train 0.10** ⭐ |

The two checkpoints with the strongest IBRL signal: **Robometer-FT s3000 + ICL + τ=0.0192** and **Qwen3.5-FT s5000 no-ICL + τ=0**. Different families, different recipes, both showed real learning curves on a single seed.

---

## What we still need to do (priority-ordered)

1. **Robometer-4B baseline IBRL with calibrated τ=0.434** — job 23025533 running now, fills in the apples-to-apples baseline comparison.
2. **Multi-seed for Config A (Robometer-FT s3000 + ICL + τ=0.0192)** — confirm 0.12 peak isn't a single-seed accident before claiming it.
3. **Test Robometer-FT s4000 + ICL in IBRL with its calibrated τ=0.131** — it had AUC=0.82 / TPR=0.32 offline; might be a stronger choice than s3000.
4. **Test Qwen3.5-FT s4000 in IBRL no-ICL with calibrated τ** — it had AUC=0.85 / TPR=0.50 offline, the best operating point of any non-ICL configuration. If this beats Config A's 0.12, it changes the paper's headline.
5. **Close the gap to Demo2Reward's ~80%**: this is the actual blocker for top-conference publication. None of our current numbers are competitive. Open question whether longer training, dense per-step reward (Chris's fallback), or a fundamentally different recipe can close 6–7×.
6. **Kendall_last 0.66 reproduction** (Task #14) — without reproducing the published baseline, even our offline-AUC comparisons are on shaky ground.

---

## How tonight's work answers the two framing questions

### Q1: How to make the VLM better?

**Things we showed help:**
- ICL on Robometer-FT family (preserves rank-order AND keeps operating point usable)
- Offline-calibrated threshold via FPR sweep (Config A vs B vs C shows the calibrated τ wins)
- Asymmetric loss + ICL on Robometer-FT (s3000 + ICL: AUC 0.85 / TPR 0.59 @ FPR=5%)

**Things we showed DON'T help (or hurt):**
- ICL on Qwen3.5-FT family (compresses predictions to where no threshold works)
- ICL on 4B baseline (4B wasn't trained with ICL → distribution shift)
- Asymmetric loss alone (compresses both classes, hurts trajectory ECE)

**Open questions:**
- Does ICL + asymmetric scale to ~80% IBRL success rate with more training/seeds, or are we structurally capped?
- Would a dense per-step reward (Robometer's own paper protocol) unlock the gap?

### Q2: Why is it not working?

**Mechanical failure modes we found and fixed:**
- embed_tokens silently random-initialized (every pre-fix 4B number was wrong)
- Qwen3.5 mm_token_type_ids not passed (Qwen3.5-FT inference was broken)
- vlm_ibrl_qwen35 env didn't exist (couldn't run Qwen3.5 in IBRL)
- Cold-GPU bf16 NaN trap on asymmetric checkpoints (mitigated, not eliminated)
- Non-sticky labeling in offline CM was the wrong protocol (fixed; numbers changed substantially)

**Methodological failure modes that remain:**
- Single-shot offline CoffeePush eval is non-deterministic — AUC swings ±0.30 across reruns of the same model. We are reporting noisy numbers in tables.
- ICL evaluation depends on which demo's frames we pick; not stable to demo choice.

**Structural gap:**
- Demo2Reward published ~80% peak success on CoffeePush IBRL. Our best is 0.12 (12%). 6–7× gap.
- The gap is not a calibration problem (we picked the threshold from offline FPR analysis). It's a *model-capability* problem: even with the best operating point we can find (TPR=0.59 at FPR=5%), the policy learning curve plateaus at 12%.
- Demo2Reward uses a different reward formulation (dense per-step, not threshold-at-truncation). Whether switching to that formulation would unlock the gap is the most consequential open question.

---

## Files where the underlying data lives

- `/scratch-shared/pkarageorgis1/vlm_ibrl_cm/*/cm_robometer_{4b,ft}_CoffeePush.json` — per-clip offline CM dumps
- `/scratch-shared/pkarageorgis1/vlm_ibrl_cm/*/cm_robometer_ft_CoffeePush_icl.json` — ICL CM dumps
- `/projects/prjs1958/pkarageorgis1/vlm_ibrl_sweep/coffeepush_robometer_ft_*/train.log` — IBRL training logs (per env-step milestones)
- `/projects/prjs1958/{Robometer,Qwen35}_FT_weights/run*_*/policy_ranking_samples/` — training-time eval dumps (used in 2×2 ablation)
- `/gpfs/home3/pkarageorgis1/Master-Thesis/loss-debug/` — figures, CSVs, README
- `/scratch-shared/pkarageorgis1/{Robometer,Qwen35}_FT_consolidated/` — consolidated FT checkpoints
- `/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl/env/robometer_utils.py` — RobometerScorer with multi-pass warm-up + fp32 fallback
- `/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl/env/vlm_envs.py` — env wrapper with ICL injection via ROBOMETER_ICL_DEMO_PATH env var
