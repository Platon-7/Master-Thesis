# Ordinal-Aware Reward Models with In-Context Learning for Robotic Task Progress Estimation

## Core Idea

Robometer trains a universal reward model with three heads: C51 progress (continuous, only supervised on successes via p_t = t/T), binary success, and pairwise preference. Progress on failures is never supervised — the preference head compensates by learning relative rankings.

We fix this directly: dense per-frame ordinal labels (1–5) on both successes and failures, a single ordinal head that replaces all three, and in-context learning where a success demonstration calibrates the progress scale per task.

## What Changes

| | Robometer | Ours |
|---|---|---|
| Progress head | C51 over [0,1], N bins | CORN ordinal over {1..5} |
| Success head | Separate binary logit | Derived: P(success) = P(y=5) |
| Preference head | Pairwise comparison | Dropped — redundant given dense ordinal supervision |
| Failure supervision | None (only via preference) | Per-frame ordinal labels from VLM+LLM pipeline |
| Input | Single trajectory | Demo (success) + query (ICL) |
| Training | Single stage | 2-stage: with ICL → without ICL |

## Loss Function

Single ordinal head per frame: 4 binary logits via CORN predicting P(y >= k) for k in {2,3,4,5}.

### Term 1 — Per-frame asymmetric ordinal loss

For each threshold k, define b_{t,k} = 1[y_t >= k]:

    L_frame = (1/|A|) * sum_t sum_k [ -beta_k * b_{t,k} * log(sigma(z_{t,k}))
                                       -alpha_k * (1-b_{t,k}) * log(1-sigma(z_{t,k})) ]

alpha_k >= beta_k: over-prediction (FP) penalized harder than under-prediction.
alpha_5 > alpha_4 > alpha_3 > alpha_2: false success is the most damaging error for RL.
Starting point: beta_k = 1, alpha_k = 1 + c(k-1), sweep c in [0.5, 3].

### Term 2 — Trajectory-level ordinal loss

Same asymmetric CORN on pooled representation against trajectory-level label Y.

### Term 3 — Consistency

KL divergence between trajectory prediction and aggregation of per-frame predictions.

### Term 4 — Calibration

Brier score on P(y=5) to keep success probabilities well-calibrated for downstream RL.

### Total

    L = L_frame + lambda_traj * L_traj + lambda_cons * L_cons + lambda_cal * L_cal

lambda_traj = 1.0, lambda_cons = 0.3, lambda_cal = 0.1.
Applied to query trajectory only. Demo is input context (ICL), not supervised.

## Why Each Component

- **CORN ordinal head**: encodes that predicting 2 when truth is 5 is worse than predicting 4. C51 bins have no ordinal semantics. Also unifies success and progress into one consistent distribution.
- **Asymmetric weighting**: formalizes the observation that false positives (over-predicted progress) cause reward hacking in RL, while false negatives just slow learning. The Bayes-optimal predictor under this loss is a conservative quantile — exactly what a reward model should be.
- **ICL**: the demonstration defines what progress 1-5 means for this specific task. Without it, the model learns a global progress scale that may not transfer across diverse tasks and embodiments.
- **2-stage training**: stage 1 learns with scaffolding (demos), stage 2 proves the model internalized the knowledge. Removes the inference-time dependency on having a demo available.

## Ablation Plan

| # | Variant | What it isolates |
|---|---|---|
| 0 | Robometer baseline (released weights) | Their method (inference only) |
| 1 | CORN symmetric, no ICL | Ordinal head vs C51 |
| 2 | CORN asymmetric, no ICL | Value of FP-penalization |
| 3 | CORN asymmetric, ICL stage 1 | Value of ICL |
| 4 | CORN asymmetric, 2-stage | Full proposal |

Runs 1-2 share a pipeline (train symmetric, save, continue asymmetric).
Runs 3-4 share a pipeline (train ICL, save, continue without demos).

## Dataset

| Source | Pairs | Labels |
|---|---|---|
| Robometer existing pairs | ~69K | Re-labeled with ordinal pipeline |
| DROID failures + success pairs | ~5.5K | VLM+LLM per-frame ordinal |
| MetaWorld / PlayWorld (sim) | TBD | Simulator ground-truth ordinal |
| Failsafe | TBD | Simulator ground-truth ordinal |
| **Total (conservative)** | **~75K–100K** | 16 frames/trajectory, all annotated |

Each pair = 1 success demo (16 frames) + 1 query trajectory (16 frames), both with ordinal labels.

## Compute Estimate (1x H100)

| Run | Steps | Time |
|---|---|---|
| CORN symmetric, no ICL | ~35K | 20–30h |
| CORN asymmetric, no ICL | ~10K (continued) | 6–8h |
| CORN asymmetric, ICL stage 1 | ~35K | 20–30h |
| CORN asymmetric, stage 2 | ~15K (continued) | 10–12h |
| **Total training** | | **55–80h** |

With LoRA: ~12–20h total. With 4x H100 (FSDP): divide by ~3.5x.

## Timeline to NeurIPS (deadline: May 6)

| Dates | Task | Days |
|---|---|---|
| Apr 16–24 | Data pipeline: label remaining pairs, extract keyframes, build unified dataset | 8 |
| Apr 24–28 | Training: all 4 ablation runs (LoRA first, full FT if needed) | 4 |
| Apr 28–30 | RL loop evaluation: plug trained model as reward into SAC/DSRL | 2 |
| May 1–3 | Paper writing | 3 |
| May 4–5 | Buffer / revisions | 2 |
| **May 6** | **Submit** | |

## Limitations (to acknowledge in paper)

- No real-world robot experiments (sim + offline real-world datasets only)
- Success trajectory labels use t/T heuristic, not VLM-labeled (acknowledged simplification)
- Failure labels from VLM+LLM pipeline — noisy, not human-annotated ground truth
- RL evaluation on limited set of tasks (Robomimic, MetaWorld)

## Contributions

1. **Dataset**: first large-scale failure dataset with dense per-frame progress labels for robotic manipulation, paired with success demonstrations
2. **Architecture + Loss**: ordinal CORN head with asymmetric FP-penalization, replacing C51 + success + preference — theoretically grounded in conservative reward estimation for RL
3. **ICL for reward models**: in-context demonstrations as task-specific progress calibration, with 2-stage training to remove inference-time demo dependency
