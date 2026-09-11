# Reward-model study — offline eval results (2026-06-02)

Self-contained study comparing 8 VLM reward models on ranking + calibration,
in-distribution and OOD, plus the downstream-RL link. All numbers follow the
data — no thumb on the scale toward any prior hypothesis (ECE was NOT confirmed
as the differentiator; see below).

## Models
- **baseline** — Robometer-4B (released, 8 frames)
- **run1** s4000/s5000 — Robometer-FT, asymmetric loss + ICL (16 frames)
- **run2** s5000 — Robometer-FT, asymmetric, no ICL
- **run3** s5000 — Robometer-FT, **paper-standard** loss, no ICL
- **run4** s6500 — Qwen3.5-FT, asymmetric + ICL
- **run5** s6500 — Qwen3.5-FT, asymmetric, no ICL
- **run6** s6500 — Qwen3.5-FT, **paper-standard**, no ICL

## Metric sourcing (bulletproof)
- **kendall / VOC-pearson (paper-exact ranking)** → Robometer harness
  (`run_baseline_eval.py policy_ranking` on rbm-1m-ood). Gate: baseline reproduces
  kendall_last **0.633** (paper ~0.66). Frames per-model (baseline 8, FT 16) —
  the original FT runs used the wrong global 8, which we corrected.
- **AUC / dense-ECE (calibration) + a cross-check kendall/pearson** → our grid
  (`reward-model-study/scripts/eval_dump.py` + `compute_metrics.py`), per-model
  frames auto-detected, both heads. Capped at 50/source for equal-n.

## Sets (FULL, not capped)
- **OOD**: 782 trajectories (666 success / 116 failure), every model.
- **In-dist**: common 3,142-trajectory intersection scored by all 8 models
  (1,344 success / 1,798 failure) — equal-n across models (the correct basis for
  cross-model AUC). [The bf16 cells reached the full 5,107; the FP32-asymmetric
  cells hit walltime at ~3,142; we report the common set every model has.]
- **ICL-on**: 129 (capped 50/source small test).

## HEADLINE FINDING — RETRACTED (2026-06-06)
The earlier claim here — *"the asymmetric loss destroys the progress head"*,
based on a near-zero progress↔GT **correlation** — is **obsolete and wrong**.
Measuring class **separability** (not correlation) shows the opposite: the
asymmetric-loss 4B FT models are the **best** success/failure separators on the
progress head (d′ ≈ 1.3–1.5 vs baseline 0.42). The asymmetric loss compresses the
score **scale** (small raw gap) but keeps — even improves — **separability**.
Authoritative, up-to-date reward-quality results now live in
`deck/VLM_reward_models.pptx` (slides 7–8).

## OOD ranking — success head (kendall from harness, AUC from grid; FULL 782)
`kendall_last` = paper-exact Robometer harness (`policy_ranking` on rbm-1m-ood, 6
robot tasks, 3-level quality); a **cross-trajectory** quality ranking of final-frame
progress — a DIFFERENT axis from the grid's **within-trajectory** progress-shape
pearson. Qwen3.5 run via `REPO_DIR=Qwen35-FT` (the Qwen3.5-aware setup_utils),
batch_size=4 (lm_head OOM guard); fidelity-gated against the grid.

| model | kendall_last (harness) | succ_AUC (grid, full) |
|---|---|---|
| baseline | **0.638** | **0.871** |
| run1 s5000 (4B asym) | 0.295 | 0.593 |
| run2 s5000 (4B asym) | 0.290 | 0.563 |
| run3 s5000 (4B paper-std) | −0.049 | 0.542 |
| run4 (Qwen3.5 asym) | −0.019 | 0.664 |
| run5 (Qwen3.5 asym) | 0.068 | 0.687 |
| run6 (Qwen3.5 paper-std) | 0.134 | 0.578 |

Baseline ≫ all FT on OOD ranking (harness 0.638 vs ≤0.30; success AUC 0.87 vs
0.54–0.69). The harness kendall does NOT cleanly separate the loss types (4B: asym
> paper-std; Qwen3.5: paper-std > asym) — it is a cross-trajectory ranking, noisier
(small per-task N) and orthogonal to within-trajectory progress shape.

## In-distribution — success head AUC (grid, ICL off, common 3,142)
| model | succ_AUC | succ_denseECE |
|---|---|---|
| baseline | 0.656 (weak — our curated data is OOD *for it*) | 0.368 |
| run1 s4000 (asym+ICL) | 0.859 | 0.378 |
| run1 s5000 (asym+ICL) | 0.836 | 0.386 |
| run2 (asym) | 0.882 | 0.382 |
| run3 (paper-std) | 0.873 | 0.415 |
| run4 (Qwen3.5 asym) | 0.657 | 0.374 |
| run5 (Qwen3.5 asym) | 0.573 | 0.340 |
| run6 (Qwen3.5 paper-std) | 0.678 | 0.408 |

**Specialization tradeoff:** the 4B FT wins in-dist (0.84–0.88 vs baseline 0.66),
loses OOD (0.54–0.59 vs 0.87). Qwen3.5 FT is weaker in-dist (0.57–0.68).

## What the data does NOT support
- **dense-ECE is NOT a differentiator** — ~0.31–0.57 across all models including
  baseline, both distributions. The "bad ECE explains RL failure" hypothesis is
  not borne out. The progress-head collapse is the real mechanism.
- **ICL at inference doesn't reliably help** — in-dist ICL-on ≈ ICL-off (sometimes
  worse). ICL was negligible on OOD (no demos there anyway).

## Downstream RL — the reward is the limiter, via EXPLOITABLE false positives
- VLM-reward IBRL on CoffeePush caps at ~0.10–0.14; **GT-reward control → 0.82** on
  the identical loop → the loop is capable, the reward is the limiter.
- **It is NOT distribution shift / perception:** the reward reads live successes
  offline (E2 AUC **0.89**, success 0.51 ≫ failure 0.20).
- **It is NOT the FP rate per se:** a controlled dose-response (GT reward + *random*
  injected FP) degrades **gradually** — 5% → 0.48, 10% → 0.28 — no cliff. A policy
  tolerates moderate *random* false positives.
- **It IS an exploitable feedback loop.** The reward's failure scores overlap the
  successes (d′ < 2), giving a ~14–17% FP **seed** that is a consistent, findable
  pattern. RL is an optimizer → the policy learns to trigger it, so on-policy FPR
  **ramps up over training** (directly observed in `data/vlm_reward_FPR`: ~8–17% at
  BC-init → **50–88%** by 10–30k steps, all 3 models), banking reward ≈ FPR with
  TPR ≈ 0 and true success 1–5% — textbook reward hacking.
  NOTE: random injected FP (any schedule) cannot model this — randomness has no
  feedback, so the dose-response only establishes the *passive* tolerance (low end);
  the **observed FPR ramp** is the separate, direct evidence of the feedback loop.
  Do NOT plot the VLM "on the random-FP curve" — different regimes.
- **Direction:** on-policy reward retraining (relabel the FPs the policy finds →
  kills the exploitable seam); TPAUC to lower the offline FP floor as a complement.
- Artifacts: `figures/rl_{variance_dprime,reward_hacking_3models,reward_hacking_proof,fp_doseresponse}.png`,
  `results/{reward_hacking_summary,fp_doseresponse}.csv`.

## Caveats / open
- **failsafe source dropped** from in-dist (eval-split ids don't match keyframe
  tars — dataset-version mismatch); in-dist covers droid/metaworld/robometer. (Accepted.)
- In-dist FP32-asymmetric cells hit walltime at ~3,142/5,107; we report the common
  3,142 every model scored (equal-n — the correct AUC basis anyway).
- ICL-on is a capped 50/source (n=129) small test, not the full set (by design).
- Downstream BC-rollout reward AUC is seed-noisy (0.66–0.88) — report as a range w/ CI.

RESOLVED this pass: Qwen3.5 OOD harness kendall (was wrongly thought "blocked by
unsloth"; real cause was the harness importing Robometer's Qwen3-VL setup_utils +
an lm_head OOM — fixed via REPO_DIR=Qwen35-FT + batch_size=4). run6 in-dist ICL-on
degeneracy fixed (bf16+warmup → real values).

## Artifacts
- `results/FULL_METRICS.csv` — full matrix (8 models × 3 cells × both heads), full sets
- `results/harness_qwen35/{run4,run5,run6}_s6500/all_metrics.json` — Qwen3.5 harness kendall
- `results/ood_kendall_harness.csv` — paper-exact Robometer (4B) OOD kendall
- `results/{full_ood,full_indist,cap_indist}_*.jsonl` — per-trajectory dumps (both heads + GT)
- `scripts/eval_dump.py`, `scripts/compute_metrics.py`, `scripts/plot_results.py` — the pipeline
