# Deck outline — VLM reward models for downstream RL
Self-contained narrative. Figures in `../figures/`, numbers in `../RESULTS.md`
and `../results/FULL_METRICS.csv`. Story follows the data.

---

## 0. Hook — "where it went wrong"
- IBRL on CoffeePush with our VLM rewards (Robometer-FT, Qwen3.5-FT, baseline)
  **caps at ~0.12 success; the policy collapses to ~2%.**
- And on the **OOD** test set, the *untrained* Robometer-4B baseline **out-ranks
  our fine-tuned models**: kendall_last **0.638 vs 0.29–0.32** (paper-exact harness).
- → Fine-tuning didn't help downstream and *hurt* OOD ranking. Why?

## 1. Pure ranking performance
- **OOD** (kendall, harness; AUC, grid): baseline ≫ all FT.
  Gate: baseline reproduces the paper (kendall_last 0.633). [confound ruled out:
  re-ran at correct per-model frames 8/16 — failure is real, not a frame artifact.]
- **In-distribution** (our curated data): FT ≫ baseline (success AUC ~0.85 vs 0.55)
  — baseline is near-random because our data is OOD *for it*.
- **fig2_specialization.png** — the tradeoff in one scatter.

## 2. THE finding — asymmetric loss destroys the progress head
- **fig1_progress_head_collapse.png** (the money slide).
- Progress-head correlation with GT progress (VOC-pearson / kendall):
  - paper-standard + baseline: **0.55–0.87** (OOD), **0.31–0.75** (in-dist) — intact.
  - **every asymmetric model (run1/2/4/5): ≈ −0.03 to −0.05** — dead, both bases,
    both distributions.
- The asymmetric C51 loss kills the head IBRL's *dense* reward depends on.

## 3. Which metric actually matters? (honesty slide)
- **dense-ECE does NOT separate the models** (~0.31–0.57 everywhere incl. baseline)
  — **fig3_ece_flat.png**. The "ECE explains the failure" hypothesis is *not*
  supported. We followed the data: the mechanism is the progress-head collapse.
- ICL at inference doesn't reliably help (in-dist ICL-on ≈ ICL-off).

## 4. Ablations (what each ingredient did)
- Loss: paper-standard keeps the progress head; asymmetric destroys it. On the
  success head, asymmetric is *slightly better* in-dist/OOD — but at the cost of
  the progress head. Net: **paper-standard is the better recipe for RL.**
- ICL: negligible at inference; main value was at training (in-dist gains).
- Base model: Robometer vs Qwen3.5 — same progress-head collapse under asymmetric.

## 5. Downstream RL — the link
- VLM-reward IBRL fails (0.12, policy → 2%).
- **GT-reward control trains to 0.56–0.82** → the IBRL loop is *capable*; the
  reward is the limiter (not the RL setup, not the task).
- Reward ranks live BC rollouts at AUC ~0.7 (seed-noisy 0.66–0.88); scoring path
  verified bug-free (inline ≡ direct).
- Progress head **inverted on rollouts** → explained by §2.

## 6. Takeaways
1. Our asymmetric loss broke the progress head — the cleanest, most consequential
   finding. Paper-standard preserves it.
2. Fine-tuning specializes: in-dist gains, OOD loss (real, frame-confound ruled out).
3. The downstream-RL bottleneck is the reward's usable signal (progress collapse +
   low success margin), NOT the RL loop (GT → 0.82) and NOT calibration/ECE.
4. Recommended next recipe: paper-standard loss (intact progress head) + on-policy
   reward data for OOD robustness.

## Open / caveats (state plainly)
- ICL-on cells for run1/run2 (FP32 OOM) and run6 (Qwen3.5 NaN) incomplete —
  environment issues, not data; finding unaffected (from ICL-off cells).
- Qwen3.5 OOD kendall via harness blocked (unsloth) — success AUC (0.65–0.78)
  covers ranking.
- BC-rollout reward AUC seed-noisy (0.66–0.88) — report as a range w/ CI.
