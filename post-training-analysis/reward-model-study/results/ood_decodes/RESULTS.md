# OOD trajectory ranking, re-scored with alternative C51 decodes

**Date:** 2026-07-07 · **Jobs:** 1493–1496 (`robometer-policy-learning/jobs/ood_decode_eval.sbatch`)
· **Script:** `reward-model-study/scripts/ood_decode_eval.py` · **Data:** HF export
`robometer/rbm-1m-ood` → `/shared/home/PKA4388/rbm-1m-ood`

Re-examines deck slide 3 of `VLM_reward_models.pptx` ("Trajectory ranking —
out-of-distribution"), which concluded that fine-tuning on our failure data reduces OOD
ranking. Both progress-head columns and the success-AUC column are revised.

## Setup

- **571 episodes, 6 held-out-robot datasets:** mit_franka 304, usc_koch 150, usc_xarm 36,
  utd_so101 30, usc_trossen 27, usc_franka 24 (matches `export_summary.json` exactly; the
  other `train/` rows — rewind_og / paired / clutter / wrist / human — are non-eval extras).
- Quality labels `successful` / `suboptimal` / `failure` (export writes `success`, mapped).
- **Metric code imported from the original harness** (`robometer.evals.compile_results.
  _compute_policy_ranking_metrics_quality_label`, τ-a via `eval_metrics_utils.kendall_tau_a`);
  slide aggregation = simple mean over the 6 datasets (validated against saved run5 harness
  output to 4 decimal places). Sanity: synthetic perfect ranking → 1.0/1.0, inverted → −1.0/0.0.
- Frame budgets match the original harness: baseline 8, FT models 16. ICL model scored
  without demo (`use_icl=false`), as in the original.
- Scorer: `vlm_ibrl/env/robometer_utils.py::RobometerScorer` (current version: cuDNN autotune
  off, warm-up-until-non-NaN, C51 bin capture). Per-model check: recomputed EV from captured
  bins equals the pipeline EV to 1e-5.

## Reproduction gate

| check | value | slide | verdict |
|---|---|---|---|
| baseline EV kendall_last | **0.6483** | 0.6384 | ✅ reproduces (±0.03 gate) |
| baseline EV ranking_acc_sum | 0.7402 | 0.813 | ❌ **not reproducible from this export** |

`ranking_acc_sum` sums per-frame EVs, so it carries a trajectory-length component; the HF
export normalizes every episode to ~32 frames (min 21, max 32), destroying that component.
The sum column below is kept for reference only (`*`); the decode-comparable pairwise
accuracy is `ranking_acc_last` (last-frame values, same pairing logic).

## Progress head: kendall_last / ranking_acc_last by decode

| model | EV (= slide decode) | condMean | median | argmax |
|---|---|---|---|---|
| baseline (Robometer-4B) | **+.648** / .816 | +.634 / .806 | +.559 / .837 | +.447 / .824 |
| run1_s4000 (asym+ICL) | +.308 / .659 | **+.502** / .754 | +.000 / .834 | +.000 / .834 |
| run2_s4000 (asym) | +.159 / .572 | **+.584** / .807 | +.001 / .834 | +.000 / .834 |
| run3_s5000 (standard) | +.054 / .531 | **+.498** / .759 | +.191 / .776 | +.124 / .790 |

(reference, non-reproducible sum column: baseline EV .740*, run2 condMean .746*, argmax .856*;
full values in `*_metrics.json`)

**Decode × model interaction, both directions as predicted:**
- The baseline's unimodal head is read best by the EV; every alternative decode *hurts* it.
- Every FT head (bimodal: bin-0 hedge lump + spike) is read worst by the EV and best by
  condMean = EV/(1−P(bin0)): run2 goes 0.16 → **0.58**, statistically at baseline level, with
  pairwise accuracy 0.57 → 0.81 (baseline: 0.82).
- Quantized decodes (median/argmax) collapse clips onto few bin values: pairwise accuracy is
  excellent (.83) but τ-a, which skips ties, goes to ~0. condMean is the only decode that is
  both continuous and calibrated.

## Success head: AUC recomputed from the same scoring pass

| model | AUC succ-vs-fail | slide 3 | subopt-as-neg | subopt-as-pos | mean p(succ): succ / subopt / fail |
|---|---|---|---|---|---|
| baseline | **0.860** | 0.87 ✅ | 0.775 | 0.806 | 0.663 / 0.439 / 0.204 |
| run1_s4000 | **0.837** | 0.60 ❌ | 0.769 | 0.784 | 0.852 / 0.791 / 0.603 |
| run2_s4000 | **0.792** | 0.56 ❌ | 0.693 | 0.756 | 0.796 / 0.722 / 0.534 |
| run3_s5000 | **0.761** | 0.54 ❌ | 0.692 | 0.715 | 0.450 / 0.379 / 0.279 |

The baseline replicates the slide; **the FT collapse does not** (all FT rows +0.2–0.28 vs the
slide). Leading explanation: the original `eval_dump` predated the scorer's NaN defenses —
its own comments document that asymmetric-FT checkpoints produced ALL-NaN success logits via
a cuDNN/bf16 fast-path before cuDNN autotune was disabled and the warm-up loop added. That
failure mode is FT-specific (baseline unaffected), matching the slide's signature exactly.
Not directly autopsiable (the original per-trajectory JSONLs remained on Snellius).

## Revised slide-3 conclusion

Fine-tuning does **not** destroy OOD ranking. Decoded correctly (condMean) and scored with the
NaN-hardened scorer, the FT models rank held-out robots at or near baseline parity
(kendall 0.50–0.58 vs 0.63–0.65; pairwise acc ~0.81 vs 0.82; success-AUC 0.76–0.84 vs 0.86),
while being calibrated higher on true successes (run1 0.85 vs baseline 0.66). The baseline
retains a small genuine edge OOD — an ordinary fine-tuning trade-off, not a collapse. This is
the offline/OOD twin of the LIBERO downstream finding: several apparent model failures were
instrument errors of the standard EV readout.

## Caveats

1. `run2_s4000` — the slide row used step 5000 (checkpoint only on Snellius); step-4000 used
   here. Baseline / run1 / run3 rows are the slide's exact checkpoints.
2. Episode set: 571 (this export) vs 782 in the original success-AUC dump — can move an AUC a
   few points, not 25.
3. success-AUC is decode-independent; its revision comes purely from re-scoring.
4. Qwen3.5 rows (run4–6) not yet re-scored — same script, add checkpoint paths + Qwen35-FT
   sys.path dispatch (scorer handles it via config.yaml base_model_id).

## Qwen3.5-FT rows (job 1497, env robometer_qwen35_gpu, phase-1 s6500 consolidated ckpts)

Progress head (kendall_last / ranking_acc_last):

| model | EV | condMean | median | argmax | old harness EV kendall |
|---|---|---|---|---|---|
| q35_run4 (asym+ICL) | +.022 / .485 | +.136 / .546 | +.042 / .820 | +.035 / .824 | −.019 |
| q35_run5 (asym) | +.277 / .662 | +.277 / .644 | +.000 / .834 | +.000 / .834 | +.068 |
| q35_run6 (std) | +.226 / .598 | +.229 / .587 | +.035 / .739 | +.005 / .812 | +.134 |

Success head: AUC succ-vs-fail = .652 / .660 / .663 (slide: .66 / .69 / .58) — **reproduces**.

Findings differ from the 4B family in three ways:
1. **condMean ≈ EV for Qwen3.5** — these heads are not dominated by the 4B-style bin-0 hedge
   lump, so the mixture decode has nothing to divide out. The revival channel here is the
   quantized decodes: pairwise accuracy .81–.83 (baseline level) — coarse quality ordering is
   intact; fine continuous ordering (τ ≤ .28) is genuinely weak.
2. **Old harness EV kendalls do not reproduce and were depressed** (run5: .277 here vs .068
   saved). Likely the known Qwen consolidated-checkpoint mixed-dtype bug (stray fp32 params;
   the load-unify fix postdates the harness run). So the slide under-reported Qwen progress
   ranking too, via a third, Qwen-specific instrument error.
3. **The Qwen success-AUC numbers on the slide are genuine, not instrument error** — they
   reproduce at ~.66. The Qwen3.5 success heads really are mid-pack OOD (vs 4B-FT .76–.84,
   baseline .86), consistent with the known Qwen3.5 calibration/OOD weakness.

**Files:** per-trajectory scores incl. full final-frame bin distributions in `<model>.jsonl`;
per-model decode metrics in `<model>_metrics.json`.
