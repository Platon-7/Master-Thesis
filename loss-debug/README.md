# Loss debug — asymmetric (ours) vs paper-standard

Quick reference for the talk: what the asymmetric loss is actually doing on
the in-distribution eval set, across both base models, at every saved
checkpoint of the no-ICL runs.

## Setup

Comparison is a clean 2×2:

| | asymmetric loss (ours, λ=0.3) | paper-standard loss |
|---|---|---|
| **Robometer-4B base** | run 2 (`run2_noicl_ours`) | run 3 (`run3_noicl_standard`) |
| **Qwen3.5-VL-4B base** | run 5 (`run5_noicl_ours`, wandb `w0otbkig`) | run 6 (`run6_noicl_standard`, wandb `u9u7seky`) |

ICL is off for all four — comparing loss recipes only, no confound from ICL.

All numbers come from the `policy_ranking_samples/step_*/<source>.json` dumps
saved at training time, on the 4-source held-out eval set
(`robometer_frames_eval_{droid,robometer,metaworld,failsafe}`).

Labels are **trajectory-level**: predictions for successful trajectories =
positives; predictions for failure trajectories = negatives.

## Headline numbers (pooled across all 4 eval sources)

Measured at each base's **last saved checkpoint** in the no-ICL runs:
Robometer-FT runs 2/3 at **step 5000**; Qwen3.5-FT runs 5/6 at **step 6500**.
(Both bases converged by then on training loss; per-step trajectories of all
metrics are in `fig3_ece_and_separation.png`.)

| Base | Loss | Checkpoint | pos_mean | neg_mean | Separation | AUC | ECE |
|---|---|---|---|---|---|---|---|
| Robometer-4B | **asymmetric (ours)** | step 5000 | 0.250 | 0.132 | +0.118 | **0.841** | **0.301** |
| Robometer-4B | paper-standard | step 5000 | 0.488 | 0.315 | **+0.173** | 0.791 | **0.113** |
| Qwen3.5-VL-4B | **asymmetric (ours)** | step 6500 | 0.284 | 0.171 | +0.113 | 0.812 | **0.266** |
| Qwen3.5-VL-4B | paper-standard | step 6500 | 0.514 | 0.307 | **+0.207** | **0.825** | **0.142** |

Take-aways:
- **Both losses learn to rank.** AUC is comparable (0.79–0.84) for both recipes on both bases.
- **Standard loss separates better.** Absolute pos–neg gap is ~1.5× wider under standard loss on both bases.
- **Standard loss is also better calibrated.** ECE is ~half on trajectory labels (0.113 vs 0.301 on Robometer-4B; 0.142 vs 0.266 on Qwen3.5).
- **Asymmetric loss compresses BOTH classes toward zero.** It looks like discrimination on the LoRA "FPR @ τ=0.5" plot because the negative tail crosses below 0.5, but the positive tail also crashes below the threshold.

## Figures

- **`fig1_per_source.{pdf,png}`** — 2×4 grid (2 base models × 4 eval sources). The pos/neg gap is the shaded band; asymmetric loss is red, standard is blue.
- **`fig2_aggregated.{pdf,png}`** — same comparison, pooled across all 4 eval sources. Cleaner version for the slide.
- **`fig3_ece_and_separation.{pdf,png}`** — ECE, AUC, and pos–neg separation tracked vs training step. Shows that the LoRA "ECE dropping with asymmetric loss" story does NOT hold on trajectory labels — ECE is higher (worse) for asymmetric here.

## Why the LoRA metrics were misleading

The LoRA per-frame curves (`Robometer-LoRA/results/step_curves_run8_progress_head.png`) showed:
- FPR @ τ=0.5 → 0 over training (looks great)
- ECE → 0.02 over training (looks great)

Both were artifacts of the same underlying compression:

| Metric | Why it dropped despite no real improvement |
|---|---|
| **FPR @ τ=0.5** | Asymmetric loss compresses *both* the pos and neg prediction distributions toward 0 (on our trajectory labels: pos_mean ≈ 0.25, neg_mean ≈ 0.13 — both well below 0.5). At τ=0.5, almost nothing crosses the threshold, so FPR ≈ 0 *and* TPR ≈ 0. The rate that drops is "fraction of any prediction above 0.5", not "fraction of false alarms among model-fired alarms." |
| **ECE** | The LoRA curve used **per-frame at-goal labels**. On per-frame data, only ~5% of frames are at-goal (the trajectory ends in success only briefly); the other ~95% are label=0. A model whose predictions are pinned to low values is approximately calibrated on that dominant 95% negative mass, so ECE ≈ 0 even though it's miscalibrated on the 5% positives. Switch to **trajectory-level labels** (~50% positives / ~50% negatives by construction) and the same compressed-low predictions are systematically wrong on the entire positive class, so ECE rises (0.27–0.30 vs the LoRA plot's 0.02–0.05). The "per-frame ECE → 0.05" is base-rate matching, not genuine improvement in calibration. |

## What still survives from the asymmetric loss story

- AUC is comparable to paper-standard, so rank discrimination is preserved.
- On the FT eval set the absolute predictions are pinned low. This *might* matter for RL when you threshold (lower FPR at a fixed τ) — but doesn't help RL when the reward consumes continuous values, because the differential between pos and neg is also smaller.
- Whether it helps downstream RL is the IBRL-side question, answered in the next section.

## What "ECE" actually means in our setting

Expected Calibration Error has the standard form

$$\mathrm{ECE} = \sum_{b=1}^{B}\frac{|B_b|}{N}\left|\overline{y}(B_b) - \overline{p}(B_b)\right|$$

where samples are bucketed into $B$ equal-width bins on $[0,1]$ by predicted
probability $p$, $|B_b|$ is the count in bin $b$ (total $N$),
$\overline{p}(B_b)$ is the mean prediction in the bin (confidence), and
$\overline{y}(B_b)$ is the mean label in the bin (empirical frequency).
We use $B = 10$ throughout.

The formula is the same; what changes between the two ECEs reported in this
project is the **choice of label paired with the prediction**.

Let $p^{(i)}_t$ be the model's `success_prob` output at frame $t$ of
trajectory $i$, and let $T_i$ be the length of trajectory $i$.

### At-goal ECE — LoRA "sparse-RL view"

- **Samples**: every $(i, t)$ pair — one per frame, $N = \sum_i T_i$.
- **Prediction**: $p^{(i)}_t$.
- **Label**: $y^{(i)}_t = 1$ iff the gripper is *at the goal state in frame
  $t$ of trajectory $i$*, else $0$.
- **Base rate**: ~5% (most frames are not at goal).
- **Interpretation**: treats `success_prob` as *"is the gripper at the goal
  right now?"* — a per-frame progress signal.

### Trajectory-outcome ECE — LoRA "dense-RL view", our "dense ECE"

- **Samples**: every $(i, t)$ pair — same $N$ as above.
- **Prediction**: $p^{(i)}_t$ — identical to above.
- **Label**: $y^{(i)} = 1$ iff *trajectory $i$ succeeded overall*, else $0$.
  The same label is reused for every frame within trajectory $i$.
- **Base rate**: ~50% (eval sets are balanced by construction).
- **Interpretation**: treats `success_prob` as *"will this trajectory
  eventually succeed?"* — a value-function-style signal.

### Why these can move in opposite directions

The predictions $p^{(i)}_t$ are the *same numbers* in both metrics; only the
label changes. If a loss recipe (asymmetric, in our case) compresses
predictions toward 0:
- on **at-goal ECE**, the ~95% of frames whose label is 0 are now correctly
  predicted as ≈ 0, so the metric *improves*;
- on **trajectory-outcome ECE**, the ~50% of frames whose label is 1 (those
  belonging to successful trajectories) are now predicted as ≈ 0, so the
  metric *worsens*.

That is the entire mechanism behind the LoRA paper's "ECE → 0.05" finding.
It is base-rate matching on a per-frame indicator, not improved calibration
on the trajectory-outcome quantity the IBRL reward actually consumes.

### Which one matters for IBRL — and the label-inversion warning

The Robometer model has **two heads** trained with different targets:

- **Success head** — sigmoid binary, trained against trajectory outcome.
  Natural calibration metric: **trajectory-outcome ECE**.
- **Progress head** — C51 or scalar regression, trained against per-frame
  progress / at-goal labels. Natural calibration metric: **at-goal ECE**.

Each head pairs naturally with one IBRL truncation mode:

| IBRL setting | Reward fires | Natural head | Calibration metric | LoRA paper's name |
|---|---|---|---|---|
| `reward_truncation=1` (sparse — one reward at episode end) | end of episode | success head | trajectory-outcome ECE | **"dense-RL view"** |
| `reward_truncation=0` (dense — reward every step) | every step | progress head | at-goal ECE | **"sparse-RL view"** |

**The labels invert across the two terminologies.** "Sparse" in IBRL means
"sparse in episode count" (one reward per episode at truncation). "Sparse"
in the LoRA plot means "sparse in time" (the at-goal indicator fires only in
the brief at-goal window). The two senses of "sparse" point to opposite
metrics:

- IBRL `truncation=1` (sparse) ↔ trajectory-outcome ECE ↔ LoRA's *dense*-RL view.
- IBRL `truncation=0` (dense)  ↔ at-goal ECE             ↔ LoRA's *sparse*-RL view.

If you internalize one mapping: **trajectory-outcome ECE is the metric for
the standard Demo2Reward IBRL setup** (`reward_truncation=1` with the
success head). At-goal ECE only matters if you switch to dense per-step
reward driven by the progress head — Robometer Appendix E-2 territory,
which we have not run.

The IBRL reward in the codebase is actually a *mix*,
$r = \beta \cdot \text{progress} + (1-\beta) \cdot \text{success\_prob}$,
so both heads contribute via the $\beta$ knob; but each head is calibrated
against a different label and the table above tells you which calibration
matters when one head dominates the mix.

## Dense ECE across recipes (continuous-reward suitability)

Dense ECE = per-frame `success_prob` calibrated to trajectory-level outcome
(positives = every frame in a successful trajectory; negatives = every frame
in a failure). ~50% base rate by construction. This is the calibration that
matters when IBRL uses `success_prob` as the reward — sparse
(`reward_truncation=1`, success_prob at the last frame against trajectory
outcome) or dense as a value-function (every frame against the same
trajectory outcome). It is *not* the at-goal calibration the LoRA paper
celebrated; that one is a progress-tracking metric and isn't what
`reward_truncation=1` IBRL consumes.

| Recipe | Eval set | Dense ECE |
|---|---|---|
| **Full FT + paper-standard** (Robometer-FT run3, step 5000) | robometer_frames_eval_* (4 sources) | **0.11** |
| Full FT + asymmetric (Robometer-FT run2, step 5000) | same | 0.30 |
| **Full FT + paper-standard** (Qwen3.5 run6, step 6500) | same | **0.14** |
| Full FT + asymmetric (Qwen3.5 run5, step 6500) | same | 0.27 |
| LoRA + asymmetric (run5 λ=0.3, step 7500) | robometer_frames_test_v3 | 0.43 |
| LoRA + paper-standard CORN (run2 c=0.05, step 7500) | same | 0.45 |
| LoRA + asymmetric + KL (run8 λ=0.3 + KL, step 7500) | same | 0.44 |

**Reads:**
- **Within a full-FT comparison at matched eval data, asymmetric ≈ 2.5× worse dense ECE than paper-standard.** Same pattern on both bases (Robometer-4B and Qwen3.5-VL-4B).
- **LoRA training erodes dense calibration regardless of loss recipe.** All three LoRA variants collapse to dense ECE ≈ 0.43–0.45 with `sp_mean` ≈ 0.001–0.020. The compression is a LoRA-training-dynamics issue, not a loss-recipe issue.
- LoRA and Full-FT rows are on different eval sets (test_v3 vs the 4-source eval); the two blocks can't be directly compared, only ranked within each block.

The continuous-reward implication: **paper-standard full FT is the only recipe we have that produces well-calibrated dense probabilities.** If we're going to try a continuous (β·progress + (1-β)·success) IBRL reward, that checkpoint is the candidate to use — not asymmetric, not LoRA.

## CoffeePush IBRL — what we got on the harness

Single-seed runs on the vlm_ibrl Demo2Reward harness (MetaWorld CoffeePush,
sparse `reward_at_truncation=1`, β·progress + (1-β)·success_prob mixing,
then optional binarize at τ). All numbers are from `train.log` in
`/projects/prjs1958/$USER/vlm_ibrl_sweep/`. **Peak train_score** = best
50-episode rolling success rate at any logged 5k-step interval.

| Family | Model / Config | β | τ | Env steps | Peak ts | nsuc at end | sp_max (last log) | Notes |
|---|---|---|---|---|---|---|---|---|
| **Robometer-4B baseline** | 22986276 | 0.5 | 0.6 | 60k | **0.14** ⭐ | 43 | 0.87 | Best 4B result on disk; cut at 60k. |
| Robometer-4B baseline | 23025602 | 0.0 | 0.434 (offline-FPR=5%) | 55k | 0.12 | 28 | NaN | VLM inference broken (cold-GPU bf16), reward ≈ 0 → BC-only |
| Robometer-4B baseline | 23028079 | 0.5 | 0.6 | 45k (cancelled) | 0.06 | 18 | 0.83 | Seed=2 repro; trailing seed=1 by ~50% on nsuc |
| **Robometer-FT (run1 = ICL + asymmetric)** | 23024658 Config A: step-3000 + ICL | 0.0 | 0.0192 (offline-FPR=5%) | 60k | **0.12** | 24 | 0.26 | Offline-calibrated τ; real VLM signal (sp ≈ 0.22) |
| Robometer-FT | 23024660 Config C: step-3000 + ICL | 0.5 | 0.05 | 60k | 0.06 | 23 | 0.05 | |
| Robometer-FT | 22907480 step-3000, no ICL | 0.0 | 0.0 | 300k | 0.12 | 63 | 0.82 | β/τ both zero — raw success_prob as reward |
| Robometer-FT | 22907485 step-3000, no ICL | 0.5 | 0.6 | 300k | 0.08 | 68 | 0.68 | Long run; final nsuc highest in sweep |
| Robometer-FT | 22907486 step-3000, no ICL | 1.0 | 0.0 | 300k | 0.08 | 59 | 0.79 | Progress-only reward |
| **Robometer-LoRA (run5 = asymmetric λ=0.3 + ICL, merged safetensors)** | 23028892 s7500, no ICL | 0.5 | 0.6 | 45k (cancelled) | **0.10** | 28 | 0.06 | sp_max << τ → reward ≈ 0 → BC-only; still climbed because BC alone gets here |
| Robometer-LoRA | 23029103 s7500, +ICL | 0.5 | 0.6 | 20k (cancelled) | 0.08 | 17 | 0.12 | Early; ICL marginally lifts sp_max but still well below τ |
| Robometer-LoRA | 23028891 s4500, no ICL | 0.5 | 0.6 | 45k (cancelled) | 0.04 | 16 | 0.07 | |
| Robometer-LoRA | 23029102 s4500, +ICL | 0.5 | 0.6 | 20k (cancelled) | 0.04 | 15 | 0.29 | |
| **Qwen3.5-FT (run4)** | — | — | — | — | — | — | — | No IBRL run completed in this study (env-build issues + cold-GPU NaN). Task #17 pending. |

Reference: **Demo2Reward published peak ≈ 0.80** on CoffeePush. Best result we have is **0.14** (4B baseline, β=0.5/τ=0.6). 6× short of the published number.

### What this table says
- **Peak train_score is bounded at ~0.12–0.14 across every model we tried.** 4B baseline, Robometer-FT+ICL with calibrated τ, LoRA s7500 — all land in the same band.
- **The 4B-with-NaN run (23025602) hit 0.12 with reward ≡ 0** (VLM was silently producing NaN/Inf, so `NaN > τ` is False every step → no reward fires). LoRA s7500 no-ICL hit 0.10 with sp_max=0.06 vs τ=0.6 (also reward ≡ 0). Both runs land in the same band as the configs where the VLM *was* firing. This makes it hard to argue the VLM is the source of any learning at this scale.
- **Calibrated offline TPR does not predict IBRL win.** FT step-3000 + ICL gives the best offline operating point we measured (AUC 0.85, TPR 0.59 @ FPR=5%) but its IBRL peak is the same 0.12 as the 4B baseline (1.7× worse offline).

### What's missing to close the gap to 0.80
Two single-run controls would convert "VLM isn't helping IBRL" from inference to measurement:
1. **Pure-BC IBRL** (no VLM reward at all) — measures the BC-bootstrap ceiling. If BC alone reaches 0.12, the VLM reward is currently invisible to RL.
2. **Demo2Reward checkpoint in our harness** — measures the achievable ceiling on this exact setup. If they reach 0.80 here, our reward model is the limiter; if they also flatten near 0.14, the harness is the limiter.

Until those two anchors land we cannot defensibly claim "VLMs can do this" or "VLMs cannot do this."

## Files

- `loss_2x2_ablation.py` — single script that produces everything in this folder
- `loss_2x2_ablation.csv` — raw per-(run, source, step) numbers
- `fig1_per_source.{pdf,png}`, `fig2_aggregated.{pdf,png}`, `fig3_ece_and_separation.{pdf,png}`

Re-run with: `python loss_2x2_ablation.py`
