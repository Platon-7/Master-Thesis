# VLM-Reward Autonomous-RL — Experiment Tracker

Single source of truth for what we've run, what's running, and what's planned.
Fair autonomous RL = the VLM reward model detects success (success_prob/progress
crosses a threshold) and terminates episodes; **no GT leak in training**.
`score/score` = true GT success in a separate plain PixelMetaWorld eval.
Last updated: 2026-06-11.

Artifacts: per-cell data `mega_results_288.csv`; verdict `MEGA_VERDICT_288.md`;
calibration `calib_multitask.json`; this tracker.

---

## Key findings so far (corrected)

1. **Only one quadrant works:** FT model + success-head detection + success-head
   reward. ~0.65–0.70 true success, ~⅓ of seeds. Everything else is dead/near-dead.
2. **Baseline (Robometer-4B) never works** anywhere (max 0.35, 0 alive).
3. **Progress head is fine as a REWARD, broken as a TERMINATOR.** detect=success +
   reward=progress reaches 0.45 (near-miss); detect=progress collapses to ~0.01
   (terminating on a 0→1 ramp ends episodes before completion — structural, not a
   threshold miss; swept {0.85,0.90,0.95}).
4. **ICL was never used in ANY of the 288 runs** (`icl_frames=None` throughout),
   despite the checkpoint being the `icl_ours` variant → train/inference asymmetry,
   untested.
5. **Offline metrics don't predict RL.** FT scores AUC ~0.88–0.97 offline but ~0.54
   whole-trajectory on fresh on-policy frames. The RL-relevant signal is the
   *streaming* causal-FPR / premature-fire, not episode-level AUC.
6. **Threshold ceiling:** success_prob rarely exceeds ~0.85 even at true success →
   at 0.90 the detector goes silent (no reward) → dead. 0.80–0.85 is forced.

---

## Table 1 — 288-run sweep (coffee-push) ✅ DONE

Full factorial × 3 seeds: model{FT,4B} × detect{success,progress} ×
reward β{0=success,1=progress} × timing{sparse,dense} × threshold × debounce.
Each cell below = 36 runs (3 thr × 2 timing × 2 debounce × 3 seed). Score = final GT.

| model | detect head | reward source | n | mean | max | #alive (≥0.5) |
|---|---|---|---|---|---|---|
| **FT** | **success** | **success** | 36 | **0.178** | **0.70** | **5** |
| FT | success | progress | 36 | 0.068 | 0.45 | 0 |
| FT | progress | success | 36 | 0.007 | 0.05 | 0 |
| FT | progress | progress | 36 | 0.013 | 0.40 | 0 |
| 4B | success | success | 36 | 0.006 | 0.10 | 0 |
| 4B | success | progress | 36 | 0.039 | 0.35 | 0 |
| 4B | progress | success | 36 | 0.013 | 0.25 | 0 |
| 4B | progress | progress | 36 | 0.028 | 0.25 | 0 |

**Within the winning quadrant (FT/success/success):** alive only at threshold
0.80–0.85 with **sparse** reward (reward_at_truncation=1); dense reward and
threshold 0.90–0.95 are dead. Debounce (1 vs 3) doesn't matter.

**The 5 alive cells:**
```
0.70  FT success/success  thr=0.85 sparse debounce=1 seed=2
0.65  FT success/success  thr=0.80 sparse debounce=3 seed=3
0.65  FT success/success  thr=0.85 sparse debounce=3 seed=3
0.55  FT success/success  thr=0.85 sparse debounce=1 seed=3
0.50  FT success/success  thr=0.80 sparse debounce=1 seed=3
```

---

## Table 2 — 54-run generalization (3 harder tasks) 🔄 RUNNING (array 413)

Recipe locked: success head, β=0 (success reward), sparse, debounce=1.
Both models. Task-tuned thresholds (from calibration). 3 seeds. EP_LEN=200.
Prediction from calibration: StickPull works, BoxClose marginal (hacking risk),
Assembly fails (no usable threshold). Fill `max GT` when done.

| model | task | thresholds | result | verdict |
|---|---|---|---|---|
| FT | StickPull | 0.70–0.90 (incl. calibrated opt 0.70) | 0.00 everywhere (12+ runs) | NEGATIVE — on-policy FP exploit at every thr (fires @~21 steps; ICL + ladder pending) |
| FT | BoxClose | 0.80/0.85/0.90 | thr0.90: 0.75/0.85 (2/4 alive); 0.85+ICL: 0.55 | **WORKS** (2nd task); ICL widens thr window |
| FT | Assembly | 0.70/0.75/0.80 | 0.00 ×7 (fires @12–17 steps) | NEGATIVE as predicted (veto case) |
| 4B | all tasks | own calibrated thrs (0.35–0.70) + FT thrs | 0.00 everywhere (15 fair runs) | baseline fails under FULL parity |

---

## Table 3 — Reward-model calibration (fresh oracle rollouts) 🔄 PARTIAL

Streaming op-point = best (causal-TPR at tolerable causal-FPR). 15s/15f per task.
Same scorer call returns BOTH heads, so each ICL setting is one pass over both heads.

### 3a. success head, NO ICL ✅ DONE
| task | ep-AUC | op-thr | causal-FPR | causal-TPR | prem-fire | usable? |
|---|---|---|---|---|---|---|
| CoffeePush | 0.536 | 0.85 | 0.27 | 0.73 | 0.00 | yes (proven in RL) |
| StickPull | 0.676 | 0.80 | 0.13 | 0.93 | 0.13 | yes (best) |
| BoxClose | 0.556 | 0.85 | 0.13 | 1.00 | 0.53 | qualified (hacking) |
| Assembly | 0.724 | — | — | — | ≥0.6 | **no usable point** |

### 3b–3d ✅ DONE (full 15s/15f, `calib_matrix_{ft,4b}_full.json`, both models)

FT model summary (op-point = best causal-TPR at causal-FPR ≤ 0.15):
| task | success no-ICL | success +ICL | progress (either ICL) |
|---|---|---|---|
| CoffeePush | thr0.85: FPR 0.27 / TPR 0.73 (leaky, proven) | WORSE (AUC 0.69→0.55) | **unusable** (fail-mean 0.54) |
| StickPull | thr0.70: FPR 0.07 / TPR 0.87 | BETTER (AUC 0.72→0.82) | **unusable** (fail-mean 0.42) |
| BoxClose | thr0.85: TPR 1.00, prem 0.60 | **prem 0.60→0.20** (key win) | **unusable** (fail-mean 0.69) |
| Assembly | no usable point | no usable point (worse) | **unusable** (fail-mean 0.33) |

Key reads: (1) progress head has NO usable detector threshold on any task — fires on
failures; confirms "fine reward, broken terminator" mechanically. (2) ICL is
task-dependent: helps BoxClose (prem-fire) + StickPull, hurts CoffeePush + Assembly.
(3) Baseline-4B lives on a LOWER threshold scale (op 0.15–0.70); its BoxClose
detection is strong (AUC 0.94, thr 0.35) → fair RL re-test = round-2 Q1.
ICL source used: real BC demo-0 PNG frames (same render pipeline as query; fmt-checked).

---

## Planned experiments / open work

- [ ] **Progress-head calibration (3b)** — confirm the premature-termination
      mechanism (predict prem-fire ≈ 1.0) and check if any high threshold (~0.97+)
      ever makes progress-as-detector viable.
- [ ] **ICL calibration (3c, 3d)** — does adding ICL at inference recover the
      on-policy signal? Directly addresses the train/inference asymmetry (Christian H2).
- [ ] **Fill Table 2** once array 413 completes; compare to calibration predictions.
- [ ] **ICL-in-RL test** — re-run the winning coffee-push recipe WITH
      `ROBOMETER_ICL_DEMO_PATH` set; does on-policy success change?
- [ ] **Progress-as-reward (0.45 near-miss)** — pull the β=1 log trace: does
      `vlm_robometer_progress` climb while GT stays flat (hacking)? (no GPU needed)
- [ ] **Ensemble / output-sampling** (Christian H1 fix) — only if Assembly fails
      systematically as predicted; inject stochasticity to break exploitable
      systematic false-positives.

## Open questions

- Why doesn't offline AUC (0.88–0.97) transfer on-policy (→0.54)? Distribution shift
  — quantify on-policy FPR vs offline.
- Is the ~0.85 success_prob ceiling fundamental? Would temperature/re-calibration
  let it reach a usable 0.90 operating point?
- Is the FT model's failure mode systematic (H1) on the tasks where it fails
  (Assembly), and does ensemble/sampling convert it to survivable noise?

---

## Queue audit — 2026-06-11 evening (decisions log)

**KEPT (answers an open question):**
- 413 FT arm (~19 tasks left): transfer of the winning recipe to 3 harder tasks at
  FT-calibrated thresholds. Tests the calibration's predictions directly.

**DUMPED (uninformative):**
- 413 4B arm (cancelled): ran the baseline at FT-scale thresholds (0.70–0.90) while
  its calibrated operating range is 0.15–0.70 → detector under-fires by construction.
  Whatever the result, it would not separate "baseline can't do RL" from "wrong threshold".

## Table 4 — Round 2 (array 501, 23 runs) 🔄 RUNNING

| # | Question | Runs | Prediction |
|---|---|---|---|
| Q1 | Fair baseline: does 4B train at ITS OWN thresholds? | 4B coffeepush thr{0.60,0.70}×3s; 4B boxclose thr0.35×3s (9) | coffeepush still fails (detector weak even when fair, cTPR≈0.33); boxclose = genuinely open (offline AUC 0.94) |
| Q2 | Does ICL in the RL loop fix premature firing / improve detection? | FT boxclose thr0.85 +ICL ×3s; FT stickpull thr0.70 +ICL ×3s (6) | boxclose improves (prem 0.60→0.20 in calib); stickpull improves (AUC 0.72→0.82) |
| Q3 | StickPull at its calibrated optimum (0.70), no-ICL control | FT stickpull thr0.70 ×3s (3) | best transfer shot (cFPR 0.07 / cTPR 0.87) |
| Q4 | Seed stability of the winning coffeepush config | FT coffeepush thr0.85 seeds 4–8 (5) | pins the hit-rate (currently 2/3 on s1–3) |

GPU budget: 413@%4 + 501@%6 = ≤10 concurrent. Ondemand only (L40S + L4 overflow).
ICL runs use the real BC demo-0 frames via ROBOMETER_ICL_DEMO_PATH (16 frames).

**Deliberately deferred:** β=0.5 reward mix (secondary lever); ensemble/output-sampling
for Assembly (needs code work — trigger if Assembly negative confirms); ICL on
coffeepush/assembly (calibration says it hurts there).

## Phase decision — 2026-06-12

**Scope locked:** make autonomous RL work WITH per-task calibration (FT) and prove the
baseline fails under the IDENTICAL protocol (own thresholds + own ICL). Zero-shot /
online threshold calibration = phase 2 (bridge already sketched: demo-anchored rule,
retro-validated — accepts CoffeePush ✓trains, vetoes Assembly ✓dead, accepts BoxClose
with ICL ✓trains; StickPull pending).

## Table 5 — Round 3 (array 524, 11 runs) 🔄

| purpose | runs |
|---|---|
| BoxClose hit-rate (2nd working task, n→8 seeds) | FT boxclose thr0.90 seeds 4–8 |
| Baseline gets the ICL privilege too (fairness loophole) | 4B boxclose thr0.15+ICL ×3; 4B coffeepush thr0.60+ICL ×3 |

Throttles: 413 %2, 501 %6, 524 %2 (≤10 GPUs). New result: stickpull FT thr0.70
(calibrated optimum, no-ICL) s3 = 0.00 — stickpull increasingly likely a true negative.
Deferred deliberately: autonomous-mode FPR logging patch (no shared-code edits while
the decisive 501 tasks are pending — triangulate via early_term × ep_len × GT instead).

## Loss/ICL ablation pipeline — 2026-06-12

Fetching from OneDrive (rclone, in progress): `run2_noicl_ours_step4000`,
`run3_noicl_standard_step5000`. 3-point decomposition (run3 alone would confound):
- run1 (icl+asym) vs run2 (noicl+asym)  → effect of ICL-TRAINING
- run2 (noicl+asym) vs run3 (noicl+standard) → effect of ASYMMETRIC LOSS
Auto-pipeline: transfer → full calibration (15s/15f, same protocol as run1) →
then RL at each run's OWN calibrated threshold (never at run1's — lesson learned).

## Table 6 — Round 4 stability levers (array 535, 13 runs) 🔄
| lever | runs | mechanism |
|---|---|---|
| confirm-vote K=4 (self-ensemble) | coffeepush thr0.85 s1–5; boxclose thr0.90 s1–5 | candidate fire needs 3/5 jittered-subsample ballots |
| output sampling (Bernoulli reward) | coffeepush thr0.85 s1–3 | terminal reward ~ Bernoulli(p); FPs stop paying deterministically |
Compare hit-rates vs the no-lever seeds (same task/thr/seeds). Veto/pass printed per fire.
## Results snapshot — regenerated from disk 2026-06-12 (40k finals; alive = >=0.50)

| task | model | thr | levers | seeds: finals | alive |
|---|---|---|---|---|---|
| assembly | 4B | 0.80 | — | s3:0.40 | 0/1 |
| assembly | FT | 0.70 | — | s1:0.00 s3:0.00 | 0/2 |
| assembly | FT | 0.75 | — | s2:0.00 s3:0.00 | 0/2 |
| assembly | FT | 0.80 | — | s1:0.00 s2:0.00 s3:0.00 | 0/3 |
| boxclose | 4B | 0.35 | — | s1:0.00 s2:0.00 s3:0.00 | 0/3 |
| boxclose | 4B | 0.90 | — | s1:0.00 s2:0.05 s3:0.05 | 0/3 |
| boxclose | FT | 0.80 | — | s1:0.35 s2:0.20 s3:0.15 | 0/3 |
| boxclose | FT | 0.85 | — | s1:0.20 s2:0.05 s3:0.15 | 0/3 |
| boxclose | FT | 0.85 | +ICL | s1:0.55 | 1/1 |
| boxclose | FT | 0.90 | — | s1:0.75 s2:0.20 s3:0.85 s5:0.25 | 2/4 |
| coffeepush | 4B | 0.60 | — | s1:0.00 s2:0.00 s3:0.00 | 0/3 |
| coffeepush | 4B | 0.70 | — | s1:0.00 s2:0.00 s3:0.00 | 0/3 |
| coffeepush | FT | 0.85 | — | s5:0.70 s6:0.80 s7:0.45 | 2/3 |
| coffeepush | run2 | 0.75 | — | s1:0.25 | 0/1 |
| stickpull | 4B | 0.80 | — | s1:0.00 | 0/1 |
| stickpull | FT | 0.70 | — | s1:0.00 s2:0.00 s3:0.00 | 0/3 |
| stickpull | FT | 0.75 | — | s1:0.00 s2:0.00 s3:0.00 | 0/3 |
| stickpull | FT | 0.80 | — | s1:0.00 s2:0.00 s3:0.00 | 0/3 |
| stickpull | FT | 0.85 | — | s1:0.00 s2:0.00 s3:0.00 | 0/3 |

(288-sweep runs not in the snapshot table — they live in `mega_results_288.csv`.
CoffeePush FT thr0.85 combined hit-rate incl. 288 seeds s1–3: 4/6 ≈ 67%.)

## Findings log — 2026-06-12 (afternoon)
1. **BoxClose = 2nd working task** (thr0.90: 0.75/0.85, 2/4 seeds). Existence result, no ICL needed.
2. **ICL widens the threshold window**: thr0.85 dead without ICL (0.05/0.05/0.20) → 0.55 with ICL (1 seed, 2 pending).
3. **CoffeePush hit-rate revised UP: ~67%** (was "~1/3" from n=3).
4. **StickPull true negative**: exploited at every threshold; on-policy FPR ≈100% (fires @~21 steps vs 53-step demos) while offline calibration said FPR 7% → offline calibration does NOT predict RL (demoted to scale-finding + veto only).
5. **Fair-baseline complete**: 4B = 0.00 in all 15 runs at its own thresholds (and ICL-parity runs queued). FT≫baseline is airtight.
6. **Retro-analysis of online guards** (381 runs): fire-rate anomaly rule (early-term>0.85 & GT≈0 by 20k) flags exploits with 1 false-positive total (99.4% precision); min-length gate (0.5×demo) blocks all fast exploits (11–28-step fires) but misses slow ones (4B@0.90 fired @150–182 steps). → layered online guard is the path to one-shot; per-step score tracing being added for quantile-bar validation.

## Qwen3.5 investigation — RESOLVED 2026-06-13

**Bug fixed:** Qwen35-FT/robometer/{configs,data,evals, trainers/__init__.py, trainers/rewind_trainer.py, utils/fsdp} were symlinks to the dead Snellius path (/gpfs/...). Repointed to local Robometer. Model now loads (run4_step6500).

**NOT a scoring-path bug.** Verified the get_robometer_4b path on Qwen3.5's OWN
in-distribution eval keyframes (robometer eval set, corner2): success-label mean
0.26 vs failure 0.067 → DISCRIMINATES. The earlier "dead success head" was MY test
error — I fed CoffeePush BC-demo PNGs (out-of-distribution rendering), not the
robometer eval frames. Lesson: run the in-distribution control FIRST.

**Real issue for IBRL = weak/uneven calibration + on-policy OOD** (same theme as the
whole study): even in-distribution, CoffeePush success_prob ≈ 0.037 (bin_picking 0.64).
So autonomous IBRL on CoffeePush is near-silent → no reward. Not a code bug.

**To use Qwen3.5 in IBRL later:** per-task threshold calibration (CoffeePush ~0.037 is
likely too weak; try stronger-signal tasks) and/or a phase-2 checkpoint. Normal experiment.
Env: vlm_ibrl_qwen35 (scoring) — has transformers 5.7, NO gymnasium; rendering needs a
gymnasium env. In-dist eval keyframes cached at /shared/home/PKA4388/eval_keyframes.

## Qwen3.5 run4 — IBRL calibration + run (2026-06-13)

**Env fix (AWS):** qwen35 IBRL needs gymnasium+metaworld+transformers5.7 in ONE env.
Added gymnasium 0.29.1 + metaworld 2.0.0 (editable, --no-deps) to `vlm_ibrl_qwen35`
(torch/transformers/numpy untouched). metaworld_repo uses the NEW `mujoco` (3.1.3,
already present) — mujoco_py is NOT in the MetaWorld import chain (only referenced in
comments/robosuite). IBRL env already supports `--vlm qwen35_ft` natively (QWEN35_FT_PATH
→ get_robometer_4b). Eval keyframes confirmed corner third-person 480x480 = same family
as IBRL corner2_default (view is NOT the OOD source).

**Calibration (run4_step6500, in-distribution eval keyframes = BEST case; job 598):**
op-point = max success-TPR at failure-FPR<=0.15.

| task | succ_mean | succ_max | fail_mean | operating point | verdict |
|---|---|---|---|---|---|
| coffee_push | 0.079 | 0.094 | 0.074 | thr 0.10 TPR 0.00 | NO-GO |
| box_close | 0.177 | 0.387 | 0.151 | thr 0.32 TPR 0.33 | NO-GO |
| stick_pull | 0.145 | 0.287 | 0.069 | thr 0.10 **TPR 0.71** FPR 0.10 | **GO** |
| assembly | 0.005 | 0.005 | 0.024 | thr 0.06 TPR 0.00 | NO-GO |

run4's success head operates on a LOW prob scale (successes 0.05-0.39) and overlaps
failures heavily — gripper-POV failures score 0.25-0.32 (false positives). Only StickPull
separates (and at a fragile thr 0.10). Notably StickPull is where Robometer-FT FAILED →
qwen35 may be complementary IF the offline GO survives on-policy.

**IBRL run:** StickPull, run4, thr 0.10, winning recipe (success/beta0/sparse/debounce1),
EP_LEN 200, 40k steps. Job 599 (seed 1 smoke); scale to 3 seeds if env runs clean.

## Morning batch results — 2026-06-13 (all COMPLETED)

### Stability levers (job 535) — confirm-K=4 ensemble vs Bernoulli reward-sample
- CoffeePush + confirm-K=4 (k4r0): 0.85,0.75,0.75,0.40,0.00 → 3/5 alive (best), peak 0.85.
- CoffeePush + Bernoulli (k0r1): 0.60,0.50,0.10 → 2/3, weaker than ensemble.
- BoxClose + confirm-K=4: 0.70,0.30,0.25,0.20,0.20 → 1/5 marginal.
Read: ensemble holds CoffeePush; Bernoulli weaker; neither rescues hard tasks.

### Min-length gate g0.8 (job 575) — STANDOUT STABILIZER
- CoffeePush: 0.65,0.75,0.70 → 3/3 alive, mean 0.70 (most robust CoffeePush yet).
- BoxClose: 0.40,0.35,0.50 → 2/3 near-alive; early-term suppressed to ~0.13.
- StickPull: 0.00x3, early-term ~0.99 → gate does NOT rescue (fires on everything post-gate).
- Assembly: 0.00x3 → fails.
Read: gate helps WORKING tasks, cannot fix the broken ones.

### Ablation — loss vs data (551 run2=asym loss, 552 run3=standard loss)
- run2 (asym): CoffeePush 0.70,0.50,0.25 → 2/3 ; BoxClose maxes 0.45/0.60/0.30 weak.
- run3 (standard): CoffeePush 0.40,0.00,0.00 → 0/3 ; BoxClose(dsucc0.20) 0.50,0.10,0.50 (max .60/.70) → 2/3.
VERDICT: asymmetric loss is NOT the cause — run3 (no asym) still trains BoxClose 2/3.
It's the data/fine-tuning; each checkpoint works on a different task.

### Assembly ceiling (job 573) — true veto confirmed
- thr 0.85 → GT 0, early-term 0.93 (poisons). thr 0.90 → GT 0, early-term 0.00 (silent).
- No usable threshold exists for Assembly.

## Gate + Ensemble combined (job 603) — RUNNING 2026-06-13
Stack both FP-suppressors: min-length gate g0.8 + confirm-K=4, winning recipe, 3 seeds.
- StickPull thr0.70 (rescue target — neither lever fixed alone)
- CoffeePush thr0.85 (control — does stacking preserve gate's 3/3?)
Prediction: helps working tasks; likely does NOT rescue StickPull (its FP is a PERSISTENT
high success_prob → passes K-ballot vote; gate+ensemble kill transient/early FPs, not
sustained Goodhart). Fill results when done.

## Results — qwen35 run4 IBRL + gate×ensemble combo (2026-06-13, all DONE)

### qwen35 run4 → StickPull (600/601), 40k, thr 0.10 — DEAD (0/3)
s1 GT0.00 ET0.995 (fires on everything/FP) ; s2 GT0.00 ET0.00 (silent) ; s3 GT0.00 ET0.00 (silent).
Offline GO (in-dist thr0.10 TPR0.71) did NOT transfer on-policy → silent or false-firing.
run4 unusable for autonomous IBRL as-is. Reinforces: validate calibration on ON-POLICY frames.

### CoffeePush + gate g0.8 + confirm-K=4 (603_3,4,5) — reached only 35k (4x scoring cost hit 5h walltime)
GT@35k: 0.70 / 0.00 / 0.50 → 2/3 alive, ONE DEAD SEED.
vs gate-ALONE (575): 0.65/0.75/0.70 → 3/3, NO dead seed.
=> For the no-dead-seed stabilization goal, GATE-ALONE > gate+ensemble. Ensemble added cost
+ a collapse; not the stabilizer. (StickPull combo arm + BoxClose combo were cancelled/NODE_FAIL,
not relaunched.)

## Gate-alone stabilization sweep (job 612) — RUNNING 2026-06-13
GOAL: certify "no dead seed" (every seed GT>0) for the WORKING tasks under the leading
stabilizer (gate g0.8, NO ensemble). CoffeePush + BoxClose × 6 seeds × 3 FT checkpoints.
Per-checkpoint calibrated thresholds:
  run1_icl_ours_step4000:       coffee 0.85 / box 0.90
  run2_noicl_ours_step4000:     coffee 0.75 / box 0.85
  run3_noicl_standard_step5000: coffee 0.25 / box 0.20
36 runs, --array=0-35%10 (<=10 GPU). 40k steps. Fill no-dead-seed verdict per (ckpt,task).

NOTE: job 612 cancelled — 12 tasks NODE_FAILed on multi-GPU (g6e-4gpu/8gpu) AWS nodes.
Resubmitted as job 647 restricted to 1-GPU partitions only (g6e-1gpu-l40s, g6-1gpu-l4) —
clean spin-up, no failures. Same 36-run config.

## Gate-alone stabilization (job 647) — RESULTS (all 36 COMPLETED, no node fails)
DEAD = final GT == 0.00. Gate g0.8, no ensemble, 6 seeds, per-ckpt thresholds.

| ckpt | CoffeePush dead/6 (finals) | BoxClose dead/6 (finals) | stable both? |
|---|---|---|---|
| run2 noicl_ours   | 0/6 (.50/.90/.50/.60/.55/.65) | 0/6 (.95/.90/.80/.80/.95/.45) | YES (winner) |
| run1 icl_ours     | 2/6 (.50/0/.55/.05/.45/0)     | 0/6 (.70/.50/.50/.90/.35/.55) | no (coffee) |
| run3 noicl_std    | 5/6 (0/0/0/0/.60/0)           | 0/6 (.40/.55/.40/.20/.35/.15) | no (coffee) |

VERDICT: run2 (no-ICL, asymmetric "ours" loss) is the most stabilizable — 0 dead seeds
on BOTH working tasks under the gate. BoxClose robust under ALL ckpts. CoffeePush only
fully stable under run2 (run1 drops 2 seeds, run3 nearly dead). => For no-dead-seed
stabilization, gate g0.8 + run2 is the recipe. (run1 = main ICL ckpt is NOT the best here.)

## REFINED loss-vs-data verdict (from gated 6-seed sweep 647) — 2026-06-13
Earlier 3-seed ablation (551/552, gate-off) concluded "it's the data, not the loss"
(each ckpt worked on a different task). The better-powered 647 sweep (gate g0.8,
per-ckpt calibrated thr, 6 seeds) overturns the *strength* of that claim:

run2 vs run3 = controlled LOSS comparison (both noicl; ours/asym vs standard):
  CoffeePush: run2 0/6 dead vs run3 5/6 dead   BoxClose: run2 .80-.95 vs run3 .15-.55
=> run2 (asymmetric loss) >= run3 (standard loss) on BOTH tasks, large CoffeePush gap.
=> The asymmetric loss HELPS (broader, more robust stability). NOT only the data.

Reconciled: standard-loss data alone yields A working model (run3 stabilizes BoxClose),
so asym loss isn't strictly REQUIRED — but it is what makes stability generalize across
tasks. Both matter. Caveat: run2=step4000 vs run3=step5000 (duration confound; run3
trained longer yet worse, which strengthens the read), single ckpt per condition.
Airtight test would be ours-vs-standard at matched step + multiple FT seeds.

## run1+ICL (686) and gate+ICL wake test (693) — RESULTS 2026-06-14 (all COMPLETED)

### 686: run1 +ICL vs run1 no-ICL (647), gate g0.8, 6 seeds
| task | run1 +ICL (dead/6) | run1 no-ICL (dead/6) |
|---|---|---|
| CoffeePush | 2/6 (0/.75/.55/.20/0/.50) | 2/6 (.50/0/.55/.05/.45/0) |
| BoxClose   | 1/6 weak (.05/.25/.15/0/.45/.10) | 0/6 strong (.70/.50/.50/.90/.35/.55) |
VERDICT: matched ICL does NOT rescue run1 — CoffeePush unchanged, BoxClose HURT
(0 dead/strong -> 1 dead/collapsed). The missing-ICL concern was valid methodology but
is NOT the cause of run1's weakness. ICL helped BoxClose OFFLINE (prem 0.60->0.20) yet
hurt it in RL = another offline!=on-policy. => run2 (noicl, asym loss) stays the winner.

### 693: gate g0.8 + ICL on StickPull / Assembly (run1, 3 seeds) — WAKE TEST
- StickPull: 0.00 x3, early-term ~0.97-1.0 (persistent FP; NOT woken).
- Assembly: 0.05/0.05/0.00 (noise, NOT woken). BUT early-term dropped 0.93 -> ~0.5-0.75
  => gate+ICL DOES suppress Assembly FPs, just not enough for success.
VERDICT: neither dead task revived. StickPull/Assembly are not fixable by stabilization
levers (gate, ensemble, ICL, all combos) — they need a better reward model/calibration.

## Task-description hypothesis (job 723) — REJECTED 2026-06-14
Re-ranked StickPull+Assembly eval clips with run1 success head under 4 descriptions
(training prompt + rich/goal/simple variants). Separation = succ_mean - fail_mean:
- StickPull: D0(training) BEST sep=+0.600, thr0.38 TPR1.00 FPR0.10; variants WORSE (-0.03..-0.07).
- Assembly:  D0 sep=+0.494 TPR1.00 FPR0.11; D3 marginally +0.021 (noise). 
=> Wording is NOT the bottleneck — current prompt already optimal/near-optimal.

KEY: success head separates StickPull/Assembly CLEANLY on eval clips (TPR 1.00 @ FPR ~0.10)
yet both DIE in RL (GT 0.00, early-term ~1.0). Distribution-shift gradient:
eval clips (perfect) -> oracle rollouts (Table 3: AUC 0.68) -> on-policy learner frames (broken).
StickPull/Assembly are an ON-POLICY ROBUSTNESS problem, not prompt or threshold.
Fix direction: on-policy/adversarial negatives in RM training, or a detector robust to OOD
on-policy frames (gate/ensemble already shown insufficient for StickPull's persistent FP).
