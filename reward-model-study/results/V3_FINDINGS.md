# MetaWorld v3 port + on-policy reward-collapse root cause (2026-06-05)

## TL;DR
Built a full **MetaWorld v3** copy of the IBRL harness to test whether the
on-policy reward collapse on CoffeePush is a v2-vs-v3 rendering-domain mismatch.
It is **not**. Matching the engine, camera, resolution and codec leaves the
on-policy AUC at ~0.54. The real cause is **trajectory structure**: the Robometer
reward model recognizes v3 CoffeePush successes (success_prob ~0.76) only when
the trajectory ends shortly after success (the curated recipe: run to success +8
steps). A run-on trajectory (fixed horizon, long post-success tail) collapses
success_prob to ~0.30 — reproducing the IBRL on-policy failure.

## What was built
- **conda env** `/gpfs/scratch1/shared/pkarageorgis1/envs/demo2reward_v3`:
  clone of `demo2reward` + gymnasium 0.29.1 + mujoco 3.1.1 + Farama MetaWorld v3
  (pin 58e32b4d, `MetaWorld/metaworld_repo`); mujoco-py removed. transformers
  stays 4.57.2 so one env both renders v3 AND scores the Qwen3-VL Robometer-FT.
- **folder** `vlm_ibrl_v3/`: copy of `vlm_ibrl/` with `env/metaworld_wrapper.py`
  ported v2->v3 (MT1 + gymnasium 5-tuple->4-tuple + `SawyerCoffeePushV3Policy`).
  Render path: `env.mujoco_renderer.render("rgb_array", camera_name=cam)[::-1]`.

## Evidence (FT run1 s3000, task="Push a mug under a coffee machine.", no ICL)
| measurement | AUC | succ_mean | fail_mean |
|---|---|---|---|
| v3 curated eval frames (control, scored in v3 env) | **0.985** | 0.772 | 0.128 |
| v3 live, in-domain corner2 (diag_v3_cameras) | 0.541 | 0.327 | 0.329 |
| v2 live GT-policy (prior) | 0.53 | 0.25 | — |

Fresh CoffeePush v3 **successes**, by generation method (regen_v3_pipeline):
| method | succ_mean |
|---|---|
| A curated pipeline, MT1 frozen tasks (run to success +8, linspace-16) | **0.762** |
| B curated pipeline, **random goals** | 0.761 |
| C fixed-175-step rollout, random goals (= diag_v3_cameras) | **0.305** |

A≈B≈curated (0.76–0.77); C collapses to 0.30. A vs C differ only in trajectory
length / truncation, so **truncation-at-success is the lever**. Goal distribution
(A vs B) is irrelevant.

## Ruled out (do not re-chase)
- Render engine (v2 vs v3): in-domain v3 corner2 still 0.54.
- Camera/zoom: corner2 confirmed in-domain by pixel match to curated
  (render_match_v3: corner2+flip MSE 415 vs >3800 any other camera).
- Resolution / h264 codec: v2 rescore AUCs all ~0.5 (sp_mp4 0.459, sp_cur 0.555).
- Scorer / env correctness: curated control reproduces 0.985 in the v3 env.
- Goal distribution: random vs frozen MT1 tasks → identical succ_mean.

## Mechanism / implication for IBRL
The reward model's success head expects the curated trajectory shape (monotonic
approach→progress→success, ending at success). On-policy IBRL queries the reward
on the *whole growing episode buffer* (`vlm_envs.py`: "Robometer consumes the
full video"); with `reward_at_truncation` the buffer is the full horizon. For a
successful episode that buffer is run-on past success → success_prob collapses →
the reward fails to mark the success → the policy can't learn it (CoffeePush caps
at ~0.1). The reward model is fine; it is being **queried with out-of-distribution
trajectory structure** on-policy.

## Recommended next step (the fix)
Query the reward on a trajectory windowed to match training structure — e.g.
score the buffer ending at the success/most-recent state, a fixed sliding window
of recent frames, or truncate-after-success — and re-measure on-policy AUC, then
re-run a short IBRL. Variant A already shows the reward recovers to ~0.76 under
the right structure; the open question is wiring that windowing into the IBRL
reward query (`vlm_ibrl*/env/vlm_envs.py::vlm_reward`).

## Repro
- env scoring control: `reward-model-study/jobs/score_cur_cp.job`
- render-domain camera match: `reward-model-study/jobs/render_match_v3.job`
- in-domain on-policy AUC: `reward-model-study/jobs/diag_v3cam.job`
- root-cause A/B/C: `reward-model-study/jobs/regen_v3.job`

---

## ON-POLICY RESULT — the fix works (2026-06-06)

The cheap fix (no BC retrain): render two cameras so policy AND reward are both
in-domain in v3, and end episodes at success:
  - V3_CORNER2_ZOOM=1            -> policy rl_camera="corner2" is zoomed (v2 BC in-domain)
  - ROBOMETER_REWARD_CAMERA=corner2_default -> reward sees default corner2 (its domain)
  - TRAIN_END_ON_SUCCESS=1       -> reward query sees the success-ending trajectory shape
(jobs ibrl_sweep_robometer.job / ibrl_gt_control.job now take these as env vars.)

Why each is needed (drop one -> back to the ~0.10 ceiling):
  - v2 render -> reward blind to successes; v3 alone (run-on traj) still AUC 0.54
  - no end_on_success -> reward collapses on run-on trajectories (0.30 vs 0.76)
  - no zoom -> v2-trained BC policy OOD on v3 default corner2 (BC succ 0.00 vs 0.40)

### Reward configuration (exact — which head, how mixed)
The reward is `mixed = beta*progress + (1-beta)*success_prob`, then
`reward = mixed` if `threshold<=0` else `1.0 if mixed>threshold else 0.0`
(`env/vlm_envs.py::vlm_reward`). The runs used **beta=0.0, threshold=0.0,
reward_at_truncation=1**, i.e.:
  - **SUCCESS head only, full weight** (beta=0 -> 1-beta=1 on success_prob); the
    C51 PROGRESS head was NOT used.
  - **continuous** reward = the success_prob probability itself (threshold=0 -> no
    binarization). The "graded / partial-credit" density vs the GT control comes
    from this continuous probability (a partial-progress failure scores ~0.2-0.4,
    not a hard 0), NOT from the progress head.
  - given **once at truncation** (reward_at_truncation=1); with end_on_success the
    truncation point is the success frame.
Untested on-policy: thresholded/binary success (threshold>0, ~the GT-control
density) and the progress head (beta>0).

### CoffeePush, seed 1, 60k steps, eval = true env success
| reward            | late mean (35-60k) | peak | 60k wall |
|-------------------|--------------------|------|----------|
| GT oracle (floor) | 0.43               | 0.70 | 0:34:45 (no VLM) |
| Robometer-FT s3000| 0.83               | 0.85 | 1:33:50 |
| Robometer-4B base | 0.77               | 0.90 | 1:18:39 |
Before fix (default corner2, no end_on_success): GT and FT both FLAT at 0.00-0.10.

Read: BOTH learned VLM rewards drive IBRL to ~0.8 success (vs old ~0.10 ceiling),
and both beat the sparse GT oracle's late eval. FT vs baseline is a TIE at one
seed (0.83 vs 0.77, within ~0.1 noise) — NOT yet a win. Multi-seed needed to
claim FT > off-the-shelf (note FT LOSES to baseline offline, so on-policy parity
is already a shift). Next: multi-seed (seeds 2,3 x FT,baseline) + other tasks.

### Runtime note (for sizing jobs on the next cluster)
With reward_at_truncation=1 the 4B reward is queried ~once per episode. A DEAD
policy = ~167s / 5k-block, flat -> 60k in ~33 min. An ALIVE policy slows to
~580s/block as success rises -> 60k in ~90 min. So a VLM run finishing in ~30 min
likely stayed dead; ~90 min means it learned. GT (no VLM) is ~35 min regardless.

---

## CRITICAL CAVEAT on the on-policy result — success signal is GROUND TRUTH, not the model (2026-06-06)

The seed-1 on-policy runs (FT 0.83, baseline 0.77, GT-floor 0.43) used the **simulator's
ground-truth success** for detection / termination / reward-gating — NOT the reward model.
The model only supplied the reward *magnitude*. So these runs do NOT yet validate the
paper's autonomous-RL protocol (Robometer App. E-2: terminate when the model's
success_prob > 0.6). Mechanism, in `vlm_ibrl_v3/env/vlm_envs.py`:

- L167 `reward_at_truncation = kwargs.pop(...)` — popped by VLMCritic, NOT forwarded to the
  base env -> base `PixelMetaWorld` reward = `SparseRewardWrapper` = `float(info["success"])`
  = GT success flag.
- L372 `truncated = info["truncated"] or (reward == 1)` — `reward==1` is GT success; this is
  what gates when the model reward is computed (reward_at_truncation).
- L409-414 `vlm_terminal` is True only if `vlm_reward == 1`. With `robometer_threshold=0.0`
  the reward is the *continuous* success_prob (never exactly 1) -> that branch never fires ->
  the base's GT-success terminal is overridden -> **episodes ran to the time limit** (matches
  the flat `other/episode`=50/block). I.e. `end_on_success=1` did NOT actually truncate at
  success; GT success only gated the reward value.
- L418 returns the base `success` (GT) -> `score/train_score` and eval are GT success.

Net: the model differentiated success (~0.76) from failure (~0.13) as a *value* signal, but
*when* reward was given and *whether* an episode counted as success came from GT. There is GT
signal in the loop. The honest claim is "FT/baseline are useful reward VALUES under GT-gated
episodes," NOT "the model autonomously detects success and drives RL."

### Also: don't read "VLM > GT oracle" as super-oracle
The GT control reward is SPARSE BINARY (`float(info["success"])` at truncation); the VLM
reward is GRADED CONTINUOUS (success_prob, beta=0, threshold=0). Graded != better-than-truth —
it's a denser optimization target. FT (0.83) ~= baseline (0.77) at one seed (within ~0.1 eval
noise); both above the sparse-GT late mean (0.43) partly due to reward density + GT's late
single-seed instability (GT peaked 0.70). For a fair oracle ceiling, run a GT control with a
DENSE reward (reward_at_truncation=0 + shaped/per-step reward).

## NEXT EXPERIMENTS (new cluster)
1. **Paper-faithful autonomous RL** — set `robometer_threshold>0` so the MODEL's success_prob
   crossing the threshold detects success AND terminates (vlm_reward becomes binary -> L411
   `vlm_reward==1` fires -> model-terminated). Also change L372's GT gate (`reward==1`) to key
   off `vlm_reward` so GT no longer leaks, and add the ~240-step timeout. This is the run that
   actually tests "model enables autonomous RL."
2. **Calibrate the threshold PER MODEL — do NOT reuse 0.6.** 0.6 is the paper's value for their
   model; FT (success~0.76/fail~0.13) and the baseline live on different scales. Calibrate for
   a TARGET LOW FALSE-POSITIVE RATE (~2-5%), not the balanced midpoint: a false positive ->
   the policy reward-hacks a model blind spot (collapse risk); a false negative -> episode just
   times out (benign/slower). Measure PER-STEP / causally (does success_prob cross the
   threshold at a non-success frame), not just per-episode — the scorer's `detailed=True`
   returns per-frame success_probs.
   Cheap pipeline (~tens of min, << one RL run): roll out the GT-reward policy (balanced
   success/fail mix) in the v3 dual-render env, score each model per-step, dump
   `(GT_label, success_prob)`, sweep thresholds -> pick per-model threshold at target FPR, read
   off TPR. FT is already half-calibrated (0.76 vs 0.13 from the curated control); the baseline
   needs a fresh rollout. Run this BEFORE the model-terminated RL.
   Likely outcome of switching to model-terminated: probably somewhat lower (FN sparsity) with
   a tail risk of reward-hacking collapse (FP) — NOT the old ~0.10 wipeout (that cause is fixed
   by dual-render + v3). Calibration predicts which.
3. **Multi-seed** FT vs baseline (seeds 2,3) for error bars; FT loses offline, so on-policy
   parity/advantage is the result to nail down.
