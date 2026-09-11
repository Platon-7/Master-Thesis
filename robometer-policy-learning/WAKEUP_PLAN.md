# Waking up the reward model on ManiSkill — diagnosis and run plan

Written on hipster (AlmaLinux/L4) after the first 5-variant sweep on PullCube-v1
returned 0% success on every arm. Everything below is measured from those runs'
own logs, not assumed.

## 1. What the first sweep actually showed

All five arms (`thresh010`, `beta05`, `normalize`, `potential`, `utd2`) ran ~300k
steps from scratch with run2 (`run2_noicl_ours_step4000`) and ended at 0% eval
success. But they were not five independent tests — four of them were dominated
by the same artefact.

### 1.1 The success-threshold fix backfired: termination became the exploit

`NEXT_EXPERIMENTS.md` #1 lowered `success_detection_threshold` 0.65 -> 0.10 because
the detector had never fired. It fired — and every single fire was false:

| run | fires | false_rate | mean len of fired ep | mean ep len |
|---|---|---|---|---|
| thresh010 | 1176 (19% of eps) | 1.00 | 11.1 | 41.6 |
| beta05 | 7911 (61%) | 1.00 | 7.4 | 20.2 |
| normalize | 42 (1%) | 1.00 | 26.4 | 50.6 |
| potential | 0 (0%) | — | — | 51.0 |
| utd2 | 8985 (65%) | 1.00 | 7.3 | 18.1 |

With `reward = VLM` and `reward_shift = -1`, ending an episode stops the cost.
The policy learned to trip the detector within ~7 steps of a 50-step episode.
That is the reward hacking the doc warned about, and `success_detection_min_ep_steps`
(the prescribed guard) was left at its default of 0.

### 1.2 The success head is NOT inverted — that reading was a selection artefact

Episode-level AUROC of max success-prob vs GT success, computed from `[SP-EP]`:

| run | AUROC (all eps) | AUROC (unfired eps only) |
|---|---|---|
| thresh010 | 0.540 | **0.628** |
| beta05 | 0.363 | **0.652** |
| normalize | 0.614 | **0.618** |
| potential | 0.621 | **0.621** |
| utd2 | 0.333 | **0.606** |

The sub-0.5 numbers are caused by termination itself: firing selects high-sp
FAILURE episodes and truncates them, so the negative class gets enriched with
exactly the high-sp episodes. Remove fired episodes and the head is a consistent
**0.61-0.65 across all five runs** — degraded relative to run2's historical 0.785
on MetaWorld, but real, positive, and above run3's 0.597. The head is not broken;
it is operating on a much worse policy distribution than IBRL ever gave it.
This is the thesis' own claim (off-policy != on-policy) showing up as a number.

Termination also poisons the buffer, independent of the measurement problem.
Turning VLM termination off is therefore justified by the data, not just taste.

### 1.3 Real successes happened and were never amplified

GT successes per run: 68 / 129 / 28 / 19 / 90 episodes (0.4-1% background rate,
from exploration alone). The policy does reach the goal occasionally. Nothing in
the reward makes those episodes worth repeating.

### 1.4 The progress head is alive, and farming as before

Mean progress rose 0.045 -> 0.15-0.19 in every run — the same signature as
Snellius (0.056 -> 0.425 with GT success flat). Gradients flow; the objective is
just not aligned with finishing.

### 1.5 Entropy collapsed before any success was found

`ent_coef` ended at 1e-4 .. 5e-3 in all five runs with success still ~0. SAC's
auto-entropy stopped exploring long before there was anything to exploit.

### 1.6 `progress_as_potential` alone cannot work in this configuration

With `add_estimated_reward=false` the VLM reward REPLACES the env reward, so with
`progress_as_potential=true` the ONLY reward is `gamma*Phi(s') - Phi(s)`. Potential
shaping is policy-invariant by construction: on its own it encodes no preference
for reaching the goal. That arm was structurally incapable of learning the task,
which matches its outcome (19 successes, all exploration, zero detector fires).
Potential shaping needs a base task reward to shape.

## 2. Structural difference from where the head succeeded

MetaWorld, Robomimic and LIBERO results were produced with **IBRL**, which
bootstraps from a behaviour-cloning policy: the policy is near-expert from step 0,
so the reward model is queried on states resembling its training distribution, and
successes are frequent enough to reinforce. `maniskill_online_rl.yaml` is plain SAC
from scratch — no BC anchor, ~1% success. Same model, far harder query distribution.
That gap is the single most likely explanation for 0.785 -> 0.62, and it is worth
stating in the paper rather than treating as a bug.

Also note the recipe difference: vlm_ibrl ships `robometer_beta = 0` — the reward
IS the success head, binarised by a threshold. Every ManiSkill arm so far has used
the PROGRESS head as the reward. **The configuration that worked on the other three
benchmarks has never been run on ManiSkill.**

## 3. Runs launched

All on PullCube-v1, run2 checkpoint, 300k steps, `RPL_LOG_DISCRIM=1` (episode-level
reward-vs-GT discrimination, the number comparable to the historical 0.785).

### Measurement (hours, not days)

| id | what | why |
|---|---|---|
| M1 | `causal_calib_maniskill.py` over the GT run's 13 checkpoints (25k..300k, 0%->100% success) scored with run2 | Two answers at once: (a) canary — does the head reach its historical 0.7-0.9 per-episode max on TRUE successes on this rebuilt graphics stack; (b) head quality as a FUNCTION of policy quality, i.e. the off-policy -> on-policy degradation curve. If (a) fails, rendering is wrong and no RL tuning matters. |

### Track A — the clean regime (the only config that ever produced 15%)

No VLM termination, `reward_shift=0.0`. The -1 shift exists to punish loitering
when episodes can end early; with no early termination on a fixed 50-step horizon
it is a constant offset on every trajectory and cannot change any policy ranking.

| id | config | why |
|---|---|---|
| A0 | seed 0 (job 344122, no DISCRIM) | already running |
| A1 | seed 1 | 15% was ONE seed of ONE event; this tests seed luck |
| A2 | seed 2 | same |

### Track B — reward design, all on top of the clean regime

| id | config | why |
|---|---|---|
| B1 | `progress_beta=0.0` (reward = success head, continuous) | the head that worked on MW/Robomimic/LIBERO, used as the reward, never tried here |
| B2 | `progress_beta=0.0` + `progress_binarize_threshold=0.10` | the vlm_ibrl recipe verbatim (beta=0, binarised); 0.10 is the on-policy operating point measured in 1.2 |
| B3 | `gamma=0.95` | gamma=0.8 gives an effective horizon of ~5 steps on a 50-step task, which structurally rewards sitting in a high-progress pose over finishing. gamma was "ruled out" on run3 — an arm that never learned anything, so that null says nothing about run2 |
| B4 | `progress_beta=0.5` | rerun of the beta05 arm without the 61%-firing confound |

### Track C — termination, done properly

| id | config | why |
|---|---|---|
| C1 | TERMINATE=1, thr=0.10, `duration=3`, `min_ep_steps=15`, shift=-1.0 | the cost regime is only coherent WITH termination. The gate makes the step-2..7 exploit impossible, so this is the first fair test of the success head as a terminator |

### Track D — exploration

| id | config | why |
|---|---|---|
| D1 | clean regime + `target_entropy=-1.0` (auto = -4) | direct response to the entropy collapse in 1.5: keep the policy stochastic long enough to find the 1% successes |

## 4. What to read, and when

* **M1 first.** If the head scores ~0.1 instead of 0.7-0.9 on the GT policy's own
  successes, stop everything and fix rendering.
* At ~150k steps, compare A0/A1/A2 against the historical 2%->15% breakthrough
  window (125-150k). Three flat seeds there is a much stronger negative than the
  single seed we have.
* Re-derive the operating threshold from each run's own `[SP-EP]` periodically —
  separation degraded 0.820 -> 0.708 historically as the policy improved.
* Judge B1/B2 on whether ANY success appears, not on final number.

## 5. If everything here still flatlines

In rough order of expected value:

1. **BC / demo warm start** — close the structural gap in section 2. ManiSkill
   ships demonstrations and this repo already has a BC algorithm, an offline H5
   buffer and `buffer.sample_ratio` for mixed replay. This makes the ManiSkill
   result methodologically comparable to the other three benchmarks instead of a
   different and much harder experiment.
2. **ICL arm (run1)** — the untested thesis contribution, and the intervention
   aimed exactly at out-of-distribution query frames. Needs an ICL bank built by
   `scripts/generate_maniskill_icl_demos.py`.
3. **Potential shaping done right** — dense unfarmable shaping PLUS a sparse
   VLM-success bonus (not GT), i.e. fix 1.6 rather than abandon it.
4. **An easier task** — LiftPegUpright-v1 is the only other task whose GT arm
   clears the bar (88% and rising at 500k).

---

# RESULTS (2026-08-20)

## The reward model is awake: 55% on PullCube-v1, vs 15% best on Snellius

| run | config | eval success |
|---|---|---|
| **gamma095 (344147)** | noterm, shift 0, **gamma=0.95** | **55%** (20-ep eval, 146k steps, still rising) |
| clean_s2 (344144) | noterm, shift 0, gamma=0.8 | ~42% |
| noterm_rawreward (344122) | noterm, shift 0, gamma=0.8 | ~17% — **reproduces the historical 15%** |
| clean_s1 (344143) | noterm, shift 0, gamma=0.8 | 0% |
| beta0 / beta0_bin010 | success head AS the reward | 0% |

Training-episode GT success by 500-episode block confirms the trend:
gamma095 `1 -> 8 -> 25 -> 39 -> 49 -> 58%`; clean_s2 `1 -> 2 -> 1 -> 0 -> 3 -> 23 -> 40%`;
clean_s1 `0 -> 1 -> 1 -> 0 -> 0 -> 0%`.

### 1. gamma was the lever, and the old null result was measured on the wrong arm

`gamma=0.8` gives an effective horizon of ~5 steps on a 50-step task, which structurally
rewards sitting in a high-progress pose over finishing. `gamma=0.95` (~20 steps) lets the
return see the end of the episode. NEXT_EXPERIMENTS.md listed gamma=0.95 under "ruled out
— do not re-spend", but that test was run on **run3 with normalization**, an arm whose eval
never left 0. A null on a dead arm says nothing about run2.

### 2. Seed variance at gamma=0.8 is severe, and explains the fragile history

Three seeds of the identical gamma=0.8 clean regime gave 0%, ~17%, ~42%. The historical
"2% -> 15% breakthrough at 125-150k" was one draw from that distribution. gamma=0.95 seeds
1 and 2 are running to test whether the higher gamma also tightens this spread.

### 3. M1 canary: the head is NOT degraded on this cluster, and thr=0.65 was right all along

Scoring run2 on the GT policy's own rollouts (13 checkpoints, 25k..300k):

* successful episodes: **max success_prob 0.797-0.852** — exactly the documented 0.7-0.9.
  The rebuilt graphics stack (extracted NVIDIA ICD + conda-forge Vulkan loader, no
  FlashAttention) is faithful. Rendering was never the problem.
* causal threshold sweep: **best threshold 0.6875, TPR 1.00, FPR 0.00, J 1.00** — perfect
  separation.
* gate guidance: real fires at step **6-12** (median 7); zero fake fires.

**This invalidates NEXT_EXPERIMENTS.md idea #1.** That doc reasoned "on-policy successes
peak at 0.19, so 0.65 is 6x too high — lower it to 0.10". The correct reading is the
opposite: the head was correctly reporting *no canonical success present*, because the RL
policy's rare successes are exploration flukes that do not look like task completion. The
head is semantically strict, not miscalibrated. Lowering the threshold to 0.10 is what
produced `false_rate=1.00` and the 0% sweep.

`min_ep_steps=15` would also have been wrong: genuine successes fire at steps 6-12, so a
15-step gate blocks every true detection. Correct gate is ~5.

### 4. The success head as a *reward* does not work from scratch (and that is consistent)

`beta=0` (reward = success_prob) and `beta=0` + binarize both sat at 0%. Until the policy
succeeds, success_prob is ~0.08 everywhere, so the reward is a near-constant with no
gradient. The success head needs a competent policy to be informative — which is exactly
why it works under IBRL (BC warm start) on MetaWorld/Robomimic/LIBERO and not here. The
progress head is what carries learning from scratch. This is a clean, defensible
on-policy/off-policy result rather than a failure.

## Running now

| job | config |
|---|---|
| 344147 | gamma095 s0 PullCube (55%, continuing) |
| 344144 | gamma0.8 s2 PullCube (~42%) |
| 344122 | gamma0.8 s0 PullCube (~17%, historical repro) |
| 344239/344240 | **gamma095 s1 / s2** — reproducibility |
| 344241 | **gamma095 LiftPegUpright-v1** — task 2 |
| 344242 | gamma099 s0 — is even longer horizon better |
| 344243 | gamma095 + termination at the canary threshold (0.6875, gate 5, duration 2) |

Cancelled as invalidated by the canary or answered: 344143, 344145, 344146, 344148,
344149, 344150, 344151-153.
