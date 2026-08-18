# Raising run2 above ~13% on ManiSkill — ranked experiment list

State of play, all measured on PullCube-v1 (GT SAC reaches 100% by 175k):

| model | offline AUROC | on-policy AUROC | RL success |
|---|---|---|---|
| **run2** (noicl, asym) | 0.500 | **0.785** [0.747, 0.824] | **10-15%**, plateaus after 150k |
| run3 (noicl, standard) | 0.397 | 0.597 [0.518, 0.676] | 2% |
| base (Robometer-4B) | 0.500 | 0.443 [0.352, 0.533] | 0% |

Canonical config: `env.use_full_state=false env.terminate_on_success=false
env.reward_shift=-1.0 gamma=0.8 tau=0.01 batch_size=1024 learning_starts=4000
num_envs=4`, CPU physics, 50-step episodes, reward model replaces the env reward
(`add_estimated_reward=false`). ~5 steps/s => 150k ~ 8h ~ 1000 SBU.

---

## 1. Fix the success threshold: 0.65 -> ~0.10  [highest value, zero new code]

Termination has been enabled the whole time (`use_success_detection=true`) but has
NEVER FIRED: 0 `[DETECT]` events across every run. The threshold was calibrated
offline on GT-actor rollouts, where run2's successful episodes peak at 0.83. On the
RL policy's OWN successful episodes it never exceeds 0.19.

On-policy calibration from `[SP-EP]` logs (per-episode max success_prob + GT label):

| thr | run2 @150k TPR/FPR | run2 resumed TPR/FPR |
|---|---|---|
| 0.08 | 72% / 20% | 83% / 50% |
| **0.10** | **48% / 8%** | 66% / 33% |
| 0.12 | 23% / 2% | 38% / 19% |
| 0.65 | 0% / 0% | 0% / 0% |

Why this should raise success: with `reward = VLM - 1`, ending an episode at step ~8
instead of 50 avoids ~40 steps of cost. That return difference dwarfs the reward
model's own success-vs-failure gap (mean return 8.65 vs 8.04, i.e. 7.5%). Stopping the
cost is the dominant signal in LIBERO/IBRL; here it has been unreachable.

    reward_model.success_detection_threshold=0.10
    reward_model.success_detection_duration=2      # majority vote, blunts false fires

Watch `[DETECT] ... gt_success=0 ... false_rate=` — a rising false_rate with fires at
small `step_in_ep` means reward hacking; apply `success_detection_min_ep_steps`.

NOTE: separation degrades as the policy improves (on-policy AUROC 0.820 -> 0.708
between the 150k run and its continuation) because failures become near-misses.
Re-derive the threshold from the run's own `[SP-EP]` logs periodically rather than
fixing it once.

## 2. beta-mix the two heads

The progress head drives RL but plateaus; the success head has on-policy AUROC
0.71-0.82. `vlm_ibrl/env/vlm_envs.py` mixes them and ships with beta=0 (pure
success_prob), binarized by a calibrated threshold:

    mixed = beta * progress + (1 - beta) * success_prob
    reward = 1.0 if mixed > threshold else 0.0

Not implemented in this repo (the buffer uses progress only, via
`extract_rewards_from_output`). Try beta in {0, 0.5}. beta=0 with the 0.10 threshold
is the exact MetaWorld/Robomimic recipe.

## 3. Reward normalization  [implemented, `normalize_reward=true`, untested on run2]

Maps the reward onto [0,1] via running p1/p99 BEFORE the -1 shift. Without it the
agent sees a model-specific constant offset (-0.90 for run2) with the signal only ~7%
of the magnitude. Tested on run3: did not unlock learning (eval stayed 0%). Untested
on run2, which unlike run3 has a working policy to improve.

## 4. Relative / potential progress  [implemented, `progress_as_potential=true`]

`gamma*Phi(s') - Phi(s)` instead of the progress level; the level is farmable and the
policy demonstrably farmed it (predicted reward rose 7x while GT success stayed 0-2%).
An OFFLINE pre-check on GT-actor rollouts ranked failures ABOVE successes (AUROC
0.17-0.22), which is why it was not run -- but that check has the same offline->online
gap this project keeps hitting, and GT-actor episodes undo the task after success
(progress declines, tau vs time flips to -0.18 on successes). Worth re-checking on
ON-POLICY `[DISCRIM]` data before dismissing.

## 5. Updates-to-data ratio

One gradient update per env step is tuned for GT's strong reward; a weak reward may
need more updates per sample. Also the fix for the `num_envs=16` collapse (more envs
at fixed updates = fewer updates per transition). Cheap to sweep, low information.

## 6. Seeds

13% is ONE seed, and the 2% -> 15% breakthrough at 125-150k is one event. Two more
seeds distinguish "the reward's ceiling" from "one run's local optimum".

---

## Ruled out — do not re-spend on these

* **gamma 0.95** — tested on run3 with normalization: training successes DECLINED
  3.7% -> 0.0%, eval stayed 0.
* **Normalization on run3** — 1.0% -> 2.7% training successes, eval never left 0.
* **Input resolution** — once frames are padded to `max_frames`, 224 and 480 give
  byte-identical outputs. The working Robomimic pipeline asserts 224.
* **DINO video_embeddings / text_embedding** — inert; `emb=None` and `emb=DINO` both
  give 0.0045. Calibration and training compute the same thing.
* **cuDNN / bf16 / fp32 numerics** — byte-identical outputs.
* **More steps alone, for run3/base** — eval GT reward pinned at 0.000 across 150k
  with no partial progress to extrapolate. (Does NOT apply to run2, which does learn.)

## Read first

`REWARD_MODEL_SCORING.md` — the canary (a fine-tuned head should read 0.7-0.9
per-episode max on a true success), the frame-count requirement, and the loading
gotchas. Three separate bugs in this project were pipeline mismatches that looked like
model failures.

## Known bug, fixed here — pull before resuming anything

`Algorithm.load()` rebinds `algorithm.actor`, but `train.py` passed the LOCAL `actor`
to the rollout worker and runner, so a resumed run collected and evaluated with an
UNTRAINED policy while training the restored one. `log_ent_coef` masked it (restores
in-place, so it looked correctly resumed). Fixed in c4486d7. Always set
`eval.eval_on_first_step=true` on a resume as a guard.
