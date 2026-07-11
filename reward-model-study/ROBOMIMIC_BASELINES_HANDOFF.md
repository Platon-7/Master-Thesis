# Running the VLM Reward-Model Baselines on Robomimic — Hand-off Guide

This is a hand-off from the MetaWorld baseline runs. Everything below was learned
building these on MetaWorld (`vlm_ibrl_v3/env/vlm_envs.py` + `metaworld_wrapper.py`).
On Robomimic you plug the same scorers into `env/robosuite_vlm_env.py` +
`robosuite_wrapper.py`. Tasks: Lift / PickPlaceCan / NutAssemblySquare (etc.).

--------------------------------------------------------------------------------
## 0. THE #1 RULE — validate the scorer OFFLINE before spending RL compute
--------------------------------------------------------------------------------
For EVERY baseline, before any RL:
1. Roll out a few oracle **success** and **failure** trajectories, render the
   reward camera(s).
2. Score them with the model and confirm **success scores > failure scores**
   (print the numbers).
3. Only launch RL if it discriminates.

This single discipline caught every real bug on MetaWorld:
- RoboReward looked "flat/OOD" → actually a **feeding bug** (full video collapsed
  spatial resolution to 4×4). Fixed by 16-frame subsample.
- LRM looked "dead" → actually **needed the initial-image anchor**, then a **wrong
  reward FORM** for off-policy RL (see §3).
Do NOT conclude "OOD / model is weak" from a flat RL curve until the offline check
is done. Cheap smoke run (≤600 steps) after that, then the full seeds.

Also: **the RL job may hardcode `NUM_TRAIN_STEP`** — check it reads from env before
assuming a `--export NUM_TRAIN_STEP=…` "smoke" is actually short.

--------------------------------------------------------------------------------
## 1. THE BASELINES + PROTOCOL (group by MODEL TYPE, not by vendor)
--------------------------------------------------------------------------------
| Baseline            | Signal                    | Protocol        | Demos? | Size |
|---------------------|---------------------------|-----------------|--------|------|
| Demo2Reward         | binary 0/1 success        | **autonomous**  | prompt opt. w/ demos | Qwen3-VL-8B |
| RoboReward-4B       | video 1–5 → [0,1]         | non-auton (sparse, end-of-ep) | zero-shot | 4B |
| RoboDopamine-4B     | multi-view hop progress   | non-auton (dense, delta) | goal image = 1 demo | 4B |
| LRM-Progress        | single-frame progress     | non-auton (dense, **delta**) | zero-shot | 8B |
| (our FT / Robometer)| success head              | autonomous      | ICL/gate variants | 4B |

Rule: **detectors → autonomous** (terminate the episode the moment success is
detected, like our FT). **scorers → non-autonomous** (full episode). Match our FT
autonomous recipe exactly for detectors (gate = 0.8×median demo length, threshold,
fire-on-detection) so the ONLY variable is the reward model.

--------------------------------------------------------------------------------
## 2. PER-BASELINE RECIPE
--------------------------------------------------------------------------------

### RoboReward-4B  (`teetone/RoboReward-4B` — NOT 8B, to match our 4B)
- Loader/scorer: `env/roboreward_utils.py`; scored via `single_prompt_eval(use_video=True)`.
- **Feed = video**, but **SUBSAMPLE to 16 frames** (`ROBOREWARD_NFRAMES=16`, final frame
  always included). Full rollout → spatial res collapse to 4×4 + timestamp-token flood.
  `max_new_tokens=24` (5 is too few to reach `ANSWER:`).
- Prompt already correct (`roboreward_prompt`, 1–5 rubric → `ANSWER:<n>` → (n-1)/4).
- Protocol: NON-autonomous, reward at truncation (`AUTONOMOUS_SUCCESS=0`, `reward_at_truncation=1`).
- **Robomimic is IN-DOMAIN for RoboReward** (their paper evaluates on Robomimic Lift/Can/Square)
  → expect it to WORK here, unlike MetaWorld where it was 0.00. Don't be surprised if it's decent.

### RoboDopamine-4B  (`tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview`)
- Scorer: `env/robodopamine_utils.py` (env-agnostic — feed PIL frames + goal image).
- **8 images per call, fixed order**: ref-start-front, goal-front, before×3-views, after×3-views.
- Needs a **GOAL image** (a success keyframe = 1 demo → this is a "≥1 demo" baseline) and
  **multi-view** (front + 2 wrist). Robomimic cams: `agentview` (front) + `robot0_eye_in_hand`
  (wrist) — for the 2nd wrist, reuse the wrist cam or a second angle (`sideview`/`frontview`).
- Parse `<score>±NN%</score>` — accept FLOATS (`0.0%`), fallback to split-on-tag parse.
- `eval_mode="forward"`; reward = **potential-based DELTA hop** (dense), scored every ~16 steps.
- Protocol: NON-autonomous. Render views on-demand via the base env (see `_base_env()` walk —
  stop at the robosuite env whose `render(camera_name=…)` exists, NOT the raw gym env).
- Generate the goal image once per task (oracle success final frame, front cam).

### LRM-Progress  (`USC-PSI-Lab/LRM-models`, `subfolder="progress"`, Qwen3-VL-**8B**)
- Scorer: `env/lrm_utils.py`. Single **current frame + INITIAL anchor** (frame 0). Output
  JSON `{"completion_progress": v}` → parse (handle floats + `NA`→0).
- **INITIAL ANCHOR IS REQUIRED** — pure single-frame is flat ~0.2 (no separation). With the
  episode's own frame-0 as anchor it calibrates (`LRM_INCLUDE_INITIAL=1`). Still zero-shot
  (frame 0 is not a demo).
- **⚠️ REWARD FORM — the big lesson (see §3): use `LRM_REWARD_MODE=delta`, NOT hold.**
- `LRM_CALL_INTERVAL=10`, `LRM_RES=256` (≥ their 256² min). Protocol: NON-autonomous.
- 8B → needs a **48 GB GPU** (see §4).
- LRM's benchmark is ManiSkill (robosuite-adjacent) → Robomimic is closer than MetaWorld but
  still OOD; expect weak-but-maybe-non-zero. Do the offline check.

### Demo2Reward  (base `Qwen/Qwen3-VL-8B-Instruct` + optimized prompt — frozen VLM)
- Optimized per-task prompts already exist: `ROBOMIMIC_DEMO2REWARD_REPLIES` (PickPlaceCan,
  NutAssemblySquare, …). These judge the **final/current frame** → binary 0/1.
- Protocol: **AUTONOMOUS** (this is a success detector — terminate on detection, match FT):
  `AUTONOMOUS_SUCCESS=1`, `ROBOMETER_SUCCESS_THRESHOLD=0.5`, `ROBOMETER_SUCCESS_CONSECUTIVE=1`,
  `ROBOMETER_MIN_EP_FRAC=0.8` (same gate as FT), `reward_at_truncation=1`.
- The env exposes the binary verdict as the detection signal: in `vlm_reward`'s demo2reward
  branch, `self._last_success_prob = float(reward)` (already done in v3). Confirm the robosuite
  env has the same hook.
- Queries the generative VLM **every step** (slow) — that's correct and matches how our FT
  autonomous detector queries every step; just slower per call (generative 8B). Budget time.
- Demo-informed (prompt optimized w/ labeled demos) → "≥1 demo" group.

--------------------------------------------------------------------------------
## 3. THE REWARD-FORM × RL-ALGORITHM LESSON (do not skip)
--------------------------------------------------------------------------------
Dense progress models (LRM, RoboDopamine) output an ABSOLUTE progress ∈ [0,1].
How you turn that into the per-step reward matters and depends on the RL algorithm:

- Their papers use **PPO** (on-policy, advantage-based). An absolute reward "held"
  every step is fine — the value baseline cancels the constant part.
- Our IBRL is **off-policy Q-learning**. An absolute-held reward (~0.25/step) is a
  **survival bonus**: Q inflates uniformly, no gradient toward completion → flat/dead.
- FIX: deliver a **potential-based DELTA** — reward = progress_t − progress_{t−interval},
  0 between scored steps. Credit lands on progress-making transitions.

Evidence: on MetaWorld, RoboDopamine (delta) = 0.13 vs LRM (absolute-held) = 0.03,
near-identical weak models; switching LRM to delta unlocked per-seed learning
(Box seed hit 0.55) that hold NEVER produced. (It still collapsed on MetaWorld —
weak reward — but on in-domain Robomimic the delta form is what gives it a chance.)
**So: any dense progress reward → delta form under IBRL. Absolute-held is a trap.**

--------------------------------------------------------------------------------
## 4. GPU / OOM
--------------------------------------------------------------------------------
- **8B models (LRM, Demo2Reward)**: ~16 GB weights + training + generate activations
  → **must run on 48 GB (L40S)**, NOT 24 GB (L4) or they OOM. Restrict the SLURM
  `--partition` to the l40s partitions only.
- 4B models (RoboReward, RoboDopamine): fine on either L4 or L40S.
- HF models download on first load (compute node has internet). RoboReward/RoboDopamine/LRM
  are ~8–16 GB each.

--------------------------------------------------------------------------------
## 5. ROBOMIMIC-SPECIFIC ADAPTATIONS
--------------------------------------------------------------------------------
- **Reward camera**: pick the in-domain third-person view (`agentview` is the natural
  analog of MetaWorld's `corner2_default`). Wrist = `robot0_eye_in_hand`. Validate the
  view shows the task clearly (offline check).
- **Episode length / gate**: robomimic horizons differ; set the gate to 0.8×median demo
  length per task (Can≈116, Square≈149 on our data — the gate we used was 120; document it).
- **Task strings + optimized prompts**: use the robomimic task language and
  `ROBOMIMIC_DEMO2REWARD_REPLIES`.
- **Goal images (RoboDopamine)**: generate one success keyframe per robomimic task
  (oracle/BC success final frame, front cam).
- **In-domain expectation**: RoboReward (and to a lesser extent the others) were built/
  evaluated on Robomimic-style sim, so they may actually PERFORM here — this is the FAIR
  comparison ground (unlike MetaWorld, which was OOD for all of them). Frame results
  accordingly.

--------------------------------------------------------------------------------
## 6. REUSABLE CODE (env-agnostic — feed robomimic PIL frames)
--------------------------------------------------------------------------------
- `env/roboreward_utils.py`   — RoboReward loader + prompt.
- `env/robodopamine_utils.py` — `RoboDopamineScorer` (takes 3-view frames + goal img).
- `env/lrm_utils.py`          — `LRMProgressScorer` (takes current frame, optional initial).
All three know NOTHING about MetaWorld — they take PIL images. The MetaWorld-specific
glue in `vlm_envs.py` you re-implement in `robosuite_vlm_env.py`:
  - `_base_env()` walk to the render-capable env (stop before the raw gym env),
  - per-episode reset (build scorer, capture frame-0 anchor / goal),
  - the `robodopamine_4b` / `lrm_progress_8b` / `roboreward_4b` / `demo2reward_*` branches
    in reset()/step(), guarded so nothing else changes.

--------------------------------------------------------------------------------
## 7. SUGGESTED ORDER OF WORK
--------------------------------------------------------------------------------
1. RoboReward first (simplest: video + subsample, sparse) — and it's in-domain, so it's
   the confidence check that the pipeline works.
2. Demo2Reward (autonomous detector; prompts already exist).
3. RoboDopamine (multi-view + goal + delta).
4. LRM (single-frame + initial anchor + **delta**; 8B/L40S).
For each: offline discrimination check → 600-step smoke → 5 seeds × 2–3 tasks.
