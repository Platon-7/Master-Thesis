# ManiSkill + RoboRef handoff: reward-hacking follow-ups and running on another cluster

Written 2026-09-15, after the workshop submission, for continuing on a cluster other
than Snellius. It replaces the pre-paper handoff, whose recipe (gamma 0.8, reward shift
-1, terminate on success) was superseded; that version is in git history. Section 9
keeps the parts of it that still hold.

The supervisor's point was that the paper's reward-hacking evidence is correlational.
This file covers the analyses proposed in response:

| id | what | needs | status |
|---|---|---|---|
| A | Goodhart curves: proxy reward, GT success and FP rate over training | existing logs, CPU | **done**, section 3 |
| D | per-seed scatter, late success vs late FP | existing logs, CPU | **done**, section 3 |
| B | roll out saved policies, score every step, watch the high-scoring failures | GPU, under 1 h per run | ready, section 4 |
| C1 | oracle SAC on the GT reward, same recipe | GPU | ready, section 5 |
| C2 | inject false positives vs false negatives into the GT reward | GPU + new transform | not implemented, section 6 |
| C3 | real reward models with their false positives masked by GT | GPU + new knob | not implemented, section 7 |

## 1. Where everything is

Both drives hold identical copies (verified with `rclone check`, 2026-09-14):
OneDrive `myonedrive:Master-Thesis-transfer/` and Google Drive `robometer_dataset:Robo-Ref/`.

| path on the drive | contents |
|---|---|
| `env/maniskill_rl.tar.zst` | the Python 3.12 venv (identical to the live one on Snellius apart from editable-install paths) |
| `env/requirements_frozen.txt`, `env/REBUILD.md` | fallback freeze; older copy of the rebuild notes (the repo copy wins) |
| `env/assets_transfer.tar.zst` | `maniskill_assets/`: task assets, demos, PPO checkpoints, calibration trajectories |
| `checkpoints/Robometer_FT_consolidated/run2_noicl_ours_step4000` | **RoboRef**, the paper checkpoint |
| `checkpoints/Robometer_FT_consolidated/run3_noicl_standard_step5000` | Robometer-4B + Failures |
| `maniskill_policies/maniskill_paper_runs.tar.zst` | the 30 paper seeds (52 legs incl. resumes): `actor.pt` + `log_ent_coef.pt` every 25k steps, `episodes.jsonl`, `training.log`, `.hydra/`. See its README |
| `reward_hacking_plots/` | plot A and D figures and the scripts that made them |

Baselines come from Hugging Face on first use (into `$HF_HOME`): `robometer/Robometer-4B`,
`teetone/RoboReward-4B`, `tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview`,
`USC-PSI-Lab/LRM-models`, plus `Qwen/Qwen3-VL-4B-Instruct` and `facebook/dinov2-base`.

On Snellius only: full run folders incl. critic and optimizer states in
`/projects/prjs1958/pkarageorgis1/backups/maniskill_runs/<group>.tar` (52 GB), and the
8-frame run2 checkpoints 1000..5000 (model shards only) in
`/projects/prjs1958/Robometer_FT_weights_8f_modelonly/`.

## 2. The paper setup, and what "comparable" means

PullCube-v1, SAC from scratch, 300k env steps, the reward model's progress head
**replaces** the env reward, no VLM termination, no GT termination, 50-step episodes.

```
env.terminate_on_success=false   env.reward_shift=0.0   training.num_envs=4 (CPU physics)
online_algorithm.gamma=0.95  tau=0.01  batch_size=1024  learning_starts=4000
reward_model.add_estimated_reward=false  reward_model.use_success_detection=false
training.save_interval=25000 (yaml default)
```

Seeds: RoboRef 3, 9, 0, 1, 2 (drawn from a larger pool); every other model 0-4.

**Use `jobs/snellius_maniskill_sac.job` for anything compared against the paper.** It
bakes the recipe in and knows all six reward models (`MODEL=run2|run3|base|roboreward|robodopamine|lrm`)
plus the GT arm. Two traps:

* `maniskill_online_rl.yaml` alone gives a different algorithm: `terminate_on_success: true`,
  `reward_shift: -1.0`, and `sac.yaml`'s gamma 0.99 / tau 0.005 / batch 128.
* `jobs/hipster_maniskill_sac.job` also differs: gamma 0.8 by default, success detection
  on by default (`TERMINATE=1`), the GT arm terminates on success, and it only knows the
  Robometer reward group. Fine for its own sweeps, not for paper comparisons.

Measured wall-clock per 300k-step seed on one Snellius A100: Robometer-4B ~10.3 h,
RoboRef ~10.4 h (max 14.1), + Failures ~14.0 h, LRM ~46 h. Robo-Dopamine and RoboReward
exceeded 24 h and were resumed. The GT arm has no VLM in the loop and is several times
faster (18-30 steps/s measured, so roughly 3-5 h).

## 3. A and D: what the existing logs show (done)

Figures in `reward_hacking_plots/` on both drives and
`/projects/prjs1958/pkarageorgis1/reward_hacking_plots/`:

* `goodhart_curves_stepFP.png`: one row per model, columns = mean per-step reward (what
  SAC maximizes), GT solve rate, and FP with **Table 3's definition** (threshold = 5th
  percentile of scores at successful steps, pooled over a model's seeds; FP = fraction
  of steps in never-solved episodes at or above it). Pooled, it reproduces Table 3
  (0.393 / 0.835 / 0.893 / 0.639 / 0.957 / 0.850; the paper has 0.956 for RoboReward).
* `goodhart_curves.png`: same layout, FP judged on each episode's **peak** score.
* `seed_scatter.png`: per seed, last-20% solve rate vs last-20% peak FP.

Regenerate anywhere (numpy + matplotlib only, no GPU):

```bash
export ROBOREF_RUNS=/path/to/extracted/maniskill_paper_runs
python paper-analysis/reward-hacking/goodhart_steps.py  out/   # step FP, Table 3 definition
python paper-analysis/reward-hacking/goodhart_curves.py out/   # peak FP + scatter
```

Step FP and solve rate, first 20% of training -> last 20%, per seed range:

| model | step FP | GT solve rate |
|---|---|---|
| RoboRef, live seeds 3, 9 | 0.06 -> 0.55-0.61 | 0.02-0.07 -> 0.82-0.86 |
| RoboRef, dead seeds 0, 1, 2 | 0.06 -> 0.54-0.61 | 0.01-0.02 -> 0.00 |
| Robometer-4B + Failures | 0.51-0.54 -> 0.91-0.92 | 0.02-0.05 -> 0.00-0.10 |
| Robometer-4B | 0.66-0.69 -> 0.93-0.95 | 0.02-0.05 -> 0.00 |
| LRM-8B | 0.72-0.76 -> 0.88-0.90 | 0.04-0.14 -> 0.01-0.07 |
| Robo-Dopamine 2.0 | 0.50-0.59 -> 0.61-0.72 | 0.03-0.12 -> 0.03-0.17 |
| RoboReward-4B | 0.94 -> 0.95-0.97 | 0.01-0.03 -> 0.01-0.15 |

Reading:

* **Baselines (Robometer-4B, + Failures, LRM)** show the Goodhart signature over time:
  the proxy climbs, FP climbs with it to ~0.9 within ~70k steps, and GT success falls
  from its random-exploration level to ~0. LRM is the cleanest case (success peaks near
  50k and decays while FP keeps rising).
* **RoboReward** sits at ~0.95 FP from step 0: an uninformative reward, not hacking
  learned during training. **Robo-Dopamine** is flat at 0.6-0.7 and does not separate
  its learning and non-learning seeds. Neither supports nor refutes hacking.
* **RoboRef** starts at 0.06 (baselines 0.3-0.7 at their first bins), ends lower (~0.55
  vs ~0.9), and its reward is correctly ordered: seeds that solve earn more reward
  (0.187 per step late) than seeds that stall (0.155).
* **RoboRef's dead seeds are not explained by false positives.** Their step FP is the
  same as the live seeds', and by the peak metric they never reach a true success's
  level (peak on failures ~0.18 vs threshold 0.217; peak FP 0.000-0.004 all run).
  They stall at partial progress: an exploration / local-optimum failure.

Consequence for the paper: the sentences at `main.tex` lines 400 and 476 that attribute
the three dead seeds to "states RoboRef still scores as successful" are not supported;
cutting that clause is the minimal fix. The rest of the hacking section is supported,
now with a time course.

Limits: observational (no intervention, hence C1-C3); each model's threshold comes from
its own successes, which are few for Robometer-4B and + Failures (0.7% of steps);
different policies visit different states; RoboRef's late ~0.55 may be near-misses
(B answers this); one task.

## 4. B: roll out saved policies with a reward trace

**Question.** What do RoboRef's high-scoring failed states look like (near-miss or
exploit)? What do the dead seeds converge to? What do baseline policies do in the
states their model scores as success?

**Tool.** `scripts/causal_calib_maniskill.py` already does most of it: it loads a ladder
of saved actors from a run's `checkpoints/`, rolls each out, scores the growing video
at every step with a Robometer-family model, and writes per-episode `success_prob` and
`progress` series with the GT outcome and GT-success step to
`<out-dir>/<task>__<tag>_causal.json`.

```bash
python scripts/causal_calib_maniskill.py --task PullCube-v1 \
  --model $CKPT_ROOT/run2_noicl_ours_step4000 --tag run2_s0dead \
  --actor-dir $ROBOREF_RUNS/ms_PullCube-v1_run2_dense_s0_20260904_234048/checkpoints \
  --n-checkpoints 13 --episodes 16
```

Two small additions are needed before it answers B:

1. **Save frames.** The episode's frames are already in `video` (around line 155);
   `imageio.mimsave` a few episodes per checkpoint, prioritising never-solved episodes
   whose peak score clears the model's threshold.
2. **Log the GT dense reward per step.** The script creates the env with
   `reward_mode="normalized_dense"`; recording that reward separates near-misses (high
   GT progress, not solved) from exploits (high model score, low GT progress).

Suggested runs, each scored by its own model: RoboRef s0 (dead) and s3 (live),
Robometer-4B s0, + Failures s0. The strongest single result is **cross-scoring**: score
Robometer-4B's policy with RoboRef. If RoboRef rates the states Robometer-4B's policy
converged to as low, that is direct evidence of the asymmetric objective working.
Only Robometer-family models load in this script; LRM, Robo-Dopamine and RoboReward
would need the adapters in `utils/baseline_reward_adapters.py`.

Cost: roughly 0.5-1 GPU-hour per (run, model) pair at 13 checkpoints x 16 episodes.

## 5. C1: oracle, SAC on the GT reward

Same recipe with the env's `normalized_dense` reward (`ARM=gt`, no termination), seeds
0-4. It establishes that the task and recipe are learnable at 300k, gives the ceiling
for every panel of plot A, and tests the dead-seed reading: if some GT seeds also stall,
the stalls are an SAC exploration property and not a reward property.

```bash
sbatch ... --export=ALL,ARM=gt,SEED=0,STEPS=300000 jobs/snellius_maniskill_sac.job
```

Cost: ~3-5 GPU-hours per seed (no VLM), so 15-25 for five seeds.

## 6. C2: controlled false positives vs false negatives (not implemented)

The causal test of the paper's premise, independent of any VLM. Start from the GT
reward and corrupt it with a known error type at a known rate p:

* **FP arm (spurious goal).** In a fixed decoy region of state space that is not success
  (for PullCube, e.g. the cube pulled sideways past a set distance), pay success-level
  reward with probability p. This models a false positive a policy can find and seek.
* **FP control.** Same rate, but uniformly random failed steps. Not exploitable; separates
  "wrong reward" from "exploitable wrong reward".
* **FN arm.** On successful steps, zero the reward with probability p.

Prediction (Gumbsch et al. 2026): the decoy FP arm collapses toward the decoy, the FN arm
only slows. Grid: p in {0.1, 0.3}, 5 seeds each, GT speed.

Implementation: a `TransitionTransform` next to `RewardShiftTransform` in
`utils/transitions_transforms.py`, appended as a post-transform where `reward_shift_fn`
is built (`utils/training_utils.py` ~line 700). The GT arm uses the plain `ReplayBuffer`,
so the transform sees only the transition; the success flag and cube position must reach
it through the transition's info (add them in `envs/maniskill_wrapper.py` if absent).

Cost: 4 arms x 2 rates x 5 seeds = 40 runs at ~3-5 h, roughly 120-200 GPU-hours. Two
arms (decoy FP, FN) at one rate is the minimum useful version, ~30-50 GPU-hours.

## 7. C3: real reward models with their false positives masked (not implemented)

Run the normal reward-model arm, but where GT says the task is not solved and the score
reaches the model's threshold, clip the score below the threshold. This removes exactly
the errors Table 3 counts and nothing else. If masked Robometer-4B / + Failures now
learn, their false positives caused the failure. If RoboRef's dead seeds still stall,
that confirms section 3.

Step thresholds (Table 3 definition, pooled over the paper seeds): RoboRef 0.1610,
+ Failures 0.2109, Robometer-4B 0.6538, Robo-Dopamine 0.6670, RoboReward 0.25, LRM 0.20.

Design caveat: online you cannot know whether an episode will succeed later, so a mask
on "not solved at this step" also clips the approach steps just before a real success.
The better mask caps the model score by GT progress (from `normalized_dense`), i.e. it
clips only scores the true state does not justify.

Implementation: in `RobometerReplayBuffer._add`, just before
`kwargs["reward"] = avg_reward` (`buffers/robometer_replay_buffer.py:748`), where the GT
success flag (`kwargs["is_success"]` / `kwargs["success"]`) and the incoming env reward
(`_gt_in`) are both available. Expose it as a new `reward_model.fp_mask_tau` knob. For
the GT-progress mask, run the env with `reward_mode=normalized_dense`; the reward model
still overwrites it because `add_estimated_reward=false`. It is an oracle diagnostic,
not a method.

Cost: Robometer-family 10-14 GPU-hours per seed; 3 models x 3 seeds is ~100-130.
LRM / Robo-Dopamine / RoboReward are 24-46 hours per seed.

## 8. Setting up another cluster

1. **Code.** `git clone git@github.com:Platon-7/RoboRef.git` (branch `main`).
2. **Fetch from a drive** (needs rclone configured there), e.g.
   `rclone copy myonedrive:Master-Thesis-transfer/env $SCRATCH_ROOT/transfer/env`, and the
   same for `checkpoints/Robometer_FT_consolidated/run2_noicl_ours_step4000`, `.../run3_noicl_standard_step5000`
   and `maniskill_policies/`.
3. **Environment.** Extract `maniskill_rl.tar.zst` to `$SCRATCH_ROOT/envs/`. On a
   non-Snellius node, follow `REBUILD.md` "Rebuilding ON hipster": pipe through `zstd -dc`
   if `tar --zstd` is missing, repoint `bin/python3` and `pyvenv.cfg` at a local Python
   3.12, and repoint the three `_editable_impl_*.pth` files at your checkout.
4. **Assets.** Extract `assets_transfer.tar.zst` so that `$SCRATCH_ROOT/maniskill_assets` exists.
5. **Paper runs** (for A and B). `zstd -dc maniskill_paper_runs.tar.zst | tar -xf -`, then
   `sha256sum -c maniskill_paper_runs.tar.zst.sha256`.
6. **Vulkan.** Only if compute nodes ship no Vulkan userspace (hipster did): build
   `$SCRATCH_ROOT/nvidia_driver` and `$SCRATCH_ROOT/vulkan_loader` per `REBUILD.md`, then
   pass `VULKAN_FIX=1`.
7. **Verify** on a GPU node: `python scripts/verify_maniskill_env.py --task PullCube-v1`,
   and check `import robometer; print(robometer.__file__)` points at `Robometer/`, not the
   in-repo submodule (else pass `SAFEPATH=1`).
8. **Launch** with the paper job, overriding partition, logs and paths at submit time:

```bash
sbatch -p <partition> -t 48:00:00 --mem=47G \
  -o logs/ms_sac_%j.out -e logs/ms_sac_%j.err \
  --export=ALL,SCRATCH_ROOT=/scratch/$USER,ENV_PREFIX=/scratch/$USER/envs/maniskill_rl,CKPT_ROOT=/scratch/$USER/ckpts,VULKAN_FIX=1,SAFEPATH=1,ARM=dense,MODEL=run2,SEED=0,STEPS=300000 \
  jobs/snellius_maniskill_sac.job
```

`SCRATCH_ROOT` moves the caches (`MS_ASSET_DIR`, `HF_HOME`, `TORCH_HOME`,
`SAPIEN_CACHE_DIR`) and the run dirs; `CKPT_ROOT` holds the consolidated RoboRef
checkpoints. If Triton fails partway into a run with `Python.h: No such file`, export a
`CPATH` with Python 3.12 headers before submitting (hipster used EESSI's, see
`hipster_maniskill_sac.job`). Runs that need more than the walltime resume with
`RESUME_FROM=<run>/checkpoints/<step>`; the replay buffer is not restored, so `STEPS` is
the remaining budget.

## 9. Still valid from the previous handoff

* **Truncation is bootstrapped, termination is not**: `base_replay_buffer.py:778` does
  `done = done * (1 - truncated)`, so running out of time never looks like success.
* **CPU physics is the recommended backend.** GPU (`env.env_kwargs.sim_backend=physx_cuda`)
  is equivalent at `num_envs=4` and not faster; `num_envs=16` collapsed LiftPegUpright
  because more envs at a fixed step budget means fewer updates per transition. The GPU
  wrapper merges `final_info` back, without which a GPU-trained policy scored 0/64.
* **Operational.** Keep SLURM logs and run dirs off `$HOME` (each `.err` reached
  150-160 MB; the 200 GiB quota killed runs twice). `sbatch --export` splits on commas,
  so Hydra list overrides arrive truncated. `ROBOMETER_DISABLE_UNSLOTH=1` for
  inference-only checkpoint loading. `moviepy` is required at the first eval.
* `RPL_LOG_REWARD=1` prints `gt_in / vlm / final_train_reward` for the first 30 steps,
  which proves no GT reward leaks into a reward-model arm.

| script | question it answers |
|---|---|
| `verify_maniskill_env.py` | is the environment importable and the vector path sane |
| `verify_with_maniskill_ppo.py` | does a known-good policy score the same through our stack |
| `verify_gpu_equivalence.py` | per-task CPU vs GPU with binomial error bars |
| `eval_actor_ckpt.py` | GT success of one saved actor, no reward model |
| `causal_calib_maniskill.py` | reward-model scores along rollouts of saved actors (B) |
| `analyze_episode_log.py` | on-policy metrics from one run's `episodes.jsonl` |
