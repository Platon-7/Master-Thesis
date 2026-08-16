# ManiSkill + RoboRef: state, fixes, and how to continue

Written to move this work to another cluster. Everything below is measured, not
assumed; where something is unverified it says so.

## 1. What works now

Ground-truth SAC on ManiSkill, proprio+DINO observation (the LIBERO/IBRL-equivalent
input), 500k steps, `num_envs=4`, CPU physics:

| task | GT eval success | verdict |
|---|---|---|
| PullCube-v1 | **100%** (stable 98-100 over 13 evals) | benchmark |
| LiftPegUpright-v1 | **88%** and still rising at 500k | benchmark |
| PokeCube-v1 | ~42%, noisy | marginal |
| RollBall-v1 | ~0% | unusable as-is |

## 2. The bug that mattered: dense reward + terminate-on-success

ManiSkill's `normalized_dense` reward is POSITIVE (~+0.9/step near the goal) and the
episode ENDS on success. Finishing therefore forfeits the rest of the episode's
reward. With `gamma=0.8` the continuation value of loitering is ~0.9/(1-0.8) = 4.5
against ~1 for finishing, so the optimal policy parks just outside the success
condition.

Measured on PullCube GT before the fix: episode return rose **4.71 -> 21.43** while
eval success fell **56% -> 10%** and `ent_coef` collapsed to 0.002. The policy was
optimising correctly; the reward was misspecified.

Why this never appeared in the other suites:

| suite | per-step GT reward | on success | loiter pays? |
|---|---|---|---|
| LIBERO | **-1** (`libero_pi0_wrapper.py:128`) | 0 | no, it costs |
| IBRL / MetaWorld | **0** (sparse) | +1 | no, earns nothing |
| ManiSkill | **+0.9** | +0.9, episode ends | **yes** |

IBRL additionally set `train_end_on_success=0` (no termination during training) and
used a BC anchor. ManiSkill was the first setup with all three ingredients.

**Fix**: treat dense rewards as costs. `env.reward_shift: -1.0` puts every step in
[-1, 0] via `RewardShiftTransform`, applied as a `post_transform` — i.e. to the FINAL
reward, AFTER `RobometerReplayBuffer` replaces the env reward with the VLM score.
LIBERO applied its `-1` in the env wrapper, where the reward model overwrote it, so
only its GT arm was ever protected.

Properties: with no early termination the shift is a constant offset on a fixed
horizon and changes nothing; with termination it penalises dithering, which is
exactly where it is needed.

**Truncation is bootstrapped, termination is not** (verified):
`base_replay_buffer.py:778` does `done = done * (1 - truncated)`, and SAC's target is
`r + (1 - done) * gamma * Q'`. So running out of time does NOT look like success.
This is essential with costs — otherwise timing out would carry no future cost.

## 3. Canonical recipe

```
env.use_full_state=false          # proprio + DINO, same as LIBERO/IBRL
env.terminate_on_success=true     # GT arm only; FALSE for reward-model arms
env.reward_shift=-1.0
online_algorithm.gamma=0.8        # 0.95 for RollBall (ManiSkill's own per-task value)
online_algorithm.tau=0.01
online_algorithm.batch_size=1024
online_algorithm.learning_starts=4000
training.num_envs=4
eval.eval_num_episodes=50
eval.eval_freq=25000
```

Reward-model arms must use `env.terminate_on_success=false`: if the environment ends
the episode on TRUE success, `done` leaks ground truth into the value function even
when the reward comes from the VLM.

## 4. Backend: CPU vs GPU physics

`ManiSkillGPUVectorWrapper` (`envs/maniskill_gpu_wrapper.py`) works; select it with
`env.env_kwargs.sim_backend=physx_cuda`.

Adapter correctness, ManiSkill's own trained PPO checkpoint through our stack,
256 episodes/cell:

| task | CPU | GPU |
|---|---|---|
| LiftPegUpright | 99.6% | 99.6% (identical) |
| PullCube | 48.0% | 58.2% |
| RollBall | 4.7% | 10.2% |
| PokeCube | 14.5% | 0.4% |

GPU matching exactly on one task and BEATING CPU on two proves the adapter does not
mangle observations, actions, or success reporting. The differences are genuine
CPU/GPU contact-solver differences; PokeCube is the most sensitive.

Training comparison at matched `num_envs=4`: PullCube CPU 100% / GPU 96-100%,
LiftPegUpright CPU 88% / GPU 78% — equivalent. But `num_envs=16` COLLAPSES
LiftPegUpright to 0%, because more envs at a fixed step budget means fewer gradient
updates per transition. Raising `num_envs` therefore requires raising updates per
step too. At `num_envs=4` GPU is not faster (13.0 vs 18.5 steps/s), so **CPU is the
recommended backend**.

GPU-specific bug fixed along the way: `ManiSkillVectorEnv` auto-resets with
gymnasium's OLD convention, putting the terminal `success` flag in `final_info`.
Gymnasium 1.3's `SyncVectorEnv` (the CPU path) returns it directly. The wrapper now
merges `final_info` back; without it a GPU-trained policy scored 0/64.

## 5. Success-threshold calibration (for the stop-on-detection regime)

`scripts/calibrate_success_threshold.py`: `collect` -> `score` -> `fit`.

Necessary because `success_detection_threshold` is applied to a raw probability, and
run2/run3 were trained with `bce_asymmetric`, which deliberately moves the operating
point. A shared 0.65 sits at a different false-positive rate per model, so policy
differences would partly reflect threshold placement rather than reward quality.
`fit` picks each model's threshold at a MATCHED false-positive rate (default 2%).

Cached trajectories (90 episodes each: 60 expert via ManiSkill's PPO checkpoint,
30 random) live in `$MS_ASSET_DIR/calibration/`. Results land in
`calibration/thresholds.json` and are copied to `results/` in this repo.

## 6. Throughput and budget

Measured, `num_envs=4`, CPU:

| arm | rate | 500k steps |
|---|---|---|
| GT | ~18-30 steps/s | 6-10 h |
| reward model (VLM in loop) | ~4.9-5.8 steps/s | ~28 h |

Snellius `gpu_a100` bills ~128 SBU per GPU-hour, so a reward-model run at 500k costs
~3,600 SBU. Budget the matrix before launching: 7 models x 3 seeds x 2 regimes at
500k is ~1,200 GPU-hours.

## 7. Operational notes

* SLURM logs must NOT go to `$HOME`. Each training `.err` reached 150-160 MB of tqdm
  output; home quota (200 GiB) hit 105% twice and killed runs mid-training, once as
  `[Errno 122] Disk quota exceeded` and once as `torch.save` writing truncated files
  (`unexpected pos X vs Y`). The job script writes run artifacts to
  `/scratch-shared/$USER/roboref_runs/...` via `hydra.run.dir`.
* `sbatch --export` splits on COMMAS, so a Hydra list override (`key=[a,b]`) arrives
  truncated. Use the `STATE_ONLY=1` style named knobs instead.
* Checkpoints bake `use_unsloth: true` into their `config.yaml`; set
  `ROBOMETER_DISABLE_UNSLOTH=1` for inference-only use.
* `moviepy` is required — the logger demands it at the first eval interval, minutes
  into a run, and kills the job rather than degrading.

## 8. Verification tools

| script | question it answers |
|---|---|
| `verify_maniskill_env.py` | is the environment importable and the vector path sane |
| `verify_with_maniskill_ppo.py` | does a known-good policy score the same through our stack |
| `verify_gpu_equivalence.py` | per-task CPU vs GPU with binomial error bars |
| `calibrate_success_threshold.py` | per-(model,task) success thresholds at matched FPR |
| `bench_maniskill_gpu.py` | transitions/s by backend and env count |

`RPL_LOG_REWARD=1` prints `gt_in / vlm / final_train_reward` for the first 30 steps —
use it to prove no ground-truth reward is leaking into a reward-model arm.
