# maniskill_rl environment — transfer notes

Target: SAME cluster (Snellius), DIFFERENT account. This is the easy case — the
CUDA driver, glibc and CPU arch are identical, so every compiled extension
(torch 2.13+cu130, sapien 3.0.3, physx) works as-is. Only paths change.

> **READ THIS FIRST — the code moved on after this env was packed.**
> The venv itself is unchanged and still correct. What changed is the repo: new
> reward knobs, new instrumentation, and — most importantly — a config trap that
> will silently give you a different RL algorithm if you miss it. See
> "§ What changed since this tar was built" below before launching anything.
> If this file and the copy in the repo ever disagree, **the repo copy wins**
> (`robometer-policy-learning/REBUILD.md`).

## Contents
* `maniskill_rl.tar.zst` — the venv (Python 3.12.4, ~6.8 GB raw)
* `requirements_frozen.txt` — exact pip freeze (194 pkgs), fallback only

## Install (3 steps)

```bash
mkdir -p /scratch-shared/$USER/envs
tar --zstd -xf maniskill_rl.tar.zst -C /scratch-shared/$USER/envs
# re-point the editable installs at YOUR checkout (they reference the old account's home)
/scratch-shared/$USER/envs/maniskill_rl/bin/python -m pip install -e /path/to/Master-Thesis/Robometer --no-deps
/scratch-shared/$USER/envs/maniskill_rl/bin/python -m pip install -e /path/to/Master-Thesis/robometer-policy-learning --no-deps
```

That is genuinely all. `bin/python` is a symlink to the cluster's Anaconda
(`/sw/arch/RHEL9/EB_production/2024/software/Anaconda3/2024.06-1/bin/python3`), which
every account can read, and `pyvenv.cfg` points there too. Python resolves
`sys.prefix` from the executable's location, so the venv does not care where it lives.

## Why no shebang rewriting is needed
The 60 console scripts in `bin/` (pip, wandb, tensorboard, ...) DO carry the old
absolute path in their shebang. They only matter if you invoke them BY NAME. All job
scripts call `$ENV_PREFIX/bin/python -u <script>`, which ignores shebangs entirely.
If you want the CLIs back:

```bash
cd /scratch-shared/$USER/envs/maniskill_rl
sed -i "1s|^#!.*python.*|#!$PWD/bin/python|" bin/* 2>/dev/null
```

Or just use `python -m pip`, `python -m wandb`, etc.

## Verify
```bash
export MS_ASSET_DIR=/scratch-shared/$USER/maniskill_assets
export HF_HOME=/scratch-shared/$USER/hf_cache
export SAPIEN_CACHE_DIR=/scratch-shared/$USER/sapien_cache
export MUJOCO_GL=egl ROBOMETER_DISABLE_UNSLOTH=1
python scripts/verify_maniskill_env.py --task PullCube-v1     # needs a GPU node
```
Expect all checks PASS, including "robometer not shadowed" and "vectorized rollout path".

## Also copy across (not in this archive)
* `maniskill_assets/` — task assets, ManiSkill demos + PPO checkpoints, ICL demo
  banks, and the cached calibration trajectories
* `roboref_checkpoints/` — run1/run2/run3 reward models
* HF cache for `Qwen/Qwen3-VL-4B-Instruct`, `robometer/Robometer-4B`,
  `facebook/dinov2-base` (or let them re-download)

## Non-obvious requirements
* `moviepy` + `imageio-ffmpeg` — the logger kills the job at the FIRST eval without them.
* `transformers` pinned <5.0 (4.57.6 here); 5.x breaks the Qwen3-VL processor path.
* Point `MS_ASSET_DIR`, `HF_HOME`, `SAPIEN_CACHE_DIR` at scratch, never `$HOME` —
  the 200 GiB home quota filled twice and killed runs mid-training.
* `ROBOMETER_DISABLE_UNSLOTH=1` for inference-only checkpoint loading.

---

# What changed since this tar was built

Work continued on a second cluster ("hipster": AlmaLinux 8.10, L4 GPUs). **None of
the env changed** — the package set there was rebuilt from this very
`requirements_frozen.txt`. What changed is the repo, plus what we now know about
which configuration actually works.

Get the code:

```bash
git fetch origin
git checkout maniskill-hipster-instrumentation      # or main, once merged
```

Two commits matter. `106913b` added three files that existed only on the old
account and had never been committed (`envs/maniskill_wrapper.py`,
`envs/maniskill_utils.py`, `configs/maniskill_online_rl.yaml`) — without them
nothing ManiSkill runs at all. The instrumentation branch adds the rest below.

## ⚠ TRAP 1 — the canonical recipe is NOT in the yaml

`maniskill_online_rl.yaml` still inherits `algorithm/sac.yaml`'s LIBERO-tuned
values: **gamma=0.99, tau=0.005, batch_size=128, learning_starts=5000**. The
numbers in `MANISKILL_HANDOFF.md` ("canonical recipe") were passed as CLI
overrides at submit time and were never written into the config.

`jobs/maniskill_sac.job` does **not** add them either. So unless you pass them
explicitly you are running a different algorithm from every published result:

```bash
online_algorithm.gamma=0.95 online_algorithm.tau=0.01 \
online_algorithm.batch_size=1024 online_algorithm.learning_starts=4000
```

(`gamma=0.95`, not the old 0.8 — see "what works" below.)

## ⚠ TRAP 2 — do NOT port the `hipster_*.job` scripts

`jobs/hipster_maniskill_sac.job`, `hipster_verify.job` and
`hipster_causal_calib.job` are in the repo, but they carry workarounds that are
specific to that cluster and are wrong or pointless on Snellius:

| hack | why it exists on hipster | on Snellius |
|---|---|---|
| `LD_LIBRARY_PATH` → extracted NVIDIA driver + conda-forge vulkan loader | its compute nodes ship no Vulkan userspace at all | harmful |
| `VK_ICD_FILENAMES` → hand-built ICD json | same | harmful |
| `CPATH` → EESSI Python headers | system python3.12 has no dev headers, so Triton's JIT fails mid-run | unnecessary |
| `--partition=capacity`, `/scratch/$USER` | different scheduler + filesystem layout | wrong |

Use `jobs/maniskill_sac.job` (gpu_a100, `/scratch-shared`) and add just one line
after `RUN_DIR` is set, to enable the new instrumentation:

```bash
export RPL_EPISODE_LOG="$RUN_DIR"
```

## ⚠ TRAP 3 — check for the shadowed `robometer` package

On hipster, `python scripts/train.py` run from the repo root imported the stale
in-repo `robometer/` submodule (which has no `robometer.data`) instead of the
real fork at `Master-Thesis/Robometer`, because Python prepends the cwd to
`sys.path`. The old account did not hit this, so it may be fine for you — but
verify once, on arrival:

```bash
python -c "import robometer, robometer.data.dataset_types as d; print(robometer.__file__)"
# must print .../Master-Thesis/Robometer/robometer/__init__.py
```

If it prints the in-repo submodule path instead, export `PYTHONSAFEPATH=1` in the
job script.

## New knobs (`reward_model/robometer.yaml`)

* `progress_beta` — `beta*progress + (1-beta)*success_prob`, the same mix as
  `vlm_ibrl`'s `robometer_beta`. **1.0 (default) = pure progress head, unchanged
  behaviour.** 0.0 = pure success head.
* `progress_binarize_threshold` — binarises that mix exactly as vlm_ibrl does
  (`reward = 1.0 if mixed > thr else 0.0`). `null` (default) = off.
  `progress_beta=0.0` + this set IS the MetaWorld/Robomimic/LIBERO recipe.

## New instrumentation (`RPL_EPISODE_LOG`)

Writes one JSON record per training episode to `$RUN_DIR/episodes.jsonl`:
`vlm_return`, `gt_solved_anytime`, `gt_first_solve_step`, unthrottled
`score_per_step` / `sp_per_step` / `reward_per_step` / `gt_per_step`, detector
fields (`fired`, `fire_step`, `gt_solved_at_fire`, `gate_suppressed` — null when
detection is off), and threshold provenance.

It runs in **every** regime, including dense no-termination where no detector
ever fires — that regime is where reward overoptimisation actually happens, and
it is invisible to a fired/not-fired metric. Because the per-step scores are
stored raw, any threshold-dependent metric can be recomputed offline without
re-running anything.

Analyse with:

```bash
python scripts/analyze_episode_log.py "$RUN_DIR/episodes.jsonl" --window 500
```

which reports `d'_onpolicy`, `farm_ratio`, `rho` (dense deployments),
FP/TP/miss/lead_time (detector deployments), and an FP sweep over a threshold
grid. Sanity-check the code any time with `python tests/test_episode_log.py`
(30 assertions, no GPU, ~5 s).

## What actually works — use this as the starting config

On PullCube-v1 with `run2_noicl_ours_step4000`, 300k steps, 5 seeds per model:

| model | eval success |
|---|---|
| **run2** (progress head) | **92%** and **85%** — 2 of 5 seeds; 3 seeds flat at 0% |
| **Robometer-4B base** | **0%** on all 5 seeds (max 5%, on 1–3 of 30 evals) |

Previous best on this benchmark was 15%. The configuration:

```
ARM=dense                                   # progress head, replaces env reward
reward_model.use_success_detection=false    # NO VLM termination
env.terminate_on_success=false              # no GT leak through `done`
env.reward_shift=0.0                        # raw positive reward, NOT -1
online_algorithm.gamma=0.95                 # NOT 0.8
online_algorithm.tau=0.01  batch_size=1024  learning_starts=4000
training.num_envs=4                         # CPU physics
```

Things that were tried and failed, so you don't repeat them:

* **`success_detection_threshold=0.10`** (suggested by `NEXT_EXPERIMENTS.md` #1) is
  catastrophic — every fire was false (`false_rate=1.00`) at step 2–7 of a 50-step
  episode, and all five arms went to 0%. Causal calibration on GT-policy rollouts
  puts the correct threshold at **0.6875** (TPR 1.00, FPR 0.00). The old 0.65 was
  right all along; the head reads 0.797–0.852 on true successes.
* **`min_ep_steps=15`** would also be wrong — real successes fire at steps 6–12.
* **`progress_beta=0.0`** (success head as the reward) gives 0% from scratch: until
  the policy succeeds, `success_prob` is ~0.08 everywhere, so the reward is a
  near-constant with no gradient. It needs a BC warm start, which is why it works
  under IBRL on the other three benchmarks and not here.
* **`progress_as_potential=true` alone** cannot work: with
  `add_estimated_reward=false` the shaping term is the ONLY reward, and potential
  shaping is policy-invariant by construction — it encodes no preference for
  finishing.
* **raising `target_entropy`** did not rescue a dead seed (flat through 184k).

Seed variance is the main open weakness: only ~2 of 5 seeds take off, and the
failures show the farming signature (VLM reward climbing 2.8 → 8.0 while GT
success stays 0%, versus ~9.0 for seeds that actually solve the task). The
reward's global optimum is correctly ordered — solving pays more than farming —
so this is an RL optimisation problem, not reward misspecification.

## Second task

`LiftPegUpright-v1` is the only other viable task found. GT screening at 300k:
LiftPeg 65% and still climbing (75–95% by 750k), PokeCube 40–45% and noisy,
PlaceSphere / PullCubeTool / RollBall all **0%** — unusable.

LiftPeg needs **750k steps**, not 300k: ground truth does not leave 0% until
~110k. run2 runs there were still at 0% at 275k and showing the farming pattern;
that work was unfinished at the time of writing.
