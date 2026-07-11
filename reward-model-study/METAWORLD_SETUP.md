# MetaWorld RL setup (environment / entrypoint layer)

Companion to `ROBOMIMIC_BASELINES_HANDOFF.md` (which has the full per-baseline
recipes). This documents the MetaWorld side: where the code lives, how to launch,
and which env vars control behavior. Written 2026-07-08; source: user's canonical
spec + the interval-hop shaping addition of the same date.

## Repo / env

- **Working dir:** `/shared/home/PKA4388/Master-Thesis/vlm_ibrl_v3`
  (NOT `vlm_ibrl` — that is the older MetaWorld-**v2** / mujoco-py branch kept for
  IBRL's original assumptions. v3 uses Farama MetaWorld v3 on new mujoco bindings.)
- **Conda env:** `/shared/home/PKA4388/miniconda3/envs/demo2reward_v3`
  (gymnasium + MetaWorld v3; mujoco 3.1.1). Activate via the job pattern:
  `export IBRL_CONDA_ENV=.../demo2reward_v3; source set_env.sh; export MUJOCO_GL=egl`.
  No mujoco-py / `~/.mujoco/mujoco210` needed (that's v2-only).

## RL algorithm / entrypoint

IBRL (Imitation-Bootstrapped RL):

```bash
python mw_main/train_rl_vlm_mw.py --config_path release/cfgs/metaworld/ibrl_long.yaml \
    --episode_length 200 --bc_policy <task> --vlm <vlm_name> \
    --num_train_step 40000 --seed <s> --reward_at_truncation <0|1> --save_dir <dir>
```

- Tasks: `coffeepush`, `boxclose` (also `assembly`, `stickpull`).
- `--vlm` ∈ {`robometer_ft`, `robometer_4b`, `qwen35_ft`, `roboreward_4b`,
  `robodopamine_4b`, `lrm_progress_8b`, `demo2reward_qwen3_8b`, …}.
- FT checkpoints: `/shared/home/PKA4388/checkpoints/Robometer_FT_consolidated/`
  via `ROBOMETER_FT_PATH` (Qwen3.5: `QWEN35_FT_PATH`).

## Reward-model env (the key file)

`env/vlm_envs.py` → `VLMCritic_PixelMetaWorld` (wraps MetaWorld with a frozen VLM
reward). Under it: `env/metaworld_wrapper.py` (`MetaWorldEnv`, v3) with
**dual-render**: the policy sees zoomed corner2 (`V3_CORNER2_ZOOM=1`), the reward
model sees `corner2_default` (`ROBOMETER_REWARD_CAMERA=corner2_default`) — the
validated scoring camera. Per-VLM scorers: `env/{robometer,roboreward,robodopamine,
lrm}_utils.py`. Scorers want **PIL images** — numpy arrays silently score ~0.

## Behavior env vars (no code edits)

- `AUTONOMOUS_SUCCESS`, `ROBOMETER_DETECT_HEAD`, `ROBOMETER_SUCCESS_THRESHOLD`,
  `ROBOMETER_SUCCESS_CONSECUTIVE`, `ROBOMETER_MIN_EP_FRAC` (gate),
  `ROBOMETER_DYNAMIC_THR` / OTSU / demo-anchor knobs — autonomous success detection.
- `LRM_CALL_INTERVAL`, `LRM_REWARD_MODE=hold|delta` — LRM dense delivery.
- `ROBODOPAMINE_*` (stride etc.) — RoboDopamine dense delivery.
- **`RBM_CALL_INTERVAL` (added 2026-07-08)** — interval-hop shaping for the
  Robometer family, mirroring the `_rd`/`_lrm` delivery: score the last
  `RBM_LASTK` (default 4) frames every N steps, decode **condMean** from the C51
  bins (capture added to `env/robometer_utils.py`), pay the potential hop
  `prog_t − prog_{t−interval}` (weight `RBM_DELTA_WEIGHT`); at truncation add the
  sparse success-head anchor when `RBM_ANCHOR=1` (default). `0` = disabled →
  stock behavior. Non-autonomous: episodes end on timeout only.
  Also fixed same date in `robometer_utils.py`: C51 loss-type match
  (`c51_asymmetric` was falling to the raw-logit path → garbage progress).

## SLURM

`jobs/*.job` (account `rob-tme-gaia`; 8B models → L40S-48GB only). The stacking
job is `jobs/mw_rbmhop_aws.job` (L40S-pinned, dual-render vars + episode_length
set; knobs via `--export`). Run outputs: `/shared/home/PKA4388/vlm_ibrl_runs/
<name>/train.log` — **the GT eval metric is the `score/score` line** (the reward
model is NOT used at eval; clean GT env).

## Stacking arms (the dense-labels experiment)

- (a) sparse-only reference = the existing money-plot runs (multi-seed).
- (b) sparse + our shaping: `VLM_NAME=robometer_ft`, `RBM_CALL_INTERVAL=10`.
- (c) control: `VLM_NAME=robometer_4b`, same shaping vars — isolates whether the
  dense failure labels (not the delivery trick) carry the effect.
