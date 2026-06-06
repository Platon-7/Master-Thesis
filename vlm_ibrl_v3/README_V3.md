# vlm_ibrl_v3 — IBRL harness ported to MetaWorld v3

This is a copy of `vlm_ibrl/` whose MetaWorld environment renders **MetaWorld v3**
(Farama-Foundation, gymnasium + the modern `mujoco` bindings) instead of v2
(rlworkgroup metaworld 0.1.0 / mujoco_py / mujoco210).

**Why:** the Robometer reward model was trained/evaluated on MetaWorld **v3**
curated frames (ids `metaworld_*_v3_*`), but the original IBRL env rendered
**v2**. That rendering-domain mismatch is the leading suspect for the on-policy
reward collapse (live CoffeePush AUC ~0.53 vs curated v3 AUC ~0.97). This folder
lets the reward model be evaluated/used in its own training domain.

## What changed vs vlm_ibrl/
Only `env/metaworld_wrapper.py::MetaWorldEnv` was rewritten. Everything else
(the RL agent, all the other gym.Wrapper layers, vlm_envs, tools) is unchanged
and still runs on **old `gym` (0.26)** with the old 4-tuple interface —
`MetaWorldEnv` translates the gymnasium 5-tuple back to `(obs, reward, done,
info)` at that single boundary.

Key v2 -> v3 differences handled in `MetaWorldEnv`:
- env build: `metaworld.MT1("coffee-push-v3").train_classes[...]` +
  `env_cls(render_mode="rgb_array", camera_name=...)` + `set_task(...)`, then
  `_freeze_rand_vec = False` for per-reset goal randomization.
- step/reset: gymnasium 5-tuple `(obs, reward, terminated, truncated, info)`
  -> old `(obs, reward, done, info)`.
- scripted policy: `SawyerCoffeePushV3Policy` (the `*V3Policy` family).
- render: `env.mujoco_renderer.render("rgb_array", camera_name=cam)[::-1]`,
  native 480x480 resized to the requested size. NOTE: the v3 dataset generator
  switched cameras via a `renderer.camera_id` *attribute* that gymnasium 0.29.1
  ignores; cameras are switched here via the `camera_name` *argument* instead.
  The in-domain camera (the one the reward model actually saw) is determined
  empirically by `reward-model-study/scripts/render_match_v3.py`.

`release/` is a symlink to `../vlm_ibrl/release` (shared BC policies + demo
data; those are v2-rendered 96x96 demos — full v3 retraining of BC/IBRL would
need v3-rendered demos, not yet done).

## Conda env
`/gpfs/scratch1/shared/pkarageorgis1/envs/demo2reward_v3` — a clone of
`demo2reward` (full IBRL stack, torch 2.4 cu121, transformers 4.57.2) PLUS
`gymnasium==0.29.1`, `mujoco==3.1.1`, and editable Farama metaworld pinned to
`58e32b4d` (`MetaWorld/metaworld_repo`). `mujoco-py` was removed so gymnasium
falls back cleanly to the new `mujoco`. One env both renders v3 AND scores the
Qwen3-VL Robometer-FT (transformers 4.57.2 — does NOT collapse outputs).

Run with: `MUJOCO_GL=egl` (needs a GPU), `PYTHONPATH=<this dir>:<Robometer>`.
See `reward-model-study/jobs/diag_v3cam.job` and `render_match_v3.job`.
