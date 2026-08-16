# ManiSkill3 + SAC setup (Snellius)

Replaces the LIBERO downstream-RL benchmark. Two things make this different
from the MetaWorld/Robomimic runs in the thesis:

* **Plain SAC, trained from scratch.** No behaviour-cloning warm start, unlike
  IBRL. Nothing is pre-seeded, so a dense per-step reward is the only thing
  that can carry learning — which is what makes this a real test of the
  progress head rather than the success head.
* **ManiSkill3, not LIBERO.** LIBERO-90 is in the RoboRef training corpus, so
  the old comparison was unfair in our favour (the thesis says as much).

## Task constraint (enforced in code)

FailSafe contributed fault-injected versions of ManiSkill's **PickCube**,
**PushCube** and **StackCube** to the training corpus. Those three families are
in-distribution and are rejected at env-construction time by
`assert_task_allowed()` in `robometer_policy_learning/envs/maniskill_utils.py`
— a config typo cannot silently reintroduce them.

Shortlist (all outside that set, easiest first):

| Task | Kind | Notes |
|---|---|---|
| `PullCube-v1` | pull | Easiest. Direct analogue of the forbidden PushCube. Start here. |
| `PokeCube-v1` | push | Poke a cube to a goal with a peg. |
| `RollBall-v1` | push | Roll a ball to a goal region. |
| `LiftPegUpright-v1` | pick | Single object, simple goal. |
| `PlaceSphere-v1` | place | Place a sphere in a bin. |
| `PickSingleYCB-v1` | pick-and-place | Real object diversity — best VLM-reward story, harder. |
| `PullCubeTool-v1` | tool use | Hardest; keep as a stretch goal. |

## Environment

ManiSkill is deliberately **not** in the core dependency list: these runs need
neither LIBERO nor openpi nor unsloth, and a full `uv sync` is the fastest way
to hit an unrelated build failure. Build a lean env instead:

```bash
# on a compute node (needs a GPU visible for the renderer check)
python -m venv /scratch-shared/$USER/envs/maniskill_rl
source /scratch-shared/$USER/envs/maniskill_rl/bin/activate
pip install --upgrade pip
pip install "mani-skill>=3.0.0" torch torchvision gymnasium numpy \
            opencv-python-headless pillow hydra-core omegaconf loguru wandb \
            transformers sentence-transformers h5py tqdm rich \
            moviepy imageio-ffmpeg
# moviepy/imageio-ffmpeg are NOT optional: the logger raises
# "moviepy not found, videos cannot be logged" at the first eval interval, which
# is far enough into a run (minutes, after model load) to waste a node.
# the reward model itself:
pip install -e ./robometer
```

### Task assets

Some tasks ship no meshes and download them on first use, prompting
interactively (which hangs a batch job). `PickSingleYCB-v1` needs the YCB set.
Fetch it once, up front:

```bash
export MS_ASSET_DIR=/scratch-shared/$USER/maniskill_assets
python -m mani_skill.utils.download_asset ycb -y
```

`MS_ASSET_DIR` is exported by both job scripts and keeps ~25 MB of meshes off
the home quota. `PullCube-v1` / `PokeCube-v1` need no assets.

### Rendering — verified working

ManiSkill renders through SAPIEN. It *prefers* a CUDA device, but **it falls
back to CPU rendering and still produces correct frames** — the adapter smoke
test passes end-to-end on a Snellius login node with no GPU at all
(`WARNING - Requested to use render device "sapien_cuda", but CUDA device was
not found. Falling back to "cpu"`). So a driver problem degrades speed, not
correctness. Confirm on a GPU node anyway with `jobs/maniskill_smoke.job`,
since CPU rendering is far too slow for a real training run.

If GPU rendering does fail, check:

```bash
python -c "import mani_skill.envs, gymnasium as gym; \
e=gym.make('PullCube-v1', num_envs=1, obs_mode='state', render_mode='rgb_array', \
sim_backend='physx_cpu'); e.reset(); print(e.render().shape)"
```

If that fails with a Vulkan/device error, options in order of preference:
1. `module load` a driver/Vulkan module if Snellius provides one;
2. set `VK_ICD_FILENAMES` to the NVIDIA ICD JSON
   (usually `/usr/share/vulkan/icd.d/nvidia_icd.json`);
3. run inside the official ManiSkill container/Apptainer image.

The smoke test below catches a silently-broken renderer (it fails if frames
come back near-uniform), so it is worth running before any long job.

## Verify the adapter before training

```bash
python scripts/smoke_test_maniskill.py --task PullCube-v1 --save-frames /tmp/ms
```

This imports only gymnasium + ManiSkill + the wrapper (not the RL stack), so it
works in the lean env. It checks the forbidden-task guard, that observations
come back as unbatched numpy (ManiSkill natively returns *batched torch*, which
`SyncVectorEnv` cannot consume), that `info["success"]` is a real bool, and that
frames are non-degenerate.

**Verified passing** on `PullCube-v1`, `PokeCube-v1` and `PickSingleYCB-v1`
(mani-skill 3.0.1, sapien 3.0.3, gymnasium 1.3.0, torch 2.13.0+cu130):

| | observed |
|---|---|
| state | `(9,)` float32 = Panda qpos |
| image | `(224,224,3)` uint8, non-degenerate |
| action | `(4,)` for `pd_ee_delta_pos`, `(7,)` for `pd_ee_delta_pose` |
| reward / terminated / truncated | python `float` / `bool` / `bool` |
| `info["success"]` | python `bool` |
| vector env | batches to `(N,224,224,3)` / `(N,9)` |

Saved frames were inspected: right-side up, with the goal marker and the
manipulated object both in view. This is what validates the two deliberate
departures from `ImageDictObsWrapper` (no vertical flip — SAPIEN, unlike
MuJoCo, renders upright; and resize instead of center-crop — a 224 crop of
ManiSkill's wider frame cuts the goal marker out of the picture entirely).

## Running

`jobs/maniskill_sac.job` takes `TASK`, `ARM`, `SEED`, `STEPS`, `REWARD_CKPT`.
Remember Snellius drops inline env vars — use `--export`:

```bash
# 1. ceiling: ground-truth dense reward, no reward model.  RUN THIS FIRST.
sbatch --export=ALL,ARM=gt,TASK=PullCube-v1 jobs/maniskill_sac.job

# 2. sparse success head (the IBRL-style signal)
sbatch --export=ALL,ARM=succ,TASK=PullCube-v1,REWARD_CKPT=/path/to/roboref jobs/maniskill_sac.job

# 3. dense progress head (the point of this benchmark)
sbatch --export=ALL,ARM=prog,TASK=PullCube-v1,REWARD_CKPT=/path/to/roboref jobs/maniskill_sac.job
```

If arm 1 does not learn the task within the step budget, the task is unusable
as a benchmark and no reward-model result from it means anything — pick an
easier task or raise `STEPS` before spending compute on arms 2 and 3.

## Reward-model compatibility (checked)

* **Frame count needs no configuration.** `robometer_replay_buffer.py` reads
  `max_frames` from the *checkpoint's own* config
  (`getattr(reward_model_config.data, "max_frames", 16)`), so run2/run3 get 16
  and the released baseline gets 8 automatically. Nothing to set.
* **The asymmetric progress head is already handled.** The buffer treats any
  `c51*` progress loss type as discrete mode; matching only the literal
  `"discrete"` would leave those reads on the raw-logit path and produce
  garbage reward. Already fixed in-tree.
* **ICL (run1) works, but needs the right `robometer` install.** The mechanism
  already exists in the RoboRef fork: `ProgressSample.context_trajectory`
  (`robometer/data/dataset_types.py`) carries a demonstration, and the collator
  (`robometer/data/collators/rbm_heads.py`) emits
  `[demo frames] <|demo_end|> [query frames]` when it is populated — the same
  layout used at training time. `vlm_ibrl_v3/env/robometer_utils.py` is a
  working reference call site.

  **The catch:** the pinned `robometer` *submodule* is upstream and has **zero**
  `context_trajectory` references, while the `Robometer/` fork has them. Install
  the fork, not the submodule:

  ```bash
  pip install -e /gpfs/home4/$USER/DAS-5/Master-Thesis/Robometer
  ```

  `RobometerReplayBuffer` checks this at startup and raises a message pointing
  here rather than failing with an opaque pydantic error mid-rollout.

  Wiring added in this repo: `reward_model.icl_demo_path` loads a bank built by
  `scripts/generate_maniskill_icl_demos.py`; the buffer samples one demo per
  query and attaches it as `context_trajectory` (both the single and batched
  reward paths, online buffer only — the offline H5 buffer is untouched).

Local checkpoint (already fetched):
`/scratch-shared/$USER/roboref_checkpoints/run2_noicl_ours_step4000`
(verified: `max_frames: 16`, `c51_asymmetric` + `bce_asymmetric`, `lambda 0.3`,
`use_icl: false`).

## Known gaps (not yet done)

* `training.num_envs: 4` is a guess; tune it to the node once the sim runs.
* The `robometer` submodule must point at the RoboRef code, not upstream.
