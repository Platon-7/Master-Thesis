# Demo2Reward: IBRL with VLM rewards

[![Paper](https://img.shields.io/badge/Paper-%20%F0%9F%93%84-blue)](#)
[![Website](https://img.shields.io/badge/Website-%F0%9F%8C%90-orange)](#)

This repository is a fork of the official [IBRL codebase](https://github.com/hengyuan-hu/ibrl) by Hu et al. (2024), extended with **Demo2Reward**: test-time prompt optimization that turns a frozen Vision-Language Model into a reliable sparse-reward signal for IBRL/RLPD-style RL.

Use it to reproduce the main simulation experiments on RoboMimic and MetaWorld with the following reward sources: ground-truth, VLM-SD (zero-shot VLM), Demo2Reward, RoboReward, and GVL.

---

## Install

### 1. Clone

`pybind11` is a git submodule (required to build the C++ replay-buffer extension), so clone with `--recursive`:

```shell
git clone --recursive <REPO_URL>
cd <REPO_DIR>
```

If you already cloned without `--recursive`, run `git submodule update --init --recursive` from the repo root before continuing. (`install.sh` runs this for you, so this is only needed if you skip the script.)

### 2. MuJoCo

Download the MuJoCo 2.1 binaries for [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) and extract them to `~/.mujoco/mujoco210`. If you place them elsewhere, export `MUJOCO_PATH` before sourcing `set_env.sh`.

### 3. Conda env

```shell
conda create -n demo2reward python=3.9 -y
conda activate demo2reward
```

(If you use a different env name, export `IBRL_CONDA_ENV=<your_name>` before sourcing `set_env.sh`.)

### 4. Python dependencies

Either run the install script,

```shell
bash install.sh
```

or run the same commands manually:

```shell
# 1) PyTorch + CUDA 12.1
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 2) Hugging Face stack for Qwen3-VL (transformers from main; Qwen3-VL classes
#    are not yet in a tagged release)
pip install git+https://github.com/huggingface/transformers
pip install accelerate
pip install qwen-vl-utils==0.0.14

# 3) flash-attn build prerequisites
pip install packaging ninja

# 4) flash-attn. --no-build-isolation reuses the torch + ninja installed above
#    instead of pulling them into a build sandbox.
python -m pip install flash-attn==2.7.3 --no-build-isolation --no-cache-dir

# 5) Remaining project dependencies (includes the correct triton pin for torch 2.4)
pip install -r requirements.txt

# 6) Compile the C++ replay-buffer extension
make -C common_utils
```

### 5. Activate the shell

Every new shell needs:

```shell
source set_env.sh
```

This adds the repo root to `PYTHONPATH`, activates the conda env, and points `LD_LIBRARY_PATH` / `MUJOCO_PY_MUJOCO_PATH` at MuJoCo.

#### Troubleshooting

If you hit `ImportError: .../libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`, force conda to use the system C++ runtime (replace `PATH_TO_CONDA_ENV` with `$(echo ${CONDA_PREFIX:-"$(dirname $(which conda))/../"})`):

```shell
ln -sf /lib/x86_64-linux-gnu/libstdc++.so.6 PATH_TO_CONDA_ENV/lib/libstdc++.so
ln -sf /lib/x86_64-linux-gnu/libstdc++.so.6 PATH_TO_CONDA_ENV/lib/libstdc++.so.6
```

---

## Data and pre-trained BC policies

Download the dataset bundle and pre-trained BC policies from [Google Drive](https://drive.google.com/file/d/1F2yH84Iqv0qRPmfH8o-kSzgtfaoqMzWE/view?usp=sharing) and place them under `release/`. The final layout should be:

```
release/
├── cfgs/    # shipped with the repo
├── data/    # from the zip
└── model/   # from the zip
```

(Optional) Retrain BC policies from scratch:

```shell
# RoboMimic
python train_bc.py --config_path release/cfgs/robomimic_bc/can.yaml
python train_bc.py --config_path release/cfgs/robomimic_bc/square.yaml

# MetaWorld
python mw_main/train_bc_mw.py --dataset.path Assembly --save_dir <SAVE_DIR>
# repeat for BoxClose, CoffeePush, StickPull
```

### (Optional) Regenerate RoboMimic data with third-person frames

The shipped `processed_data96.hdf5` files contain only the **policy camera** (`robot0_eye_in_hand` for `PickPlace-Can`, `agentview` for `NutAssembly-Square`) at 96×96. That is sufficient to train any of the IBRL / RLPD policies in this repo — including the VLM-reward variants, since the VLM-critic env renders `agentview` live every step and never touches the preload data.

You only need to regenerate the data if you want **third-person demo frames cached on disk**, e.g. to add GVL on RoboMimic or to run a Demo2Reward prompt-optimization pass on `PickPlace-Can`. Both consume demo frames at 224×224.

1. Download the raw [RoboMimic v0.1 proficient-human demos](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html) and unpack them somewhere, e.g. `~/robomimic_raw/{can,square}/demo.hdf5`.
2. Regenerate the 224×224 demo file (both cameras, full resolution):

```shell
# PickPlace-Can
python tools/dataset_states_to_obs.py \
    --dataset ~/robomimic_raw/can/demo.hdf5 \
    --output_folder release/data/robomimic/can_224 \
    --camera_names agentview robot0_eye_in_hand \
    --image_size 224 --obs_size 224 \
    --done_mode 1 --copy_dones

# NutAssembly-Square
python tools/dataset_states_to_obs.py \
    --dataset ~/robomimic_raw/square/demo.hdf5 \
    --output_folder release/data/robomimic/square_224 \
    --camera_names agentview robot0_eye_in_hand \
    --image_size 224 --obs_size 224 \
    --done_mode 1 --copy_dones
```

This produces `release/data/robomimic/{can,square}_224/processed_data224.hdf5` with both cameras at 224×224. The shipped 96×96 file is unchanged and remains what the policy loads.

---

## Reproduce results

All runs below use the paper's sparse, end-of-episode reward setup (see Suppl. B.1):

- `--train_end_on_success 0`: don't terminate training episodes the moment the ground-truth success signal fires. Let them play out to the horizon to avoid leaking GT success.
- `--reward_at_truncation 1`: query the (VLM or GT) reward only once at the end of each episode, instead of every step.

The flags `--save_dir` and `--seed` are omitted below; pick whatever you like. Add `--use_wb 0` to any command to disable Weights & Biases logging.

### RoboMimic (IBRL)

Two tasks: `PickPlace-Can` (`can_ibrl.yaml`) and `NutAssembly-Square` (`square_ibrl.yaml`).

```shell
RM_CFG=release/cfgs/robomimic_rl/can_ibrl.yaml   # or square_ibrl.yaml
```

**Standard IBRL** (original dense-reward, early-termination setup from Hu et al. 2024) — for reference:
```shell
python train_rl.py --config_path $RM_CFG
```

**Ground-truth rewards** (sparse end-of-episode; upper bound for the VLM-reward runs below):
```shell
python train_rl.py --config_path $RM_CFG \
    --train_end_on_success 0 --reward_at_truncation 1
```

**VLM-SD** (zero-shot Qwen3-VL-8B as success detector):
```shell
python train_vlm_rl.py --config_path $RM_CFG --vlm vlm_sd_qwen3_8b \
    --train_end_on_success 0 --reward_at_truncation 1
```

**Demo2Reward** (optimized prompt):
```shell
python train_vlm_rl.py --config_path $RM_CFG --vlm demo2reward_qwen3_8b \
    --train_end_on_success 0 --reward_at_truncation 1
```

**RoboReward** baseline:
```shell
python train_vlm_rl.py --config_path $RM_CFG --vlm roboreward_8b \
    --train_end_on_success 0 --reward_at_truncation 1
```

### RoboMimic (RLPD)

For RLPD robustness on `PickPlace-Can`, swap the config to `can_rlpd.yaml`:

```shell
python train_vlm_rl.py --config_path release/cfgs/robomimic_rl/can_rlpd.yaml \
    --vlm demo2reward_qwen3_8b \
    --train_end_on_success 0 --reward_at_truncation 1
```

### MetaWorld (IBRL)

Four tasks via `--bc_policy`: `assembly`, `boxclose`, `coffeepush`, `stickpull`.

```shell
MW_CFG=release/cfgs/metaworld/ibrl_long.yaml
TASK=assembly   # or boxclose, coffeepush, stickpull
```

**Ground-truth rewards**:
```shell
python mw_main/train_rl_mw.py --config_path $MW_CFG --bc_policy $TASK \
    --train_end_on_success 0 --reward_at_truncation 1
```

**VLM-SD**:
```shell
python mw_main/train_rl_vlm_mw.py --config_path $MW_CFG --bc_policy $TASK \
    --vlm vlm_sd_qwen3_8b \
    --train_end_on_success 0 --reward_at_truncation 1
```

**Demo2Reward**:
```shell
python mw_main/train_rl_vlm_mw.py --config_path $MW_CFG --bc_policy $TASK \
    --vlm demo2reward_qwen3_8b \
    --train_end_on_success 0 --reward_at_truncation 1
```

**RoboReward**:
```shell
python mw_main/train_rl_vlm_mw.py --config_path $MW_CFG --bc_policy $TASK \
    --vlm roboreward_8b \
    --train_end_on_success 0 --reward_at_truncation 1
```

**GVL** (paper uses Qwen3-VL-32B; pre-loads 3 in-context demos from `release/data/metaworld/<TASK>_frame_stack_1_224x224_modem/dataset.hdf5`):
```shell
python mw_main/train_rl_vlm_mw.py --config_path $MW_CFG --bc_policy $TASK \
    --vlm gvl_qwen3_32b \
    --train_end_on_success 0 --reward_at_truncation 1
```

### MetaWorld (RLPD)

Swap the config to `rlpd_long.yaml`:

```shell
python mw_main/train_rl_vlm_mw.py --config_path release/cfgs/metaworld/rlpd_long.yaml \
    --bc_policy boxclose --vlm demo2reward_qwen3_8b \
    --train_end_on_success 0 --reward_at_truncation 1
```

---

## Citation

```bibtex
@misc{hu2023imitation,
    title={Imitation Bootstrapped Reinforcement Learning},
    author={Hengyuan Hu and Suvir Mirchandani and Dorsa Sadigh},
    year={2023},
    eprint={2311.02198},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

(Demo2Reward citation to be added.)
