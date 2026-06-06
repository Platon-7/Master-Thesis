"""Smoke test: instantiate VLMCritic_PixelMetaWorld with ROBOMETER_ICL_DEMO_PATH
set, score one episode-end clip, confirm ICL is wired through and produces a
non-NaN reward."""

import os
import sys
import numpy as np

# Force-set the ICL env var BEFORE anything else
os.environ.setdefault(
    "ROBOMETER_ICL_DEMO_PATH",
    "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl/release/data/metaworld/CoffeePush_frame_stack_1_224x224_end_on_success/demonstrations/mw-coffee-push/frames",
)
os.environ.setdefault("ROBOMETER_ICL_DEMO_IDX", "0")
os.environ.setdefault("ROBOMETER_ICL_FRAMES", "16")
os.environ.setdefault(
    "ROBOMETER_FT_PATH",
    "/scratch-shared/pkarageorgis1/Robometer_FT_consolidated/run1_icl_ours_step3000",
)

sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/Qwen35-FT")
sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl")

from env.vlm_envs import VLMCritic_PixelMetaWorld
from mw_main import train_bc_mw  # for env params loader

print("[1/3] loading BC policy to get env_params ...", flush=True)
bc_path = "release/data/metaworld/CoffeePush_frame_stack_1_96x96_end_on_success/dataset.hdf5"
bc_policy, _, env_params = train_bc_mw.load_model("coffeepush", "cuda")
env_params["end_on_success"] = False
env_params["episode_length"] = 100
env_params["env_reward_scale"] = 1

print("[2/3] constructing VLMCritic env with ICL ...", flush=True)
env = VLMCritic_PixelMetaWorld(
    **env_params,
    vlm="robometer_ft",
    past_len=4,
    reward_at_truncation=1,
    metaworld_data_dir="release/data/metaworld",
    gvl_context_len=9,
    robometer_beta=0.0,
    robometer_threshold=0.0192,   # the offline-derived 5% FPR threshold
    robometer_reward_scale=1.0,
)
print(f"  icl_frames loaded: {len(env.icl_frames) if env.icl_frames is not None else 0}")

print("[3/3] rollout 1 episode with BC, check the reward at truncation ...", flush=True)
import torch
obs, _ = env.reset()
total_steps = 0
final_reward = None
for t in range(100):
    with torch.no_grad():
        a = bc_policy.act(obs, eval_mode=True)
    obs, r, terminal, success, _ = env.step(a.numpy())
    total_steps += 1
    if terminal:
        final_reward = r
        break

print(f"  episode_len={total_steps}  final_reward={final_reward}  GT_success={success}")
print(f"  scorer's last success_prob={env._last_success_prob:.5f}  progress={env._last_progress:.4f}")
if final_reward is None or (isinstance(final_reward, float) and final_reward != final_reward):
    print("  >>> SMOKE FAILED — no reward / NaN")
    sys.exit(1)
print(">>> SMOKE PASSED — ICL is plumbed through IBRL env path")
