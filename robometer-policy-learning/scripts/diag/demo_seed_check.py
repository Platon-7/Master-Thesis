"""Is low replay fidelity caused by initial-state mismatch, or by physics divergence?

Compares the object pose after reset(seed=episode_seed) against the demo's own
env_states[0]. If they differ, seed-based reset does not reproduce the demo start
(the demos were generated with num_envs=1024 on physx_cuda) and replay-by-action is
meaningless regardless of physics. The fix is then to SET the first env state.
"""
import os, sys, json
os.environ.setdefault("MUJOCO_GL","egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, h5py, torch
import gymnasium as gym, mani_skill.envs  # noqa
from mani_skill.trajectory import utils as traj_utils
from robometer_policy_learning.envs.maniskill_utils import get_task_spec

task = sys.argv[1] if len(sys.argv)>1 else "PokeCube-v1"
src  = sys.argv[2]
spec = get_task_spec(task)
f = h5py.File(src,"r"); meta = json.load(open(src.replace(".h5",".json")))
eps = {e["episode_id"]: e for e in meta["episodes"]}

env = gym.make(task, num_envs=1, obs_mode="state", control_mode=spec.control_mode,
               sim_backend="physx_cpu", max_episode_steps=spec.max_episode_steps,
               reward_mode="normalized_dense", render_mode="rgb_array")
u = env.unwrapped
names = sorted(f.keys(), key=lambda s:int(s.split("_")[1]))[:8]
print(f"{task}: comparing reset(seed) start-state vs demo env_states[0]\n")
print(f"{'traj':8s} {'seeded start (xy)':26s} {'demo start (xy)':26s} {'delta':>8}")
mism=0
for tn in names:
    eid=int(tn.split("_")[1]); ep=eps[eid]
    env.reset(seed=ep.get("episode_seed"))
    st = traj_utils.dict_to_list_of_dicts(f[tn]["env_states"])[0]
    # first movable actor that isn't the table
    key = next(k for k in st["actors"] if "table" not in k.lower())
    demo_xy = np.asarray(st["actors"][key])[:2]
    live = u.get_state_dict()
    lv = np.asarray(live["actors"][key]).reshape(-1)[:2]
    d = float(np.linalg.norm(demo_xy-lv))
    mism += d > 0.01
    print(f"{tn:8s} {str(np.round(lv,4)):26s} {str(np.round(demo_xy,4)):26s} {d:8.4f}")
print(f"\n  mismatched starts: {mism}/{len(names)}")
print("  => " + ("SEED RESET DOES NOT REPRODUCE THE DEMO START -- must set env_states[0]"
                 if mism else "starts match; low fidelity is genuine physics divergence"))
