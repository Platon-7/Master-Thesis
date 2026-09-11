#!/usr/bin/env python3
"""Convert ManiSkill demonstrations into the robomimic-style h5 this repo's
offline buffer expects, regenerating observations by replay.

ManiSkill ships demos with obs_mode="none": actions + env_states only, no
observations. Our BC actor consumes {state, image, dino_embedding, language}, so
the trajectory has to be re-executed inside the SAME wrapper stack training uses,
otherwise the BC inputs would not match the RL inputs.

Replay is by ACTION rather than by set_state_dict, for two reasons: it exercises
the exact wrapper/observation path used online, and it is self-validating -- the
demos were generated on physx_cuda while we run physx_cpu, so a trajectory that
no longer succeeds under our physics is one we should not clone. Only replays
that still report success are written.

Output layout (h5_replay_buffer.H5ReplayBuffer):
    /data/{demo}/actions          (T, A)
    /data/{demo}/obs/{key}        (T, ...)   -- obs[t] pairs with actions[t]
    /data/{demo}/rewards          (T,)
    /data/{demo}/dones            (T,)
"""
from __future__ import annotations
import argparse, json, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, h5py, torch



def _base_env(env):
    """Walk the wrapper/vector stack down to the ManiSkill env that owns set_state_dict.

    make_env returns SyncVectorEnv(ManiSkillSingleEnvWrapper(...)) plus language /
    DINO wrappers, so neither .unwrapped nor .envs[0] alone reaches it.
    """
    seen, stack = set(), [env]
    while stack:
        e = stack.pop(0)
        if id(e) in seen:
            continue
        seen.add(id(e))
        if hasattr(e, "set_state_dict") and hasattr(e, "get_state_dict"):
            return e
        for attr in ("unwrapped", "env", "_env"):
            if hasattr(e, attr):
                stack.append(getattr(e, attr))
        if hasattr(e, "envs"):
            stack.extend(list(e.envs))
    raise RuntimeError("could not locate a ManiSkill env exposing set_state_dict")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--demo-h5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-demos", type=int, default=100)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--max-attempts", type=int, default=0,
                    help="trajectories to try (0 = 3x num-demos)")
    args = ap.parse_args()

    from robometer_policy_learning.envs.maniskill_utils import get_task_spec
    from robometer_policy_learning.utils.env_utils import make_env
    from robometer_policy_learning.rollouts.rollout_worker import extract_info_for_env
    from mani_skill.trajectory import utils as traj_utils
    from transformers import AutoImageProcessor, AutoModel

    spec = get_task_spec(args.task)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
    dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    # identical wrapper stack to training (proprio + DINO + language)
    env, _ = make_env(
        env_name=f"maniskill/{args.task}", num_envs=1, chunk_size=None,
        max_episode_steps=spec.max_episode_steps, use_full_state=False,
        dinov2_model=dino, dinov2_processor=dproc, device=str(dev),
        terminate_on_success=False,
        env_kwargs={"sim_backend": "physx_cpu", "image_size": args.image_size,
                    "control_mode": spec.control_mode, "reward_mode": "normalized_dense"},
    )
    instr = env.get_language_instruction()
    base = _base_env(env)

    f = h5py.File(args.demo_h5, "r")
    meta = json.load(open(args.demo_h5.replace(".h5", ".json")))
    eps = {e["episode_id"]: e for e in meta["episodes"]}
    traj_names = sorted(f.keys(), key=lambda s: int(s.split("_")[1]))
    max_attempts = args.max_attempts or args.num_demos * 3

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out = h5py.File(args.out, "w")
    data = out.create_group("data")

    kept = attempted = 0
    for tn in traj_names:
        if kept >= args.num_demos or attempted >= max_attempts:
            break
        attempted += 1
        eid = int(tn.split("_")[1])
        ep = eps.get(eid)
        if ep is None:
            continue
        acts = np.array(f[tn]["actions"])
        obs, _ = env.reset(seed=ep.get("episode_seed"))
        # reset(seed) does NOT reproduce the demo's start: these demos were generated
        # with num_envs=1024 on physx_cuda and the object lands 5-12cm away here
        # (measured 8/8 mismatched). Set the recorded first state, then replay actions.
        st0 = traj_utils.dict_to_list_of_dicts(f[tn]["env_states"])[0]
        base.set_state_dict(st0)
        # The obs returned by reset() describes the PRE-set state, so it cannot be
        # recorded. Burn action[0] to obtain an observation produced by the real
        # dynamics from the demo's start; recording then begins at t=1, costing one
        # transition out of 50 and keeping every (obs, action) pair correctly aligned.
        obs, _r0, _te0, _tr0, _in0 = env.step(acts[0][None, :])
        start_t = 1
        rec = {k: [] for k in obs.keys()}
        rewards, dones, solved = [], [], False
        for t in range(start_t, len(acts)):
            for k, v in obs.items():
                arr = np.asarray(v)
                rec[k].append(arr[0] if arr.ndim and arr.shape[0] == 1 else arr)
            a = acts[t][None, :]
            obs, r, term, trunc, infos = env.step(a)
            info_i = extract_info_for_env(infos, 0, 1)
            s = bool(info_i.get("success", False))
            solved = solved or s
            rewards.append(float(np.asarray(r).reshape(-1)[0]))
            dones.append(bool(s))
            if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                break
        if not solved:
            continue                                  # physics diverged -- don't clone it
        g = data.create_group(f"demo_{kept}")
        n = len(rewards)
        # ManiSkill records the PRE-CLIP Gaussian sample, so ~5% of demo actions fall
        # outside the [-1,1] action space (observed range [-3.05, 1.93]). The env
        # executed the clipped action, so the clipped value is what actually produced
        # the next observation -- and atanh(|a|>1) is undefined, which makes the BC
        # NLL loss NaN. Clip just inside the bound to keep the tanh inverse finite.
        _a = np.clip(acts[start_t:start_t + n], -1.0 + 1e-6, 1.0 - 1e-6)
        g.create_dataset("actions", data=_a.astype(np.float32))
        g.create_dataset("rewards", data=np.array(rewards, dtype=np.float32))
        g.create_dataset("dones", data=np.array(dones, dtype=bool))
        og = g.create_group("obs")
        for k, vals in rec.items():
            a = np.stack(vals[:n])
            if k == "language":
                og.create_dataset(k, data=np.array([instr] * n, dtype=h5py.string_dtype()))
            else:
                og.create_dataset(k, data=a, compression="gzip" if a.nbytes > 1e6 else None)
        g.attrs["num_samples"] = n
        kept += 1
        if kept % 10 == 0:
            print(f"  kept {kept}/{args.num_demos} (attempted {attempted})", flush=True)
    out.attrs["env_id"] = args.task
    out.attrs["instruction"] = instr
    out.close(); f.close()
    print(f"\nDONE {args.task}: kept {kept} of {attempted} attempted "
          f"(replay success rate {kept/max(attempted,1)*100:.0f}%)")
    print(f"wrote {args.out}  ({os.path.getsize(args.out)/1e6:.0f} MB)")
    return 0 if kept > 0 else 3


if __name__ == "__main__":
    sys.exit(main())
