#!/usr/bin/env python3
"""Generate BC demonstrations by running ManiSkill's PPO expert in OUR simulator.

ManiSkill ships trained PPO checkpoints alongside its demo trajectories. Replaying
the shipped trajectories fails for tasks whose success is fragile across physics
backends -- RollBall is an open-loop 1.4m shot into a 0.10m radius, and 0 of 300
recorded trajectories survived physx_cuda -> physx_cpu. Running the expert POLICY
instead sidesteps that entirely: every trajectory is generated under the physics we
train in, so it is correct by construction and there is no fidelity loss.

The expert consumes ManiSkill's flat state observation while our BC dataset needs
{state, image, dino_embedding}; both come from the same env, the former from the
base env and the latter from our wrapper stack.

Output matches scripts/convert_maniskill_demos.py exactly.
"""
from __future__ import annotations
import argparse, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, h5py, torch, torch.nn as nn


def build_ppo_actor(state_dict, obs_dim, act_dim):
    """ManiSkill's PPO baseline actor: Linear/Tanh x3 then a linear head."""
    net = nn.Sequential(
        nn.Linear(obs_dim, 256), nn.Tanh(),
        nn.Linear(256, 256), nn.Tanh(),
        nn.Linear(256, 256), nn.Tanh(),
        nn.Linear(256, act_dim),
    )
    sd = {k[len("actor_mean."):]: v for k, v in state_dict.items() if k.startswith("actor_mean.")}
    net.load_state_dict(sd)
    net.eval()
    return net


def _base_env(env):
    seen, stack = set(), [env]
    while stack:
        e = stack.pop(0)
        if id(e) in seen:
            continue
        seen.add(id(e))
        if hasattr(e, "set_state_dict") and hasattr(e, "get_obs"):
            return e
        for a in ("unwrapped", "env", "_env"):
            if hasattr(e, a):
                stack.append(getattr(e, a))
        if hasattr(e, "envs"):
            stack.extend(list(e.envs))
    raise RuntimeError("no base ManiSkill env with get_obs/set_state_dict")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-demos", type=int, default=100)
    ap.add_argument("--max-episodes", type=int, default=1500)
    ap.add_argument("--image-size", type=int, default=224)
    args = ap.parse_args()

    from robometer_policy_learning.envs.maniskill_utils import get_task_spec
    from robometer_policy_learning.utils.env_utils import make_env
    from robometer_policy_learning.rollouts.rollout_worker import extract_info_for_env
    from transformers import AutoImageProcessor, AutoModel

    spec = get_task_spec(args.task)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
    dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

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

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    act_dim = int(sd["actor_mean.6.weight"].shape[0])
    obs_dim = int(sd["actor_mean.0.weight"].shape[1])
    actor = build_ppo_actor(sd, obs_dim, act_dim).to(dev)
    print(f"[ppo] expert loaded: obs_dim={obs_dim} act_dim={act_dim}", flush=True)

    def raw_state():
        o = base.get_obs()
        if isinstance(o, dict):
            from mani_skill.utils import common
            o = common.flatten_state_dict(o, use_torch=True)
        t = o if torch.is_tensor(o) else torch.as_tensor(np.asarray(o))
        return t.reshape(1, -1).float().to(dev)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out = h5py.File(args.out, "w"); data = out.create_group("data")
    kept = ep = 0
    while kept < args.num_demos and ep < args.max_episodes:
        ep += 1
        obs, _ = env.reset(seed=10000 + ep)
        rec = {k: [] for k in obs.keys()}
        acts, rewards, dones, solved = [], [], [], False
        for t in range(spec.max_episode_steps):
            s = raw_state()
            if s.shape[1] != obs_dim:
                print(f"ERROR: env state dim {s.shape[1]} != expert obs_dim {obs_dim}", file=sys.stderr)
                return 2
            with torch.no_grad():
                a = actor(s).cpu().numpy()
            a = np.clip(a, -1.0 + 1e-6, 1.0 - 1e-6)      # expert head is unbounded
            for k, v in obs.items():
                arr = np.asarray(v)
                rec[k].append(arr[0] if arr.ndim and arr.shape[0] == 1 else arr)
            acts.append(a[0])
            obs, r, term, trunc, infos = env.step(a)
            info_i = extract_info_for_env(infos, 0, 1)
            sc = bool(info_i.get("success", False)); solved = solved or sc
            rewards.append(float(np.asarray(r).reshape(-1)[0])); dones.append(sc)
            if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                break
        if not solved:
            continue
        g = data.create_group(f"demo_{kept}"); n = len(acts)
        g.create_dataset("actions", data=np.stack(acts).astype(np.float32))
        g.create_dataset("rewards", data=np.array(rewards, dtype=np.float32))
        g.create_dataset("dones", data=np.array(dones, dtype=bool))
        og = g.create_group("obs")
        for k, vals in rec.items():
            arr = np.stack(vals[:n])
            og.create_dataset(k, data=arr, compression="gzip" if arr.nbytes > 1e6 else None)
        g.attrs["num_samples"] = n
        kept += 1
        if kept % 10 == 0:
            print(f"  kept {kept}/{args.num_demos} (episodes {ep}, success {kept/ep*100:.0f}%)", flush=True)
    out.attrs["env_id"] = args.task; out.attrs["instruction"] = instr
    out.close()
    print(f"\nDONE {args.task}: {kept} demos from {ep} episodes "
          f"(expert success {kept/max(ep,1)*100:.0f}%)")
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.0f} MB)")
    return 0 if kept > 0 else 3


if __name__ == "__main__":
    sys.exit(main())
