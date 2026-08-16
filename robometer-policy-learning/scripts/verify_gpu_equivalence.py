#!/usr/bin/env python3
"""Per-task CPU-vs-GPU equivalence for the ManiSkill adapter.

Runs ManiSkill's own trained PPO checkpoint through our stack on one backend and
writes the success rate to JSON. A separate invocation does the other backend
(PhysX cannot switch backends inside one interpreter), and `compare` prints the
verdict per task.

Sized for a real answer rather than a hint: an earlier 64-episode version had the
CPU baseline swinging 6%-27% run to run, which is why "GPU is broken on PokeCube"
could not be established. At 256 episodes the binomial standard error at p=0.15 is
~2%, so a genuine gap is separable from noise.

    python scripts/verify_gpu_equivalence.py run --task PokeCube-v1 --backend physx_cpu
    python scripts/verify_gpu_equivalence.py compare
"""
from __future__ import annotations
import argparse, glob, json, os, sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = os.path.join(os.environ.get("MS_ASSET_DIR", "/tmp"), "gpu_equivalence")


def run(args) -> int:
    import numpy as np, torch
    from scripts.verify_with_maniskill_ppo import build_actor
    from robometer_policy_learning.envs.maniskill_utils import get_task_spec
    from robometer_policy_learning.utils.env_utils import make_env
    from robometer_policy_learning.rollouts.rollout_worker import extract_info_for_env

    spec = get_task_spec(args.task)
    ck = glob.glob(os.path.join(os.environ["MS_ASSET_DIR"], "demos", args.task, "**",
                                f"ppo_{spec.control_mode}_ckpt.pt"), recursive=True)
    if not ck:
        print(f"SKIP {args.task}: no PPO checkpoint for {spec.control_mode}")
        return 2
    sd = torch.load(ck[0], map_location="cpu", weights_only=False)
    actor = build_actor(sd, sd["actor_mean.0.weight"].shape[1], sd["actor_mean.6.weight"].shape[0])

    n_envs = args.num_envs
    env, eval_env = make_env(
        env_name=f"maniskill/{args.task}", num_envs=n_envs, chunk_size=None,
        max_episode_steps=spec.max_episode_steps, use_full_state=True,
        env_kwargs={"sim_backend": args.backend, "image_size": 224,
                    "control_mode": spec.control_mode, "reward_mode": "normalized_dense"},
    )
    obs, _ = env.reset(seed=args.seed)
    done_eps = succ_eps = 0
    hit = np.zeros(n_envs, bool); steps = np.zeros(n_envs, int)
    while done_eps < args.episodes:
        with torch.no_grad():
            a = actor(torch.as_tensor(np.asarray(obs["state"]), dtype=torch.float32)).numpy()
        obs, _r, term, trunc, infos = env.step(a)
        steps += 1
        for i in range(n_envs):
            if bool(extract_info_for_env(infos, i, n_envs).get("success", False)):
                hit[i] = True
            if term[i] or trunc[i] or steps[i] >= spec.max_episode_steps:
                done_eps += 1; succ_eps += int(hit[i]); hit[i] = False; steps[i] = 0
                if done_eps >= args.episodes:
                    break
    env.close()
    if eval_env is not None and eval_env is not env:
        eval_env.close()

    os.makedirs(OUT, exist_ok=True)
    rec = dict(task=args.task, backend=args.backend, num_envs=n_envs,
               success=succ_eps, episodes=done_eps, rate=succ_eps / max(1, done_eps))
    json.dump(rec, open(os.path.join(OUT, f"{args.task}__{args.backend}.json"), "w"))
    print(f"EQUIV {args.task:20s} {args.backend:11s} {succ_eps}/{done_eps} = {100*rec['rate']:.1f}%")
    return 0


def compare(args) -> int:
    import math
    files = sorted(glob.glob(os.path.join(OUT, "*.json")))
    by_task = {}
    for f in files:
        r = json.load(open(f))
        by_task.setdefault(r["task"], {})[r["backend"]] = r
    print(f"\n{'task':20s} {'CPU':>16s} {'GPU':>16s}  verdict")
    all_ok = True
    for t, d in sorted(by_task.items()):
        c, g = d.get("physx_cpu"), d.get("physx_cuda")
        if not c or not g:
            print(f"{t:20s} {'--':>16s} {'--':>16s}  INCOMPLETE"); all_ok = False; continue
        # binomial SE of the difference; flag only gaps beyond ~2 sigma
        def se(r):
            return math.sqrt(max(r["rate"] * (1 - r["rate"]), 1e-9) / max(1, r["episodes"]))
        diff = c["rate"] - g["rate"]
        sigma = math.sqrt(se(c) ** 2 + se(g) ** 2)
        ok = abs(diff) <= max(2 * sigma, 0.05)
        all_ok = all_ok and ok
        print(f"{t:20s} {100*c['rate']:6.1f}% ({c['episodes']:4d}) {100*g['rate']:6.1f}% ({g['episodes']:4d})  "
              f"diff={100*diff:+5.1f}pp  2sig={100*2*sigma:4.1f}pp  {'MATCH' if ok else 'MISMATCH'}")
    print("\n" + ("ALL TASKS EQUIVALENT -- GPU usable" if all_ok else "NOT equivalent on every task"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--task", required=True)
    r.add_argument("--backend", required=True); r.add_argument("--episodes", type=int, default=256)
    r.add_argument("--num-envs", type=int, default=16); r.add_argument("--seed", type=int, default=0)
    r.set_defaults(func=run)
    c = sub.add_parser("compare"); c.set_defaults(func=compare)
    a = ap.parse_args(); sys.exit(a.func(a))
