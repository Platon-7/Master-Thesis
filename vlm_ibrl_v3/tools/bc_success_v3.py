"""Measure the v2-trained BC `coffeepush` policy's env success rate in the v3
env, with and without the v2 corner2 zoom (env var V3_CORNER2_ZOOM).

Decides the cheap-vs-expensive fork after the GT-reward IBRL floor came up flat
in v3 (the v2 BC policy is out-of-domain on v3-default-corner2):
  - if zoom ON recovers BC success (>~0.3) -> the BC failure is CAMERA/zoom, so a
    dual-render setup (policy sees v2-zoom corner2, reward sees default corner2)
    could unblock IBRL cheaply — no BC retrain.
  - if zoom ON stays ~0 -> the failure is the v3 render ENGINE (textures/meshes),
    and a v3 BC retrain on v3 demos is required.

Uses the GT trainer's Workspace (train_rl_mw -> plain PixelMetaWorld, NO VLM),
so no 4B model load. Roll out ws.bc_policy on ws.eval_env (plain).

  V3_CORNER2_ZOOM=0 python tools/bc_success_v3.py --run-dir <gt_run_dir> --num-episodes 50
  V3_CORNER2_ZOOM=1 python tools/bc_success_v3.py --run-dir <gt_run_dir> --num-episodes 50
"""
from __future__ import annotations
import argparse, dataclasses, os, sys, tempfile
from pathlib import Path
import numpy as np, torch, yaml

sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl_v3")
sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl_v3/mw_main")

from common_utils import ibrl_utils as utils
from train_rl_mw import MainConfig, Workspace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path, help="a GT IBRL run dir with cfg.yaml")
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=999)
    args = p.parse_args()

    zoom = os.environ.get("V3_CORNER2_ZOOM", "0")
    raw = yaml.safe_load(open(args.run_dir / "cfg.yaml"))
    valid = {f.name for f in dataclasses.fields(MainConfig)}
    raw = {k: v for k, v in raw.items() if k in valid}
    tf = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(raw, tf); tf.close()
    import pyrallis
    cfg = pyrallis.load(MainConfig, open(tf.name))
    cfg.use_wb = 0
    cfg.save_dir = str(args.run_dir.parent / f"bc_success_v3_zoom{zoom}_{args.run_dir.name}")

    print(f"V3_CORNER2_ZOOM={zoom}  bc_policy={cfg.bc_policy}  building Workspace (no VLM)…", flush=True)
    ws = Workspace(cfg)
    if ws.bc_policy is None:
        raise RuntimeError("workspace has no bc_policy (cfg.use_bc must be 1)")
    actor = ws.bc_policy
    env = ws.eval_env  # plain PixelMetaWorld, no VLM

    n_succ = 0
    with torch.no_grad(), utils.eval_mode(actor):
        for ep in range(args.num_episodes):
            np.random.seed(args.seed + ep)
            obs, image_obs = env.reset()
            terminal = False
            ep_s = 0
            while not terminal:
                action = actor.act(obs, eval_mode=True).numpy()
                obs, reward, terminal, success, image_obs = env.step(action)
                if success:
                    ep_s = 1
            n_succ += ep_s
            if (ep + 1) % 10 == 0:
                print(f"  [{ep+1}/{args.num_episodes}] success so far: {n_succ}", flush=True)

    rate = n_succ / args.num_episodes
    print(f"\n==== BC `{cfg.bc_policy}` in v3, V3_CORNER2_ZOOM={zoom}: "
          f"{n_succ}/{args.num_episodes} = {rate:.2f} success ====")
    print("  ref: v2 BC in v2 env is competent (v2 GT IBRL pulses to ~0.76).")
    print("  zoom=1 high -> camera/zoom fix (dual-render viable); zoom=1 ~0 -> need v3 BC retrain.")


if __name__ == "__main__":
    main()
