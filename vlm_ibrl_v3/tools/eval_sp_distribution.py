"""Roll out a saved IBRL policy for N episodes and capture per-episode
(env_success, vlm_success_prob, vlm_progress) to CSV.

Used to test the hypothesis that the VLM's `success_prob` distribution
shifts when scored on IBRL-policy-generated trajectories vs the
offline-eval distribution.

Usage:
    python tools/eval_sp_distribution.py \
        --run-dir /projects/prjs1958/.../coffeepush_qwen35_ft_b0.0_t0.25_23217978 \
        --num-episodes 200 \
        --out-csv /scratch-shared/$USER/sp_dist_eval/qwen35_ft_run4_s6500.csv
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyrallis
import torch
import yaml

sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl")
sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl/mw_main")
sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer")

from common_utils import ibrl_utils as utils
from train_rl_vlm_mw import MainConfig, Workspace


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path,
                   help="IBRL run directory (must contain cfg.yaml; latest.pt required unless --bc-only)")
    p.add_argument("--num-episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out-csv", required=True, type=Path)
    p.add_argument("--bc-only", action="store_true",
                   help="Skip loading latest.pt; roll out with BC policy only (no IBRL drift). "
                        "Use to isolate distribution shift from BC vs IBRL exploration.")
    p.add_argument("--dump-frames-dir", default=None, type=Path,
                   help="If set, save each episode's raw scorer frame buffer (npz: frames,label,sp,prog) "
                        "here, for offline re-scoring through the curated mp4 pipeline.")
    p.add_argument("--score-vlm", default=None,
                   help="Override cfg.vlm: roll out the saved policy but SCORE with this VLM "
                        "(e.g. robometer_ft). Lets us roll out a GT-reward policy — which has a "
                        "balanced success/failure mix — and measure another model's on-policy AUC "
                        "on enough positives to be trustworthy.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = args.run_dir / "cfg.yaml"
    pol_path = args.run_dir / "latest.pt"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing {cfg_path}")
    if not args.bc_only and not pol_path.exists():
        raise FileNotFoundError(f"missing {pol_path} (use --bc-only to roll out with BC policy)")

    # Load cfg via pyrallis. Older run dirs (e.g. the GT-reward control) were written
    # by a code version whose MainConfig had extra top-level fields (e.g.
    # `too_early_reward`) that the current MainConfig dropped; pyrallis strict-rejects
    # unknown fields, so sanitize the yaml to the current schema before loading.
    raw = yaml.safe_load(open(cfg_path))
    valid = {f.name for f in dataclasses.fields(MainConfig)}
    dropped = [k for k in list(raw) if k not in valid]
    if dropped:
        print(f"  dropping cfg fields not in current MainConfig: {dropped}")
        raw = {k: v for k, v in raw.items() if k in valid}
    _tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(raw, _tmp); _tmp.close()
    cfg = pyrallis.load(MainConfig, open(_tmp.name))
    cfg.use_wb = 0
    cfg.save_dir = str(args.run_dir.parent / f"eval_sp_dist_{args.run_dir.name}")
    if args.score_vlm is not None:
        print(f"  OVERRIDE scoring VLM: {cfg.vlm} -> {args.score_vlm} "
              f"(rolling out {args.run_dir.name}'s policy, scoring with {args.score_vlm})")
        cfg.vlm = args.score_vlm
    print(f"loaded cfg from {cfg_path}")
    print(f"  bc_policy:   {cfg.bc_policy}")
    print(f"  vlm:         {cfg.vlm}")
    print(f"  beta / thr:  {cfg.robometer_beta} / {cfg.robometer_threshold}")

    print("\n[1/3] building Workspace (env + agent)…")
    ws = Workspace(cfg)

    if args.bc_only:
        if ws.bc_policy is None:
            raise RuntimeError("--bc-only requires the workspace to have a BC policy; cfg.use_bc must be 1")
        print(f"\n[2/3] BC-only mode — skipping latest.pt; will roll out with BC policy")
        actor = ws.bc_policy
    else:
        print("\n[2/3] loading saved policy …")
        state_dict = torch.load(str(pol_path), map_location="cuda", weights_only=False)
        ws.agent.load_state_dict(state_dict)
        print(f"  loaded {pol_path}")
        actor = ws.agent

    if args.dump_frames_dir is not None:
        args.dump_frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"  dumping per-episode raw frame buffers to {args.dump_frames_dir}")

    print(f"\n[3/3] rolling out {args.num_episodes} episodes (seed={args.seed})…")
    # IMPORTANT: ws.eval_env is a plain PixelMetaWorld — no VLM. Use ws.train_env
    # which is VLMCritic_PixelMetaWorld with the actual scorer wired in.
    env = ws.train_env

    rows = []
    with torch.no_grad(), utils.eval_mode(actor):
        for ep in range(args.num_episodes):
            np.random.seed(args.seed + ep)
            obs, image_obs = env.reset()
            terminal = False
            ep_env_success = 0
            while not terminal:
                action = actor.act(obs, eval_mode=True).numpy()
                obs, reward, terminal, success, image_obs = env.step(action)
                if success:
                    ep_env_success = 1
            sp = float(getattr(env, "_last_success_prob", float("nan")))
            prog = float(getattr(env, "_last_progress", float("nan")))
            # Capture the EXACT frame buffer the scorer saw this episode (raw 224x224
            # RGB straight from fetch_img — no mp4 encode/decode). Lets us re-score
            # these same trajectories offline through the curated-data pipeline
            # (mp4 round-trip) to test whether the on-policy AUC collapse is a
            # frame-format artifact rather than a real content shift.
            if args.dump_frames_dir is not None and getattr(env, "current_video", None):
                vid = np.stack([np.asarray(f) for f in env.current_video]).astype(np.uint8)
                np.savez_compressed(
                    Path(args.dump_frames_dir) / f"ep{ep:04d}.npz",
                    frames=vid, label=ep_env_success, sp=sp, prog=prog)
            rows.append({
                "episode": ep,
                "env_success": ep_env_success,
                "sp": sp,
                "progress": prog,
            })
            if (ep + 1) % 25 == 0:
                n_succ_so_far = sum(r["env_success"] for r in rows)
                print(f"  [{ep+1}/{args.num_episodes}] n_succ={n_succ_so_far}  last: env={ep_env_success} sp={sp:.4f} prog={prog:.4f}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "env_success", "sp", "progress"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {args.out_csv}")

    n_succ = sum(r["env_success"] for r in rows)
    sp_succ = [r["sp"] for r in rows if r["env_success"] == 1]
    sp_fail = [r["sp"] for r in rows if r["env_success"] == 0]
    print(f"\n=== summary ===")
    print(f"  {n_succ}/{len(rows)} successful episodes ({100*n_succ/len(rows):.1f}%)")
    if sp_succ:
        sp_s = np.asarray(sp_succ)
        print(f"  sp on success: n={len(sp_s)} mean={sp_s.mean():.4f} std={sp_s.std():.4f} min={sp_s.min():.4f} max={sp_s.max():.4f}")
    if sp_fail:
        sp_f = np.asarray(sp_fail)
        print(f"  sp on failure: n={len(sp_f)} mean={sp_f.mean():.4f} std={sp_f.std():.4f} min={sp_f.min():.4f} max={sp_f.max():.4f}")


if __name__ == "__main__":
    main()
