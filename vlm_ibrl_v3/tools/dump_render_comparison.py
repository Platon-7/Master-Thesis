"""Dump live IBRL-env rollout frames next to curated keyframes for the same
task, to decide whether the curated->live AUC collapse is render-domain shift
(frames look different) or trajectory-content shift (frames look the same style
but the policy does OOD things).

Builds the env from an IBRL run cfg, rolls out the BC policy for a couple
episodes, grabs the 16 frames the scorer actually sees (env.current_video
subsampled), and tiles them against curated CoffeePush keyframes.
"""
from __future__ import annotations

import argparse
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import pyrallis
import torch
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl")
sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl/mw_main")
sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer")

from common_utils import ibrl_utils as utils
from train_rl_vlm_mw import MainConfig, Workspace

CURATED_TAR = "/projects/prjs1958/robometer_frame_dataset/metaworld/keyframes_success/metaworld_coffee_push_v3/shard-00000.tar"


def load_curated(n=8):
    with tarfile.open(CURATED_TAR) as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(".jpg")]
        ep = members[0].name.split("/")[0]
        ep_members = sorted([m for m in members if m.name.startswith(ep + "/")], key=lambda m: m.name)
        frames = [np.asarray(Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB"))
                  for m in ep_members]
    idx = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-frames", type=int, default=8)
    args = ap.parse_args()

    cfg = pyrallis.load(MainConfig, open(args.run_dir / "cfg.yaml"))
    cfg.use_wb = 0
    cfg.save_dir = str(args.run_dir.parent / f"dump_{args.run_dir.name}")
    ws = Workspace(cfg)
    env = ws.train_env
    actor = ws.bc_policy if ws.bc_policy is not None else ws.agent

    # roll out one episode, grab the accumulated video
    live_frames = None
    with torch.no_grad(), utils.eval_mode(actor):
        np.random.seed(7)
        obs, image_obs = env.reset()
        terminal = False
        while not terminal:
            action = actor.act(obs, eval_mode=True).numpy()
            obs, reward, terminal, success, image_obs = env.step(action)
        # env.current_video holds every frame of the episode at 224x224
        vid = env.current_video
        idx = np.linspace(0, len(vid) - 1, args.n_frames, dtype=int)
        live_frames = [np.asarray(vid[i]) for i in idx]
        print(f"live episode: {len(vid)} frames, success={success}")

    curated = load_curated(args.n_frames)

    n = args.n_frames
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.4))
    for j in range(n):
        axes[0, j].imshow(curated[j]); axes[0, j].axis("off")
        axes[1, j].imshow(live_frames[j]); axes[1, j].axis("off")
        if j == 0:
            axes[0, j].set_title("curated keyframe (480px)", loc="left", fontsize=10)
            axes[1, j].set_title("live IBRL env render (224px)", loc="left", fontsize=10)
    fig.suptitle("CoffeePush — curated keyframes (top) vs live MuJoCo env renders (bottom)", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
