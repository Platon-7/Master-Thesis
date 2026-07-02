"""Rung-2 BC eval: load the shipped robomimic BC policy and run GT-success eval.

Validates env construction + EGL rendering + policy load end-to-end before any
VLM/RL. Reference success (from the BC train.log): can best 0.56 / ckpt ~0.35-0.38;
square best 0.42 / ckpt ~0.34. A healthy run should land in that band.

Usage: python jobs/rung2_bc_eval.py <path/to/model0.pt>
Env vars: NUM_GAME (default 25), EVAL_SEED (default 1).
"""
import os
import sys

import numpy as np

import train_bc
from evaluate import run_eval


def main():
    bc_path = sys.argv[1]
    num_game = int(os.environ.get("NUM_GAME", "25"))
    seed = int(os.environ.get("EVAL_SEED", "1"))

    print(f"[RUNG2] loading BC policy: {bc_path}", flush=True)
    policy, _, env_params = train_bc.load_model(bc_path, "cuda")
    print("[RUNG2] env_params:",
          {k: v for k, v in env_params.items() if k != "device"}, flush=True)

    scores = np.asarray(run_eval(env_params, policy, num_game, seed, verbose=True),
                        dtype=float)
    sr = float((scores > 0).mean())
    print(f"[RUNG2-RESULT] bc={bc_path} num_game={num_game} "
          f"success_rate={sr:.3f} mean_score={scores.mean():.3f} "
          f"n_success={int((scores > 0).sum())}/{len(scores)}", flush=True)


if __name__ == "__main__":
    main()
