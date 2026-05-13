"""Smoke test the Robometer-4B loader and a single forward pass.

Run from the repo root after ``source set_env.sh`` and with
``Robometer/`` on ``PYTHONPATH``::

    PYTHONPATH="$PWD:$HOME/Master-Thesis/Robometer:$PYTHONPATH" \\
        python tools/robometer_smoke.py
"""

from __future__ import annotations

import argparse

import numpy as np

from env.robometer_utils import get_robometer_4b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="robometer/Robometer-4B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--task", default="pick up a nut and place it onto a peg")
    args = parser.parse_args()

    print(f"Loading {args.model_path} on {args.device} ...")
    scorer = get_robometer_4b(model_path=args.model_path, device=args.device)
    print(f"OK. model_type={scorer._model_type} "
          f"discrete={scorer._is_discrete} num_bins={scorer._num_bins} "
          f"max_frames={scorer.max_frames} device={scorer.device}")

    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
              for _ in range(args.num_frames)]
    out = scorer(frames, task=args.task)
    print(f"Forward pass OK: progress={out['progress_reward']:.4f} "
          f"success_prob={out['success_prob']:.4f}")

    for key in ("progress_reward", "success_prob"):
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of [0, 1]"
    print("All assertions passed.")


if __name__ == "__main__":
    main()
