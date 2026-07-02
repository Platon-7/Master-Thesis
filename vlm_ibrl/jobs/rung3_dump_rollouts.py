"""Rung-3 stage A: dump BC-rollout agentview videos + GT success labels.

Runs the shipped BC policy (acting on its own rl_camera, exactly as rung 2) while
ALSO rendering `agentview` at high-res, and saves one subsampled video per episode
with its GT success label. These labelled videos are the offline-scoring gate set:
positives = success episodes, negatives = timeout/failure episodes.

Usage: python jobs/rung3_dump_rollouts.py <bc_model0.pt> <out.npz>
Env: NUM_GAME (default 50), DUMP_FRAMES (uniform frames/episode, default 32), EVAL_SEED (default 1).
"""
import json
import os
import sys

import numpy as np
import torch

import train_bc
from env.robosuite_wrapper import PixelRobosuite
from common_utils import ibrl_utils as utils


def to_hwc_uint8(chw_tensor):
    arr = chw_tensor.detach().cpu().numpy()          # (C,H,W) uint8
    return np.ascontiguousarray(np.transpose(arr, (1, 2, 0)))  # (H,W,C)


def subsample(frames, n):
    if len(frames) <= n:
        return frames
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx]


def main():
    bc_path, out_path = sys.argv[1], sys.argv[2]
    num_game = int(os.environ.get("NUM_GAME", "50"))
    n_frames = int(os.environ.get("DUMP_FRAMES", "32"))
    seed = int(os.environ.get("EVAL_SEED", "1"))

    policy, _, env_params = train_bc.load_model(bc_path, "cuda")
    env_name = env_params["env_name"]
    pol_cams = list(env_params["rl_cameras"])
    # render agentview (for scoring) in addition to the policy camera; policy obs unchanged.
    env_params = dict(env_params)
    env_params["camera_names"] = list(dict.fromkeys(pol_cams + ["agentview"]))
    env_params["rl_cameras"] = pol_cams
    print(f"[dump] env={env_name} policy_cam={pol_cams} cameras={env_params['camera_names']} "
          f"num_game={num_game} frames/ep={n_frames}", flush=True)

    env = PixelRobosuite(**env_params)
    store, labels, lengths = {}, [], []
    with torch.no_grad(), utils.eval_mode(policy):
        for ep in range(num_game):
            np.random.seed(seed + ep)
            obs, high = env.reset()
            vid = [to_hwc_uint8(high["agentview"])]
            terminal, succ = False, 0.0
            while not terminal:
                action = policy.act(obs, eval_mode=True)
                obs, reward, terminal, success, high = env.step(action)
                vid.append(to_hwc_uint8(high["agentview"]))
                succ = max(succ, float(success), float(reward > 0))
            v = np.stack(subsample(vid, n_frames)).astype(np.uint8)
            store[f"ep{ep}"] = v
            labels.append(int(succ > 0))
            lengths.append(env.time_step)
            print(f"[dump] ep{ep:02d} success={int(succ>0)} len={env.time_step} "
                  f"saved_frames={v.shape[0]}", flush=True)

    labels = np.asarray(labels, dtype=np.int64)
    np.savez_compressed(out_path, labels=labels, lengths=np.asarray(lengths),
                        env_name=env_name, **store)
    meta = dict(env_name=env_name, num_game=num_game, n_success=int(labels.sum()),
                n_fail=int((labels == 0).sum()), frames_per_ep=n_frames)
    with open(out_path.replace(".npz", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[dump] DONE {out_path} : {meta}", flush=True)


if __name__ == "__main__":
    main()
