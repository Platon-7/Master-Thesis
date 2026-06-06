"""Run Robometer-FT step-3000 on actual training-set frames.

Hypothesis: feeding 16-frame, 480x480 JPEG inputs (matching training format)
yields strong success_prob separation between success/failure trajectories.
If true, the IBRL "flat output" failure is a pipeline mismatch (224x224 +
5 frames), not a model limitation.
"""
import os
import sys
import tarfile
import io
import json

import numpy as np
from PIL import Image

# repo imports
sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl")
sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer")

from env.robometer_utils import get_robometer_4b

CKPT = "/scratch-shared/pkarageorgis1/Robometer_FT_consolidated/run1_icl_ours_step3000"

# Pick one success + one failure trajectory from the training data shards.
SUCCESS_TAR = "/projects/prjs1958/robometer_frame_dataset/metaworld/keyframes_success/metaworld_coffee_push_v3/shard-00000.tar"
FAILURE_TAR = "/projects/prjs1958/robometer_frame_dataset/metaworld/keyframes/metaworld_coffee_push_v3/shard-00000.tar"


def load_first_episode(tar_path, want_corner="corner2"):
    """Return list of (frame_idx, np.uint8 (H,W,3)) for the first corner2 trajectory."""
    with tarfile.open(tar_path, "r") as tf:
        # Group members by episode prefix
        episodes = {}
        for m in tf.getmembers():
            if not m.isfile() or not m.name.endswith(".jpg"):
                continue
            ep = m.name.split("/")[0]
            if want_corner not in ep:
                continue
            episodes.setdefault(ep, []).append(m)
        if not episodes:
            raise RuntimeError(f"No {want_corner} episodes in {tar_path}")
        ep = sorted(episodes.keys())[0]
        members = sorted(episodes[ep], key=lambda m: m.name)
        frames = []
        for m in members:
            f = tf.extractfile(m)
            img = np.asarray(Image.open(io.BytesIO(f.read())).convert("RGB"))
            frames.append(img)
        # Pull the meta too
        meta_member = next((m for m in tf.getmembers() if m.name == f"{ep}/meta.json"), None)
        meta = None
        if meta_member:
            meta = json.loads(tf.extractfile(meta_member).read())
        return ep, frames, meta


def main():
    print("Loading Robometer-FT step-3000 ...")
    scorer = get_robometer_4b(model_path=CKPT)
    task = "Push a mug under a coffee machine."

    print(f"max_frames={scorer.max_frames}  (training had n_keyframes=16)\n")

    for label, tar in [("SUCCESS", SUCCESS_TAR), ("FAILURE", FAILURE_TAR)]:
        ep, frames, meta = load_first_episode(tar)
        print(f"=== {label} ===")
        print(f"  episode_id   : {ep}")
        print(f"  num frames   : {len(frames)}")
        print(f"  frame shape  : {frames[0].shape}  dtype={frames[0].dtype}")
        if meta:
            print(f"  meta.label   : {meta.get('label')}  terminal_reward={meta.get('terminal_reward')}")
            print(f"  frame_labels : {meta.get('frame_labels')}")

        # Run scorer with the FULL trajectory (16 frames at 480x480, JPEG-decoded)
        out = scorer(frames, task=task)
        print(f"  >>> 16-frame 480x480  : progress={out['progress_reward']:.4f}  "
              f"success_prob={out['success_prob']:.4f}")

        # Compare: same trajectory but downsized to 224 (matches IBRL res)
        frames_224 = [np.array(Image.fromarray(f).resize((224, 224), Image.BILINEAR), dtype=np.uint8)
                       for f in frames]
        out_224 = scorer(frames_224, task=task)
        print(f"  >>> 16-frame 224x224  : progress={out_224['progress_reward']:.4f}  "
              f"success_prob={out_224['success_prob']:.4f}")

        # Compare: only 5 frames at 480 (IBRL count, training res)
        if len(frames) >= 5:
            idx = np.linspace(0, len(frames) - 1, 5, dtype=int)
            sub5 = [frames[i] for i in idx]
            out_5_480 = scorer(sub5, task=task)
            print(f"  >>>  5-frame 480x480  : progress={out_5_480['progress_reward']:.4f}  "
                  f"success_prob={out_5_480['success_prob']:.4f}")

        # Compare: 5 frames at 224 (full IBRL config)
        if len(frames_224) >= 5:
            sub5_224 = [frames_224[i] for i in idx]
            out_5_224 = scorer(sub5_224, task=task)
            print(f"  >>>  5-frame 224x224  : progress={out_5_224['progress_reward']:.4f}  "
                  f"success_prob={out_5_224['success_prob']:.4f}")
        print()


if __name__ == "__main__":
    main()
