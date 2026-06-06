"""ICL-augmented offline CM eval: prepend 16 uniform frames from demo 0 of
CoffeePush as the in-context demonstration, then score the same 300-clip
sweep against demos 1 and 2 (skipping demo 0 to avoid label leakage)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from env.robometer_utils import get_robometer_4b
from tools.robometer_offline_cm import (
    load_demo_rgb_and_rewards,
    find_success_t,
    build_clip,
    TASK_DESCRIPTIONS,
)


def load_icl_frames(demo_idx: int = 0, n: int = 16, task: str = "CoffeePush") -> np.ndarray:
    """Pick `n` uniform frames from demo `demo_idx` in the demonstrations/ tree.
    Filenames are `{demo_idx}_{0..100}.png`."""
    frames_dir = Path("release/data/metaworld") / \
        f"{task}_frame_stack_1_224x224_end_on_success" / \
        "demonstrations" / "mw-coffee-push" / "frames"
    available = sorted(
        p for p in frames_dir.iterdir()
        if p.name.startswith(f"{demo_idx}_") and p.suffix == ".png"
    )
    if len(available) == 0:
        raise FileNotFoundError(f"no frames for demo {demo_idx} in {frames_dir}")
    idx = np.linspace(0, len(available) - 1, n).round().astype(int)
    selected = [available[i] for i in idx]
    out = np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in selected])
    print(f"ICL: demo={demo_idx}  picked {n} frames from {len(available)} → "
          f"indices {idx.tolist()}; shape={out.shape}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task", default="CoffeePush", choices=list(TASK_DESCRIPTIONS.keys()))
    parser.add_argument("--data-dir", default="release/data/metaworld")
    parser.add_argument("--camera", default="corner2")
    parser.add_argument("--num-demos", type=int, default=3)
    parser.add_argument("--icl-demo", type=int, default=0,
                        help="Which demo to pull ICL frames from (default 0)")
    parser.add_argument("--icl-frames", type=int, default=16)
    parser.add_argument("--skip-icl-source-demo", action="store_true",
                        help="If set, do NOT evaluate on the demo we used as ICL context")
    parser.add_argument("--past-len", type=int, default=4)
    parser.add_argument("--full-clip", action="store_true")
    parser.add_argument("--prefix-stride", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset_path = os.path.join(
        args.data_dir, f"{args.task}_frame_stack_1_224x224_end_on_success", "dataset.hdf5"
    )
    task_desc = TASK_DESCRIPTIONS[args.task]

    print("=" * 64)
    print(f"Model:     {args.model_path}")
    print(f"Task:      {args.task}  —  {task_desc!r}")
    print(f"ICL:       demo={args.icl_demo}, n_frames={args.icl_frames}")
    print(f"Skip ICL src demo from eval: {args.skip_icl_source_demo}")
    print(f"full_clip: {args.full_clip}, prefix_stride: {args.prefix_stride}")
    print("=" * 64)

    icl_frames = load_icl_frames(demo_idx=args.icl_demo, n=args.icl_frames, task=args.task)

    print(f"\nLoading scorer ...")
    scorer = get_robometer_4b(model_path=args.model_path, device=args.device)
    print(f"OK. max_frames={scorer.max_frames}")

    results = []
    tp = fp = tn = fn = 0
    for demo_idx in range(args.num_demos):
        if args.skip_icl_source_demo and demo_idx == args.icl_demo:
            continue
        frames_hwc, rewards = load_demo_rgb_and_rewards(dataset_path, demo_idx, args.camera)
        T = len(frames_hwc)
        success_t = find_success_t(rewards)
        print(f"demo {demo_idx}: T={T:>4}  success_t={success_t}")

        prefix_ends = list(range(args.prefix_stride, T + 1, args.prefix_stride))
        if T not in prefix_ends:
            prefix_ends.append(T)

        for end_t in prefix_ends:
            gt = int(rewards[end_t - 1] >= 1.0)
            if args.full_clip:
                clip = [frames_hwc[i] for i in range(end_t)]
            else:
                clip = build_clip(frames_hwc, end_t, args.past_len)
            out = scorer(clip, task=task_desc, episode_id=demo_idx, icl_frames=list(icl_frames))
            pred = int(out["success_prob"] > args.threshold)
            results.append(dict(
                demo=demo_idx, end_t=end_t, success_t=success_t,
                gt=gt, pred=pred,
                progress_reward=out["progress_reward"], success_prob=out["success_prob"],
            ))
            if gt == 1:
                tp += pred == 1; fn += pred == 0
            else:
                fp += pred == 1; tn += pred == 0

    n_pos = tp + fn; n_neg = fp + tn
    cm = dict(tp=tp, fp=fp, tn=tn, fn=fn,
              tpr=(tp / n_pos if n_pos else float("nan")),
              fpr=(fp / n_neg if n_neg else float("nan")),
              tnr=(tn / n_neg if n_neg else float("nan")),
              fnr=(fn / n_pos if n_pos else float("nan")))
    print(f"\n=== ICL-augmented CM @ τ={args.threshold} ===")
    print(json.dumps(cm, indent=2))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({
            "args": vars(args),
            "confusion_matrix": cm,
            "results": results,
        }))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
