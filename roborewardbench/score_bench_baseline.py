"""Score RoboRewardBench with RoboDopamine or LRM using our own adapters.

Upstream's robodopamine baseline needs a vLLM venv (torch 2.9) that conflicts with
our stack, and upstream has no LRM baseline at all. Both models already have
transformers-based scorers in robometer_policy_learning that were validated
end-to-end on Robomimic, MetaWorld and ManiSkill, so this reuses those.

Each bench clip is scored once, first frame against last frame, which is the
"how much of the task got done" quantity the 1-5 label encodes. Output is written
in the same record shape the Robometer harness writes, so the MAE tooling reads it
unchanged.

  python score_bench_baseline.py --model lrm|robodopamine --out <dir> [--limit N]
"""
import argparse, json, os, sys, time
import numpy as np
from PIL import Image


def load_frames(path):
    with np.load(path) as z:
        key = "frames" if "frames" in z else list(z.keys())[0]
        return z[key]


def to_pil(a):
    a = np.asarray(a)
    if a.dtype != np.uint8:
        a = (255 * np.clip(a, 0, 1)).astype(np.uint8)
    return Image.fromarray(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lrm", "robodopamine"])
    ap.add_argument("--cache", default=f"/scratch-shared/{os.environ['USER']}/roborewardbench/step2/roboreward_test/processed_dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model-path", default=None)
    args = ap.parse_args()

    import torch
    from datasets import load_from_disk
    from robometer_policy_learning.utils.baseline_reward_adapters import load_baseline_reward_model

    ds = load_from_disk(args.cache)
    n = args.limit or ds.num_rows
    print(f"scoring {n}/{ds.num_rows} clips with {args.model}", flush=True)

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, _, _, wrapped = load_baseline_reward_model(args.model, args.model_path, dev)
    model, processor = wrapped.model, wrapped.processor

    os.makedirs(args.out, exist_ok=True)
    recs, t0 = [], time.time()
    for i in range(n):
        r = ds[i]
        try:
            fr = load_frames(r["frames"]) if isinstance(r["frames"], str) else np.asarray(r["frames"])
            first, last = to_pil(fr[0]), to_pil(fr[-1])
            task = r.get("task") or ""
            if args.model == "lrm":
                res = int(os.environ.get("LRM_RES", "256"))
                from robometer_policy_learning.utils.lrm_utils import LRMProgressScorer
                s = LRMProgressScorer(model, processor, task=task,
                                      initial_img=first.resize((res, res)),
                                      max_new_tokens=wrapped.max_new_tokens)
                score = float(s.score(last.resize((res, res))))
            else:
                from robometer_policy_learning.utils.robodopamine_utils import (
                    RoboDopamineScorer, accumulate_progress)
                s = RoboDopamineScorer(model, processor, task=task, goal_img=first,
                                       ref_start_img=first, eval_mode=wrapped.eval_mode,
                                       max_new_tokens=wrapped.max_new_tokens)
                raw = s.score([first] * 3, [last] * 3)
                score = 0.0 if raw is None else float(accumulate_progress(wrapped.eval_mode, raw, 0.0))
        except Exception as e:
            print(f"  [{i}] FAILED {type(e).__name__}: {e}", flush=True)
            score = 0.0
        recs.append({"id": r.get("id"), "data_source": r.get("data_source"),
                     "task": r.get("task"), "quality_label": r.get("quality_label"),
                     "partial_success": r.get("partial_success"),
                     "progress_pred": [score]})
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{n}  {el/(i+1):.2f}s/clip  eta {(n-i-1)*el/(i+1)/60:.0f}min", flush=True)

    d = os.path.join(args.out, "reward_alignment")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "roboreward_test_results.json"), "w") as f:
        json.dump(recs, f)
    print(f"wrote {len(recs)} records to {d}", flush=True)


if __name__ == "__main__":
    main()
