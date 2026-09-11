"""Reproduce RoboReward's own RoboRewardBench protocol.

The paper is explicit (Section: Video Analysis):

    "We sample the rollout video at 1 FPS and provide the resulting image frames
     (including the final frame) along with the original task description."

So the frame count is proportional to clip DURATION, not fixed. The source clips
are 10 fps, so a 30-frame bridge clip yields 3 frames while a 396-frame
stanford_hydra clip yields 40. Our cache handed every clip a fixed 16, which is
why we beat the published number on short subsets and lost on long ones.

This reads the ORIGINAL mp4 (no re-encode, native resolution), samples at 1 FPS
with the true final frame always included, and scores with the harness's own
RoboReward, whose prompt already matches the paper's rubric verbatim.

  python score_roboreward_1fps.py --out <dir> [--limit N] [--model teetone/RoboReward-4B]
"""
import argparse, json, os, sys, time
import numpy as np


def sample_1fps(path):
    """Frames at 1 FPS, final frame always included. Returns (N,H,W,3) uint8."""
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps)))
    want = list(range(0, n, step))
    if n and (n - 1) not in want:
        want.append(n - 1)               # "including the final frame"
    want = set(want)
    out, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    if not out:
        return np.empty((0, 0, 0, 3), np.uint8), fps, n
    return np.asarray(out, np.uint8), fps, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=f"/scratch-shared/{os.environ['USER']}/RoboReward/test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="teetone/RoboReward-4B")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, "/home/pkarageorgis1/Master-Thesis/Robometer")
    from robometer.evals.baselines.roboreward import RoboReward

    rows = []
    with open(f"{args.src}/metadata.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e.get("file_name") and e.get("task") and e.get("reward") is not None:
                rows.append(e)
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} clips", flush=True)

    # the harness defaults to use_unsloth=True; RRB_UNSLOTH=0 falls back to stock
    # transformers, the last uncontrolled variable in our inference path
    use_unsloth = os.environ.get("RRB_UNSLOTH", "1") != "0"
    print(f"use_unsloth={use_unsloth}", flush=True)
    model = RoboReward(model_path=args.model, use_unsloth=use_unsloth)
    recs, t0, nf = [], time.time(), []
    for i, e in enumerate(rows):
        fn = e["file_name"]
        p = os.path.join(args.src, fn)
        ds = fn.split("/")[0]
        if ds == "robo_arena":
            ds = "roboarena"
        ps = (int(e["reward"]) - 1) / 4.0
        try:
            frames, fps, n = sample_1fps(p)
            nf.append(len(frames))
            scores = model.compute_progress(frames, e["task"])
            s = next((x for x in reversed(scores or []) if x is not None), None)
            # compute_progress ALREADY returns (score-1)/4 in [0,1] (roboreward.py:259),
            # despite a docstring that claims 1.0-5.0. Converting again gave -0.25 for score 1.
            pred = float("nan") if s is None else float(s)
        except Exception as ex:
            print(f"  [{i}] FAILED {type(ex).__name__}: {ex}", flush=True)
            pred = float("nan")
        recs.append({"id": os.path.splitext(os.path.basename(fn))[0],
                     "data_source": f"roboreward_{ds}", "task": e["task"],
                     "partial_success": ps,
                     "progress_pred": [None if pred != pred else pred]})
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(rows)}  {el/(i+1):.2f}s/clip  "
                  f"mean frames {np.mean(nf):.1f}  eta {(len(rows)-i-1)*el/(i+1)/60:.0f}min", flush=True)

    d = os.path.join(args.out, "reward_alignment")
    os.makedirs(d, exist_ok=True)
    json.dump(recs, open(os.path.join(d, "roboreward_test_results.json"), "w"))
    print(f"wrote {len(recs)} records | frames per clip: mean {np.mean(nf):.1f} "
          f"min {min(nf)} max {max(nf)}", flush=True)


if __name__ == "__main__":
    main()
