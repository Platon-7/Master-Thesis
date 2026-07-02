"""Phase-2 scorer (Qwen3.5): score ALREADY-SAVED rollout frames — no robosuite needed.

The neutral rollout (clean BC + noisy BC) was produced by the robosuite env in
demo2reward_robomimic and saved to disk. Qwen3.5 (run4/run5) cannot load there
(needs transformers>=5.7), so we score the saved frames here in the Qwen3.5 env.
GT labels come from gt_labels.json (recovered from the Robometer run 951, same seeds).

Env vars: SAVE_ROOT (has CLEAN_xx/OOD_xx dirs + gt_labels.json), MODELS ("name:path,..."),
          ROBOMETER_ICL_DEMO_PATH/IDX/FRAMES.
Run with the Qwen3.5 env's python; only needs numpy/torch/PIL + get_robometer_4b.
"""
import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image

VLM_IBRL = os.environ.get("VLM_IBRL_DIR", "/shared/home/PKA4388/Master-Thesis/vlm_ibrl")
sys.path.insert(0, os.path.join(VLM_IBRL, "env"))
import robometer_utils  # noqa: E402  (standalone import avoids the robosuite-laden env package)
get_robometer_4b = robometer_utils.get_robometer_4b

TASK = ("A can is placed in a bin in front of a single robot arm. There are four containers "
        "next to the bin. The robot must place the can into its corresponding container.")
KSTEP = 10
SAVE_ROOT = os.environ["SAVE_ROOT"]
MODELS = [tuple(m.split(":", 1)) for m in os.environ["MODELS"].split(",") if m.strip()]


def load_icl():
    p = os.environ.get("ROBOMETER_ICL_DEMO_PATH", "")
    if not p:
        return None
    idx = int(os.environ.get("ROBOMETER_ICL_DEMO_IDX", "0"))
    n = int(os.environ.get("ROBOMETER_ICL_FRAMES", "16"))
    avail = sorted(q for q in Path(p).iterdir() if q.name.startswith(f"{idx}_") and q.suffix == ".png")
    picks = np.linspace(0, len(avail) - 1, n).round().astype(int)
    return [np.asarray(Image.open(avail[i]).convert("RGB"), dtype=np.uint8) for i in picks]


def frames_of(d):
    fs = sorted(Path(d).glob("*.png"), key=lambda q: int(q.stem))
    return [np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8) for f in fs]


def causal_max(scorer, frames, icl):
    mx = 0.0
    for t in range(KSTEP, len(frames) + 1, KSTEP):
        mx = max(mx, float(scorer(frames[:t], task=TASK, icl_frames=icl)["success_prob"]))
    return mx


def main():
    gt = json.load(open(os.path.join(SAVE_ROOT, "gt_labels.json")))
    icl = load_icl()
    eps = {}
    for split in ["CLEAN", "OOD"]:
        for i, lab in enumerate(gt[split]):
            eps[(split, i)] = (int(lab), frames_of(os.path.join(SAVE_ROOT, f"{split}_{i:02d}")))
    print(f"[scoreqwen] models={[m[0] for m in MODELS]} icl={len(icl) if icl else 0}f eps={len(eps)}", flush=True)

    results = []
    for name, path in MODELS:
        scorer = get_robometer_4b(model_path=path)
        for tag, icl_arg in [("+ICL", icl), ("-noICL", None)]:
            label = f"{name}{tag}"
            res = {}
            for k, (g, fr) in eps.items():
                mx = causal_max(scorer, fr, icl_arg)
                res[k] = (g, mx)
                print(f"  [{label}] {k[0]}{k[1]:02d} GT={'S' if g else 'F'} maxSP={mx:.2f}", flush=True)
            results.append((label, res))
        del scorer
        torch.cuda.empty_cache()

    print("\n=== FAIR SUMMARY (real-success vs NEUTRAL OOD-fail) ===", flush=True)
    for label, res in results:
        cs = [v[1] for k, v in res.items() if k[0] == "CLEAN" and v[0] == 1]
        of = [v[1] for k, v in res.items() if k[0] == "OOD" and v[0] == 0]
        ms = f"{np.mean(cs):.2f}" if cs else "--"
        mf = f"{np.mean(of):.2f}" if of else "--"
        gap = f"{np.mean(cs) - np.mean(of):.2f}" if (cs and of) else "--"
        sep = "YES" if (cs and of and min(cs) > max(of)) else "no/overlap"
        wo = f"{min(cs):.2f}" if cs else "--"
        wb = f"{max(of):.2f}" if of else "--"
        print(f"  [{label.ljust(11)}] real-success n={len(cs)} mean={ms} (min={wo}) | "
              f"neutral-OOD-fail n={len(of)} mean={mf} (max={wb}) | GAP={gap} | separable={sep}", flush=True)
    print("[scoreqwen] done", flush=True)


if __name__ == "__main__":
    main()
