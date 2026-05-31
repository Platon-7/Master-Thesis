"""
Diagnose where the NaN in success_prob comes from for Qwen3.5-FT step-4000/5000
on Chris's CoffeePush release demos (where the failure was observed).

Strategy:
  1. Load step-4000 in default dtype (bf16). Score one real CoffeePush demo
     (the first clip from job 23021951 that produced NaN). Capture intermediate
     activations via forward hooks. Find the first layer where NaN appears.

  2. Try the same clip in fp32. If fp32 has no NaN, the bug is a bf16 overflow
     and the fix is fp32 inference (or selective fp32 for the head).

  3. Print the success_logits scalar for step-3000 (works) on the same clip,
     so we can see its range — confirms whether the problem is "logits are
     normal" (algorithmic bug) vs "logits push into overflow" (precision bug).

Run via SLURM (needs GPU).
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


# Reproduce the same clip the failed job (23021951) processed.
# robometer_offline_cm.py builds clips by stepping prefix_stride=5 across the
# trajectory, full_clip=1 → clip is frames[0:end_t]. We replay the first clip:
# demo=0, end_t=5 → frames 0..4 from demo 0.

def load_first_clip(task: str = "CoffeePush", demo_idx: int = 0, end_t: int = 5) -> np.ndarray:
    """Reproduce exactly what robometer_offline_cm.py does for the first clip
    (demo 0, end_t=5, camera=corner2, 224x224 release demo set)."""
    import h5py
    path = Path("release/data/metaworld") / f"{task}_frame_stack_1_224x224_end_on_success" / "dataset.hdf5"
    with h5py.File(path, "r") as f:
        demo = f[f"data/demo_{demo_idx}"]
        img = demo["obs/corner2_image"][:end_t]  # (T, C, H, W) frame-stacked (C=6 typically)
    if img.ndim != 4 or img.shape[1] < 3:
        raise ValueError(f"unexpected image shape {img.shape}")
    cur = img[:, -3:, :, :]  # current-frame slice
    frames_hwc = np.transpose(cur, (0, 2, 3, 1)).astype(np.uint8)
    return frames_hwc


def install_hooks(model: torch.nn.Module):
    """Forward hooks on every nn.Module — record first NaN occurrence."""
    seen_nan = []

    def make_hook(name):
        def _h(_, _in, out):
            if seen_nan:  # already found earliest, skip cheap
                return
            for t in (out if isinstance(out, (tuple, list)) else (out,)):
                if torch.is_tensor(t):
                    if t.dtype.is_floating_point and not torch.isfinite(t).all():
                        n_nan = int(torch.isnan(t).sum())
                        n_inf = int(torch.isinf(t).sum())
                        n_total = int(t.numel())
                        # peek a sample of finite values
                        finite = t[torch.isfinite(t)]
                        mn = float(finite.min()) if finite.numel() else float("nan")
                        mx = float(finite.max()) if finite.numel() else float("nan")
                        seen_nan.append({
                            "name": name, "shape": list(t.shape),
                            "n_nan": n_nan, "n_inf": n_inf, "n_total": n_total,
                            "finite_min": mn, "finite_max": mx,
                        })
                        break
        return _h

    handles = []
    for name, mod in model.named_modules():
        handles.append(mod.register_forward_hook(make_hook(name)))
    return seen_nan, handles


def diag_one(model_path: str, frames: np.ndarray, dtype: str):
    """Load model in given dtype, run scorer on the clip, report first NaN site."""
    sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/Qwen35-FT")
    sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl")

    from env.robometer_utils import RobometerScorer

    print(f"=== loading {model_path}  (dtype={dtype}) ===", flush=True)
    if dtype == "fp32":
        os.environ["ROBOMETER_FORCE_FP32"] = "1"  # marker; not yet hooked
    scorer = RobometerScorer(model_path=model_path, device="cuda")
    # Force-cast the model to fp32 if requested (override what setup_utils picked)
    if dtype == "fp32":
        scorer.model.to(torch.float32)
        for m in scorer.model.modules():
            for p in m.parameters(recurse=False):
                p.data = p.data.float()
        # processor stays the same
        # Hide layernorm autocast: also force compute dtype
    print(f"  model dtype: {next(scorer.model.parameters()).dtype}")

    seen_nan, handles = install_hooks(scorer.model)
    try:
        out = scorer([f for f in frames], task="push a mug under a coffee machine", episode_id=0)
        print(f"  success_prob = {out.get('success_prob')}")
        print(f"  progress     = {out.get('progress_reward')}")
    except Exception as e:
        print(f"  ERROR during scoring: {e}")
    finally:
        for h in handles:
            h.remove()

    if not seen_nan:
        print("  no NaN/Inf detected anywhere ✓")
    else:
        rec = seen_nan[0]
        print(f"  FIRST non-finite output at layer:")
        print(f"    name           : {rec['name']}")
        print(f"    shape          : {rec['shape']}")
        print(f"    n_nan / n_total: {rec['n_nan']} / {rec['n_total']}")
        print(f"    n_inf          : {rec['n_inf']}")
        print(f"    finite range   : [{rec['finite_min']:+.3e}, {rec['finite_max']:+.3e}]")
    print()
    # release GPU memory for next load
    del scorer
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end-t", type=int, default=5,
                    help="prefix length to reproduce (default 5 = first clip of demo 0)")
    args = ap.parse_args()

    frames = load_first_clip(end_t=args.end_t)
    print(f"loaded {len(frames)} frames, shape={frames.shape}, dtype={frames.dtype}")
    print(f"pixel stats: min={frames.min()} max={frames.max()} mean={frames.mean():.1f}")
    print()

    # 1) step-3000 bf16 — should work, sanity baseline
    diag_one("/scratch-shared/pkarageorgis1/Qwen35_FT_consolidated/run4_step3000", frames, "bf16")

    # 2) step-4000 bf16 — expected to produce NaN
    diag_one("/scratch-shared/pkarageorgis1/Qwen35_FT_consolidated/run4_step4000", frames, "bf16")

    # 3) step-4000 fp32 — does the NaN go away?
    diag_one("/scratch-shared/pkarageorgis1/Qwen35_FT_consolidated/run4_step4000", frames, "fp32")


if __name__ == "__main__":
    main()
