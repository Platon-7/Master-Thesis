"""v2 NaN diag — hooks ONLY on the success_head + model output, not every module.
Also dumps the raw model_output.success_logits scalar (pre-sigmoid)."""
import sys, json, numpy as np, torch
from pathlib import Path

def load_clip(end_t=5):
    import h5py
    path = Path("release/data/metaworld/CoffeePush_frame_stack_1_224x224_end_on_success/dataset.hdf5")
    with h5py.File(path, "r") as f:
        demo = f["data/demo_0"]
        img = demo["obs/corner2_image"][:end_t]
    cur = img[:, -3:, :, :]
    return np.transpose(cur, (0, 2, 3, 1)).astype(np.uint8)

def run(model_path):
    sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/Qwen35-FT")
    sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl")
    from env.robometer_utils import RobometerScorer
    print(f"=== {model_path} ===", flush=True)
    scorer = RobometerScorer(model_path=model_path, device="cuda")

    # SMALL set of targeted hooks — only the heads + frame_pool_attn
    captured = {}
    def grab(name):
        def _h(_, _in, out):
            t = out if torch.is_tensor(out) else (out[0] if isinstance(out,(list,tuple)) else None)
            if t is None or not torch.is_tensor(t): return
            captured[name] = {
                "shape": list(t.shape), "dtype": str(t.dtype),
                "n_nan": int(torch.isnan(t).sum()),
                "n_inf": int(torch.isinf(t).sum()),
                "n_total": int(t.numel()),
                "min_finite": float(t[torch.isfinite(t)].min()) if torch.isfinite(t).any() else float("nan"),
                "max_finite": float(t[torch.isfinite(t)].max()) if torch.isfinite(t).any() else float("nan"),
            }
        return _h

    for name, mod in scorer.model.named_modules():
        if any(s in name for s in ["success_head", "progress_head", "frame_pool_attn"]):
            mod.register_forward_hook(grab(name))

    frames = load_clip(end_t=5)
    out = scorer([f for f in frames], task="push a mug under a coffee machine", episode_id=0)
    print(f"  success_prob = {out.get('success_prob')}")
    print(f"  progress     = {out.get('progress_reward')}")
    for k, v in captured.items():
        flag = ""
        if v["n_nan"] > 0: flag = f" ⚠NaN ({v['n_nan']}/{v['n_total']})"
        elif v["n_inf"] > 0: flag = f" ⚠Inf ({v['n_inf']}/{v['n_total']})"
        print(f"  {k:35s} shape={str(v['shape']):20s} finite=[{v['min_finite']:+.3e}, {v['max_finite']:+.3e}]{flag}")
    print()
    del scorer; torch.cuda.empty_cache()

for step in [3000, 4000, 5000]:
    run(f"/scratch-shared/pkarageorgis1/Qwen35_FT_consolidated/run4_step{step}")
