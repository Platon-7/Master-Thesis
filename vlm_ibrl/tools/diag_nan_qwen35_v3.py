"""v3 NaN diag — score 20 prefix lengths (5,10,...,100) in sequence per model,
then re-score end_t=5 to detect state pollution."""
import sys, h5py, numpy as np, torch
from pathlib import Path

def load_demo(demo_idx=0):
    p = "release/data/metaworld/CoffeePush_frame_stack_1_224x224_end_on_success/dataset.hdf5"
    with h5py.File(p, "r") as f:
        img = f[f"data/demo_{demo_idx}"]["obs/corner2_image"][:]   # (T, C, H, W)
    cur = img[:, -3:, :, :]
    return np.transpose(cur, (0, 2, 3, 1)).astype(np.uint8)   # (T, H, W, 3)

def run(model_path):
    sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/Qwen35-FT")
    sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl")
    from env.robometer_utils import RobometerScorer
    print(f"=== {model_path} ===", flush=True)
    scorer = RobometerScorer(model_path=model_path, device="cuda")
    frames = load_demo(demo_idx=0)
    print(f"  demo frames: {frames.shape}")
    prev = None
    for end_t in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
        clip = [frames[i] for i in range(end_t)]
        out = scorer(clip, task="push a mug under a coffee machine", episode_id=0)
        sp = out.get("success_prob"); pr = out.get("progress_reward")
        flag = " ⚠NaN" if (sp != sp) else ""   # nan != nan
        print(f"  end_t={end_t:>3}  sp={sp}  progress={pr}{flag}")
        prev = sp
    # Re-score end_t=5 to test state pollution
    print("  -- re-score end_t=5 after the sweep --")
    clip = [frames[i] for i in range(5)]
    out = scorer(clip, task="push a mug under a coffee machine", episode_id=0)
    flag = " ⚠NaN" if (out["success_prob"] != out["success_prob"]) else ""
    print(f"  end_t=  5 (re)  sp={out['success_prob']}  progress={out['progress_reward']}{flag}")
    print()
    del scorer; torch.cuda.empty_cache()

for step in [3000, 4000, 5000]:
    run(f"/scratch-shared/pkarageorgis1/Qwen35_FT_consolidated/run4_step{step}")
