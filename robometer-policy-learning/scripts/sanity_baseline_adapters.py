#!/usr/bin/env python3
"""Sanity-check the RoboDopamine / LRM adapters on REAL ManiSkill demo frames.

The useful property to test is monotonicity: a demo is a successful trajectory,
so a clip covering more of it should score higher progress than a clip covering
only its opening. A wired-up-but-broken adapter typically returns a constant
(often 0.0), which this catches; a correct one separates early from late.

  python scripts/sanity_baseline_adapters.py --model lrm
  python scripts/sanity_baseline_adapters.py --model robodopamine
"""
import argparse, os
import h5py, numpy as np

DEMOS = "/scratch-shared/pkarageorgis1/maniskill_assets/demos_converted/PullCube-v1_bc_clip.h5"
TASK = "Pull the cube to the goal region"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lrm", "robodopamine", "roboreward"])
    ap.add_argument("--h5", default=DEMOS)
    ap.add_argument("--episodes", type=int, default=3)
    a = ap.parse_args()

    from robometer_policy_learning.utils.baseline_reward_adapters import load_baseline_reward_model
    path = {"lrm": "USC-PSI-Lab/LRM-models",
            "robodopamine": "tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview",
            "roboreward": "teetone/RoboReward-4B"}[a.model]
    _, _, _, rm = load_baseline_reward_model(a.model, path, device="cuda")
    print(f"loaded {a.model}; model_type={rm.model_type}", flush=True)

    f = h5py.File(a.h5, "r")
    keys = list(f["data"].keys())[: a.episodes]
    ok = 0
    for k in keys:
        frames = np.array(f["data"][k]["obs"]["image"])       # (T, H, W, C) uint8
        T = len(frames)
        early = frames[: max(2, T // 3)]
        full = frames
        r_early, s_early = rm.score_clip(early, TASK)
        r_full, s_full = rm.score_clip(full, TASK)
        rose = r_full > r_early
        ok += bool(rose)
        print(f"  {k}: T={T:3d}  progress(early {len(early)}f)={r_early:.3f}  "
              f"progress(full)={r_full:.3f}  {'RISES' if rose else 'flat/falls'}  "
              f"succ_prob={s_full:.3f}", flush=True)

    print(f"\n{ok}/{len(keys)} episodes scored higher on the fuller clip")
    print("VERDICT:", "PLAUSIBLE" if ok >= max(1, len(keys) // 2) else
          "SUSPECT -- adapter may be returning a constant")


if __name__ == "__main__":
    main()
