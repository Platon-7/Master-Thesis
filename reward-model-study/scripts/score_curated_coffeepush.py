"""CONTROL: score the actual curated v3 CoffeePush eval frames through the SAME
scorer + env we use for the on-policy v3 rollouts (demo2reward_v3). Must
reproduce the reference AUC ~0.97 (succ_mean ~0.77, fail_mean ~0.13).

If it does -> the scorer path is sound in this env, and the on-policy collapse
(diag_v3_cameras corner2 AUC 0.54) is a property of the freshly-generated
frames, NOT the scorer/env.
If it does NOT -> the v3-env scorer itself is off and the comparison is invalid.

Also re-scores the curated frames a second way: rebuilt as a plain list of
PIL.Image (the exact call shape diag_v3_cameras uses) to rule out any
input-format difference.
"""
import os
import sys
import numpy as np

sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl_v3")
sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer")

from env.robometer_utils import get_robometer_4b
import datasets
import imageio.v3 as iio
from PIL import Image
import scipy.stats as st

TASK = "Push a mug under a coffee machine."
MODEL = "/scratch-shared/pkarageorgis1/Robometer_FT_consolidated/run1_icl_ours_step3000"
BAK = ("/projects/prjs1958/robometer_frames_hf_full/eval_metaworld_raw/"
       "robometer_frames_eval_metaworld.bak_pre_drop_metaworld_success_labels")


def auc(l, s):
    l = np.asarray(l); s = np.asarray(s); P = (l == 1).sum(); N = (l == 0).sum()
    if P == 0 or N == 0:
        return float('nan')
    r = st.rankdata(s)
    return float((r[l == 1].sum() - P * (P + 1) / 2) / (P * N))


def main():
    ds = datasets.load_from_disk(BAK)
    cp = [r for r in ds if "coffee_push" in r["id"]]
    print(f"curated coffee_push rows: {len(cp)}")
    sc = get_robometer_4b(model_path=MODEL)
    print(f"scorer max_frames = {getattr(sc, 'max_frames', '?')}", flush=True)

    labels, sps = [], []
    for r in cp:
        mp4 = os.path.join(BAK, r["frames"].split("robometer_frames_eval_metaworld/", 1)[1])
        frames = [np.asarray(f).astype(np.uint8) for f in iio.imiter(mp4, plugin="pyav")]
        pil = [Image.fromarray(f) for f in frames]
        out = sc(pil, task=r["task"], icl_frames=None)
        lab = 1 if r["quality_label"] == "successful" else 0
        labels.append(lab); sps.append(float(out["success_prob"]))
        print(f"  {r['id']:60s} label={lab} sp={out['success_prob']:.3f} "
              f"task={r['task'][:40]!r}", flush=True)

    L = np.array(labels); S = np.array(sps)
    print(f"\n==== curated CoffeePush via v3-env scorer: {len(L)} rows "
          f"({int(L.sum())}s/{int((L == 0).sum())}f) ====")
    print(f"  AUC={auc(L, S):.3f}  succ_mean={S[L == 1].mean():.3f}  fail_mean={S[L == 0].mean():.3f}")
    print(f"  ref: AUC~0.97, succ_mean~0.77, fail_mean~0.13")
    print("  If reproduced -> scorer sound here; on-policy collapse is a frame-content gap.")


if __name__ == "__main__":
    main()
