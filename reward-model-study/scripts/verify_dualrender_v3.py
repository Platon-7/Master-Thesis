"""Verify dual-render: with V3_CORNER2_ZOOM=1, the policy camera "corner2" is
zoomed (v2 framing) while the pseudo-camera "corner2_default" renders the
un-zoomed corner2 (reward model's in-domain view). Confirms BEFORE the long
pulse runs that the reward camera is NOT accidentally zoomed.

Expect: corner2_default ~ low MSE vs a curated CoffeePush success frame
(in-domain, ~400 like render_match), corner2(zoom) ~ much higher MSE.
"""
import os
import sys
import numpy as np

os.environ.setdefault("V3_CORNER2_ZOOM", "1")
sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl_v3")
from env.metaworld_wrapper import MetaWorldEnv
import datasets, imageio.v3 as iio
from PIL import Image

OUT = "/gpfs/home3/pkarageorgis1/Master-Thesis/reward-model-study/results/v3_sample_frames"
BAK = ("/projects/prjs1958/robometer_frames_hf_full/eval_metaworld_raw/"
       "robometer_frames_eval_metaworld.bak_pre_drop_metaworld_success_labels")


def main():
    os.makedirs(OUT, exist_ok=True)
    print("V3_CORNER2_ZOOM =", os.environ.get("V3_CORNER2_ZOOM"))
    env = MetaWorldEnv("CoffeePush", camera_name="corner2", width=240, height=240)
    env.reset()
    for _ in range(40):
        a = env.get_heuristic_action()
        _, _, d, info = env.step(a)
        if d:
            break
    zoom = env.render(camera_name="corner2", width=240, height=240)        # policy view (zoomed)
    deflt = env.render(camera_name="corner2_default", width=240, height=240)  # reward view (default)
    Image.fromarray(zoom).save(os.path.join(OUT, "dual_corner2_zoom.png"))
    Image.fromarray(deflt).save(os.path.join(OUT, "dual_corner2_default.png"))
    # sanity: the two views must differ (zoom actually applied)
    diff = float(np.mean((zoom.astype(np.float32) - deflt.astype(np.float32)) ** 2))
    print(f"MSE(zoom vs default) = {diff:.1f}  (should be large -> zoom is applied)")

    ds = datasets.load_from_disk(BAK)
    succ = [r for r in ds if "coffee_push" in r["id"] and r["quality_label"] == "successful"]
    r = succ[0]
    mp4 = os.path.join(BAK, r["frames"].split("robometer_frames_eval_metaworld/", 1)[1])
    fr = [np.asarray(f) for f in iio.imiter(mp4, plugin="pyav")]
    cur = fr[len(fr) // 2].astype(np.float32)
    m_def = float(np.mean((deflt.astype(np.float32) - cur) ** 2))
    m_zoom = float(np.mean((zoom.astype(np.float32) - cur) ** 2))
    print(f"MSE(corner2_default vs curated) = {m_def:.1f}   (expect LOW ~400 -> reward in-domain)")
    print(f"MSE(corner2_zoom    vs curated) = {m_zoom:.1f}   (expect HIGH -> policy view, OOD for reward)")
    ok = m_def < 1500 and m_zoom > m_def * 2
    print("DUALRENDER_OK" if ok else "DUALRENDER_CHECK_FAILED")


if __name__ == "__main__":
    main()
