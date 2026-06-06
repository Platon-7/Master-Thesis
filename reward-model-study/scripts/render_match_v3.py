"""Find which (camera_name, flip) of the v3 env reproduces the curated-data
rendering domain for CoffeePush. Renders all 6 model cameras raw + vertically
flipped, saves PNGs, and ranks each by MSE against a real curated v3 CoffeePush
success frame (background/table framing dominates the MSE since object/goal
placement differs between episodes).

Run on a GPU node (EGL). Inspect the saved PNGs + the printed MSE ranking to
pick the camera+orientation that matches `curated_coffeepush_mid.png`.
"""
import os
import sys
import numpy as np

sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl_v3")
import mujoco
import metaworld
from PIL import Image
import imageio.v3 as iio

OUT = "/gpfs/home3/pkarageorgis1/Master-Thesis/reward-model-study/results/v3_sample_frames"
BAK = ("/projects/prjs1958/robometer_frames_hf_full/eval_metaworld_raw/"
       "robometer_frames_eval_metaworld.bak_pre_drop_metaworld_success_labels")
CAMS = ["topview", "corner", "corner2", "corner3", "behindGripper", "gripperPOV"]
RES = 240  # curated frames are 240x240


def main():
    os.makedirs(OUT, exist_ok=True)
    mt1 = metaworld.MT1("coffee-push-v3", seed=42)
    cls = mt1.train_classes["coffee-push-v3"]
    env = cls(render_mode="rgb_array", camera_name="corner2")
    env.set_task(list(mt1.train_tasks)[0])
    env._freeze_rand_vec = False
    obs, _ = env.reset()
    import metaworld.policies as pol
    policy = pol.SawyerCoffeePushV3Policy()
    # advance to a mid-episode state so the scene has the mug/machine clearly placed
    for _ in range(40):
        a = policy.get_action(obs).clip(-1, 1)
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break

    rends = {}
    for cam in CAMS:
        try:
            img = env.mujoco_renderer.render("rgb_array", camera_name=cam)
            img = np.asarray(img)
            raw = np.asarray(Image.fromarray(img).resize((RES, RES), Image.BILINEAR))
            flp = np.asarray(Image.fromarray(img[::-1].copy()).resize((RES, RES), Image.BILINEAR))
            rends[(cam, "raw")] = raw
            rends[(cam, "flip")] = flp
            Image.fromarray(raw).save(os.path.join(OUT, f"v3render_{cam}_raw.png"))
            Image.fromarray(flp).save(os.path.join(OUT, f"v3render_{cam}_flip.png"))
            print(f"rendered {cam}: native {img.shape}", flush=True)
        except Exception as e:
            print(f"camera {cam} FAILED: {type(e).__name__}: {e}", flush=True)

    # load a curated success frame to rank against
    import datasets
    ds = datasets.load_from_disk(BAK)
    succ = [r for r in ds if "coffee_push" in r["id"] and r["quality_label"] == "successful"]
    cur = None
    if succ:
        r = succ[0]
        mp4 = os.path.join(BAK, r["frames"].split("robometer_frames_eval_metaworld/", 1)[1])
        fr = [np.asarray(f) for f in iio.imiter(mp4, plugin="pyav")]
        cur = fr[len(fr) // 2].astype(np.float32)
        print(f"\ncurated ref: {r['id']} ({len(fr)} frames @ {fr[0].shape})")

    if cur is not None:
        scores = []
        for (cam, mode), img in rends.items():
            mse = float(np.mean((img.astype(np.float32) - cur) ** 2))
            scores.append((mse, cam, mode))
        scores.sort()
        print("\n==== MSE vs curated CoffeePush success frame (lower=closer framing) ====")
        for mse, cam, mode in scores:
            print(f"  {cam:14s} {mode:4s}  MSE={mse:9.1f}")
        print(f"\nBEST: {scores[0][1]} ({scores[0][2]})")
    print(f"\nsaved candidate renders + curated frames to {OUT}")


if __name__ == "__main__":
    main()
