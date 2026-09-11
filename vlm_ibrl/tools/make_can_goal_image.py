#!/usr/bin/env python3
"""Render a RoboDopamine goal (reference-END) image for Robomimic PickPlace-Can.

The MetaWorld goal generator (reward-model-study/scripts/gen_robodopamine_goals.py)
rolls out a scripted expert, which Robomimic has no equivalent of. Instead we take a
SUCCESSFUL demonstration from release/data/robomimic/can/processed_data96.hdf5 and
restore its FINAL simulator state, then render agentview at full resolution.

The demo file's own images are the 96x96 wrist camera, which is the wrong view and
too small to anchor on; restoring the state and re-rendering gives a proper
agentview frame that matches what the reward model sees at run time.

  python tools/make_can_goal_image.py --out /projects/.../can_goal.png
"""
import argparse, os
import numpy as np
import h5py
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="release/data/robomimic/can/processed_data96.hdf5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--camera", default="agentview")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--demo", default=None, help="demo key; default = first successful")
    ap.add_argument("--also-start", default=None,
                    help="also render the demo's FIRST state here, to verify the can moved")
    a = ap.parse_args()

    f = h5py.File(a.h5, "r")
    demos = list(f["data"].keys())
    # a Robomimic demo is successful iff its last reward is 1
    if a.demo:
        key = a.demo
    else:
        key = next((d for d in demos if float(np.array(f[f"data/{d}/rewards"])[-1]) == 1.0), None)
        if key is None:
            raise SystemExit(f"no successful demo among {len(demos)}")
    states = np.array(f[f"data/{key}/states"])
    rew = np.array(f[f"data/{key}/rewards"])
    print(f"demo={key}  T={len(states)}  final_reward={rew[-1]}")

    from env.robosuite_wrapper import PixelRobosuite
    env = PixelRobosuite(
        env_name="PickPlaceCan", robots="Panda", episode_length=200,
        image_size=a.image_size, rl_image_size=96,
        camera_names=[a.camera], rl_cameras=[a.camera],
        end_on_success=False, use_state=False,
    )
    env.reset()
    # restore the demonstration's terminal simulator state and settle the renderer
    sim = env.env.sim
    sim.set_state_from_flattened(states[-1])
    sim.forward()
    def render_state(st):
        sim.set_state_from_flattened(st); sim.forward()
        return env.env.sim.render(width=a.image_size, height=a.image_size,
                                  camera_name=a.camera)[::-1]   # mujoco renders bottom-up

    if a.also_start:
        s0 = render_state(states[0])
        os.makedirs(os.path.dirname(os.path.abspath(a.also_start)), exist_ok=True)
        Image.fromarray(s0.astype(np.uint8)).save(a.also_start)
        print(f"wrote {a.also_start} (START state, for comparison)")
    frame = render_state(states[-1])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    Image.fromarray(frame.astype(np.uint8)).save(a.out)
    print(f"wrote {a.out}  shape={frame.shape}  mean={frame.mean():.1f} std={frame.std():.1f}")


if __name__ == "__main__":
    main()
