"""Verify that reward_response_probe's scene staging actually reaches the renderer.

Writes a filmstrip (rows = tasks, cols = 0/25/50/75/100% completion) plus a
changed-pixel count. Mean-absolute-pixel-diff is a poor validity metric for small
objects -- a 4cm cube at 1m fills ~10x10 of a 224^2 frame -- so this reports the
number of pixels changing by >30 levels, which does not wash out with object size.
"""
import math, sys, os, numpy as np, torch
import gymnasium as gym, mani_skill.envs  # noqa
from mani_skill.utils.structs.pose import Pose
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from robometer_policy_learning.envs.maniskill_utils import get_task_spec
import imageio.v3 as iio

GEOM = {"RollBall-v1": ("ball", "goal_region", False),
        "PullCube-v1": ("obj", "goal_region", False),
        "PushT-v1":    ("tee", "goal_tee", True)}

def rgb(u, size=224):
    o = u.render()
    if hasattr(o, "detach"): o = o.detach().cpu().numpy()
    a = np.asarray(o); a = a.reshape(-1, *a.shape[-3:])[0]
    if a.shape[0] != size:
        from PIL import Image
        a = np.asarray(Image.fromarray(a.astype(np.uint8)).resize((size, size)))
    return a.astype(np.uint8)

rows = []
for task in ["PullCube-v1", "PushT-v1", "RollBall-v1"]:
    spec = get_task_spec(task); on, gn, rot = GEOM[task]
    env = gym.make(task, num_envs=1, obs_mode="state", control_mode=spec.control_mode,
                   sim_backend="physx_cpu", max_episode_steps=spec.max_episode_steps,
                   reward_mode="normalized_dense", render_mode="rgb_array")
    env.reset(seed=4000); u = env.unwrapped
    obj = getattr(u, on); goal = getattr(u, gn)
    p0 = obj.pose.p.clone(); g = goal.pose.p.clone()
    q = obj.pose.q.clone(); qg = goal.pose.q.clone()
    def za(qq):
        w, x, y, z = [float(qq[0, i]) for i in range(4)]
        return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    a0, ag = za(q), za(qg); da = (ag - a0 + math.pi) % (2*math.pi) - math.pi
    print(f"\n{task}: object start p={p0[0,:2].tolist()}  goal p={g[0,:2].tolist()}")
    fr = []; base = None
    for k in (0.0, 0.25, 0.5, 0.75, 1.0):
        p = p0.clone(); p[0,0] = p0[0,0] + k*(g[0,0]-p0[0,0]); p[0,1] = p0[0,1] + k*(g[0,1]-p0[0,1])
        qk = q
        if rot:
            an = a0 + k*da
            qk = torch.tensor([[math.cos(an/2), 0., 0., math.sin(an/2)]], dtype=q.dtype, device=q.device)
        obj.set_pose(Pose.create_from_pq(p=p, q=qk))
        im = rgb(u); fr.append(im)
        act = obj.pose.p[0,:2].tolist()
        if base is None:
            base = im; print(f"    k=0.00 pose_after_set={[round(v,3) for v in act]}")
        else:
            d = np.abs(base.astype(np.int16) - im.astype(np.int16))
            print(f"    k={k:.2f} pose_after_set={[round(v,3) for v in act]} "
                  f"mean|diff|={d.mean():5.2f} changed_px(>30)={int((d.max(axis=2)>30).sum())}")
    rows.append(np.concatenate(fr, axis=1)); env.close()
h = min(r.shape[0] for r in rows); w = min(r.shape[1] for r in rows)
img = np.concatenate([r[:h,:w] for r in rows], axis=0)
img = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)
out = "/scratch/%s/staging_check.png" % os.environ.get("USER", "15552055")
iio.imwrite(out, img); print("\nwrote", out, img.shape, "(rows: PullCube, PushT, RollBall; cols 0/25/50/75/100%)")
