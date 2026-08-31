#!/usr/bin/env python3
"""Does the progress head's score RISE as the task is completed?

The RL runs can only tell us the joint outcome of (reward model x explorer). This
isolates the reward model: it STAGES the scene at known fractions of task
completion -- teleporting the manipulated object along the object->goal line --
renders each staged trajectory, and scores it. If the score does not increase with
completion fraction, the reward encodes no gradient toward the goal and no amount
of RL tuning can help. It also re-runs the sweep under an alternative camera, since
RollBall's default viewpoint is 1.6x further from the scene than PullCube's.

    python scripts/reward_response_probe.py --task RollBall-v1 --model /path/to/run2
"""
from __future__ import annotations
import argparse, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import numpy as np, torch


def _render_rgb(u, size):
    """One HWC uint8 frame. render() may return a CUDA torch tensor, batched."""
    out = u.render()
    if hasattr(out, "detach"):
        out = out.detach().cpu().numpy()
    arr = np.asarray(out)
    arr = arr.reshape(-1, *arr.shape[-3:])[0]          # drop any batch dim
    if arr.shape[0] != size:
        from PIL import Image
        arr = np.asarray(Image.fromarray(arr.astype(np.uint8)).resize((size, size)))
    return arr.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="RollBall-v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=224)
    args = ap.parse_args()

    import gymnasium as gym, mani_skill.envs  # noqa
    from mani_skill.utils.structs.pose import Pose
    from mani_skill.utils import sapien_utils
    from mani_skill.sensors.camera import CameraConfig
    from robometer.evals.eval_utils import raw_dict_to_sample
    from robometer.evals.eval_server import process_batch_helper
    from robometer.utils.save import load_model_from_hf
    from robometer.utils.setup_utils import setup_batch_collator
    from robometer_policy_learning.utils.robometer_utils import extract_rewards_from_output
    from robometer_policy_learning.envs.maniskill_utils import get_task_spec

    spec = get_task_spec(args.task)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, tok, proc, model = load_model_from_hf(model_path=args.model, device=dev)
    if not getattr(cfg.data, "use_multi_image", False):
        cfg.data.use_multi_image = True
    coll = setup_batch_collator(proc, tok, cfg, is_eval=True)
    MF = int(getattr(cfg.data, "max_frames", 16))
    _plt = str(cfg.loss.progress_loss_type).lower()
    is_disc = _plt == "discrete" or "c51" in _plt
    nb = cfg.loss.progress_discrete_bins

    def score(frames, instr):
        fr = np.stack(frames)
        if len(fr) < MF:
            fr = fr[np.linspace(0, len(fr) - 1, MF).round().astype(int)]
        raw = dict(frames=fr, task=instr, id=0,
                   metadata=dict(subsequence_length=len(fr)),
                   video_embeddings=None, text_embedding=None)
        s = raw_dict_to_sample(raw_data=raw, max_frames=MF, sample_type="progress")
        out = process_batch_helper(model_type=cfg.model.model_type, model=model, tokenizer=tok,
                                   batch_collator=coll, device=model.device, batch_data=[s.model_dump()],
                                   job_id=0, is_discrete_mode=is_disc, num_bins=nb)
        return float(extract_rewards_from_output(out)[0])

    # (moved object attr, goal attr, whether ORIENTATION must be interpolated too).
    # PushT is the odd one out: its goal is a second T-shaped actor and success is
    # 90% AREA OVERLAP, so staging must rotate the block into the target pose as
    # well as translate it -- position-only staging would never reach "done".
    GEOM = {"RollBall-v1":  ("ball", "goal_region", False),
            "PullCube-v1":  ("obj",  "goal_region", False),
            "PokeCube-v1":  ("cube", "goal_region", False),
            "PushT-v1":     ("tee",  "goal_tee",    True)}
    objname, goalname, interp_rot = GEOM.get(args.task, ("obj", "goal_region", False))

    CAMS = {"default": None,
            "pullcube_like": CameraConfig("render_camera",
                sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35]), 512, 512, 1, 0.01, 100)}

    fracs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for camname, cam in CAMS.items():
        kw = dict(num_envs=1, obs_mode="state", control_mode=spec.control_mode,
                  sim_backend="physx_cpu", max_episode_steps=spec.max_episode_steps,
                  reward_mode="normalized_dense", render_mode="rgb_array")
        if cam is not None:
            kw["human_render_camera_configs"] = dict(pose=cam.pose)
        env = gym.make(args.task, **kw)
        rows = {f: [] for f in fracs}
        pixdiff = []
        for ep in range(args.episodes):
            env.reset(seed=4000 + ep)
            u = env.unwrapped
            obj = getattr(u, objname)
            goal = getattr(u, goalname)
            p0 = obj.pose.p.clone()
            g = goal.pose.p.clone()
            q = obj.pose.q.clone()
            qg = goal.pose.q.clone()
            # z-euler of start and goal, for tasks needing rotation staging
            def _zang(quat):
                w, x, y, z = [float(quat[0, i]) for i in range(4)]
                return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            a0, ag = _zang(q), _zang(qg)
            da = (ag - a0 + math.pi) % (2 * math.pi) - math.pi   # shortest path
            for f in fracs:
                frames = []
                for k in np.linspace(0.0, f, MF):      # a clip showing progress up to f
                    p = p0.clone()
                    p[0, 0] = p0[0, 0] + k * (g[0, 0] - p0[0, 0])
                    p[0, 1] = p0[0, 1] + k * (g[0, 1] - p0[0, 1])
                    qk = q
                    if interp_rot:
                        ang = a0 + k * da
                        qk = torch.tensor(
                            [[math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2)]],
                            dtype=q.dtype, device=q.device)
                    obj.set_pose(Pose.create_from_pq(p=p, q=qk))
                    if hasattr(u.scene, "_gpu_apply_all"):
                        try: u.scene._gpu_apply_all()
                        except Exception: pass
                    frames.append(_render_rgb(u, args.image_size))
                rows[f].append(score(frames, spec.instruction))
                if f == 0.0:
                    _first = frames[-1]
                elif f == 1.0:
                    # SANITY: staging must actually change the picture. If the frames
                    # at 0% and 100% completion are identical, set_pose never reached
                    # the renderer and a "flat" score would be an artefact, not a result.
                    _d = float(np.mean(np.abs(_first.astype(np.int16) - frames[-1].astype(np.int16))))
                    pixdiff.append(_d)
        print(f"\n=== {args.task}  camera={camname}  instr={spec.instruction!r} ===")
        print(f"  {'completion':>11} {'mean progress':>14} {'sd':>7}")
        for f in fracs:
            v = rows[f]
            print(f"  {f*100:10.0f}% {np.mean(v):14.4f} {np.std(v):7.4f}")
        md = float(np.mean(pixdiff)) if pixdiff else float("nan")
        print(f"  [staging check] mean |pixel diff| between 0% and 100% frames = {md:.2f}"
              f"   {'OK' if md > 1.0 else '*** STAGING DID NOT RENDER -- result invalid ***'}")
        base, top = np.mean(rows[0.0]), np.mean(rows[1.0])
        print(f"  --> score at DONE minus score at START = {top-base:+.4f}"
              f"   ({'RESPONDS' if top-base > 0.05 else 'FLAT -- no gradient toward the goal'})")
        env.close()


if __name__ == "__main__":
    sys.exit(main())
