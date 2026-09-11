"""Same policy, greedy vs sampled. Isolates action noise from everything else.

The BC warm start measures ~25% under evaluation_worker (deterministic=True) but
~4% in rollout_worker (deterministic=False). Shrinking log_std_init to -2.0 did not
close that gap, so either the sampling is still the cause and sigma is not what I
think it is, or the difference lives elsewhere in the rollout path. This loads ONE
checkpoint and evaluates it both ways in the same process and same env.
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL","egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch

ap=argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--ckpt", required=True, help="dir containing actor.pt")
ap.add_argument("--episodes", type=int, default=50)
ap.add_argument("--num-envs", type=int, default=1)
a=ap.parse_args()

from robometer_policy_learning.envs.maniskill_utils import get_task_spec
from robometer_policy_learning.utils.env_utils import make_env
from robometer_policy_learning.rollouts.rollout_worker import extract_info_for_env
from robometer_policy_learning.utils.gpu_utils import convert_to_tensor, move_to_device
from transformers import AutoImageProcessor, AutoModel

spec=get_task_spec(a.task)
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
dino=AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
dproc=AutoImageProcessor.from_pretrained("facebook/dinov2-base")
env,_=make_env(env_name=f"maniskill/{a.task}", num_envs=a.num_envs, chunk_size=None,
    max_episode_steps=spec.max_episode_steps, use_full_state=False,
    dinov2_model=dino, dinov2_processor=dproc, device=str(dev), terminate_on_success=False,
    env_kwargs={"sim_backend":"physx_cpu","image_size":224,
                "control_mode":spec.control_mode,"reward_mode":"normalized_dense"})
actor=torch.load(os.path.join(a.ckpt,"actor.pt"), map_location=dev, weights_only=False)
actor.eval()

# report the policy's actual sigma
try:
    b=actor.log_std_layer.bias.detach().cpu().numpy()
    print(f"[ckpt] log_std bias={np.round(b,3)} -> sigma={np.round(np.exp(b),4)}")
    print(f"[ckpt] log_std weight |max|={float(actor.log_std_layer.weight.abs().max()):.5f} "
          f"(0 => state-independent)")
except Exception as e:
    print("[ckpt] could not read log_std:", e)

N=a.num_envs
for det in (True, False):
    succ=0; total=0
    rounds=max(1, a.episodes//N)
    for ep in range(rounds):
        obs,_=env.reset(seed=50000+ep*N)
        solved=[False]*N
        for t in range(spec.max_episode_steps):
            with torch.no_grad():
                ot=move_to_device(convert_to_tensor(obs), dev)
                act,_=actor.act(ot, actor_state=None, deterministic=det)
            obs,r,term,trunc,infos=env.step(act.detach().cpu().numpy())
            for i in range(N):
                if bool(extract_info_for_env(infos,i,N).get("success",False)): solved[i]=True
            if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]): break
        succ+=sum(solved); total+=N
    print(f"[eval] num_envs={N} deterministic={str(det):5s} -> {succ}/{total} = {succ/total*100:.1f}%")
print("\n  If greedy>>sampled: it IS the action noise.")
print("  If both similar   : the gap lives in the rollout path, not the policy.")
