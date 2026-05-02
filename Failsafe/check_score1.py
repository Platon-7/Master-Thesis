"""Quick replay of FailPushCube score-1 seed=1 to check tcp/obj distances."""
import sys, os, numpy as np
os.environ["MUJOCO_GL"] = "egl"
FAILSAFE_PATH = "/home/pkarageorgis1/Master-Thesis/Failsafe/failsafe_repo"
sys.path.insert(0, FAILSAFE_PATH)
import importlib
for mod in ["failgen.tasks.fail_push_cube"]:
    importlib.import_module(mod)
import gymnasium as gym

seed = 1; max_steps = 25
env = gym.make("FailPushCube-v1", obs_mode="rgb", render_mode="rgb_array", sensor_configs=dict())
env.action_space.seed(seed)
obs, info = env.reset(seed=seed)
uw = env.unwrapped
obj_p = np.asarray(uw.obj.pose.p.cpu().numpy()).reshape(-1)[-3:]
tcp_p = np.asarray(uw.agent.tcp.pose.p.cpu().numpy()).reshape(-1)[-3:]
away = tcp_p - obj_p
away_xy = away[:2]
n = float(np.linalg.norm(away_xy))
if n < 1e-6:
    away_xy = np.array([1.0, 0.0])
else:
    away_xy = away_xy / n
print(f"Initial: obj={obj_p}, tcp={tcp_p}, away_xy={away_xy}")
rng = np.random.default_rng(seed)
lo, hi = env.action_space.low, env.action_space.high
keyframes = [0,4,7,11,14,18,21,25]
for step in range(max_steps+1):
    action = env.action_space.sample()
    if len(action) >= 3:
        action[0] = float(np.clip(away_xy[0] * 0.95 + rng.normal(0, 0.08), lo[0], hi[0]))
        action[1] = float(np.clip(away_xy[1] * 0.95 + rng.normal(0, 0.08), lo[1], hi[1]))
        action[2] = float(np.clip(0.5 + rng.normal(0, 0.08), lo[2], hi[2]))
    obs, reward, terminated, truncated, info = env.step(action)
    obj_p = np.asarray(uw.obj.pose.p.cpu().numpy()).reshape(-1)[-3:]
    tcp_p = np.asarray(uw.agent.tcp.pose.p.cpu().numpy()).reshape(-1)[-3:]
    dist = float(np.linalg.norm(tcp_p - obj_p))
    if step in keyframes:
        print(f"  step {step}: tcp={tcp_p.round(3)}, obj={obj_p.round(3)}, dist={dist:.4f}")
    if terminated or truncated:
        break
env.close()
