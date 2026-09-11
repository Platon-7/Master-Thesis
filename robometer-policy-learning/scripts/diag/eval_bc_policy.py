"""Does a BC policy trained on our demos actually SOLVE the task?

The training runs never produced this number: eval_freq (10000) exceeds
num_offline_steps (5000), so the offline evaluation never fires. Without it we
cannot distinguish "BC failed to learn a competent policy" from "BC learned one
and SAC destroyed it in the first updates against an untrained critic" -- the two
have completely different fixes.

Trains BC exactly as train.py does, then evaluates the resulting actor greedily.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL","egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch, argparse

ap=argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--demo-h5", required=True)
ap.add_argument("--bc-steps", type=int, default=5000)
ap.add_argument("--episodes", type=int, default=20)
a=ap.parse_args()

from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from robometer_policy_learning.envs.maniskill_utils import get_task_spec
from robometer_policy_learning.utils.training_utils import setup_training, create_buffer, build_actor_critic_models
from robometer_policy_learning.rollouts.rollout_worker import extract_info_for_env
from robometer_policy_learning.utils.gpu_utils import convert_to_tensor, move_to_device

initialize_config_dir(config_dir=os.path.abspath("robometer_policy_learning/configs"), version_base=None)
cfg = compose(config_name="maniskill_online_rl", overrides=[
    f"env.env_name=maniskill/{a.task}", "algorithm@online_algorithm=sac", "alg.online_alg_name=sac",
    "algorithm@offline_algorithm=bc", "alg.offline_alg_name=bc",
    f"+env.h5_dataset_path={a.demo_h5}", f"training.num_offline_steps={a.bc_steps}",
    "offline_algorithm.loss_type=mse", "buffer.sample_ratio=0",
    "training.num_rollouts=0", "env.reward_shift=0.0",
] + ([f"env.max_episode_steps=80"] if a.task=="RollBall-v1" else []))

comp = setup_training(cfg)
actor, env, eval_env = comp.actor, comp.env, comp.eval_env
dev = next(actor.parameters()).device
buf = create_buffer(sampler=None, use_eval_server=False, eval_server_url=None, eval_server_timeout=None,
    reward_model=None, reward_model_exp_cfg=None, use_gt_rewards=False, use_relative_rewards=False,
    capacity=0, remove_obs_keys=list(cfg.env.extra_keys_to_drop or []), post_transforms=[],
    h5_paths=[a.demo_h5], use_full_state=False, sentence_model=None, dinov2_model=None,
    dinov2_processor=None, reward_relabeling_keys=["image"])
from robometer_policy_learning.algorithms.bc.configuration_bc import BCConfig
bc_dict = OmegaConf.to_container(cfg.offline_algorithm)
bc_cfg = BCConfig(**bc_dict); bc_cfg.env=env; bc_cfg.actor=actor; bc_cfg.critic=comp.critic
bc_cfg.buffer=buf; bc_cfg.logger=None
bc = bc_cfg.create()
print(f"[bc-eval] training BC for {a.bc_steps} steps ...", flush=True)
for i in range(a.bc_steps):
    m = bc.train_step()
    if (i+1) % 1000 == 0: print(f"   step {i+1}: mse={m.get('mse_error', float('nan')):.5f}", flush=True)

spec = get_task_spec(a.task)
actor = bc.actor; actor.eval()
succ = 0
for ep in range(a.episodes):
    obs,_ = eval_env.reset(seed=90000+ep); done=False
    for t in range(spec.max_episode_steps):
        with torch.no_grad():
            ot = move_to_device(convert_to_tensor(obs), dev)
            act,_ = actor.act(ot, actor_state=None, deterministic=True)
        obs, r, term, trunc, infos = eval_env.step(act.detach().cpu().numpy())
        info_i = extract_info_for_env(infos, 0, 1)
        if bool(info_i.get("success", False)): done=True
        if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]): break
    succ += int(done)
print(f"\n[bc-eval] {a.task}: BC policy success = {succ}/{a.episodes} = {succ/a.episodes*100:.0f}%")
print("  >50%  -> BC works; a 0% first online eval means SAC destroyed the warm start")
print("  ~0%   -> BC cloned actions but the policy does not solve the task")
