#!/usr/bin/env python3
"""
Generate failure trajectories for ManiSkill tasks using FailSafe's
stage-based motion-planning failure injection.

See RUBRIC.md for the complete failure specification.
Each slot = one unique episode with a specific failure mode.

Modes:
  random_away     — away-biased random actions (score 1)
  zero_freeze     — zero actions, arm stays still (score 1)
  freeze          — solver with noise=0, truncate at fraction (scores 2-4)
  fail_inject     — solver with noise, accept failures (scores 2-4)
  success         — solver with noise=0, accept success (score 0/5)
  success_extend  — solver to success, then extend with bad actions (bonus)
  overshoot       — solver with noise, accept overshoot pattern (bonus)

Usage:
    python generate_failures.py --output-dir output/sample_v1
"""

import sys
import os
import json
import shutil
import argparse
import numpy as np
from pathlib import Path
from contextlib import contextmanager
from PIL import Image

# ---------------------------------------------------------------------------
# FailSafe import (needs FAILSAFE_PATH env var)
# ---------------------------------------------------------------------------

FAILSAFE_PATH = os.environ.get("FAILSAFE_PATH", "")
if FAILSAFE_PATH:
    sys.path.insert(0, FAILSAFE_PATH)

try:
    from failgen.env_wrapper import FailgenWrapper
    import yaml as _yaml
    FAILSAFE_AVAILABLE = True
except ImportError:
    FAILSAFE_AVAILABLE = False
    print("WARNING: FailSafe not importable. Set FAILSAFE_PATH.\n")


# ---------------------------------------------------------------------------
# Task slot configurations — see RUBRIC.md for full specification
# ---------------------------------------------------------------------------

TASK_CONFIGS = {
    "FailPickCube-v1": {
        "description": "Pick up the red cube and lift it to the goal position.",
        "slots": [
            # --- Score 1 ---
            {"id": "pick_s1_away",   "target_score": 1, "mode": "random_away",
             "stage_name": "random_away", "notes": "arm moves away from cube"},
            {"id": "pick_s1_freeze", "target_score": 1, "mode": "zero_freeze",
             "stage_name": "total_freeze", "notes": "arm does not move at all"},
            # --- Score 2 ---
            {"id": "pick_s2_approach_freeze", "target_score": 2, "mode": "freeze",
             "freeze_frac": 0.40, "stage_name": "approach_freeze",
             "notes": "arm approaches cube then freezes"},
            {"id": "pick_s2_approach_retreat", "target_score": 2, "mode": "fail_inject",
             "fail_stage": 0, "fail_type": "trans_x", "noise": 0.20,
             "accept_peak": 2, "accept_drop": True, "max_attempts": 30,
             "stage_name": "approach_retreat",
             "notes": "arm approaches then moves away (2->1)"},
            # --- Score 4 ---
            {"id": "pick_s4_grasp_freeze", "target_score": 4, "mode": "freeze",
             "freeze_frac": 0.65, "stage_name": "grasp_freeze",
             "notes": "cube grasped, arm freezes holding it"},
            {"id": "pick_s4_grasp_release", "target_score": 4, "mode": "fail_inject",
             "fail_stage": 3, "fail_type": "trans_x", "noise": 5.0,
             "extend_action": "open_gripper", "extend_steps": 30,
             "accept_peak": 4, "accept_drop": True, "max_attempts": 30,
             "stage_name": "grasp_release",
             "notes": "cube grasped, lift plan fails, gripper opens -> drop (4->2)"},
            # --- Bonus ---
            {"id": "pick_bonus_drop", "target_score": 5, "mode": "success_extend",
             "extend_action": "open_gripper", "extend_steps": 30,
             "accept_peak": 5, "accept_drop": True, "max_attempts": 20,
             "stage_name": "success_then_drop",
             "notes": "pick up, lift to goal (5), then drop (5->2)"},
            # --- Success ---
            {"id": "pick_success", "target_score": 5, "mode": "success",
             "stage_name": "success_reference",
             "notes": "clean successful pick"},
        ],
    },

    "FailPushCube-v1": {
        "description": "Push the red cube to the green goal position.",
        "slots": [
            # --- Score 1 ---
            {"id": "push_s1_away",   "target_score": 1, "mode": "random_away",
             "stage_name": "random_away", "notes": "arm moves away from cube"},
            {"id": "push_s1_freeze", "target_score": 1, "mode": "zero_freeze",
             "stage_name": "total_freeze", "notes": "arm does not move at all"},
            # --- Score 2 ---
            {"id": "push_s2_approach_freeze", "target_score": 2, "mode": "freeze",
             "freeze_frac": 0.55, "stage_name": "approach_freeze",
             "notes": "arm reaches behind cube then freezes"},
            {"id": "push_s2_approach_retreat", "target_score": 2, "mode": "fail_inject",
             "fail_stage": 1, "fail_type": "trans_x", "noise": 0.20,
             "accept_peak": 2, "accept_drop": True, "max_attempts": 30,
             "stage_name": "approach_retreat",
             "notes": "arm approaches then deflects sideways (2->1)"},
            # --- Score 4 ---
            {"id": "push_s4_push_freeze", "target_score": 4, "mode": "freeze",
             "freeze_frac": 0.85, "stage_name": "push_freeze",
             "notes": "cube pushed most of the way, arm freezes"},
            # --- Success ---
            {"id": "push_success", "target_score": 5, "mode": "success",
             "stage_name": "success_reference",
             "notes": "clean successful push"},
        ],
    },

    "FailStackCube-v1": {
        "description": "Pick up the red cube and stack it on top of the green cube.",
        "slots": [
            # --- Score 1 ---
            {"id": "stack_s1_away",   "target_score": 1, "mode": "random_away",
             "stage_name": "random_away", "notes": "arm moves away from cube"},
            {"id": "stack_s1_freeze", "target_score": 1, "mode": "zero_freeze",
             "stage_name": "total_freeze", "notes": "arm does not move at all"},
            # --- Score 2 ---
            {"id": "stack_s2_approach_freeze", "target_score": 2, "mode": "freeze",
             "freeze_frac": 0.35, "stage_name": "approach_freeze",
             "notes": "arm near cube then freezes before grasping"},
            {"id": "stack_s2_approach_retreat", "target_score": 2, "mode": "fail_inject",
             "fail_stage": 2, "fail_type": "trans_x", "noise": 0.20,
             "accept_peak": 2, "accept_drop": True, "max_attempts": 30,
             "stage_name": "approach_retreat",
             "notes": "arm approaches cube then deflects (2->1)"},
            # --- Score 3 ---
            {"id": "stack_s3_grasp_freeze", "target_score": 3, "mode": "freeze",
             "freeze_frac": 0.55, "stage_name": "grasp_freeze",
             "notes": "cube grasped, arm freezes holding it"},
            {"id": "stack_s3_carry_no_drop", "target_score": 3, "mode": "fail_inject",
             "fail_stage": 5, "fail_type": "trans_x", "noise": 5.0,
             "accept_peak": 3, "max_attempts": 30,
             "stage_name": "carry_no_drop",
             "notes": "cube grasped, plan fails far from cubeB, held grasped (stays 3)"},
            {"id": "stack_s3_carry_drop", "target_score": 3, "mode": "fail_inject",
             "fail_stage": 5, "fail_type": "trans_x", "noise": 5.0,
             "extend_action": "open_gripper", "extend_steps": 20,
             "accept_peak": 3, "accept_drop": True, "max_attempts": 30,
             "stage_name": "carry_drop",
             "notes": "cube grasped+lifted, plan fails at align -> gripper opens -> cube falls (3->2)"},
            # --- Score 4 ---
            {"id": "stack_s4_drop_near_goal", "target_score": 4, "mode": "fail_inject",
             "fail_stage": 5, "fail_type": "trans_y", "noise": 0.30,
             "accept_peak": 4, "accept_drop": True, "max_attempts": 40,
             "stage_name": "drop_near_goal",
             "notes": "cube reaches near cubeB (peak 4), then drops (4->2)"},
            {"id": "stack_s4_align_freeze", "target_score": 4, "mode": "freeze",
             "freeze_frac": 0.88, "stage_name": "align_freeze",
             "notes": "cube above cubeB, arm freezes before release"},
            # --- Success ---
            {"id": "stack_success", "target_score": 5, "mode": "success",
             "stage_name": "success_reference",
             "notes": "clean successful stack"},
        ],
    },
}


# ---------------------------------------------------------------------------
# FailSafe YAML patching
# ---------------------------------------------------------------------------

@contextmanager
def patch_failsafe_save_path(task_name, save_path):
    cfg_path = Path(FAILSAFE_PATH) / "failgen" / "configs" / f"{task_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"FailSafe config not found: {cfg_path}")
    original_text = cfg_path.read_text()
    cfg = _yaml.safe_load(original_text)
    cfg["save_path"] = str(save_path)
    cfg_path.write_text(_yaml.dump(cfg))
    try:
        yield
    finally:
        cfg_path.write_text(original_text)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _numpy_to_pil(np_img):
    if hasattr(np_img, 'cpu'):
        np_img = np_img.cpu().numpy()
    if hasattr(np_img, 'detach'):
        np_img = np_img.detach().numpy()
    np_img = np.asarray(np_img)
    if np_img.ndim == 4:
        np_img = np_img[0]
    if np_img.ndim == 3 and np_img.shape[-1] == 4:
        np_img = np_img[:, :, :3]
    if np_img.dtype != np.uint8:
        if np_img.max() <= 1.0 + 1e-6:
            np_img = (np_img * 255).clip(0, 255).astype(np.uint8)
        else:
            np_img = np_img.clip(0, 255).astype(np.uint8)
    return Image.fromarray(np_img)


def _import_failsafe_tasks():
    import importlib
    for mod in ["failgen.tasks.fail_pick_cube",
                "failgen.tasks.fail_push_cube",
                "failgen.tasks.fail_stack_cube"]:
        try:
            importlib.import_module(mod)
        except Exception:
            pass


def _vec3(p):
    if hasattr(p, "cpu"):
        p = p.cpu().numpy()
    p = np.asarray(p).reshape(-1)
    return p[-3:].astype(np.float64)


def _obj_pos(unwrapped, task_name):
    if task_name == "FailStackCube-v1":
        return _vec3(unwrapped.cubeA.pose.p)
    if task_name == "FailPickCube-v1":
        return _vec3(unwrapped.cube.pose.p)
    return _vec3(unwrapped.obj.pose.p)


# ---------------------------------------------------------------------------
# Per-step state capture
# ---------------------------------------------------------------------------

def _read_state(env, task_name):
    try:
        if task_name == "FailStackCube-v1":
            grasped = env.agent.is_grasping(env.cubeA)
            if hasattr(grasped, "any"):
                grasped = bool(grasped.any().item() if hasattr(grasped, "item") else grasped.any())
            else:
                grasped = bool(grasped)
            return {
                "cubeA":      _vec3(env.cubeA.pose.p),
                "cubeB":      _vec3(env.cubeB.pose.p),
                "tcp":        _vec3(env.agent.tcp.pose.p),
                "is_grasped": grasped,
            }
        elif task_name == "FailPushCube-v1":
            return {
                "obj":  _vec3(env.obj.pose.p),
                "goal": _vec3(env.goal_region.pose.p),
                "tcp":  _vec3(env.agent.tcp.pose.p),
            }
        elif task_name == "FailPickCube-v1":
            grasped = env.agent.is_grasping(env.cube)
            if hasattr(grasped, "any"):
                grasped = bool(grasped.any().item() if hasattr(grasped, "item") else grasped.any())
            else:
                grasped = bool(grasped)
            return {
                "obj":        _vec3(env.cube.pose.p),
                "goal":       _vec3(env.goal_site.pose.p),
                "tcp":        _vec3(env.agent.tcp.pose.p),
                "is_grasped": grasped,
            }
    except Exception as e:
        return {"_error": str(e)}
    return {}


def _install_state_logger(fw, task_name):
    states = []
    env = fw._env.unwrapped
    _orig_step = fw._env.step

    def logged_step(action):
        result = _orig_step(action)
        states.append(_read_state(env, task_name))
        return result

    fw._env.step = logged_step

    def restore():
        fw._env.step = _orig_step

    return states, restore


# ---------------------------------------------------------------------------
# Non-monotonic frame labelers (with score 5 = success)
# ---------------------------------------------------------------------------

_EE_NEAR = 0.08


def _score_frame_stack(s, initial):
    if "_error" in s or not s:
        return 1
    cubeA, cubeB, tcp = s["cubeA"], s["cubeB"], s["tcp"]
    grasped = s["is_grasped"]
    horiz = float(np.linalg.norm(cubeA[:2] - cubeB[:2]))
    cube_z = float(cubeA[2])
    target_z = float(cubeB[2]) + 0.04
    ee_to_cube = float(np.linalg.norm(tcp - cubeA))

    # 5 — stacked on cubeB
    if not grasped and horiz < 0.03 and abs(cube_z - target_z) < 0.02:
        return 5
    # 4 — cube held close above cubeB
    if grasped and horiz < 0.07 and cube_z > float(cubeB[2]):
        return 4
    # 3 — grasped or elevated
    if grasped:
        return 3
    if cube_z > 0.045:
        return 3
    # 2 — EE near cube
    if ee_to_cube < _EE_NEAR:
        return 2
    return 1


def _score_frame_push(s, initial):
    if "_error" in s or not s or "_error" in initial or not initial:
        return 1
    obj, goal, tcp = s["obj"], s["goal"], s["tcp"]
    ee_to_obj = float(np.linalg.norm(tcp - obj))
    obj_to_goal = float(np.linalg.norm(obj[:2] - goal[:2]))
    init_dist = float(np.linalg.norm(initial["obj"][:2] - initial["goal"][:2]))
    covered = max(0.0, (init_dist - obj_to_goal) / init_dist) if init_dist > 1e-6 else 0.0

    # 5 — at the bullseye (solver stops ~0.075m out, so threshold is generous)
    if obj_to_goal < 0.08:
        return 5
    # 4 — meaningful push
    if covered > 0.4 or obj_to_goal < 0.12:
        return 4
    # 3 — some progress
    if covered > 0.2:
        return 3
    # 2 — EE near cube
    if ee_to_obj < _EE_NEAR:
        return 2
    return 1


def _score_frame_pick(s, initial):
    if "_error" in s or not s or not initial:
        return 1
    obj, goal, tcp = s["obj"], s["goal"], s["tcp"]
    grasped = s.get("is_grasped", False)
    ee_to_obj = float(np.linalg.norm(tcp - obj))
    obj_to_goal = float(np.linalg.norm(obj - goal))
    init_z = float(initial.get("obj", obj)[2])
    lifted = float(obj[2]) > init_z + 0.05

    # 5 — cube at goal AND actually lifted (not just a near-table goal_site)
    if obj_to_goal < 0.05 and lifted:
        return 5
    # 4 — grasped
    if grasped:
        return 4
    # 2 — EE near cube
    if ee_to_obj < _EE_NEAR:
        return 2
    return 1


SCORERS = {
    "FailStackCube-v1": _score_frame_stack,
    "FailPushCube-v1":  _score_frame_push,
    "FailPickCube-v1":  _score_frame_pick,
}


def compute_frame_labels(states, frame_step_indices, task_name):
    if not states:
        return [1] * len(frame_step_indices)
    scorer = SCORERS.get(task_name, _score_frame_push)
    initial = states[0]
    labels = []
    for step in frame_step_indices:
        idx = min(max(step - 1, 0), len(states) - 1)
        labels.append(scorer(states[idx], initial))
    return labels


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def load_existing(jsonl_path):
    done = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                ep = json.loads(line)
                done.add(ep["episode_id"])
    return done


def write_episode(jsonl_path, record):
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Mode: random_away (score 1)
# ---------------------------------------------------------------------------

def collect_random_away(task_name, seed=0, max_steps=25):
    import gymnasium as gym
    if FAILSAFE_PATH:
        _import_failsafe_tasks()

    env = gym.make(task_name, obs_mode="rgb", render_mode="rgb_array", sensor_configs=dict())
    env.action_space.seed(seed)
    obs, info = env.reset(seed=seed)

    unwrapped = env.unwrapped
    try:
        obj_p = _obj_pos(unwrapped, task_name)
        tcp_p = _vec3(unwrapped.agent.tcp.pose.p)
        away = tcp_p - obj_p
        away_xy = away[:2]
        n = float(np.linalg.norm(away_xy))
        away_xy = np.array([1.0, 0.0]) if n < 1e-6 else away_xy / n
    except Exception:
        away_xy = np.array([1.0, 0.0])

    rng = np.random.default_rng(seed)
    lo, hi = env.action_space.low, env.action_space.high

    frames = [_numpy_to_pil(env.render())]
    for _ in range(max_steps):
        action = env.action_space.sample()
        if len(action) >= 3:
            action[0] = float(np.clip(away_xy[0] * 0.95 + rng.normal(0, 0.08), lo[0], hi[0]))
            action[1] = float(np.clip(away_xy[1] * 0.95 + rng.normal(0, 0.08), lo[1], hi[1]))
            action[2] = float(np.clip(0.5 + rng.normal(0, 0.08), lo[2], hi[2]))
        obs, _, terminated, truncated, _ = env.step(action)
        frames.append(_numpy_to_pil(env.render()))
        if terminated or truncated:
            break
    total = len(frames) - 1
    env.close()
    return frames, total


# ---------------------------------------------------------------------------
# Mode: zero_freeze (score 1)
# ---------------------------------------------------------------------------

def collect_zero_freeze(task_name, seed=0, max_steps=10):
    import gymnasium as gym
    if FAILSAFE_PATH:
        _import_failsafe_tasks()

    env = gym.make(task_name, obs_mode="rgb", render_mode="rgb_array", sensor_configs=dict())
    obs, info = env.reset(seed=seed)

    action = np.zeros(env.action_space.shape)
    frames = [_numpy_to_pil(env.render())]
    for _ in range(max_steps):
        obs, _, terminated, truncated, _ = env.step(action)
        frames.append(_numpy_to_pil(env.render()))
        if terminated or truncated:
            break
    total = len(frames) - 1
    env.close()
    return frames, total


# ---------------------------------------------------------------------------
# Extension helper: open gripper after success
# ---------------------------------------------------------------------------

def _extend_open_gripper(fw, task_name, n_steps=30):
    """Step the env with gripper-open actions. Cube drops from gripper.

    Action space is 8D: 7 arm joints + 1 normalized gripper command.
    Gripper: 1 = open, -1 = closed (NOT raw joint positions).
    The planner uses 6 steps per open/close call.
    """
    import torch
    env = fw._env
    unwrapped = env.unwrapped
    qpos = unwrapped.agent.robot.get_qpos()
    if hasattr(qpos, 'cpu'):
        qpos = qpos.cpu().numpy()
    qpos = np.asarray(qpos).flatten()

    # Action = [7 arm joint positions, 1 gripper command (normalized)]
    # qpos has 9 dims (7 arm + 2 finger joints), action has 8 (mirrored gripper)
    act_dim = env.action_space.shape[-1]  # should be 8
    arm_qpos = qpos[:7]  # 7 arm joints
    action = np.zeros(act_dim, dtype=np.float32)
    action[:7] = arm_qpos
    action[7] = 1.0  # OPEN = 1 (normalized), not 0.04 (raw joint pos)
    print(f"    [extend] act_dim={act_dim} arm_qpos={arm_qpos.round(3)} gripper_cmd=1.0(OPEN)")

    action_t = torch.tensor(action).unsqueeze(0)
    extended = 0
    for i in range(n_steps):
        try:
            env.step(action_t)
            extended += 1
        except Exception as e:
            print(f"    [extend] step {i} error: {e}")
            break
    print(f"    [extend] completed {extended}/{n_steps} steps")


def _extend_forward_push(fw, task_name, n_steps=15):
    """After a successful push, drive the arm further forward to overshoot the goal.

    Heuristic: nudge shoulder pitch (joint 1) and elbow (joint 3) to extend reach,
    which pushes the contact point further along the push direction.
    """
    import torch
    env = fw._env
    unwrapped = env.unwrapped
    qpos = unwrapped.agent.robot.get_qpos()
    if hasattr(qpos, 'cpu'):
        qpos = qpos.cpu().numpy()
    qpos = np.asarray(qpos).flatten()

    act_dim = env.action_space.shape[-1]
    action = np.zeros(act_dim, dtype=np.float32)
    action[:7] = qpos[:7]
    gripper_pos = qpos[7] if len(qpos) > 7 else 0.0
    action[7] = 1.0 if gripper_pos > 0.02 else -1.0
    print(f"    [extend] forward_push from arm_qpos={qpos[:7].round(3)}")

    extended = 0
    for i in range(n_steps):
        action[1] += 0.02   # shoulder pitch forward
        action[3] += 0.02   # elbow extend
        action_t = torch.tensor(action).unsqueeze(0)
        try:
            env.step(action_t)
            extended += 1
        except Exception as e:
            print(f"    [extend] step {i} error: {e}")
            break
    print(f"    [extend] completed {extended}/{n_steps} forward push steps")


# ---------------------------------------------------------------------------
# FailSafe-based collection (freeze, fail_inject, success, success_extend, overshoot)
# ---------------------------------------------------------------------------

def _anchor_slot(cfg):
    """Find first real (non-freeze, non-random) slot to use as FailSafe anchor."""
    for s in cfg["slots"]:
        if s.get("fail_stage") is not None:
            return s
    return None


def collect_failsafe_episode(task_name, cfg, slot, failsafe_save_dir):
    """Returns (png_files, kf_idx, labels, ep_states, total_steps) or None."""
    if not FAILSAFE_AVAILABLE:
        return None

    mode = slot["mode"]

    # Determine FailSafe noise parameters
    if mode in ("freeze", "success", "success_extend"):
        noise_val = 0.0
        anchor = _anchor_slot(cfg)
        fail_stage = slot.get("fail_stage", anchor["fail_stage"] if anchor else 0)
        fail_type = slot.get("fail_type", anchor["fail_type"] if anchor else "trans_x")
    else:
        noise_val = slot["noise"]
        fail_stage = slot["fail_stage"]
        fail_type = slot["fail_type"]

    with patch_failsafe_save_path(task_name, failsafe_save_dir):
        fw = FailgenWrapper(task_name=task_name, headless=True, save_video=True)
        fw._fail_plan_wrapper.set_active_type(fail_type)
        fw._fail_plan_wrapper.set_active_stage(fail_stage)
        af = fw._fail_plan_wrapper._active_fail
        if af is not None:
            af.noise = noise_val
            if fail_stage not in af.stages:
                af.stages = list(af.stages) + [fail_stage]

        states, restore = _install_state_logger(fw, task_name)

        try:
            max_attempts = slot.get("max_attempts", 40)
            best_candidate = None  # (score, png_files, kf_idx, labels, ep_states, total_steps)
            for attempt in range(max_attempts):
                states.clear()
                success = fw.get_failure(num_gt=attempt)

                print(f"    attempt {attempt+1}: states={len(states)} success={success}")

                # Acceptance logic per mode
                want_success = mode in ("freeze", "success", "success_extend")
                want_failure = mode == "fail_inject"
                # overshoot: accept either, filter by labels later

                if want_success and not success:
                    try: fw.save_video(save=False)
                    except OSError: pass
                    continue
                if want_failure and success:
                    try: fw.save_video(save=False)
                    except OSError: pass
                    continue

                # Extension: open gripper to cause drop (for success_extend OR fail_inject with extend_action)
                extend_action = slot.get("extend_action")
                if extend_action:
                    n_ext = slot.get("extend_steps", 30)
                    if extend_action == "open_gripper":
                        _extend_open_gripper(fw, task_name, n_steps=n_ext)
                    elif extend_action == "forward_push":
                        _extend_forward_push(fw, task_name, n_steps=n_ext)

                # Save video to get PNGs
                ep_idx = attempt + 1
                try:
                    fw.save_video(save=True, num_gt=attempt, ep_idx=ep_idx)
                except OSError as e:
                    print(f"    save_video error: {e}")
                    continue

                png_dir = (Path(failsafe_save_dir) / task_name / str(attempt)
                           / f"{fail_stage}_{fail_type}_{ep_idx}" / "front")
                if not png_dir.exists():
                    print(f"    PNG dir missing: {png_dir}")
                    continue

                png_files = sorted(
                    [p for p in png_dir.iterdir() if p.suffix == ".png"],
                    key=lambda p: int(p.stem))
                if len(png_files) < 2:
                    shutil.rmtree(png_dir.parent, ignore_errors=True)
                    continue

                ep_states = list(states)

                # Freeze: truncate + pad
                if mode == "freeze":
                    frac = slot.get("freeze_frac", 0.70)
                    cutoff = max(2, int(len(png_files) * frac))
                    png_files = png_files[:cutoff]
                    ep_states = ep_states[:cutoff]

                total_steps = len(png_files)
                kf_idx = [round(i * (total_steps - 1) / 7) for i in range(8)]

                # For freeze, last 2 keyframes repeat the cutoff frame
                if mode == "freeze":
                    end_idx = total_steps - 1
                    kf_idx[-1] = end_idx
                    kf_idx[-2] = end_idx

                labels = compute_frame_labels(ep_states, kf_idx, task_name)

                # Score this attempt vs target. Higher score = better match.
                # peak_ok + drop_ok -> ideal (return immediately).
                # Otherwise keep the best candidate and fall back after max_attempts.
                if mode == "overshoot":
                    accept_peak = 5
                    accept_drop = True
                else:
                    accept_peak = slot.get("accept_peak")
                    accept_drop = slot.get("accept_drop", False)

                peak = max(labels)
                final = labels[-1]
                peak_ok = (accept_peak is None) or (peak == accept_peak)
                drop_ok = (not accept_drop) or (final < peak)

                # Score: 0 = perfect, more negative = worse
                score = 0
                if accept_peak is not None:
                    score -= abs(peak - accept_peak) * 10
                if accept_drop and not drop_ok:
                    score -= 5

                candidate = (score, png_files, kf_idx, labels, ep_states, total_steps, png_dir)

                if peak_ok and drop_ok:
                    # Ideal — clean up previous best, return now
                    if best_candidate is not None:
                        try:
                            shutil.rmtree(best_candidate[6].parent, ignore_errors=True)
                        except Exception:
                            pass
                    print(f"    ACCEPT labels={labels}")
                    return png_files, kf_idx, labels, ep_states, total_steps

                # Not ideal — compare to best so far
                if best_candidate is None or score > best_candidate[0]:
                    if best_candidate is not None:
                        try:
                            shutil.rmtree(best_candidate[6].parent, ignore_errors=True)
                        except Exception:
                            pass
                    best_candidate = candidate
                    print(f"    kept as best (score={score}) labels={labels}")
                else:
                    try:
                        shutil.rmtree(png_dir.parent, ignore_errors=True)
                    except Exception:
                        pass
                    print(f"    REJECT (score={score} <= best={best_candidate[0]}) labels={labels}")

            # Exhausted attempts — fall back to best candidate if any
            if best_candidate is not None:
                s, pf, ki, lb, es, ts, _ = best_candidate
                print(f"    FALLBACK to best (score={s}) labels={lb}")
                return pf, ki, lb, es, ts
            return None
        finally:
            restore()


# ---------------------------------------------------------------------------
# Write a single episode to disk
# ---------------------------------------------------------------------------

def save_episode(slot, task_name, cfg, png_files_or_frames, kf_idx, labels,
                 total_steps, output_dir, jsonl_path, is_pil=False):
    kf_root = output_dir / "keyframes"
    base_id = slot["id"]
    ep_dir = kf_root / base_id
    ep_dir.mkdir(parents=True, exist_ok=True)

    frame_records = []
    for f_idx, (step, label) in enumerate(zip(kf_idx, labels)):
        t = step / max(total_steps, 1)
        fname = f"frame_{f_idx}_{t:.2f}s.jpg"
        if is_pil:
            png_files_or_frames[step].save(ep_dir / fname)
        else:
            img = Image.open(png_files_or_frames[step]).convert("RGB")
            img.save(ep_dir / fname)
        frame_records.append({
            "frame_idx":   f_idx,
            "step":        int(step),
            "timestamp":   round(t, 4),
            "score":       label,
            "frame_label": label,
            "image_path":  f"{base_id}/{fname}",
        })

    record = {
        "episode_id":       base_id,
        "episode_base_id":  base_id,
        "task":             task_name,
        "task_description": cfg["description"],
        "target_score":     slot["target_score"],
        "fail_stage":       slot.get("fail_stage"),
        "stage_name":       slot.get("stage_name", ""),
        "fail_type":        slot.get("fail_type", slot["mode"]),
        "stage_notes":      slot.get("notes", ""),
        "noise":            slot.get("noise", 0),
        "total_steps":      total_steps,
        "final_score":      labels[-1],
        "frames":           frame_records,
    }
    write_episode(jsonl_path, record)
    print(f"  -> saved {base_id} labels={labels}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output/sample_v1")
    parser.add_argument("--tasks", nargs="+", default=list(TASK_CONFIGS.keys()))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    kf_dir = output_dir / "keyframes"
    jsonl_path = output_dir / "maniskill_failures.jsonl"
    failsafe_save_dir = output_dir / "_failsafe_tmp"

    output_dir.mkdir(parents=True, exist_ok=True)
    kf_dir.mkdir(exist_ok=True)
    failsafe_save_dir.mkdir(exist_ok=True)

    done_ids = load_existing(jsonl_path)
    print(f"Resuming from {len(done_ids)} existing episodes\n")

    total_saved = 0
    total_skipped = 0

    for task_name in args.tasks:
        if task_name not in TASK_CONFIGS:
            print(f"Unknown task: {task_name}")
            continue

        cfg = TASK_CONFIGS[task_name]
        print(f"\n{'='*60}\nTask: {task_name}\n  {cfg['description']}\n{'='*60}")

        for slot in cfg["slots"]:
            slot_id = slot["id"]
            mode = slot["mode"]

            if slot_id in done_ids:
                print(f"\n[{slot_id}] already done, skipping")
                total_skipped += 1
                continue

            print(f"\n[{slot_id}] mode={mode} target={slot['target_score']}")
            print(f"  {slot['notes']}")

            if args.dry_run:
                print("  (dry run)")
                continue

            # ---- random_away ----
            if mode == "random_away":
                seed = hash(slot_id) % 10000
                frames, total_steps = collect_random_away(task_name, seed=seed)
                n_f = len(frames)
                kf_idx = [round(i * (n_f - 1) / 7) for i in range(8)]
                labels = [1] * 8
                save_episode(slot, task_name, cfg, frames, kf_idx, labels,
                             total_steps, output_dir, jsonl_path, is_pil=True)
                total_saved += 1

            # ---- zero_freeze ----
            elif mode == "zero_freeze":
                seed = hash(slot_id) % 10000
                frames, total_steps = collect_zero_freeze(task_name, seed=seed)
                n_f = len(frames)
                kf_idx = [round(i * (n_f - 1) / 7) for i in range(8)]
                labels = [1] * 8
                save_episode(slot, task_name, cfg, frames, kf_idx, labels,
                             total_steps, output_dir, jsonl_path, is_pil=True)
                total_saved += 1

            # ---- FailSafe modes ----
            elif mode in ("freeze", "fail_inject", "success",
                          "success_extend", "overshoot"):
                result = collect_failsafe_episode(
                    task_name, cfg, slot, failsafe_save_dir)
                if result is None:
                    print(f"  FAILED: could not generate {slot_id} after max attempts")
                    continue
                png_files, kf_idx, labels, ep_states, total_steps = result
                save_episode(slot, task_name, cfg, png_files, kf_idx, labels,
                             total_steps, output_dir, jsonl_path, is_pil=False)
                # Clean up PNGs
                for pf in png_files:
                    try: shutil.rmtree(pf.parent.parent, ignore_errors=True)
                    except Exception: pass
                total_saved += 1
            else:
                print(f"  Unknown mode: {mode}")

    # ----------- summary -----------
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"  Saved:   {total_saved}")
    print(f"  Skipped: {total_skipped} (already done)")
    print(f"  Output:  {jsonl_path}")
    print(f"  Frames:  {kf_dir}/")

    if failsafe_save_dir.exists():
        shutil.rmtree(failsafe_save_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
