#!/usr/bin/env python3
"""
Generate diverse failure trajectories from all 48 Metaworld tasks.

Uses MetaWorld's built-in scripted expert policies with category-specific
noise injection and scoring:

  - grasp_move: full 1-2-3-4 scoring using grasp_success + obj_to_target
    Score 3 modes: drop, wrong_dir, freeze
    Score 4 modes: misplace, late_drop, freeze
  - push_sweep: full 1-2-3-4 scoring using relative in_place_reward
  - mechanism (drawer, door, window, etc.): skip score 3, go 1→2→4
  - press (button, coffee-button): skip score 3, go 1→2→4 with stricter thresholds

Each episode produces 8 keyframes with progress labels matching RoboMeter format.

Usage:
    python generate_failures.py --episodes-per-score 20 --camera corner2
    python generate_failures.py --tasks pick-place-v3 --episodes-per-score 5
"""

import json
import re
import argparse
import inspect
import numpy as np
from pathlib import Path
from PIL import Image

import gymnasium as gym
import metaworld
import metaworld.policies as mw_policies


# ============================================================
# Auto-discover all 50 scripted policies
# ============================================================

def _build_policy_map():
    policy_map = {}
    for name, cls in inspect.getmembers(mw_policies):
        if inspect.isclass(cls) and name.startswith("Sawyer") and name.endswith("V3Policy"):
            core = name[len("Sawyer"):-len("V3Policy")]
            kebab = re.sub(r"([A-Z])", r"-\1", core).strip("-").lower()
            policy_map[kebab + "-v3"] = cls
    return policy_map

SCRIPTED_POLICIES = _build_policy_map()


# ============================================================
# Task categories
# ============================================================

GRASP_MOVE_TASKS = {
    "assembly-v3", "basketball-v3", "bin-picking-v3", "box-close-v3",
    "coffee-pull-v3", "hammer-v3", "hand-insert-v3",
    "peg-insertion-side-v3", "peg-unplug-side-v3",
    "pick-out-of-hole-v3", "pick-place-v3", "pick-place-wall-v3",
    "shelf-place-v3", "stick-pull-v3", "stick-push-v3",
}

PUSH_SWEEP_TASKS = {
    "coffee-push-v3", "plate-slide-v3", "plate-slide-side-v3",
    "plate-slide-back-v3", "plate-slide-back-side-v3",
    "push-v3", "push-wall-v3", "push-back-v3",
    "soccer-v3", "sweep-v3", "sweep-into-v3",
}

# Single manipulation action — skip score 3, go 1→2→4
MECHANISM_TASKS = {
    "disassemble-v3",  # single-pull task; score 3 = near success, skip it
    "dial-turn-v3", "door-close-v3", "door-lock-v3", "door-open-v3",
    "door-unlock-v3", "drawer-close-v3", "drawer-open-v3",
    "faucet-close-v3", "faucet-open-v3",
    "handle-press-v3", "handle-press-side-v3",
    "handle-pull-v3", "handle-pull-side-v3",
    "lever-pull-v3", "window-close-v3", "window-open-v3",
}

# Even simpler — skip score 3, stricter 2→4 threshold
PRESS_TASKS = {
    "button-press-v3", "button-press-wall-v3",
    "button-press-topdown-v3", "button-press-topdown-wall-v3",
    "coffee-button-v3",  # gr_progress-based scoring avoids spring-back issues
}

# grasp_move tasks where the object springs back to start when arm freezes
# (e.g. attached by a joint). Needs gentle continued pull during noise phase.
SPRING_PULL_TASKS = {"coffee-pull-v3"}

# Tasks that are push_sweep category but visually have no meaningful score 3
# (only 1→2→4 makes sense). Handled like mechanism for valid_scores only.
SKIP_SCORE3_TASKS = {"soccer-v3"}

# Mechanism/press tasks where 50% expert force triggers task success for score 4
# (mechanism moves to completion before episode can be captured as a failure).
# Use pure freeze instead to hold the mechanism at the injection point.
EASY_COMPLETE_MECHANISM_TASKS = {
    "faucet-open-v3", "faucet-close-v3",
    "window-close-v3", "window-open-v3",
    "coffee-button-v3",
}

# Mechanism tasks where spring-back overwhelms 50% expert force for score 4
# (mechanism springs to start, dropping progress below detection threshold 0.15).
# Use 75% expert force + higher injection point so frozen progress stays above 0.15.
SPRING_BACK_MECHANISM_TASKS = {
    "handle-press-v3", "handle-press-side-v3",
    "handle-pull-side-v3",
}

# Noise modes that allow score to decrease (object dropped/moved away from goal).
# Freeze/misplace/default: mechanism stays → scores should be monotonic.
DECREASE_ALLOWED_MODES = {"drop", "late_drop", "wrong_dir", "lift_away", "push_sideways"}

EXCLUDED_TASKS = {"reach-v3", "reach-wall-v3", "peg-insertion-side-v3", "peg-unplug-side-v3"}


def get_task_category(task_name):
    if task_name in GRASP_MOVE_TASKS:
        return "grasp_move"
    elif task_name in PUSH_SWEEP_TASKS:
        return "push_sweep"
    elif task_name in MECHANISM_TASKS:
        return "mechanism"
    elif task_name in PRESS_TASKS:
        return "press"
    return "push_sweep"


def get_valid_scores(category, task_name=""):
    """Which scores are valid for this category."""
    if category in ("mechanism", "press") or task_name in SKIP_SCORE3_TASKS:
        return [1, 2, 4]  # skip 3
    return [1, 2, 3, 4]


# ============================================================
# Task descriptions
# ============================================================

TASK_DESCRIPTIONS = {
    "assembly-v3": "Pick up a nut and place it onto a peg.",
    "basketball-v3": "Dunk the basketball into the basket.",
    "bin-picking-v3": "Grasp the puck from one bin and place it into another bin.",
    "box-close-v3": "Grasp the cover and close the box with it.",
    "button-press-topdown-v3": "Press a button from the top.",
    "button-press-topdown-wall-v3": "Bypass a wall and press a button from the top.",
    "button-press-v3": "Press a button.",
    "button-press-wall-v3": "Bypass a wall and press a button.",
    "coffee-button-v3": "Push a button on the coffee machine.",
    "coffee-pull-v3": "Pull a mug from a coffee machine.",
    "coffee-push-v3": "Push a mug under a coffee machine.",
    "dial-turn-v3": "Rotate a dial 180 degrees.",
    "disassemble-v3": "Pick a nut out of a peg.",
    "door-close-v3": "Close a door with a revolving joint.",
    "door-lock-v3": "Lock the door by rotating the lock clockwise.",
    "door-open-v3": "Open a door with a revolving joint.",
    "door-unlock-v3": "Unlock the door by rotating the lock counter-clockwise.",
    "drawer-close-v3": "Push and close a drawer.",
    "drawer-open-v3": "Open a drawer.",
    "faucet-close-v3": "Rotate the faucet clockwise.",
    "faucet-open-v3": "Rotate the faucet counter-clockwise.",
    "hammer-v3": "Hammer a screw on the wall.",
    "hand-insert-v3": "Insert the gripper into a hole.",
    "handle-press-side-v3": "Press a handle down sideways.",
    "handle-press-v3": "Press a handle down.",
    "handle-pull-side-v3": "Pull a handle up sideways.",
    "handle-pull-v3": "Pull a handle up.",
    "lever-pull-v3": "Pull a lever down 90 degrees.",
    "peg-insertion-side-v3": "Insert a peg sideways.",
    "peg-unplug-side-v3": "Unplug a peg sideways.",
    "pick-out-of-hole-v3": "Pick up a puck from a hole.",
    "pick-place-v3": "Pick and place a puck to a goal.",
    "pick-place-wall-v3": "Pick a puck, bypass a wall and place the puck.",
    "plate-slide-back-side-v3": "Get a plate from the cabinet sideways.",
    "plate-slide-back-v3": "Get a plate from the cabinet.",
    "plate-slide-side-v3": "Slide a plate into a cabinet sideways.",
    "plate-slide-v3": "Slide a plate into a cabinet.",
    "push-back-v3": "Pull a puck to a goal.",
    "push-v3": "Push the puck to a goal.",
    "push-wall-v3": "Bypass a wall and push a puck to a goal.",
    "shelf-place-v3": "Pick and place a puck onto a shelf.",
    "soccer-v3": "Kick a soccer ball into the goal.",
    "stick-pull-v3": "Grasp a stick and pull a box with the stick.",
    "stick-push-v3": "Grasp a stick and push a box using the stick.",
    "sweep-into-v3": "Sweep a puck into a hole.",
    "sweep-v3": "Sweep a puck off the table.",
    "window-close-v3": "Push and close a window.",
    "window-open-v3": "Push and open a window.",
}


# ============================================================
# Calibration
# ============================================================

def calibrate_task(env_cls, task, policy):
    """Run expert once to get initial signal values and task dynamics."""
    env = env_cls()
    env.set_task(task)
    obs, info = env.reset()

    init_ipr = None
    init_gr = None
    init_o2t = None
    init_gs = 0
    init_tcp_dist = float(np.linalg.norm(obs[0:3] - obs[4:7]))
    init_obj_pos = obs[4:7].copy()

    for step in range(500):
        action = policy.get_action(obs)
        obs, r, d, t, info = env.step(action)
        if init_ipr is None:
            init_ipr = info.get("in_place_reward", 0)
            init_gr = info.get("grasp_reward", 0)
            init_o2t = info.get("obj_to_target", 0.3)
            init_gs = info.get("grasp_success", 0)
        if info.get("success", 0):
            break

    peak_ipr = info.get("in_place_reward", init_ipr)
    ipr_range = max(peak_ipr - init_ipr, 0.1)
    env.close()

    return {
        "init_ipr": init_ipr,
        "init_gr": init_gr,
        "init_o2t": init_o2t,
        "init_gs": init_gs,
        "init_tcp_dist": init_tcp_dist,
        "init_obj_pos": init_obj_pos,
        "ipr_range": ipr_range,
        "expert_steps": step + 1,
    }


# ============================================================
# Scoring functions (per category)
# ============================================================

def detect_score_grasp_move(info, init_o2t, init_ipr=None, ipr_range=None,
                             obs=None, init_tcp_dist=None, init_obj_pos=None,
                             init_gs=0):
    """Grasp-move: full 1-2-3-4 using grasp_success + obj_to_target."""
    gr = info.get("grasp_reward", 0)
    gs = info.get("grasp_success", 0)
    o2t = info.get("obj_to_target", init_o2t)

    if init_o2t >= 0.01:
        # Normal path: obj_to_target is meaningful
        progress = (init_o2t - o2t) / (init_o2t + 1e-8)

        if progress > 0.4:
            return 4
        if gs > 0 or progress > 0.12:
            return 3
        if gr > 0.5 or progress > 0.05:
            return 2
        return 1

    # Fallback for tasks where obj_to_target is always 0
    # (assembly, box-close, disassemble, hammer).
    # Use TCP-to-object distance for approach and object displacement for move.
    if obs is not None and init_tcp_dist is not None and init_obj_pos is not None:
        tcp_dist = float(np.linalg.norm(obs[0:3] - obs[4:7]))
        obj_disp = float(np.linalg.norm(obs[4:7] - init_obj_pos))
        approached = tcp_dist < init_tcp_dist * 0.5

        if init_gs == 1:
            # Task starts with object already grasped (e.g. box-close).
            # Score 1 = dropped, Score 2 = held but not moving, etc.
            if gs == 0:
                return 1
            if obj_disp > 0.10:
                return 4
            if obj_disp > 0.04:
                return 3
            return 2
        else:
            if obj_disp > 0.10:
                return 4  # object moved far from start
            if gs > 0 or obj_disp > 0.04:
                return 3  # grasped or object nudged
            if approached or gr > 0.5:
                return 2  # arm near object
            return 1

    # Last resort: IPR-based
    if init_ipr is not None and ipr_range is not None:
        ipr = info.get("in_place_reward", 0)
        progress = (ipr - init_ipr) / (ipr_range + 1e-8)
        if progress > 0.4:
            return 4
        if gs > 0 or progress > 0.12:
            return 3
        if gr > 0.5:
            return 2
    return 1


def detect_score_push_sweep(info, init_o2t):
    """Push/sweep: full 1-2-3-4 using obj_to_target distance."""
    o2t = info.get("obj_to_target", init_o2t)
    gr = info.get("grasp_reward", 0)
    progress = (init_o2t - o2t) / (init_o2t + 1e-8)

    if progress >= 0.50:
        return 4
    if progress >= 0.15:
        return 3
    if gr >= 0.4 or progress >= 0.05:
        return 2
    return 1


def detect_score_mechanism(info, init_ipr, ipr_range, init_gr=0.0,
                           obs=None, init_tcp_dist=None):
    """Mechanism (drawer, door, etc.): skip 3, go 1→2→4.

    Uses in_place_reward progress (obj_to_target is 0 for some tasks like
    door-open). Wide thresholds since IPR can be noisy.
    """
    ipr = info.get("in_place_reward", 0)
    gr = info.get("grasp_reward", 0)
    progress = (ipr - init_ipr) / (ipr_range + 1e-8)

    if init_gr >= 0.8 and obs is not None and init_tcp_dist is not None:
        # Door tasks: grasp_reward ≈ 1.0 regardless of arm position, useless.
        # Use TCP-to-handle distance + higher progress threshold (door drifts
        # ~0.02 from physics alone, so 0.04 filters out noise).
        tcp_dist = float(np.linalg.norm(obs[0:3] - obs[4:7]))
        approached = tcp_dist < init_tcp_dist * 0.45
        if progress >= 0.15:
            return 4
        if progress >= 0.04 or approached:
            return 2
        return 1

    if progress >= 0.15:
        return 4
    if progress >= 0.02:
        return 2
    if gr >= 0.4:
        return 2
    return 1


def detect_score_press(info, init_ipr, ipr_range, init_gr):
    """Press tasks: skip 3, go 1→2→4 with stricter thresholds.

    Uses in_place_reward for press progress and grasp_reward for arm approach.
    """
    ipr = info.get("in_place_reward", 0)
    gr = info.get("grasp_reward", 0)
    progress = (ipr - init_ipr) / (ipr_range + 1e-8)
    gr_progress = (gr - init_gr) / max(1.0 - init_gr, 0.1)

    if progress >= 0.04 or gr_progress >= 0.30:
        return 4
    if gr_progress >= 0.10 or progress >= 0.02:
        return 2
    return 1


def detect_score(info, category, calib, ep_init_o2t, obs=None):
    """Dispatch to category-specific scoring."""
    if category == "grasp_move":
        return detect_score_grasp_move(info, ep_init_o2t,
                                       init_ipr=calib["init_ipr"],
                                       ipr_range=calib["ipr_range"],
                                       obs=obs,
                                       init_tcp_dist=calib.get("init_tcp_dist"),
                                       init_obj_pos=calib.get("init_obj_pos"),
                                       init_gs=calib.get("init_gs", 0))
    elif category == "push_sweep":
        return detect_score_push_sweep(info, ep_init_o2t)
    elif category == "mechanism":
        return detect_score_mechanism(
            info, calib["init_ipr"], calib["ipr_range"], calib["init_gr"],
            obs=obs, init_tcp_dist=calib.get("init_tcp_dist"))
    elif category == "press":
        return detect_score_press(
            info, calib["init_ipr"], calib["ipr_range"], calib["init_gr"])
    return 1


# ============================================================
# Injection triggers (per category)
# ============================================================

def should_inject_grasp_move(info, target_score, init_o2t, noise_mode,
                              init_ipr=None, ipr_range=None,
                              obs=None, init_tcp_dist=None, init_obj_pos=None,
                              init_gs=0, task_name=""):
    """When to start noise for grasp-move tasks."""
    if target_score == 1:
        return True
    gr = info.get("grasp_reward", 0)
    gs = info.get("grasp_success", 0)
    o2t = info.get("obj_to_target", init_o2t)

    if init_o2t >= 0.01:
        progress = (init_o2t - o2t) / (init_o2t + 1e-8)
    else:
        # Fallback: use object displacement as progress proxy
        if obs is not None and init_obj_pos is not None:
            obj_disp = float(np.linalg.norm(obs[4:7] - init_obj_pos))
            # Map displacement to 0-1 range (0.20m ≈ full progress)
            progress = min(obj_disp / 0.20, 1.0)
        elif init_ipr is not None and ipr_range is not None:
            ipr = info.get("in_place_reward", 0)
            progress = (ipr - init_ipr) / (ipr_range + 1e-8)
        else:
            progress = 0.0

    if target_score == 2:
        if init_o2t < 0.01 and obs is not None and init_tcp_dist is not None:
            # Tasks that start with object already grasped (e.g. box-close, init_gs=1):
            # inject immediately since the arm is already at the "approach" position.
            if init_gs == 1:
                return True
            # Fallback tasks: inject when arm is close to object
            tcp_dist = float(np.linalg.norm(obs[0:3] - obs[4:7]))
            return tcp_dist < init_tcp_dist * 0.5
        # Normal path: inject when grasping OR object has moved slightly
        # (catches tasks like stick-push where gr stays low).
        # Use 0.08 so frozen progress (0.08) > detection threshold (0.05).
        return gr > 0.5 or progress > 0.08
    if target_score == 3:
        if noise_mode == "drop":
            # Allow progress-only fallback for tasks where gs stays 0
            return (gs > 0 and progress > 0.20) or progress > 0.25
        if noise_mode == "wrong_dir":
            return (gs > 0 and progress > 0.15) or progress > 0.20
        # freeze: for fallback tasks (init_o2t=0) require displacement progress;
        # for normal tasks allow gs OR meaningful progress
        if init_o2t < 0.01:
            return progress > 0.20  # displacement proxy: ~4cm object movement
        return gs > 0 or progress > 0.15
    if target_score == 4:
        if noise_mode == "late_drop":
            return (gs > 0 and progress > 0.45) or progress > 0.55
        # Raise threshold so frozen robot stays above score-4 detection (0.40)
        if init_o2t < 0.01:
            return progress > 0.55  # displacement proxy: ~11cm movement
        # Spring-pull tasks (coffee-pull): inject later so lower freeze force
        # avoids task completion (object pulled all the way out → success).
        if task_name in SPRING_PULL_TASKS:
            return (gs > 0 and progress > 0.55) or progress > 0.60
        return (gs > 0 and progress > 0.45) or progress > 0.50
    return False


def should_inject_push_sweep(info, target_score, init_ipr, ipr_range, noise_mode="default",
                              task_name=""):
    """When to start noise for push/sweep tasks."""
    if target_score == 1:
        return True
    ipr = info.get("in_place_reward", 0)
    gr = info.get("grasp_reward", 0)
    progress = (ipr - init_ipr) / (ipr_range + 1e-8)

    if target_score == 2:
        # Allow progress-only fallback for tasks where gr stays low (e.g. soccer)
        return gr > 0.35 or progress > 0.05
    if target_score == 3:
        if noise_mode in ("lift_away", "push_sideways"):
            # Inject once past score-2 territory
            return progress > 0.10
        return progress > 0.06  # freeze: early injection
    if target_score == 4:
        # Soccer: ball has momentum — inject earlier so ball is farther from goal
        # when arm stops, giving friction time to decelerate it before goal.
        if task_name in SKIP_SCORE3_TASKS:
            return progress > 0.52
        # Inject at 0.60, well above the 0.50 detection threshold, so that
        # minor ball drift after freezing doesn't drop score back to 3.
        return progress > 0.60
    return False


def should_inject_mechanism(info, target_score, init_ipr, ipr_range, init_gr=0.0,
                            obs=None, init_tcp_dist=None, task_name=""):
    """When to start noise for mechanism tasks (no score 3)."""
    if target_score == 1:
        return True
    gr = info.get("grasp_reward", 0)
    ipr = info.get("in_place_reward", 0)
    progress = (ipr - init_ipr) / (ipr_range + 1e-8)

    if target_score == 2:
        if init_gr >= 0.4 and obs is not None and init_tcp_dist is not None:
            # Door/faucet tasks: don't freeze at start — wait for arm approach.
            tcp_dist = float(np.linalg.norm(obs[0:3] - obs[4:7]))
            if init_gr >= 0.8:
                # Door tasks: gr always ~1.0, use progress OR approach
                return progress > 0.04 or tcp_dist < init_tcp_dist * 0.45
            else:
                # Faucet-type (0.4 ≤ init_gr < 0.8): inject once arm close
                return tcp_dist < init_tcp_dist * 0.45
        # Raise from 0.01 to 0.02 so that after freeze the progress is above
        # the score-2 detection threshold (>= 0.02), preventing score-1 outcomes.
        return gr > 0.4 or progress > 0.02
    if target_score == 4:
        # Spring-back tasks need a deeper injection so that when frozen the
        # mechanism stays above the score-4 detection threshold (0.15).
        if task_name in SPRING_BACK_MECHANISM_TASKS:
            return progress > 0.40
        # Raise threshold: frozen robot at 0.08 falls below 0.15 detection →
        # always detected as score 2. Inject at 0.20 so frozen progress > 0.15.
        return progress > 0.20
    return False


def should_inject_press(info, target_score, init_ipr, ipr_range, init_gr):
    """When to start noise for press tasks (no score 3)."""
    if target_score == 1:
        return True
    gr = info.get("grasp_reward", 0)
    ipr = info.get("in_place_reward", 0)
    progress = (ipr - init_ipr) / (ipr_range + 1e-8)
    gr_progress = (gr - init_gr) / max(1.0 - init_gr, 0.1)

    if target_score == 2:
        return gr_progress > 0.10
    if target_score == 4:
        # Inject when arm is close to button (gr_progress); frozen arm stays
        # close → gr_progress stays elevated → score 4 detected (no spring-back
        # in gr unlike ipr which tracks actual button depth).
        return gr_progress > 0.30 or progress > 0.03
    return False


def should_inject(info, target_score, category, calib, ep_init_o2t, noise_mode,
                   obs=None, task_name=""):
    """Dispatch to category-specific injection."""
    if category == "grasp_move":
        return should_inject_grasp_move(info, target_score, ep_init_o2t, noise_mode,
                                        init_ipr=calib["init_ipr"],
                                        ipr_range=calib["ipr_range"],
                                        obs=obs,
                                        init_tcp_dist=calib.get("init_tcp_dist"),
                                        init_obj_pos=calib.get("init_obj_pos"),
                                        init_gs=calib.get("init_gs", 0),
                                        task_name=task_name)
    elif category == "push_sweep":
        return should_inject_push_sweep(
            info, target_score, calib["init_ipr"], calib["ipr_range"], noise_mode,
            task_name=task_name)
    elif category == "mechanism":
        return should_inject_mechanism(
            info, target_score, calib["init_ipr"], calib["ipr_range"], calib["init_gr"],
            obs=obs, init_tcp_dist=calib.get("init_tcp_dist"), task_name=task_name)
    elif category == "press":
        return should_inject_press(
            info, target_score, calib["init_ipr"], calib["ipr_range"], calib["init_gr"])
    return False


# ============================================================
# Noise modes and bias generation
# ============================================================

def get_mode_sequence(category, target_score, n_episodes):
    """Return a list of modes, one slot per desired episode.

    For small n (sample runs), guarantees exactly one of each mode.
    For larger n (production), cycles through modes proportionally.
    """
    # Score 1 (all categories): pure noise + robot goes in wrong direction
    if target_score == 1:
        base = ["noise", "wrong_dir", "wrong_dir"]
        return [base[i % len(base)] for i in range(n_episodes)]

    # Score 2 (all categories): approach without grasping — freeze near object
    if target_score == 2:
        return ["freeze"] * n_episodes

    if category == "grasp_move" and target_score == 3:
        base = ["freeze", "drop", "wrong_dir"]
    elif category == "grasp_move" and target_score == 4:
        base = ["freeze", "misplace", "late_drop"]
    elif category == "push_sweep" and target_score == 3:
        base = ["freeze", "lift_away", "push_sideways"]
    elif category == "push_sweep" and target_score == 4:
        base = ["freeze", "misplace", "push_sideways"]
    else:
        # mechanism / press score 4: freeze + default
        base = ["freeze", "default"]

    # Cycle through base modes to fill n_episodes slots
    return [base[i % len(base)] for i in range(n_episodes)]


def make_noise_bias(action_dim, noise_scale, category, target_score, noise_mode):
    """Generate constant bias adapted to category, score, and noise mode."""
    if noise_mode == "freeze":
        return np.zeros(action_dim)

    # Score 1: noise mode uses random actions (no bias), wrong_dir uses strong bias
    if target_score == 1:
        if noise_mode == "wrong_dir":
            bias = np.zeros(action_dim)
            # Strong directional bias — robot goes in completely wrong direction
            angle = np.random.uniform(0, 2 * np.pi)
            bias[0] = np.cos(angle) * noise_scale * 3.0
            bias[1] = np.sin(angle) * noise_scale * 3.0
            bias[2] = np.random.choice([-1, 1]) * noise_scale * 2.0
            if action_dim > 3:
                bias[3] = 0.0  # don't actuate gripper
            return bias
        else:  # "noise" mode
            return np.zeros(action_dim)

    bias = np.random.randn(action_dim) * noise_scale

    if category == "grasp_move":
        if noise_mode == "drop":
            # Force gripper open → object falls mid-transport
            # (injected at progress > 0.22 so object lands at score-3 position)
            bias[3] = -abs(bias[3]) - noise_scale * 0.8
            bias[0:3] *= 0.4
        elif noise_mode == "wrong_dir":
            # Robot carries object in a random lateral direction (away from goal)
            # Gripper stays closed; XY bias is strong and perpendicular-ish
            angle = np.random.uniform(0, 2 * np.pi)
            bias[0] = np.cos(angle) * noise_scale * 1.8
            bias[1] = np.sin(angle) * noise_scale * 1.8
            bias[2] *= 0.2   # minimal Z change
            bias[3] = 0.0    # keep gripper closed
        elif noise_mode == "misplace":
            # Weak offset near goal → object placed slightly off target
            bias *= 0.35
            bias[3] = 0.0
        elif noise_mode == "late_drop":
            # Gripper opens after ≥75% progress → object drops near goal
            bias[3] = -abs(bias[3]) - noise_scale * 0.6
            bias[0:3] *= 0.25

    elif category == "push_sweep":
        if noise_mode == "lift_away":
            # Arm lifts up, loses contact with object → object stays at mid-progress
            bias[2] = abs(bias[2]) + noise_scale * 1.5  # strong Z-up
            bias[0:2] *= 0.15
            bias[3] = 0.0
        elif noise_mode == "push_sideways":
            # Push perpendicular to goal: pick a random XY direction
            angle = np.random.uniform(0, 2 * np.pi)
            bias[0] = np.cos(angle) * noise_scale * 1.5
            bias[1] = np.sin(angle) * noise_scale * 1.5
            bias[2] *= 0.2
            bias[3] = 0.0
        elif noise_mode == "misplace":
            # Weak offset near goal → object pushed slightly off target
            bias *= 0.35
            bias[3] = 0.0
        else:
            bias[0:2] *= 1.3  # default: stronger XY

    elif category in ("mechanism", "press"):
        bias *= 1.1

    # Normalize to consistent magnitude
    mag = np.linalg.norm(bias)
    if mag > 1e-8:
        bias = bias / mag * noise_scale

    return bias


# ============================================================
# Environment wrapper
# ============================================================

class MetaworldWrapper(gym.Wrapper):
    def __init__(self, env_cls, task, render_mode=None, camera_name=None):
        kwargs = {}
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        if camera_name is not None:
            kwargs["camera_name"] = camera_name
        env = env_cls(**kwargs)
        env.set_task(task)
        super().__init__(env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def render_all_cameras(env, camera_names):
    """Render the current MuJoCo state from each camera in camera_names.

    Returns a dict {camera_name: np.ndarray (H,W,3)} with images already
    flipped to the correct orientation (MuJoCo renders upside-down).

    gymnasium's MujocoRenderer.render() uses self.camera_id, so we look up
    each camera name via mujoco.mj_name2id, swap camera_id, render, restore.
    """
    import mujoco as _mujoco
    images = {}
    try:
        renderer = env.env.mujoco_renderer
        model = env.env.model
        orig_id = renderer.camera_id
        for cam in camera_names:
            cam_id = _mujoco.mj_name2id(model, _mujoco.mjtObj.mjOBJ_CAMERA, cam)
            renderer.camera_id = cam_id
            img = renderer.render("rgb_array")
            images[cam] = img[::-1].copy() if img is not None else None
        renderer.camera_id = orig_id  # restore
    except Exception:
        # Fallback: use whatever camera the env was initialised with
        img = env.render()
        img = img[::-1].copy() if img is not None else None
        for cam in camera_names:
            images[cam] = img
    return images


# ============================================================
# Data collection
# ============================================================

def _run_one_episode(env, policy, target_score, category, calib,
                     noise_mode, max_steps, noise_scale, noisy_steps, cameras,
                     task_name=""):
    """Run a single episode attempt. Returns (keyframes, n_steps) or None.

    cameras: list of camera names to render from, or empty list for no render.
    Each keyframe stores images: {camera_name: np.ndarray}.
    """
    init_o2t_default = calib["init_o2t"]
    obs, info = env.reset()
    ep_init_o2t = None
    trajectory = []
    noise_active = (target_score == 1)
    noise_start_step = 0 if target_score == 1 else None
    step = 0

    bias = make_noise_bias(
        env.action_space.shape[0], noise_scale, category, target_score, noise_mode
    )

    # For score 1 on tasks that start at handle (faucet-*, door-close, etc.):
    # precompute a strong directional bias to push robot away, so gr drops
    # below 0.4 and detect_score reliably returns 1 (not 2).
    escape_bias = None
    if target_score == 1 and (calib["init_gr"] > 0.4 or
                               calib.get("init_tcp_dist", 1.0) < 0.06 or
                               (category in ("mechanism", "press") and
                                calib.get("init_tcp_dist", 1.0) < 0.25)):
        _adim = env.action_space.shape[0]
        _angle = np.random.uniform(0, 2 * np.pi)
        escape_bias = np.zeros(_adim)
        escape_bias[0] = np.cos(_angle) * noise_scale * 3.0
        escape_bias[1] = np.sin(_angle) * noise_scale * 3.0
        escape_bias[2] = np.random.choice([-1, 1]) * noise_scale * 2.0

    while step < max_steps:
        if ep_init_o2t is None and step > 0:
            ep_init_o2t = info.get("obj_to_target", init_o2t_default)

        if not noise_active:
            if should_inject(info, target_score, category, calib,
                             ep_init_o2t or init_o2t_default, noise_mode,
                             obs=obs, task_name=task_name):
                noise_active = True
                noise_start_step = step

        if noise_active:
            if target_score == 1:
                if noise_mode == "wrong_dir":
                    # Robot moves decisively in wrong direction from start
                    expert_action = policy.get_action(obs)
                    jitter = np.random.randn(*expert_action.shape) * (noise_scale * 0.05)
                    action = np.clip(bias + jitter, -1.0, 1.0)
                elif escape_bias is not None:
                    # Tasks where robot starts near handle/button — push away so
                    # detect_score reliably returns 1.
                    action = np.clip(escape_bias + np.random.randn(env.action_space.shape[0]) * 0.05, -1.0, 1.0)
                else:
                    action = env.action_space.sample()
            elif noise_mode == "freeze":
                if category in ("mechanism", "press") and target_score == 4:
                    if task_name in EASY_COMPLETE_MECHANISM_TASKS:
                        # Pure freeze: 50% expert still pushes mechanism to success.
                        # Arm holds at injection point — mechanism stays, score 4 maintained.
                        action = np.random.randn(env.action_space.shape[0]) * 0.02
                        if env.action_space.shape[0] > 3:
                            action[3] = 0.0
                    elif task_name in SPRING_BACK_MECHANISM_TASKS:
                        # Spring-back overwhelms 50% force → use 75% to hold position
                        # above the score-4 detection threshold (progress > 0.15).
                        action = policy.get_action(obs) * 0.75
                        action += np.random.randn(env.action_space.shape[0]) * 0.03
                    else:
                        # Default: 50% expert (works for most mechanism tasks such as
                        # door, drawer, dial, lever that don't complete too easily).
                        action = policy.get_action(obs) * 0.50
                        action += np.random.randn(env.action_space.shape[0]) * 0.03
                elif task_name in SPRING_PULL_TASKS and target_score >= 3:
                    # Object springs back to start when arm stops pulling.
                    if target_score == 4:
                        # Less force for score 4: object stays at injection point
                        # without being pulled all the way out (triggering success).
                        action = policy.get_action(obs) * 0.30
                    else:
                        action = policy.get_action(obs) * 0.50
                    action += np.random.randn(env.action_space.shape[0]) * 0.03
                elif task_name == "stick-push-v3" and target_score == 4:
                    # Gentle push keeps stick pressed against box so score-4 progress
                    # is maintained through all 8 sampled keyframes (without this,
                    # box drifts back and the brief score-4 window is missed).
                    action = policy.get_action(obs) * 0.15
                    action += np.random.randn(env.action_space.shape[0]) * 0.02
                else:
                    action = np.random.randn(env.action_space.shape[0]) * 0.02
                    # Keep gripper closed when object was already grasped at start
                    # (init_gs=1, e.g. box-close) OR for high-score episodes where
                    # gripper opening would drop the object.
                    if (target_score >= 3 or calib.get("init_gs", 0) == 1) and \
                            env.action_space.shape[0] > 3:
                        action[3] = 0.8
                    # Stick-push score 2: keep gripper open to prevent accidental
                    # grasp of the stick during approach freeze (which would give
                    # gs=1 → score 3 instead of the intended score 2).
                    if task_name == "stick-push-v3" and target_score == 2 and \
                            env.action_space.shape[0] > 3:
                        action[3] = -0.8
                action = np.clip(action, -1.0, 1.0)
            elif noise_mode == "drop" or noise_mode == "late_drop":
                # Freeze arm + force gripper open → object falls in place
                action = np.random.randn(env.action_space.shape[0]) * 0.03
                action[3] = -1.0
                action = np.clip(action, -1.0, 1.0)
            elif noise_mode == "wrong_dir":
                # Replace XYZ with bias direction, preserve gripper state
                expert_action = policy.get_action(obs)
                action = np.zeros_like(expert_action)
                action[0:3] = bias[0:3] + np.random.randn(3) * (noise_scale * 0.1)
                action[3] = expert_action[3]
                action = np.clip(action, -1.0, 1.0)
            elif noise_mode == "misplace":
                # Expert + weak offset near goal
                expert_action = policy.get_action(obs)
                action = expert_action + bias * 0.3
                action += np.random.randn(*expert_action.shape) * 0.05
                action = np.clip(action, -1.0, 1.0)
            elif noise_mode == "lift_away":
                # Arm lifts up, loses contact with object
                action = policy.get_action(obs)
                action[2] = 0.8  # strong upward
                action[0:2] *= 0.1
                action += np.random.randn(*action.shape) * 0.03
                action = np.clip(action, -1.0, 1.0)
            elif noise_mode == "push_sideways":
                # Push in bias direction (perpendicular to goal)
                action = np.zeros(env.action_space.shape[0])
                action[0:3] = bias[0:3] + np.random.randn(3) * (noise_scale * 0.1)
                action = np.clip(action, -1.0, 1.0)
            else:
                # default: expert + bias
                expert_action = policy.get_action(obs)
                jitter = np.random.randn(*expert_action.shape) * (noise_scale * 0.1)
                action = np.clip(expert_action + bias + jitter, -1.0, 1.0)
        else:
            action = policy.get_action(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        if ep_init_o2t is None:
            ep_init_o2t = info.get("obj_to_target", init_o2t_default)

        imgs = render_all_cameras(env, cameras) if cameras else {}

        score = detect_score(info, category, calib, ep_init_o2t, obs=obs)
        trajectory.append({"images": imgs, "step": step, "score": score})
        step += 1

        if info.get("success", 0) > 0:
            return None  # success, discard

        # Dynamic noisy_steps: match expert steps so uniform 8-frame sampling
        # gives balanced expert/failure representation.
        # Score 1: all steps are noisy, use fixed noisy_steps.
        if noise_start_step is not None:
            if target_score == 1:
                effective_noisy = noisy_steps
            else:
                effective_noisy = max(20, min(noise_start_step, 200))
            if step - noise_start_step >= effective_noisy:
                break

    n_steps = len(trajectory)
    if n_steps < 8:
        return None

    # Basketball's expert briefly moves the arm away from the ball before
    # approaching, making frame 0 look misleadingly close. Skip the first
    # 10% of steps for this task only.
    kf_start = (n_steps // 10) if task_name == "basketball-v3" else 0
    indices = np.linspace(kf_start, n_steps - 1, 8, dtype=int)
    keyframes = []
    for idx in indices:
        t = trajectory[idx]
        keyframes.append({
            "step": t["step"],
            "timestamp": t["step"] / max(n_steps - 1, 1),
            "images": t["images"],   # dict {camera_name: np.ndarray}
            # Cap at target_score: prevents score-4 detect from blocking score-3
            # collection, and ensures score-1 episodes never show higher scores.
            "score": min(t["score"], target_score),
        })

    # Monotonic non-decreasing enforcement (used for episode collection check).
    # Skip for score 1: arm wanders freely, final state reflects actual position.
    if target_score > 1:
        for i in range(1, len(keyframes)):
            if keyframes[i]["score"] < keyframes[i - 1]["score"]:
                keyframes[i]["score"] = keyframes[i - 1]["score"]

    # ------------------------------------------------------------------ #
    # Per-frame label (per user spec):                                     #
    #   score 1 → all frames labeled 1                                    #
    #   score 2 → 1…1→2…2 (monotonic, freeze only mode)                  #
    #   score 3 → 1→2→3 for freeze; 1→2→3→2 for drop/wrong_dir          #
    #   score 4 → 1→2→3→4 for freeze; 1→2→…→4→2 for drop/wrong_dir     #
    #   floor at 2 (not 1) once score 3+ has been reached                 #
    # ------------------------------------------------------------------ #
    if target_score == 1:
        for kf in keyframes:
            kf["frame_label"] = 1
    elif noise_mode not in DECREASE_ALLOWED_MODES:
        # Freeze / misplace / default: mechanism/object stays → monotonic labels
        raw = [min(trajectory[idx]["score"], target_score) for idx in indices]
        for i in range(1, len(raw)):
            if raw[i] < raw[i - 1]:
                raw[i] = raw[i - 1]
        for kf, lbl in zip(keyframes, raw):
            kf["frame_label"] = lbl
    else:
        # Drop / wrong_dir: allow backward transition from 3→2 or 4→2,
        # but floor at 2 once score 3+ has been reached (per spec).
        raw = [min(trajectory[idx]["score"], target_score) for idx in indices]
        max_reached = 1
        for i in range(len(raw)):
            max_reached = max(max_reached, raw[i])
            if max_reached >= 3 and raw[i] < 2:
                raw[i] = 2
        for kf, lbl in zip(keyframes, raw):
            kf["frame_label"] = lbl

    return keyframes, n_steps


def collect_episodes(env_cls, task, policy, target_score, n_episodes,
                     category, calib, max_steps=500, noise_scale=0.5,
                     cameras=None, noisy_steps=40,
                     freeze_fraction=0.12, task_name=""):
    """Collect failure episodes.

    Iterates through a pre-assigned mode sequence (one slot per desired episode)
    to guarantee visual diversity. Each mode slot retries up to max_per_mode
    attempts before falling back to 'freeze'.

    cameras: list of camera names to render; None/[] disables rendering.
    """
    cameras = cameras or []
    render_mode = "rgb_array" if cameras else None
    # Use the first camera as the primary for env initialisation
    primary_camera = cameras[0] if cameras else None
    env = MetaworldWrapper(env_cls, task, render_mode=render_mode, camera_name=primary_camera)

    mode_sequence = get_mode_sequence(category, target_score, n_episodes)
    max_per_mode = 100

    episodes = []
    total_attempts = 0
    mode_counts = {}

    for slot_mode in mode_sequence:
        found = False
        # Try requested mode first
        for attempt in range(max_per_mode):
            total_attempts += 1
            result = _run_one_episode(
                env, policy, target_score, category, calib,
                slot_mode, max_steps, noise_scale, noisy_steps, cameras,
                task_name=task_name
            )
            if result is None:
                continue
            keyframes, n_steps = result
            if keyframes[-1]["score"] == target_score:
                mode_counts[slot_mode] = mode_counts.get(slot_mode, 0) + 1
                episodes.append({
                    "keyframes": keyframes,
                    "total_steps": n_steps,
                    "final_score": keyframes[-1]["score"],
                    "noise_mode": slot_mode,
                })
                found = True
                break

        if not found and slot_mode != "freeze":
            # Fallback to freeze if preferred mode didn't work
            for attempt in range(max_per_mode):
                total_attempts += 1
                result = _run_one_episode(
                    env, policy, target_score, category, calib,
                    "freeze", max_steps, noise_scale, noisy_steps, cameras,
                    task_name=task_name
                )
                if result is None:
                    continue
                keyframes, n_steps = result
                if keyframes[-1]["score"] == target_score:
                    mode_counts["freeze(fb)"] = mode_counts.get("freeze(fb)", 0) + 1
                    episodes.append({
                        "keyframes": keyframes,
                        "total_steps": n_steps,
                        "final_score": keyframes[-1]["score"],
                        "noise_mode": "freeze",
                    })
                    found = True
                    break

    env.close()
    modes_str = ", ".join(f"{m}={c}" for m, c in sorted(mode_counts.items())) or "none"
    print(f"    ({total_attempts} attempts → {len(episodes)} eps, modes: {modes_str})")
    return episodes


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Task names (default: all 48)")
    parser.add_argument("--episodes-per-score", type=int, default=20)
    parser.add_argument("--output-dir", type=str,
                        default="/home/pkarageorgis/DAS-5/Master-Thesis/MetaWorld/output")
    parser.add_argument("--cameras", nargs="+", default=["corner2"],
                        help="Camera names to render (default: corner2). "
                             "Pass multiple for multi-view dataset.")
    parser.add_argument("--noise-scale", type=float, default=0.5)
    parser.add_argument("--noisy-steps", type=int, default=40)
    parser.add_argument("--scores", nargs="+", type=int, default=None,
                        help="Override scores (default: category-specific)")
    parser.add_argument("--freeze-fraction", type=float, default=0.12)
    args = parser.parse_args()

    all_tasks = sorted(t for t in SCRIPTED_POLICIES if t not in EXCLUDED_TASKS)
    tasks = args.tasks if args.tasks else all_tasks

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tasks: {len(tasks)} tasks")
    print(f"Episodes per score: {args.episodes_per_score}")
    print(f"Cameras: {args.cameras}")
    print(f"Noise scale: {args.noise_scale}, noisy steps: {args.noisy_steps}")
    print(f"Freeze fraction: {args.freeze_fraction:.0%}")
    print()

    # ------------------------------------------------------------------ #
    # Resume support: load any existing JSONL so we can skip task/score   #
    # combos that are already fully collected.                             #
    # ------------------------------------------------------------------ #
    jsonl_path = output_dir / "metaworld_failures.jsonl"
    existing_by_id = {}
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    existing_by_id[r["episode_id"]] = r
        print(f"Resume: loaded {len(existing_by_id)} existing JSONL entries from {jsonl_path}")

    # Count how many episodes already exist for each (task, score) pair.
    # Key: (task_name, target_score) → set of base episode ids already collected.
    from collections import defaultdict
    existing_base_eps = defaultdict(set)
    for r in existing_by_id.values():
        base = r.get("episode_base_id", r["episode_id"])
        existing_base_eps[(r["task"], r["target_score"])].add(base)

    all_results = list(existing_by_id.values())  # seed with existing so final write is complete
    task_stats = {}

    for task_name in tasks:
        category = get_task_category(task_name)
        desc = TASK_DESCRIPTIONS.get(task_name, task_name)
        valid_scores = get_valid_scores(category, task_name)
        task_scores = args.scores if args.scores else valid_scores

        print(f"\n{'='*60}")
        print(f"Task: {task_name} [{category}] scores={task_scores}")
        print(f"  {desc}")
        print(f"{'='*60}")

        if task_name not in SCRIPTED_POLICIES:
            print(f"  WARNING: No scripted policy, skipping")
            continue

        mt1 = metaworld.MT1(task_name, seed=42)
        env_cls = mt1.train_classes[task_name]
        task_obj = mt1.train_tasks[0]
        policy = SCRIPTED_POLICIES[task_name]()

        calib = calibrate_task(env_cls, task_obj, policy)
        print(f"  Calibration: init_ipr={calib['init_ipr']:.3f}, "
              f"range={calib['ipr_range']:.3f}, "
              f"init_o2t={calib['init_o2t']:.3f}, "
              f"expert={calib['expert_steps']} steps")

        task_stats[task_name] = {}

        for target_score in task_scores:
            # Skip score 3 for mechanism/press/special tasks even if --scores overrides
            if target_score == 3 and (category in ("mechanism", "press") or
                                       task_name in SKIP_SCORE3_TASKS):
                continue

            already_have = len(existing_base_eps[(task_name, target_score)])
            still_need = args.episodes_per_score - already_have
            ep_start_idx = already_have  # continue numbering from where we left off

            if still_need <= 0:
                print(f"\n  --- Score {target_score} --- SKIP (already have {already_have}/{args.episodes_per_score})")
                task_stats[task_name][target_score] = already_have
                continue

            if already_have > 0:
                print(f"\n  --- Score {target_score} --- RESUME (have {already_have}, need {still_need} more)")
            else:
                print(f"\n  --- Score {target_score} ---")

            episodes = collect_episodes(
                env_cls, task_obj, policy,
                target_score=target_score,
                n_episodes=still_need,
                category=category,
                calib=calib,
                noise_scale=args.noise_scale,
                noisy_steps=args.noisy_steps,
                cameras=args.cameras,
                freeze_fraction=args.freeze_fraction,
                task_name=task_name,
            )
            task_stats[task_name][target_score] = already_have + len(episodes)
            print(f"  Collected {len(episodes)}/{still_need} new episodes with score {target_score} "
                  f"(total: {task_stats[task_name][target_score]}/{args.episodes_per_score}"
                  f"{'' if task_stats[task_name][target_score] == args.episodes_per_score else ' — partial'})")

            for ep_idx, ep in enumerate(episodes):
                ep_id = f"{task_name}_score{target_score}_{ep_start_idx + ep_idx:03d}"
                ep_dir = output_dir / "keyframes" / ep_id

                # Collect frame metadata (labels/scores are camera-independent)
                frame_meta = []
                for f_idx, kf in enumerate(ep["keyframes"]):
                    frame_meta.append({
                        "frame_idx": f_idx,
                        "step": kf["step"],
                        "timestamp": kf["timestamp"],
                        "score": kf["score"],
                        "frame_label": kf["frame_label"],
                    })

                # Save images and create one JSONL entry per camera
                collected_cameras = list((ep["keyframes"][0]["images"] or {}).keys())
                if not collected_cameras:
                    # No-render run: single entry with no image paths
                    result = {
                        "episode_id": ep_id,
                        "task": task_name,
                        "task_description": desc,
                        "task_category": category,
                        "target_score": target_score,
                        "camera": None,
                        "noise_scale": args.noise_scale,
                        "noise_mode": ep.get("noise_mode", "default"),
                        "total_steps": ep["total_steps"],
                        "final_score": ep["final_score"],
                        "frames": frame_meta,
                    }
                    all_results.append(result)
                else:
                    for cam in collected_cameras:
                        cam_dir = ep_dir / cam
                        cam_dir.mkdir(parents=True, exist_ok=True)

                        frames_with_paths = []
                        for f_idx, kf in enumerate(ep["keyframes"]):
                            img = (kf["images"] or {}).get(cam)
                            rel_path = None
                            if img is not None:
                                fname = f"frame_{f_idx}_{kf['timestamp']:.2f}s.jpg"
                                abs_path = cam_dir / fname
                                Image.fromarray(img).save(str(abs_path), quality=95)
                                # Store path relative to keyframes root for portability
                                rel_path = str(Path(ep_id) / cam / fname)

                            frames_with_paths.append({
                                **frame_meta[f_idx],
                                "image_path": rel_path,
                            })

                        result = {
                            "episode_id": f"{ep_id}_{cam}",
                            "episode_base_id": ep_id,
                            "task": task_name,
                            "task_description": desc,
                            "task_category": category,
                            "target_score": target_score,
                            "camera": cam,
                            "noise_scale": args.noise_scale,
                            "noise_mode": ep.get("noise_mode", "default"),
                            "total_steps": ep["total_steps"],
                            "final_score": ep["final_score"],
                            "frames": frames_with_paths,
                        }
                        all_results.append(result)

    # Write JSONL — all_results was seeded from existing entries at startup
    # and new episodes were appended, so writing it out gives a complete merged file.
    merged = {r["episode_id"]: r for r in all_results}
    with open(jsonl_path, "w") as f:
        for r in merged.values():
            f.write(json.dumps(r) + "\n")

    # Count unique base episodes (not multiplied by cameras)
    unique_eps = len({r.get("episode_base_id", r["episode_id"]) for r in all_results})
    n_cameras = len(args.cameras)
    print(f"\n{'='*60}")
    print(f"Done! {unique_eps} episodes × {n_cameras} cameras = {len(all_results)} JSONL entries")
    print(f"Keyframes: {output_dir / 'keyframes'}")
    print(f"Metadata: {jsonl_path}")

    print(f"\nOverall score distribution (unique episodes):")
    seen = set()
    score_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for r in all_results:
        base = r.get("episode_base_id", r["episode_id"])
        if base not in seen:
            seen.add(base)
            score_counts[r["final_score"]] += 1
    for s in [1, 2, 3, 4]:
        n = score_counts[s]
        pct = n / unique_eps * 100 if unique_eps else 0
        print(f"  Score {s}: {n} episodes ({pct:.0f}%)")

    # ------------------------------------------------------------------ #
    # Completeness report: flag missing and partial task/score combos      #
    # ------------------------------------------------------------------ #
    n_target = args.episodes_per_score
    missing = []   # 0 episodes collected
    partial = []   # 1 .. n_target-1 episodes collected

    print(f"\nPer-task breakdown (target: {n_target} eps/score):")
    for tn in tasks:
        if tn not in task_stats:
            continue
        counts = task_stats[tn]
        cat = get_task_category(tn)
        valid = get_valid_scores(cat, tn)
        parts = []
        for s in valid:
            got = counts.get(s, 0)
            tag = f"s{s}={got}"
            if got == 0:
                tag += "✗"
                missing.append((tn, s))
            elif got < n_target:
                tag += f"(partial)"
                partial.append((tn, s, got))
            parts.append(tag)
        total = sum(counts.values())
        print(f"  {tn:30s} [{cat:12s}] {total:3d} eps  ({', '.join(parts)})")

    print(f"\n{'='*60}")
    print(f"COMPLETENESS SUMMARY")
    print(f"  Target: {n_target} episodes per score per task")
    print(f"  Fully collected: "
          f"{sum(1 for tn in tasks if tn in task_stats for s in get_valid_scores(get_task_category(tn), tn) if task_stats[tn].get(s, 0) == n_target)} "
          f"task/score combinations")
    if partial:
        print(f"  Partial ({len(partial)} combos):")
        for tn, s, got in partial:
            print(f"    {tn} score {s}: {got}/{n_target}")
    if missing:
        print(f"  Missing / 0 collected ({len(missing)} combos):")
        for tn, s in missing:
            print(f"    {tn} score {s}: 0/{n_target}")
    if not missing and not partial:
        print("  All task/score combinations fully collected!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
