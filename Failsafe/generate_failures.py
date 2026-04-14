#!/usr/bin/env python3
"""
Generate failure trajectories for ManiSkill tasks using FailSafe's
stage-based motion-planning failure injection.

For each task, failures are produced by injecting positional/rotational noise
at a specific motion stage (reach, grasp, lift, align, etc.). The stage at
which failure is injected determines the RoboMeter 1-4 progress score.

Stage → Score hypothesis (to be verified via HTML viewer):
  Score 1  — random actions, robot never meaningfully approaches the object
  Score 2  — failure at early approach/grasp stage (arm reached region but didn't engage)
  Score 3  — failure at transport/execution stage (object grasped/contacted, task failed midway)
  Score 4  — failure at near-completion stage (only in StackCube: cube aligned, failed to place)

Tasks covered:
  FailPushCube-v1   → Scores 1, 2, 3
  FailStackCube-v1  → Scores 1, 2, 3, 4

NOTE: FailPickCube-v1 is excluded — it uses the SO100 robot but the FailSafe
      solve function is written for PandaArmMotionPlanningSolver. The kinematic
      mismatch makes the arm never reach the cube.

Output format matches MetaWorld/generate_failures.py (JSONL + keyframes/).

Usage:
    python generate_failures.py --episodes-per-score 3 --output-dir output/sample_v1
    python generate_failures.py --episodes-per-score 20 --output-dir output/grand_v1
    python generate_failures.py --tasks FailPickCube-v1 --episodes-per-score 5

Requires:
    conda activate failsafe
    export FAILSAFE_PATH=/path/to/cloned/FailSafe_code
    export MUJOCO_GL=egl  (headless cluster rendering)
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
    print("WARNING: FailSafe not importable. Set FAILSAFE_PATH to the cloned repo root.")
    print("         Only Score 1 (random-action) episodes will be generated.\n")


# ---------------------------------------------------------------------------
# Task configurations
# ---------------------------------------------------------------------------
# For each task we define:
#   stage_to_score: {stage_idx → (target_score, stage_name, human description)}
#   fail_types_per_stage: {stage_idx → [fail_type, ...]}  (from FailSafe noise types)
#   stage_progress_score: score the robot has reached after successfully
#                         completing each stage (used for per-frame labels)
#   valid_scores: which RoboMeter scores this task can produce
#
# FAIL TYPES available in FailSafe:
#   trans_x, trans_y, trans_z  — positional offset noise on end-effector
#   rot_x, rot_y, rot_z        — rotational offset noise on end-effector
#
# NOTE: stages are 0-indexed and come from the FailSafe YAML configs:
#   FailPickCube-v1   stages [0, 1, 3]   = reach_pose, grasp_pose, lift_to_goal
#   FailPushCube-v1   stages [0, 1, 2]   = close_gripper, reach_pose, goal_pose
#   FailStackCube-v1  stages [0,1,2,3,4,5,6] = search,reach,grasp,close,lift,align,place

TASK_CONFIGS = {
    # NOTE: FailPickCube-v1 is NOT included here.
    # That task uses the SO100 robot but the solve function uses PandaArmMotionPlanningSolver.
    # The kinematic mismatch means the arm never reaches the cube. Confirmed by visual inspection.
    # Only FailPushCube-v1 and FailStackCube-v1 (both PandaWristCam) produce valid trajectories.

    "FailPushCube-v1": {
        "description": "Push the red cube to the green goal position.",
        # Stage mapping (verified against soln_push_cube.py + debug stg_index):
        #   stg_index = [close_end, reach_end, goal_end]
        #   Baseline no-failure durations: close≈5 steps, reach≈30 steps, push≈35 steps
        #
        #   stage 0 = close_gripper  — not a pose move; no trans/rot injection.
        #   stage 1 = reach_pose     — arm approaches cube from the side.
        #             Noise 0.15m keeps target reachable; arm visibly approaches but misses.
        #             YAML trans_x stages=[1] ✓, trans_y stages=[1,2] ✓
        #   stage 2 = goal_pose      — arm pushes cube toward goal.
        #             Noise 0.3m keeps target reachable; arm pushes cube off-course.
        #             YAML trans_y stages=[1,2] ✓, trans_z stages=[2] ✓
        "stage_to_score": {
            1: (2, "reach_pose",   "arm approaches cube contact point but misses engagement"),
            2: (3, "push_to_goal", "cube contacted, arm pushes off-course"),
        },
        "fail_types_per_stage": {
            1: ["trans_x", "trans_y"],
            2: ["trans_y", "trans_z"],
        },
        # Noise per stage: keep targets reachable so planner executes the stage.
        # score 2 = 0.15m (arm near cube, misses by ~7cm), score 3 = 0.3m (cube pushed wrong dir)
        "noise_per_stage": {
            1: 0.15,
            2: 0.30,
        },
        # stg_score_transitions[fail_stage] = [(stg_index_position, new_score), ...]
        # After stg_index[stg_index_position] is passed, the frame score becomes new_score.
        "stg_score_transitions": {
            1: [(0, 2)],           # after close_gripper (stg_index[0]), arm in approach region
            2: [(0, 2), (1, 3)],   # after close → approach; after reach → pushing
        },
    },

    "FailStackCube-v1": {
        "description": "Pick up the red cube and stack it on top of the green cube.",
        # Stage mapping (verified against soln_stack_cube.py + debug stg_index):
        #   stg_index = [-1, reach_end, grasp_end, close_end, lift_end, align_end, open_end]
        #   Baseline no-failure durations (seed 0): reach≈29, grasp≈15, close≈6, lift≈24, align≈26, open≈6
        #
        #   stage 0 = search (dry-run, stg_index[0]=-1, no recorded moves)
        #   stage 1 = reach_pose   — same problem as PickCube; stage 2 corrects the arm anyway.
        #             Skip as fail_stage.
        #   stage 2 = grasp_pose   — actual grasp target. Noise 0.15m → arm near cube, misses.
        #             YAML stages=[1,2,5] ✓
        #   stage 3 = close_gripper — not a pose move.
        #   stage 4 = lift_pose    — cube grasped, arm lifts to WRONG height (not unreachable).
        #             lift_pose = grasp_pose + 0.1m up. Noise 0.3m: target ~0.25m up (reachable).
        #             YAML stages=[1,2,5] excludes 4 → stages override at runtime.
        #   stage 5 = align_pose   — cube lifted, arm aligns above target but slightly off.
        #             Noise 0.15m → cube is in the air, near-miss on stacking.
        #             YAML stages=[1,2,5] ✓
        "stage_to_score": {
            2: (2, "grasp_cube",  "arm at grasp position but gripper misaligns"),
            4: (3, "lift_cube",   "cube grasped, arm lifts to wrong height before dropping"),
            5: (4, "align_cubes", "cube above target cube, fails to lower cleanly onto it"),
        },
        "fail_types_per_stage": {
            2: ["trans_x", "trans_y"],
            4: ["trans_x", "trans_y"],   # runtime stages override needed
            5: ["trans_x", "trans_y"],
        },
        "noise_per_stage": {
            2: 0.15,   # score 2: near-miss grasp
            4: 0.30,   # score 3: cube lifted to wrong height (still visible above table)
            5: 0.15,   # score 4: cube above target, slight misalignment
        },
        # After stg_index[1] (reach done): arm in grasp region → score 2
        # After stg_index[3] (close done): cube is grasped → score 3
        # After stg_index[4] (lift done): cube is in the air → score 4
        "stg_score_transitions": {
            2: [(1, 2)],
            4: [(1, 2), (3, 3)],
            5: [(1, 2), (3, 3), (4, 4)],
        },
    },
}


# ---------------------------------------------------------------------------
# FailSafe YAML patching (sets save_path at runtime)
# ---------------------------------------------------------------------------

@contextmanager
def patch_failsafe_save_path(task_name, save_path):
    """Temporarily override save_path in FailSafe's task YAML config."""
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
    """
    Convert a raw frame (numpy array or torch Tensor) to a PIL RGB image.

    ManiSkill's GPU pipeline may return frames as float32 [0,1] or uint8 [0,255],
    with 3 (RGB) or 4 (RGBA) channels, possibly with a leading batch dimension.
    """
    if hasattr(np_img, 'cpu'):        # torch.Tensor
        np_img = np_img.cpu().numpy()
    if hasattr(np_img, 'detach'):
        np_img = np_img.detach().numpy()
    np_img = np.asarray(np_img)
    # Squeeze leading batch dim if present (e.g. shape (1,H,W,C) → (H,W,C))
    if np_img.ndim == 4:
        np_img = np_img[0]
    # Drop alpha channel
    if np_img.ndim == 3 and np_img.shape[-1] == 4:
        np_img = np_img[:, :, :3]
    # Convert float → uint8
    if np_img.dtype != np.uint8:
        if np_img.max() <= 1.0 + 1e-6:          # [0, 1] float
            np_img = (np_img * 255).clip(0, 255).astype(np.uint8)
        else:                                    # already in [0, 255] float
            np_img = np_img.clip(0, 255).astype(np.uint8)
    return Image.fromarray(np_img)


def _clear_recorder(fw):
    """Reset all per-episode frame storage in the RecordEpisode wrapper."""
    fw._env._images_storage = [[], [], []]
    fw._env.render_images = []
    fw._env._video_steps = 0


def collect_random_episode(task_name, seed=0, max_steps=50):
    """
    Score 1: run pure random actions to produce a no-progress episode.
    Returns (list of PIL images at each step, total_steps).
    """
    import gymnasium as gym
    # FailSafe tasks need their module registered; import to trigger @register_env
    if FAILSAFE_PATH:
        _import_failsafe_tasks()

    env = gym.make(task_name, obs_mode="rgb", render_mode="rgb_array", sensor_configs=dict())
    obs, info = env.reset(seed=seed)
    frames = [_numpy_to_pil(env.render())]
    steps = 0
    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(_numpy_to_pil(env.render()))
        steps += 1
        if terminated or truncated:
            break
    env.close()
    return frames, steps


def _import_failsafe_tasks():
    """Import FailSafe task modules so ManiSkill registers the envs."""
    import importlib
    for mod in ["failgen.tasks.fail_pick_cube",
                "failgen.tasks.fail_push_cube",
                "failgen.tasks.fail_stack_cube"]:
        try:
            importlib.import_module(mod)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Per-frame label computation (stg_index-based, physics-grounded)
# ---------------------------------------------------------------------------

def compute_frame_labels(stg_index, stg_score_transitions, frame_step_indices):
    """
    Assign per-frame RoboMeter 1-4 labels using stg_index stage boundaries.

    stg_index: list of step indices where each stage ends (from fw.stg_index).
               -1 entries (e.g. StackCube stage 0 dry-run) are ignored.
    stg_score_transitions: [(stg_index_position, new_score), ...] in ascending order.
               After the step at stg_index[stg_index_position], score becomes new_score.
    frame_step_indices: step index for each keyframe.

    This replaces the previous time-fraction heuristic which assigned labels based
    on wall-clock proportion of the episode, not actual robot progress.
    """
    labels = []
    for step in frame_step_indices:
        score = 1
        for stg_pos, new_score in stg_score_transitions:
            if (stg_pos < len(stg_index)
                    and stg_index[stg_pos] >= 0        # ignore -1 (dry-run)
                    and step > stg_index[stg_pos]):    # AFTER stage ended
                score = new_score
        labels.append(score)
    return labels


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def load_existing(jsonl_path):
    """Return set of episode_base_ids already in the output file."""
    done = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                ep = json.loads(line)
                done.add(ep.get("episode_base_id", ep["episode_id"]))
    return done


def write_episode(jsonl_path, record):
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Main collection logic
# ---------------------------------------------------------------------------

def collect_score1(task_name, cfg, output_dir, jsonl_path, n_episodes, done_ids, dry_run=False):
    """Collect Score 1 episodes using random actions (no expert)."""
    kf_root = output_dir / "keyframes"
    collected = 0

    for ep_idx in range(n_episodes * 3):  # over-sample in case render fails
        if collected >= n_episodes:
            break

        base_id = f"{task_name}_score1_{collected:03d}"
        if base_id in done_ids:
            collected += 1
            continue

        print(f"  [score 1] episode {collected+1}/{n_episodes} ...", end=" ", flush=True)
        if dry_run:
            print("(dry run)")
            collected += 1
            continue

        try:
            frames, total_steps = collect_random_episode(task_name, seed=ep_idx)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        # Select 8 evenly-spaced keyframes
        n_f = len(frames)
        indices = [round(i * (n_f - 1) / 7) for i in range(8)]
        keyframes = [(frames[idx], indices[idx]) for idx in range(8)]

        ep_dir = kf_root / base_id
        ep_dir.mkdir(parents=True, exist_ok=True)

        frame_records = []
        for f_idx, (img, step) in enumerate(keyframes):
            t = step / max(total_steps, 1)
            fname = f"frame_{f_idx}_{t:.2f}s.jpg"
            img.save(ep_dir / fname)
            frame_records.append({
                "frame_idx":   f_idx,
                "step":        int(step),
                "timestamp":   round(t, 4),
                "score":       1,
                "frame_label": 1,
                "image_path":  f"{base_id}/{fname}",
            })

        record = {
            "episode_id":      base_id,
            "episode_base_id": base_id,
            "task":            task_name,
            "task_description": cfg["description"],
            "target_score":    1,
            "fail_stage":      None,
            "stage_name":      "random_actions",
            "fail_type":       "random",
            "total_steps":     total_steps,
            "final_score":     1,
            "frames":          frame_records,
        }
        write_episode(jsonl_path, record)
        done_ids.add(base_id)
        collected += 1
        print(f"done ({total_steps} steps)")

    return collected


def collect_stage_failures(task_name, cfg, fail_stage, fail_type, target_score,
                           stage_name, failsafe_save_dir, output_dir, jsonl_path,
                           n_episodes, done_ids, dry_run=False):
    """Collect n_episodes failures for one (task, stage, fail_type) slot.

    Key design notes vs. the naive approach:
    - We do NOT call fw.save_video(save=True) — FailSafe's flush_multi_images crashes
      when the wrist camera slice has zero width (only 2 cameras rendered, not 3).
    - Instead, frames are read directly from fw._env._images_storage[0] (front camera).
    - _images_storage is cleared before each attempt so frames don't accumulate.
    - Failure.stages is patched at runtime when fail_stage is not in the YAML stages list
      (e.g. PickCube stage 3 / StackCube stage 4 both require this).
    """
    if not FAILSAFE_AVAILABLE:
        print(f"  [score {target_score} stage {fail_stage} {fail_type}] SKIP (FailSafe unavailable)")
        return 0

    kf_root = output_dir / "keyframes"
    stg_score_transitions = cfg["stg_score_transitions"][fail_stage]
    # Task-specific noise: keeps the noisy target reachable (planner executes the stage)
    # while being large enough to cause task failure.
    noise_val = cfg["noise_per_stage"].get(fail_stage, 0.20)

    collected = 0
    attempted = 0
    max_attempts = n_episodes * 30  # generous budget
    num_gt = 0

    with patch_failsafe_save_path(task_name, failsafe_save_dir):
        fw = FailgenWrapper(task_name=task_name, headless=True, save_video=True)

        # --- Configure the failure injection ---
        fw._fail_plan_wrapper.set_active_type(fail_type)
        fw._fail_plan_wrapper.set_active_stage(fail_stage)
        af = fw._fail_plan_wrapper._active_fail
        if af is not None:
            af.noise = noise_val
            # YAML may not list the requested fail_stage; patch it in so check_active passes.
            if fail_stage not in af.stages:
                af.stages = list(af.stages) + [fail_stage]
                print(f"    [stages patched: {af.stages}]")
        print(f"    [noise={noise_val}m for {fail_type} at stage {fail_stage}]")

        while collected < n_episodes and attempted < max_attempts:
            base_id = f"{task_name}_score{target_score}_{stage_name}_{fail_type}_{collected:03d}"
            if base_id in done_ids:
                collected += 1
                continue

            print(f"  [score {target_score} {stage_name} {fail_type}] "
                  f"ep {collected+1}/{n_episodes} (attempt {attempted+1}) ...",
                  end=" ", flush=True)

            if dry_run:
                print("(dry run)")
                collected += 1
                continue

            # Clear stale frames from previous attempt
            _clear_recorder(fw)

            success = fw.get_failure(num_gt=num_gt)
            attempted += 1
            num_gt += 1  # advance seed regardless of outcome for diversity

            stg_idx = fw.stg_index   # list of step indices where each stage ended
            n_frames_cap = len(fw._env._images_storage[0])
            print(f"stg_index={stg_idx} frames={n_frames_cap} success={success}")

            if success:
                print("(task succeeded, retry)")
                continue

            # ---- Failure episode: read frames directly from the recorder ----
            raw_frames = fw._env._images_storage[0]   # front camera
            if len(raw_frames) < 2:
                print(f"WARNING: only {len(raw_frames)} frames captured, skipping")
                continue

            # Sanitize and convert to PIL
            try:
                pil_frames = [_numpy_to_pil(f) for f in raw_frames]
            except Exception as e:
                print(f"WARNING: frame conversion failed ({e}), skipping")
                continue

            total_steps = len(pil_frames)

            # Select 8 evenly-spaced keyframes
            n_f = len(pil_frames)
            kf_indices = [round(i * (n_f - 1) / 7) for i in range(8)]
            kf_pairs = [(pil_frames[idx], kf_indices[idx]) for idx in range(8)]

            # Per-frame labels using stg_index stage boundaries (not time fraction)
            frame_steps = [step for _, step in kf_pairs]
            labels = compute_frame_labels(stg_idx, stg_score_transitions, frame_steps)

            # Save keyframes to our output directory
            ep_dir = kf_root / base_id
            ep_dir.mkdir(parents=True, exist_ok=True)

            frame_records = []
            for f_idx, ((img, step), label) in enumerate(zip(kf_pairs, labels)):
                t = step / max(total_steps, 1)
                fname = f"frame_{f_idx}_{t:.2f}s.jpg"
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
                "target_score":     target_score,
                "fail_stage":       fail_stage,
                "stage_name":       stage_name,
                "fail_type":        fail_type,
                "total_steps":      total_steps,
                "final_score":      labels[-1],
                "frames":           frame_records,
            }
            write_episode(jsonl_path, record)
            done_ids.add(base_id)
            collected += 1
            print(f"done ({total_steps} steps, labels={labels})")

    return collected


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-score", type=int, default=5,
                        help="Episodes to collect per (task, score) combination")
    parser.add_argument("--output-dir", default="output/sample_v1",
                        help="Root output directory")
    parser.add_argument("--tasks", nargs="+", default=list(TASK_CONFIGS.keys()),
                        help="Which tasks to run (default: all 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print collection plan without running episodes")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    kf_dir     = output_dir / "keyframes"
    jsonl_path = output_dir / "maniskill_failures.jsonl"
    # Temporary directory for FailSafe's own image output
    failsafe_save_dir = output_dir / "_failsafe_tmp"

    output_dir.mkdir(parents=True, exist_ok=True)
    kf_dir.mkdir(exist_ok=True)
    failsafe_save_dir.mkdir(exist_ok=True)

    done_ids = load_existing(jsonl_path)
    print(f"Resuming from {len(done_ids)} existing episodes in {jsonl_path}\n")

    for task_name in args.tasks:
        if task_name not in TASK_CONFIGS:
            print(f"Unknown task: {task_name}, skipping")
            continue

        cfg = TASK_CONFIGS[task_name]
        valid_scores = sorted({1} | {s for s, _, _ in cfg["stage_to_score"].values()})
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"  {cfg['description']}")
        print(f"  Valid scores: {valid_scores}")
        print(f"{'='*60}")

        n = args.episodes_per_score

        # --- Score 1: random actions (all tasks get score-1 episodes) ---
        print(f"\n[Score 1 — random actions] target: {n} episodes")
        got = collect_score1(task_name, cfg, output_dir, jsonl_path, n,
                             done_ids, dry_run=args.dry_run)
        print(f"  Collected: {got}/{n}")

        # --- Scores 2-4: stage-based failures ---
        for fail_stage, (target_score, stage_name, notes) in sorted(cfg["stage_to_score"].items()):
            fail_types = cfg["fail_types_per_stage"].get(fail_stage, ["trans_x"])
            # Distribute n_episodes across fail_types for visual diversity
            per_type = max(1, n // len(fail_types))

            for fail_type in fail_types:
                print(f"\n[Score {target_score} — stage {fail_stage} ({stage_name}) / {fail_type}]")
                print(f"  Expected behavior: {notes}")
                print(f"  Target: {per_type} episodes")

                got = collect_stage_failures(
                    task_name, cfg, fail_stage, fail_type, target_score,
                    stage_name, failsafe_save_dir, output_dir, jsonl_path,
                    per_type, done_ids, dry_run=args.dry_run
                )
                print(f"  Collected: {got}/{per_type}")

    # --- Completeness report ---
    print(f"\n{'='*60}")
    print("COMPLETENESS SUMMARY")
    print(f"{'='*60}")

    all_eps = []
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_eps.append(json.loads(line))

    from collections import defaultdict
    counts = defaultdict(int)
    for ep in all_eps:
        key = (ep["task"], ep["target_score"])
        counts[key] += 1

    for task_name in args.tasks:
        if task_name not in TASK_CONFIGS:
            continue
        cfg = TASK_CONFIGS[task_name]
        valid_scores = sorted({1} | {s for s, _, _ in cfg["stage_to_score"].values()})
        print(f"\n  {task_name}:")
        for score in valid_scores:
            got = counts.get((task_name, score), 0)
            status = "OK" if got >= args.episodes_per_score else f"PARTIAL {got}/{args.episodes_per_score}"
            print(f"    Score {score}: {got} episodes  [{status}]")

    total = len(all_eps)
    print(f"\n  Total episodes written: {total}")
    print(f"  Output:  {output_dir}/maniskill_failures.jsonl")
    print(f"  Frames:  {output_dir}/keyframes/")

    # Cleanup tmp FailSafe output
    if failsafe_save_dir.exists():
        shutil.rmtree(failsafe_save_dir)
        print(f"  Cleaned: {failsafe_save_dir}")


if __name__ == "__main__":
    main()
