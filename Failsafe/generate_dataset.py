#!/usr/bin/env python3
"""
Generate the full FailSafe dataset in robometer_frame_dataset format.

For each task (FailPickCube, FailPushCube, FailStackCube):
  - For each slot (failure mode or success):
    - Generate N unique instances with distinct initial-condition seeds.
    - Each instance is rendered from 3 cameras (front/side/wrist); each camera
      view is emitted as its own episode (matching robometer's 1-camera format).
    - 16 keyframes per episode, labeled by the state-based scorer.

Output layout (robometer_frame_dataset compatible):
  <output>/
    keyframes/<archive>/shard-NNNNN.tar
    keyframes_success/<archive>/shard-NNNNN.tar
    manifests/<archive>_failures.jsonl
    manifests/<archive>_successes.jsonl
    scores/<archive>_scored.jsonl

Archive names: failsafe_pick, failsafe_push, failsafe_stack.
Family: failsafe.

Usage:
    # One task, full scale
    python generate_dataset.py --task FailPickCube-v1 --output-dir /path/to/out

    # One slot only (for parallelization)
    python generate_dataset.py --task FailPickCube-v1 --slot-id pick_s2_approach_retreat

    # Small smoke test
    python generate_dataset.py --task FailPickCube-v1 --slot-id pick_success --instances 2
"""

import argparse
import json
import os
import random
import shutil
import sys
import tarfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Share slot configs + helpers from generate_failures.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from generate_failures import (
    TASK_CONFIGS,
    FAILSAFE_PATH,
    FAILSAFE_AVAILABLE,
    _import_failsafe_tasks,
    _install_state_logger,
    _extend_open_gripper,
    _extend_forward_push,
    _score_frame_stack,
    _score_frame_push,
    _score_frame_pick,
    patch_failsafe_save_path,
    _numpy_to_pil,
    _vec3,
    _obj_pos,
    _read_state,
    compute_frame_labels,
    SCORERS,
)

# Robometer format: 16 keyframes per episode
N_KEYFRAMES = 16
FPS_NOMINAL = 20.0  # ManiSkill default control freq

# Archive + family names
TASK_TO_ARCHIVE = {
    "FailPickCube-v1":  "failsafe_pick",
    "FailPushCube-v1":  "failsafe_push",
    "FailStackCube-v1": "failsafe_stack",
}
FAMILY = "failsafe"


# ---------------------------------------------------------------------------
# Label slot overrides: successes get fewer instances
# ---------------------------------------------------------------------------

def slot_num_instances(slot, default_failures=100, default_success=25):
    if "num_instances" in slot:
        return slot["num_instances"]
    if slot.get("target_score") == 5 and slot.get("mode") == "success":
        return default_success
    return default_failures


def slot_is_success(slot):
    return slot.get("mode") == "success"


# ---------------------------------------------------------------------------
# 16-keyframe index picker
# ---------------------------------------------------------------------------

def pick_keyframe_indices(total_steps, n=N_KEYFRAMES, freeze_end=False):
    if total_steps < n:
        # pad by repeating last step
        idx = list(range(total_steps)) + [total_steps - 1] * (n - total_steps)
    else:
        idx = [round(i * (total_steps - 1) / (n - 1)) for i in range(n)]
    if freeze_end:
        end = total_steps - 1
        idx[-1] = end
        idx[-2] = end
    return idx


# ---------------------------------------------------------------------------
# Core generation: one instance of a slot, per-camera episodes
# ---------------------------------------------------------------------------

def _configure_failsafe(fw, slot, mode):
    """Configure FailSafe's active failure type/stage/noise for this slot."""
    if mode in ("freeze", "success", "success_extend"):
        noise_val = 0.0
        fail_stage = slot.get("fail_stage", 0)
        fail_type = slot.get("fail_type", "trans_x")
    else:
        noise_val = slot["noise"]
        fail_stage = slot["fail_stage"]
        fail_type = slot["fail_type"]

    fw._fail_plan_wrapper.set_active_type(fail_type)
    fw._fail_plan_wrapper.set_active_stage(fail_stage)
    af = fw._fail_plan_wrapper._active_fail
    if af is not None:
        af.noise = noise_val
        if fail_stage not in af.stages:
            af.stages = list(af.stages) + [fail_stage]
    return fail_stage, fail_type


def _collect_png_dirs(save_root, task_name, attempt, fail_stage, fail_type, ep_idx):
    """Return {camera: Path} for the 3 camera subdirs written by FailSafe."""
    base = Path(save_root) / task_name / str(attempt) / f"{fail_stage}_{fail_type}_{ep_idx}"
    return {c: base / c for c in ("front", "side", "wrist")}, base


def _score_and_accept(slot, mode, labels):
    """Compute a score for an attempt vs the slot's accept criteria."""
    if mode == "overshoot":
        accept_peak, accept_drop = 5, True
    else:
        accept_peak = slot.get("accept_peak")
        accept_drop = slot.get("accept_drop", False)

    peak = max(labels)
    final = labels[-1]
    peak_ok = (accept_peak is None) or (peak == accept_peak)
    drop_ok = (not accept_drop) or (final < peak)

    score = 0
    if accept_peak is not None:
        score -= abs(peak - accept_peak) * 10
    if accept_drop and not drop_ok:
        score -= 5

    return peak_ok and drop_ok, score


def generate_slot_instance(fw, task_name, slot, instance_idx, failsafe_save_dir, base_seed=0):
    """Generate ONE instance of a slot (retries for acceptance).

    Returns: dict with keys
        png_dirs (front/side/wrist path),
        kf_idx, labels, total_steps, n_source_frames, success_flag
    or None on failure.
    """
    mode = slot["mode"]
    fail_stage, fail_type = _configure_failsafe(fw, slot, mode)

    states, restore = _install_state_logger(fw, task_name)
    try:
        max_attempts = slot.get("max_attempts", 20)
        # Give success mode more unique-seed budget because we need many distinct successes
        if mode == "success":
            max_attempts = max(5, max_attempts)

        best = None  # (score, png_dirs, kf_idx, labels, n_source, base_png_dir)

        for attempt in range(max_attempts):
            states.clear()
            # Distinct seeds per (instance, attempt). base_seed supports running
            # the same slot across multiple SLURM shards without collision.
            num_gt = base_seed + instance_idx * 1000 + attempt
            success = fw.get_failure(num_gt=num_gt)

            want_success = mode in ("freeze", "success", "success_extend")
            want_failure = mode == "fail_inject"
            # overshoot: accept either (filter via labels later)

            if want_success and not success:
                try: fw.save_video(save=False)
                except OSError: pass
                continue
            if want_failure and success:
                try: fw.save_video(save=False)
                except OSError: pass
                continue

            # Apply extension action (e.g., open_gripper, forward_push)
            extend_action = slot.get("extend_action")
            if extend_action:
                n_ext = slot.get("extend_steps", 30)
                if extend_action == "open_gripper":
                    _extend_open_gripper(fw, task_name, n_steps=n_ext)
                elif extend_action == "forward_push":
                    _extend_forward_push(fw, task_name, n_steps=n_ext)

            ep_idx = attempt + 1
            try:
                fw.save_video(save=True, num_gt=num_gt, ep_idx=ep_idx)
            except OSError as e:
                print(f"      save_video error: {e}")
                continue

            png_dirs, base_png_dir = _collect_png_dirs(
                failsafe_save_dir, task_name, num_gt, fail_stage, fail_type, ep_idx)
            if not png_dirs["front"].exists():
                continue

            front_pngs = sorted(
                [p for p in png_dirs["front"].iterdir() if p.suffix == ".png"],
                key=lambda p: int(p.stem))
            if len(front_pngs) < 4:
                shutil.rmtree(base_png_dir, ignore_errors=True)
                continue

            ep_states = list(states)
            n_source = len(front_pngs)

            # Freeze: truncate
            if mode == "freeze":
                frac = slot.get("freeze_frac", 0.70)
                cutoff = max(4, int(n_source * frac))
                n_source = cutoff
                ep_states = ep_states[:cutoff]

            kf_idx = pick_keyframe_indices(n_source, freeze_end=(mode == "freeze"))
            labels = compute_frame_labels(ep_states, kf_idx, task_name)

            # Overshoot filter
            if mode == "overshoot" and (max(labels) < 5 or labels[-1] >= 5):
                shutil.rmtree(base_png_dir, ignore_errors=True)
                continue

            is_ideal, score = _score_and_accept(slot, mode, labels)

            if is_ideal:
                if best is not None:
                    try: shutil.rmtree(best[5], ignore_errors=True)
                    except Exception: pass
                print(f"    inst={instance_idx} attempt={attempt+1} ACCEPT labels={labels}")
                return {
                    "png_dirs": png_dirs,
                    "kf_idx": kf_idx,
                    "labels": labels,
                    "n_source_frames": n_source,
                    "base_png_dir": base_png_dir,
                    "fps": FPS_NOMINAL,
                }

            if best is None or score > best[0]:
                if best is not None:
                    try: shutil.rmtree(best[5], ignore_errors=True)
                    except Exception: pass
                best = (score, png_dirs, kf_idx, labels, n_source, base_png_dir)
            else:
                try: shutil.rmtree(base_png_dir, ignore_errors=True)
                except Exception: pass

        # Exhausted: fall back to best
        if best is not None:
            print(f"    inst={instance_idx} FALLBACK (score={best[0]}) labels={best[3]}")
            return {
                "png_dirs": best[1],
                "kf_idx": best[2],
                "labels": best[3],
                "n_source_frames": best[4],
                "base_png_dir": best[5],
                "fps": FPS_NOMINAL,
            }
        print(f"    inst={instance_idx} FAILED after {max_attempts} attempts")
        return None
    finally:
        restore()


# ---------------------------------------------------------------------------
# random_away / zero_freeze: bypass FailSafe entirely
# ---------------------------------------------------------------------------

def generate_simple_instance(task_name, slot, instance_idx, base_seed=0):
    """Generate a random_away or zero_freeze instance. Returns dict with frames (PIL list)."""
    import gymnasium as gym
    if FAILSAFE_PATH:
        _import_failsafe_tasks()

    seed = base_seed + instance_idx * 1000 + hash(slot["id"]) % 997
    mode = slot["mode"]

    env = gym.make(task_name, obs_mode="rgb", render_mode="rgb_array", sensor_configs=dict())
    env.action_space.seed(seed)
    obs, info = env.reset(seed=seed)

    if mode == "zero_freeze":
        action = np.zeros(env.action_space.shape)
    elif mode == "random_away":
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
        lo, hi = env.action_space.low, env.action_space.high
        rng = np.random.default_rng(seed)

    frames = [_numpy_to_pil(env.render())]
    n_steps = 20
    for _ in range(n_steps):
        if mode == "random_away":
            action = env.action_space.sample()
            if len(action) >= 3:
                action[0] = float(np.clip(away_xy[0] * 0.95 + rng.normal(0, 0.08), lo[0], hi[0]))
                action[1] = float(np.clip(away_xy[1] * 0.95 + rng.normal(0, 0.08), lo[1], hi[1]))
                action[2] = float(np.clip(0.5 + rng.normal(0, 0.08), lo[2], hi[2]))
        obs, _, terminated, truncated, _ = env.step(action)
        frames.append(_numpy_to_pil(env.render()))
        if terminated or truncated:
            break
    env.close()

    n_source = len(frames)
    kf_idx = pick_keyframe_indices(n_source)
    labels = [1] * N_KEYFRAMES  # score-1 slot

    return {
        "frames": frames,
        "kf_idx": kf_idx,
        "labels": labels,
        "n_source_frames": n_source,
        "fps": FPS_NOMINAL,
    }


# ---------------------------------------------------------------------------
# Write an episode (one camera view) in robometer format
# ---------------------------------------------------------------------------

def write_episode_dir(out_root, archive, episode_id, frames_iter, kf_idx, labels,
                      n_source, fps, task_desc, label_str, terminal_reward,
                      paired_success_id=None):
    """Write <out_root>/<archive>/<episode_id>/{frame_NN_TTs.jpg,meta.json}.
    `frames_iter` yields a PIL image per kf_idx entry."""
    ep_dir = out_root / archive / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)

    for f_idx, (step, img) in enumerate(zip(kf_idx, frames_iter)):
        t = step / max(fps, 1e-6)
        fname = f"frame_{f_idx:02d}_{t:.2f}s.jpg"
        img.convert("RGB").save(ep_dir / fname, quality=92)

    meta = {
        "episode_id":       episode_id,
        "archive":          archive,
        "family":           FAMILY,
        "task":             task_desc,
        "label":            label_str,
        "terminal_reward":  terminal_reward,
        "n_source_frames":  int(n_source),
        "n_keyframes":      N_KEYFRAMES,
        "keyframes_dir":    f"{archive}/{episode_id}",
        "paired_success_id": paired_success_id,
        "frame_labels":     [int(l) for l in labels],
        "fps":              fps,
    }
    with open(ep_dir / "meta.json", "w") as f:
        json.dump(meta, f)
    return ep_dir, meta


def load_png_as_pil(path):
    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# Main loop: iterate a task's slots, emit episodes per (instance, camera)
# ---------------------------------------------------------------------------

def run_task(task_name, output_dir, slot_filter=None, instance_override=None,
             base_seed=0, keep_tmp=False):
    assert FAILSAFE_AVAILABLE, "FailSafe not importable — set FAILSAFE_PATH"

    cfg = TASK_CONFIGS[task_name]
    archive = TASK_TO_ARCHIVE[task_name]
    task_desc = cfg["description"]

    out = Path(output_dir)
    kf_fail_root = out / "keyframes"
    kf_succ_root = out / "keyframes_success"
    manifest_dir = out / "manifests"
    scores_dir   = out / "scores"
    tmp_failsafe = out / "_failsafe_tmp" / task_name
    for d in (kf_fail_root / archive, kf_succ_root / archive, manifest_dir, scores_dir,
              tmp_failsafe):
        d.mkdir(parents=True, exist_ok=True)

    manifest_fail = manifest_dir / f"{archive}_failures.jsonl"
    manifest_succ = manifest_dir / f"{archive}_successes.jsonl"
    scored_fail   = scores_dir   / f"{archive}_scored.jsonl"

    # PASS 1: successes first (so failures can reference paired_success_id)
    success_ids_by_task = []  # list of success episode_ids for this task
    failure_records = []     # buffer for paired_success_id assignment

    # Handle FailSafe-using slots: create one FailgenWrapper per slot (reused across instances)
    from failgen.env_wrapper import FailgenWrapper

    # Iterate slots, successes first then failures
    success_slots = [s for s in cfg["slots"] if slot_is_success(s)]
    other_slots   = [s for s in cfg["slots"] if not slot_is_success(s)]

    def apply_filter(slots):
        if slot_filter is None:
            return slots
        return [s for s in slots if s["id"] in slot_filter]

    # === PASS 1: SUCCESSES ===
    for slot in apply_filter(success_slots):
        n_inst = instance_override if instance_override else slot_num_instances(slot)
        print(f"\n[{slot['id']}] SUCCESS mode, generating {n_inst} instances × 3 cameras")

        with patch_failsafe_save_path(task_name, tmp_failsafe):
            fw = FailgenWrapper(task_name=task_name, headless=True, save_video=True)
            _configure_failsafe(fw, slot, slot["mode"])

            for inst in range(n_inst):
                result = generate_slot_instance(fw, task_name, slot, inst,
                                                 tmp_failsafe, base_seed=base_seed)
                if result is None:
                    continue

                kf_idx = result["kf_idx"]
                labels = result["labels"]
                n_src  = result["n_source_frames"]

                # Emit 3 episodes (one per camera view)
                for cam in ("front", "side", "wrist"):
                    png_dir = result["png_dirs"][cam]
                    if not png_dir.exists():
                        continue
                    png_files = sorted(
                        [p for p in png_dir.iterdir() if p.suffix == ".png"],
                        key=lambda p: int(p.stem))
                    if not png_files:
                        continue

                    episode_id = f"{archive}_{slot['id']}_inst{inst:04d}_{cam}"
                    frames_iter = (load_png_as_pil(png_files[min(i, len(png_files)-1)])
                                   for i in kf_idx)
                    _, meta = write_episode_dir(
                        kf_succ_root, archive, episode_id, frames_iter, kf_idx,
                        labels, n_src, result["fps"], task_desc,
                        label_str="success", terminal_reward=5,
                        paired_success_id=None)
                    # Write to success manifest
                    with open(manifest_succ, "a") as fh:
                        fh.write(json.dumps(meta) + "\n")
                    success_ids_by_task.append(episode_id)

                # Clean up this instance's PNG tree
                try: shutil.rmtree(result["base_png_dir"], ignore_errors=True)
                except Exception: pass

    if not success_ids_by_task and manifest_succ.exists():
        # Re-read existing success manifest (e.g., when slot_filter is only failures)
        with open(manifest_succ) as f:
            success_ids_by_task = [json.loads(l)["episode_id"] for l in f if l.strip()]

    print(f"\n  -> {len(success_ids_by_task)} success episodes available for pairing")

    # === PASS 2: FAILURES & SCORE-1 ===
    rng = random.Random(base_seed + 12345)

    def pair_success():
        if not success_ids_by_task:
            return None
        return rng.choice(success_ids_by_task)

    for slot in apply_filter(other_slots):
        n_inst = instance_override if instance_override else slot_num_instances(slot)
        mode = slot["mode"]
        target = slot["target_score"]
        print(f"\n[{slot['id']}] {mode} target={target}, {n_inst} instances × 3 cameras")

        # Score-1 slots (no FailSafe wrapper needed)
        if mode in ("random_away", "zero_freeze"):
            for inst in range(n_inst):
                result = generate_simple_instance(task_name, slot, inst, base_seed=base_seed)
                if result is None:
                    continue
                kf_idx = result["kf_idx"]
                labels = result["labels"]
                n_src  = result["n_source_frames"]
                # Use the same PIL frames for all 3 cameras (no multi-cam for these).
                # We still emit 3 episode entries so the count matches the rest.
                for cam in ("front", "side", "wrist"):
                    episode_id = f"{archive}_{slot['id']}_inst{inst:04d}_{cam}_score_{labels[-1]}"
                    frames = result["frames"]
                    frames_iter = (frames[min(i, len(frames)-1)] for i in kf_idx)
                    psid = pair_success()
                    _, meta = write_episode_dir(
                        kf_fail_root, archive, episode_id, frames_iter, kf_idx,
                        labels, n_src, result["fps"], task_desc,
                        label_str="failure", terminal_reward=labels[-1],
                        paired_success_id=psid)
                    with open(manifest_fail, "a") as fh:
                        fh.write(json.dumps(meta) + "\n")
                    with open(scored_fail, "a") as fh:
                        fh.write(json.dumps({
                            "episode_id": episode_id,
                            "archive": archive,
                            "task": task_desc,
                            "terminal_reward": int(labels[-1]),
                            "frame_labels": [int(l) for l in labels],
                            "paired_success_id": psid,
                        }) + "\n")
            continue

        # FailSafe slot — create wrapper once per slot
        with patch_failsafe_save_path(task_name, tmp_failsafe):
            fw = FailgenWrapper(task_name=task_name, headless=True, save_video=True)
            _configure_failsafe(fw, slot, mode)

            for inst in range(n_inst):
                result = generate_slot_instance(fw, task_name, slot, inst,
                                                 tmp_failsafe, base_seed=base_seed)
                if result is None:
                    continue
                kf_idx = result["kf_idx"]
                labels = result["labels"]
                n_src  = result["n_source_frames"]
                terminal = int(labels[-1])

                for cam in ("front", "side", "wrist"):
                    png_dir = result["png_dirs"][cam]
                    if not png_dir.exists():
                        continue
                    png_files = sorted(
                        [p for p in png_dir.iterdir() if p.suffix == ".png"],
                        key=lambda p: int(p.stem))
                    if not png_files:
                        continue

                    episode_id = f"{archive}_{slot['id']}_inst{inst:04d}_{cam}_score_{terminal}"
                    frames_iter = (load_png_as_pil(png_files[min(i, len(png_files)-1)])
                                   for i in kf_idx)
                    psid = pair_success()
                    _, meta = write_episode_dir(
                        kf_fail_root, archive, episode_id, frames_iter, kf_idx,
                        labels, n_src, result["fps"], task_desc,
                        label_str="failure", terminal_reward=terminal,
                        paired_success_id=psid)
                    with open(manifest_fail, "a") as fh:
                        fh.write(json.dumps(meta) + "\n")
                    with open(scored_fail, "a") as fh:
                        fh.write(json.dumps({
                            "episode_id": episode_id,
                            "archive": archive,
                            "task": task_desc,
                            "terminal_reward": terminal,
                            "frame_labels": [int(l) for l in labels],
                            "paired_success_id": psid,
                        }) + "\n")

                try: shutil.rmtree(result["base_png_dir"], ignore_errors=True)
                except Exception: pass

    if not keep_tmp:
        try: shutil.rmtree(out / "_failsafe_tmp", ignore_errors=True)
        except Exception: pass


# ---------------------------------------------------------------------------
# Shard packing: loose episode dirs -> shard-NNNNN.tar
# ---------------------------------------------------------------------------

def pack_shards(archive_root, shard_size_episodes=512):
    """Pack all episode dirs under archive_root/<archive>/ into tar shards."""
    for archive_dir in archive_root.iterdir():
        if not archive_dir.is_dir():
            continue
        ep_dirs = sorted([p for p in archive_dir.iterdir() if p.is_dir()])
        if not ep_dirs:
            continue
        shard_idx = 0
        for i in range(0, len(ep_dirs), shard_size_episodes):
            batch = ep_dirs[i:i + shard_size_episodes]
            shard_path = archive_dir / f"shard-{shard_idx:05d}.tar"
            with tarfile.open(shard_path, "w") as tar:
                for ep in batch:
                    for f in sorted(ep.iterdir()):
                        tar.add(f, arcname=f"{ep.name}/{f.name}")
            for ep in batch:
                shutil.rmtree(ep, ignore_errors=True)
            print(f"  packed {shard_path.name} ({len(batch)} episodes)")
            shard_idx += 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--task", required=True, choices=list(TASK_CONFIGS.keys()))
    p.add_argument("--slot-id", action="append", default=None,
                   help="Run only these slot IDs (repeatable). If omitted, run all slots.")
    p.add_argument("--instances", type=int, default=None,
                   help="Override num_instances per slot (for smoke tests).")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--no-pack", action="store_true",
                   help="Skip tar-shard packing at the end (useful for parallel runs).")
    p.add_argument("--pack-only", action="store_true",
                   help="Only pack existing loose episode dirs into shards.")
    p.add_argument("--keep-tmp", action="store_true")
    args = p.parse_args()

    output_dir = Path(args.output_dir)

    if args.pack_only:
        print(f"Packing shards under {output_dir}...")
        if (output_dir / "keyframes").exists():
            pack_shards(output_dir / "keyframes")
        if (output_dir / "keyframes_success").exists():
            pack_shards(output_dir / "keyframes_success")
        print("Done.")
        return

    run_task(args.task, output_dir,
             slot_filter=args.slot_id,
             instance_override=args.instances,
             base_seed=args.base_seed,
             keep_tmp=args.keep_tmp)

    if not args.no_pack:
        print(f"\nPacking shards for {args.task}...")
        pack_shards(output_dir / "keyframes")
        pack_shards(output_dir / "keyframes_success")

    print("\nDone.")


if __name__ == "__main__":
    main()
