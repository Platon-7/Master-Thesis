#!/usr/bin/env python3
"""
Generate the full MetaWorld dataset in robometer_frame_dataset format.

For each task (46 MetaWorld tasks, reach/peg-insertion excluded):
  - 25 success episodes via scripted expert (randomized initial conditions
    via MetaWorld MT1 train_tasks cycling)
  - 100 failure episodes per valid target score (1-4 depending on category)
  - Each instance is rendered from 3 cameras (corner2/corner3/gripperPOV);
    each camera view is emitted as its own episode.
  - 16 keyframes per episode, labeled by category-specific scoring.

Output layout (robometer_frame_dataset compatible):
  <output>/
    keyframes/<archive>/shard-NNNNN.tar           # failures
    keyframes_success/<archive>/shard-NNNNN.tar    # successes
    manifests/<archive>_failures.jsonl
    manifests/<archive>_successes.jsonl
    scores/<archive>_scored.jsonl

Archive names: metaworld_<task_with_underscores>, e.g. metaworld_pick_place_v3.
Family: metaworld.

Usage:
    # One task, full scale
    python generate_dataset.py --task pick-place-v3 --output-dir /path/to/out

    # Smoke test (1 success + 2 failure instances on one score)
    python generate_dataset.py --task pick-place-v3 \
        --n-success 1 --n-failures 2 --output-dir /tmp/mw_smoke
"""

import argparse
import json
import random
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

import metaworld

sys.path.insert(0, str(Path(__file__).parent))
from generate_failures import (
    SCRIPTED_POLICIES,
    EXCLUDED_TASKS,
    TASK_DESCRIPTIONS,
    get_task_category,
    get_valid_scores,
    get_mode_sequence,
    calibrate_task,
    detect_score,
    _run_one_episode,
    MetaworldWrapper,
    render_all_cameras,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_KEYFRAMES = 16
FPS_NOMINAL = 10.0          # MetaWorld ~10 Hz control
FAMILY = "metaworld"
CAMERAS = ["corner2", "corner3", "gripperPOV"]
DEFAULT_NOISE_SCALE = 0.5
DEFAULT_NOISY_STEPS = 60     # longer than default 40 so 16 kf always fit


def task_archive(task_name: str) -> str:
    """pick-place-v3 -> metaworld_pick_place_v3"""
    return "metaworld_" + task_name.replace("-", "_")


# ---------------------------------------------------------------------------
# Success episode: scripted expert, no noise, category scoring
# ---------------------------------------------------------------------------

def _run_success_episode(env, policy, category, calib, cameras,
                          max_steps=500, task_name="", n_keyframes=N_KEYFRAMES):
    """Run the scripted expert to completion. Returns (keyframes, n_steps) or None."""
    init_o2t_default = calib["init_o2t"]
    obs, info = env.reset()
    ep_init_o2t = None
    trajectory = []

    succeeded_at = None
    for step in range(max_steps):
        action = policy.get_action(obs)
        action = np.clip(action, -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)

        if ep_init_o2t is None:
            ep_init_o2t = info.get("obj_to_target", init_o2t_default)

        imgs = render_all_cameras(env, cameras) if cameras else {}
        score = detect_score(info, category, calib, ep_init_o2t, obs=obs)
        trajectory.append({"images": imgs, "step": step, "score": score})

        if info.get("success", 0) > 0:
            succeeded_at = step
            # Collect a few post-success frames so final keyframe shows completion.
            for post in range(min(8, max_steps - step - 1)):
                action = policy.get_action(obs)
                action = np.clip(action, -1.0, 1.0)
                obs, _, _, _, info = env.step(action)
                imgs = render_all_cameras(env, cameras) if cameras else {}
                score = detect_score(info, category, calib, ep_init_o2t, obs=obs)
                trajectory.append({"images": imgs, "step": step + 1 + post,
                                    "score": score})
            break

    n_steps = len(trajectory)
    if n_steps < n_keyframes or succeeded_at is None:
        return None

    indices = np.linspace(0, n_steps - 1, n_keyframes, dtype=int)
    keyframes = []
    for idx in indices:
        t = trajectory[idx]
        keyframes.append({
            "step": t["step"],
            "images": t["images"],
            "score": t["score"],
        })

    # Monotonic non-decreasing labels for successes
    raw = [kf["score"] for kf in keyframes]
    for i in range(1, len(raw)):
        if raw[i] < raw[i - 1]:
            raw[i] = raw[i - 1]
    for kf, lbl in zip(keyframes, raw):
        kf["frame_label"] = lbl

    return keyframes, n_steps


# ---------------------------------------------------------------------------
# Write one episode in robometer format
# ---------------------------------------------------------------------------

def write_episode_dir(out_root, archive, episode_id, frames_steps_labels,
                       n_source, fps, task_desc, label_str, terminal_reward,
                       paired_success_id=None):
    """Write <out_root>/<archive>/<episode_id>/{frame_NN_TT.TTs.jpg, meta.json}."""
    ep_dir = out_root / archive / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    for f_idx, (img, step, lbl) in enumerate(frames_steps_labels):
        t_sec = step / max(fps, 1e-6)
        fname = f"frame_{f_idx:02d}_{t_sec:.2f}s.jpg"
        Image.fromarray(img).convert("RGB").save(ep_dir / fname, quality=92)
        labels.append(int(lbl))

    meta = {
        "episode_id": episode_id,
        "archive": archive,
        "family": FAMILY,
        "task": task_desc,
        "label": label_str,
        "terminal_reward": int(terminal_reward),
        "n_source_frames": int(n_source),
        "n_keyframes": N_KEYFRAMES,
        "keyframes_dir": f"{archive}/{episode_id}",
        "paired_success_id": paired_success_id,
        "frame_labels": labels,
        "fps": fps,
    }
    with open(ep_dir / "meta.json", "w") as fh:
        json.dump(meta, fh)
    return meta


def _frames_for_cam(keyframes, cam):
    """Extract [(img, step, frame_label)] for one camera. Returns None if any missing."""
    out = []
    for kf in keyframes:
        img = kf["images"].get(cam) if kf.get("images") else None
        if img is None:
            return None
        out.append((img, kf["step"], kf["frame_label"]))
    return out


# ---------------------------------------------------------------------------
# Per-task driver
# ---------------------------------------------------------------------------

def run_task(task_name, output_dir, n_success=25, n_failures_per_score=100,
             base_seed=0, cameras=CAMERAS, max_attempts=12,
             noise_scale=DEFAULT_NOISE_SCALE, noisy_steps=DEFAULT_NOISY_STEPS):
    if task_name in EXCLUDED_TASKS:
        print(f"Skip excluded task: {task_name}")
        return
    if task_name not in SCRIPTED_POLICIES:
        print(f"No scripted policy for {task_name}")
        return

    category = get_task_category(task_name)
    valid_scores = get_valid_scores(category, task_name)
    task_desc = TASK_DESCRIPTIONS.get(task_name, task_name)
    archive = task_archive(task_name)

    out = Path(output_dir)
    kf_fail_root = out / "keyframes"
    kf_succ_root = out / "keyframes_success"
    manifest_dir = out / "manifests"
    scores_dir = out / "scores"
    for d in (kf_fail_root / archive, kf_succ_root / archive,
              manifest_dir, scores_dir):
        d.mkdir(parents=True, exist_ok=True)

    manifest_fail = manifest_dir / f"{archive}_failures.jsonl"
    manifest_succ = manifest_dir / f"{archive}_successes.jsonl"
    scored_fail = scores_dir / f"{archive}_scored.jsonl"

    print(f"\n{'='*60}")
    print(f"Task: {task_name} [{category}] scores={valid_scores}")
    print(f"  {task_desc}")
    print(f"{'='*60}")

    mt1 = metaworld.MT1(task_name, seed=42 + base_seed)
    env_cls = mt1.train_classes[task_name]
    train_tasks = list(mt1.train_tasks)
    n_tasks = len(train_tasks)
    policy = SCRIPTED_POLICIES[task_name]()

    calib = calibrate_task(env_cls, train_tasks[0], policy)
    print(f"  Calibration: init_ipr={calib['init_ipr']:.3f}, "
          f"range={calib['ipr_range']:.3f}, init_o2t={calib['init_o2t']:.3f}, "
          f"expert={calib['expert_steps']} steps, train_tasks={n_tasks}")

    # --------------------------------------------------------------- #
    # PASS 1: SUCCESSES                                               #
    # --------------------------------------------------------------- #
    success_ids = []
    print(f"\n[success] generating {n_success} instances x {len(cameras)} cameras")
    for inst in range(n_success):
        task_obj = train_tasks[inst % n_tasks]
        env = MetaworldWrapper(env_cls, task_obj,
                                render_mode="rgb_array",
                                camera_name=cameras[0])
        try:
            result = _run_success_episode(env, policy, category, calib, cameras,
                                           task_name=task_name,
                                           n_keyframes=N_KEYFRAMES)
        finally:
            try: env.close()
            except Exception: pass

        if result is None:
            print(f"  inst={inst} FAILED (expert did not complete / too short)")
            continue
        keyframes, n_steps = result
        final_label = keyframes[-1]["frame_label"]

        for cam in cameras:
            frames = _frames_for_cam(keyframes, cam)
            if frames is None or len(frames) != N_KEYFRAMES:
                continue
            episode_id = f"{archive}_success_inst{inst:04d}_{cam}"
            meta = write_episode_dir(
                kf_succ_root, archive, episode_id, frames,
                n_source=n_steps, fps=FPS_NOMINAL, task_desc=task_desc,
                label_str="success", terminal_reward=4,
                paired_success_id=None)
            with open(manifest_succ, "a") as fh:
                fh.write(json.dumps(meta) + "\n")
            success_ids.append(episode_id)
        print(f"  inst={inst} OK final_label={final_label}")

    # Re-read from manifest if running failures-only in a later shard
    if not success_ids and manifest_succ.exists():
        with open(manifest_succ) as fh:
            success_ids = [json.loads(l)["episode_id"] for l in fh if l.strip()]
    print(f"  -> {len(success_ids)} success episodes available for pairing")

    # --------------------------------------------------------------- #
    # PASS 2: FAILURES                                                #
    # --------------------------------------------------------------- #
    rng = random.Random(base_seed + abs(hash(task_name)) % 9973)
    def pair_success():
        return rng.choice(success_ids) if success_ids else None

    for target_score in valid_scores:
        print(f"\n[score={target_score}] generating {n_failures_per_score} "
              f"instances x {len(cameras)} cameras")
        mode_sequence = get_mode_sequence(category, target_score,
                                           n_failures_per_score)

        for inst in range(n_failures_per_score):
            task_obj = train_tasks[inst % n_tasks]
            preferred_mode = mode_sequence[inst]

            env = MetaworldWrapper(env_cls, task_obj,
                                    render_mode="rgb_array",
                                    camera_name=cameras[0])
            best = None  # (abs_diff, (keyframes, n_steps), used_mode)
            try:
                for attempt in range(max_attempts):
                    # First half: preferred mode; second half: fall back to freeze
                    mode = preferred_mode if attempt < max_attempts // 2 else "freeze"
                    result = _run_one_episode(
                        env, policy, target_score, category, calib,
                        mode, max_steps=500, noise_scale=noise_scale,
                        noisy_steps=noisy_steps, cameras=cameras,
                        task_name=task_name, n_keyframes=N_KEYFRAMES)
                    if result is None:
                        continue
                    keyframes, n_steps = result
                    final_score = keyframes[-1]["score"]
                    diff = abs(final_score - target_score)
                    if diff == 0:
                        best = (0, (keyframes, n_steps), mode)
                        print(f"  inst={inst} ACCEPT attempt={attempt+1} mode={mode}")
                        break
                    if best is None or diff < best[0]:
                        best = (diff, (keyframes, n_steps), mode)
            finally:
                try: env.close()
                except Exception: pass

            if best is None:
                print(f"  inst={inst} FAILED (no valid episode in {max_attempts})")
                continue
            if best[0] != 0:
                print(f"  inst={inst} FALLBACK diff={best[0]} mode={best[2]}")

            keyframes, n_steps = best[1]
            used_mode = best[2]
            terminal = int(keyframes[-1]["frame_label"])

            for cam in cameras:
                frames = _frames_for_cam(keyframes, cam)
                if frames is None or len(frames) != N_KEYFRAMES:
                    continue
                episode_id = (f"{archive}_score{target_score}_inst{inst:04d}_"
                               f"{cam}_{used_mode}_score_{terminal}")
                psid = pair_success()
                meta = write_episode_dir(
                    kf_fail_root, archive, episode_id, frames,
                    n_source=n_steps, fps=FPS_NOMINAL, task_desc=task_desc,
                    label_str="failure", terminal_reward=terminal,
                    paired_success_id=psid)
                with open(manifest_fail, "a") as fh:
                    fh.write(json.dumps(meta) + "\n")
                with open(scored_fail, "a") as fh:
                    fh.write(json.dumps({
                        "episode_id": episode_id,
                        "archive": archive,
                        "task": task_desc,
                        "target_score": target_score,
                        "terminal_reward": terminal,
                        "frame_labels": [int(kf["frame_label"]) for kf in keyframes],
                        "noise_mode": used_mode,
                        "paired_success_id": psid,
                    }) + "\n")


# ---------------------------------------------------------------------------
# Shard packing
# ---------------------------------------------------------------------------

def pack_shards(archive_root, shard_size=512):
    for archive_dir in sorted(archive_root.iterdir()):
        if not archive_dir.is_dir():
            continue
        ep_dirs = sorted([p for p in archive_dir.iterdir() if p.is_dir()])
        if not ep_dirs:
            continue
        for i, start in enumerate(range(0, len(ep_dirs), shard_size)):
            batch = ep_dirs[start:start + shard_size]
            shard_path = archive_dir / f"shard-{i:05d}.tar"
            with tarfile.open(shard_path, "w") as tar:
                for ep in batch:
                    for f in sorted(ep.iterdir()):
                        tar.add(f, arcname=f"{ep.name}/{f.name}")
            for ep in batch:
                shutil.rmtree(ep, ignore_errors=True)
            print(f"  packed {shard_path.name} ({len(batch)} episodes)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--task", required=True,
                     help="MetaWorld task name, e.g. pick-place-v3")
    ap.add_argument("--n-success", type=int, default=25)
    ap.add_argument("--n-failures", type=int, default=100)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--cameras", nargs="+", default=CAMERAS)
    ap.add_argument("--no-pack", action="store_true")
    ap.add_argument("--pack-only", action="store_true")
    args = ap.parse_args()

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
              n_success=args.n_success,
              n_failures_per_score=args.n_failures,
              base_seed=args.base_seed,
              cameras=args.cameras)

    if not args.no_pack:
        print(f"\nPacking shards...")
        pack_shards(output_dir / "keyframes")
        pack_shards(output_dir / "keyframes_success")

    print("\nDone.")


if __name__ == "__main__":
    main()
