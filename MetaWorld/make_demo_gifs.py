#!/usr/bin/env python3
"""
Generate 4 GIFs for pick-place-v3, one per score level (1-4).

Uses the same category-specific scoring and noise as generate_failures.py.
"""

import numpy as np
from pathlib import Path
from PIL import Image

import metaworld
from metaworld.policies import SawyerPickPlaceV3Policy

import sys
sys.path.insert(0, str(Path(__file__).parent))
from generate_failures import (
    detect_score, should_inject, make_noise_bias,
    get_mode_sequence, calibrate_task,
)


def run_episode(env_cls, task, policy, target_score, seed, calib,
                max_steps=500, noise_scale=0.5, noisy_steps=40):
    env = env_cls(render_mode="rgb_array", camera_name="corner2")
    env.set_task(task)
    obs, _ = env.reset()
    np.random.seed(seed)

    category = "grasp_move"
    modes = get_mode_sequence(category, target_score, 3)
    noise_mode = modes[seed % len(modes)]
    bias = make_noise_bias(env.action_space.shape[0], noise_scale, category, target_score, noise_mode)

    init_o2t = calib["init_o2t"]
    ep_init_o2t = None

    frames = []
    step_scores = []
    noise_active = (target_score == 1)
    noise_start_step = 0 if target_score == 1 else None

    for step in range(max_steps):
        if ep_init_o2t is None and step > 0:
            ep_init_o2t = info_prev.get("obj_to_target", init_o2t)

        if not noise_active and step > 0:
            if should_inject(info_prev, target_score, category, calib,
                             ep_init_o2t or init_o2t, noise_mode):
                noise_active = True
                noise_start_step = step

        if noise_active:
            if target_score == 1:
                action = env.action_space.sample()
            elif noise_mode == "freeze":
                action = np.random.randn(env.action_space.shape[0]) * 0.02
            else:
                expert_action = policy.get_action(obs)
                jitter = np.random.randn(*expert_action.shape) * (noise_scale * 0.1)
                action = np.clip(expert_action + bias + jitter, -1.0, 1.0)
        else:
            action = policy.get_action(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        info_prev = info

        if ep_init_o2t is None:
            ep_init_o2t = info.get("obj_to_target", init_o2t)

        img = env.render()[::-1]
        frames.append(img)

        score = detect_score(info, category, calib, ep_init_o2t)
        step_scores.append(score)

        if info.get("success", 0) > 0:
            env.close()
            return None

        if noise_start_step is not None and step - noise_start_step >= noisy_steps:
            break

    env.close()

    n = len(frames)
    if n < 8:
        return None

    kf_idx = np.linspace(0, n - 1, 8, dtype=int)
    kf_scores = [step_scores[i] for i in kf_idx]

    for i in range(1, len(kf_scores)):
        if kf_scores[i] < kf_scores[i - 1]:
            kf_scores[i] = kf_scores[i - 1]

    return frames, kf_scores, kf_scores[-1]


def frames_to_gif(frames, path, fps=15, subsample=4):
    selected = frames[::subsample]
    imgs = [Image.fromarray(f) for f in selected]
    duration = 1000 // fps
    imgs[0].save(
        str(path), save_all=True, append_images=imgs[1:],
        duration=duration, loop=0, optimize=True,
    )
    print(f"  GIF: {path.name} ({len(imgs)} frames, {len(imgs)*duration/1000:.1f}s)")


def make_strip(frames, kf_scores, path, n_kf=8):
    n = len(frames)
    kf_idx = np.linspace(0, n - 1, n_kf, dtype=int)
    kf_size = 200
    strip = Image.new("RGB", (kf_size * n_kf, kf_size), (255, 255, 255))
    for i, idx in enumerate(kf_idx):
        img = Image.fromarray(frames[idx]).resize((kf_size, kf_size), Image.LANCZOS)
        strip.paste(img, (i * kf_size, 0))
    strip.save(str(path), quality=95)
    print(f"  Strip: {path.name} (scores: {kf_scores})")


def main():
    out_dir = Path("/home/pkarageorgis/DAS-5/Master-Thesis/MetaWorld/demo_gifs")
    out_dir.mkdir(parents=True, exist_ok=True)

    mt1 = metaworld.MT1("pick-place-v3", seed=42)
    env_cls = mt1.train_classes["pick-place-v3"]
    task = mt1.train_tasks[0]
    policy = SawyerPickPlaceV3Policy()

    calib = calibrate_task(env_cls, task, policy)
    print(f"Calibration: {calib}")

    collected = {}

    for target_score in [1, 2, 3, 4]:
        print(f"\nSearching for score {target_score}...")
        found = False

        noise_scales = [0.3, 0.4, 0.5, 0.6, 0.7] if target_score > 1 else [0]
        for noise_scale in noise_scales:
            if found:
                break
            for seed in range(300):
                result = run_episode(
                    env_cls, task, policy,
                    target_score=target_score, seed=seed, calib=calib,
                    noise_scale=noise_scale,
                )
                if result is None:
                    continue
                frames, kf_scores, final_score = result
                if final_score == target_score:
                    print(f"  Found! noise_scale={noise_scale}, seed={seed}, "
                          f"kf_scores={kf_scores}")
                    collected[target_score] = (frames, kf_scores)
                    found = True
                    break

        if target_score not in collected:
            print(f"  WARNING: Could not find score {target_score}")

    print(f"\n{'='*60}")
    print("Generating GIFs and strips...")
    for score in sorted(collected.keys()):
        frames, kf_scores = collected[score]
        frames_to_gif(frames, out_dir / f"pick_place_score{score}.gif")
        make_strip(frames, kf_scores, out_dir / f"pick_place_score{score}_strip.jpg")

    print(f"\nDone! Files in {out_dir}")
    for score in [1, 2, 3, 4]:
        status = "OK" if score in collected else "MISSING"
        print(f"  Score {score}: {status}")


if __name__ == "__main__":
    main()
