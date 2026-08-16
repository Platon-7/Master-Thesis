#!/usr/bin/env python3
"""Calibrate the success-detection threshold per (model, task).

Why this is not optional
------------------------
`success_detection_threshold` gates when the SUCCESS head ends an episode
(`use_success_detection=true`, the "vlmterm" regime). It is applied to a raw
probability, so its meaning depends on how each head is calibrated -- and our
fine-tuned models were trained with `bce_asymmetric`, whose entire purpose is to
move the operating point. A single shared 0.65 therefore sits at a *different*
false-positive rate for each model, and policy-success differences between models
would partly reflect threshold placement rather than reward quality. Calibration
also shifts per task, since the scenes differ.

Both failure modes are costly and asymmetric:
  * threshold too low  -> false positives end episodes that did not succeed, so the
    policy is rewarded for triggering the detector rather than solving the task;
  * threshold too high -> the head never fires and `TERMINATE=1` silently degrades
    into `TERMINATE=0`.

Method: score labelled ManiSkill trajectories with each model's success head using
the *same* code path the replay buffer uses, then pick each model's threshold at a
matched false-positive rate. Equal conservativeness across models makes the arms
comparable.

Trajectories come from ManiSkill's own trained PPO checkpoints (which succeed at a
known rate, giving positives and negatives with ground-truth labels) plus a random
policy (near-pure negatives).

    # once per task -- no VLM involved, cheap
    python scripts/calibrate_success_threshold.py collect --task PullCube-v1

    # once per (model, task)
    python scripts/calibrate_success_threshold.py score \
        --task PullCube-v1 --model /path/to/ckpt --tag run2

    # emits thresholds.json
    python scripts/calibrate_success_threshold.py fit --task PullCube-v1 --target-fpr 0.02
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

CAL_DIR = os.path.join(os.environ.get("MS_ASSET_DIR", os.path.expanduser("~/.maniskill")), "calibration")


# --------------------------------------------------------------------------
# phase 1: collect labelled trajectories (no reward model)
# --------------------------------------------------------------------------
def collect(args) -> int:
    import torch

    from robometer_policy_learning.envs.maniskill_utils import get_task_spec
    from robometer_policy_learning.utils.env_utils import make_env
    from robometer_policy_learning.rollouts.rollout_worker import extract_info_for_env
    from scripts.verify_with_maniskill_ppo import build_actor

    spec = get_task_spec(args.task)
    asset_dir = os.environ.get("MS_ASSET_DIR", os.path.expanduser("~/.maniskill"))
    ck = glob.glob(os.path.join(asset_dir, "demos", args.task, "**",
                                f"ppo_{spec.control_mode}_ckpt.pt"), recursive=True)
    actor = None
    if ck:
        sd = torch.load(ck[0], map_location="cpu", weights_only=False)
        actor = build_actor(sd, sd["actor_mean.0.weight"].shape[1], sd["actor_mean.6.weight"].shape[0])
        print(f"expert policy: {os.path.basename(ck[0])}")
    else:
        print("WARNING: no PPO checkpoint; positives will be rare (random policy only)")

    env, eval_env = make_env(
        env_name=f"maniskill/{args.task}", num_envs=1, chunk_size=None,
        max_episode_steps=spec.max_episode_steps, use_full_state=True,
        env_kwargs={"sim_backend": "physx_cpu", "image_size": 224,
                    "control_mode": spec.control_mode, "reward_mode": "normalized_dense"},
    )
    instruction = env.get_language_instruction()

    frames, succ_step, ep_success, source = [], [], [], []
    n_expert = args.episodes if actor is not None else 0
    total = n_expert + args.random_episodes
    for ep in range(total):
        use_expert = ep < n_expert
        obs, _ = env.reset(seed=5000 + ep)
        ep_frames, first_success = [], -1
        for t in range(spec.max_episode_steps):
            ep_frames.append(np.asarray(obs["image"]).reshape(224, 224, 3).astype(np.uint8))
            if use_expert:
                with torch.no_grad():
                    a = actor(torch.as_tensor(np.asarray(obs["state"]).reshape(1, -1),
                                              dtype=torch.float32)).numpy()
            else:
                a = np.stack([env.single_action_space.sample()])
            obs, _r, _term, _trunc, infos = env.step(a)
            if first_success < 0 and bool(extract_info_for_env(infos, 0, 1).get("success", False)):
                first_success = t  # GT success occurred at this step
        frames.append(np.stack(ep_frames))
        succ_step.append(first_success)
        ep_success.append(first_success >= 0)
        source.append("expert" if use_expert else "random")
        if (ep + 1) % 10 == 0:
            print(f"  {ep+1}/{total} episodes, successes so far: {sum(ep_success)}", flush=True)

    env.close()
    if eval_env is not None and eval_env is not env:
        eval_env.close()

    os.makedirs(CAL_DIR, exist_ok=True)
    out = os.path.join(CAL_DIR, f"{args.task}_trajs.npz")
    np.savez_compressed(out, frames=np.stack(frames), succ_step=np.array(succ_step),
                        ep_success=np.array(ep_success), source=np.array(source),
                        instruction=instruction)
    print(f"\nwrote {out}")
    print(f"  {len(frames)} episodes, {int(sum(ep_success))} with GT success "
          f"({100*np.mean(ep_success):.0f}%)")
    if sum(ep_success) < 5:
        print("WARNING: very few positives -- ROC will be unreliable.")
    return 0


# --------------------------------------------------------------------------
# phase 2: score with one model's success head (same path as the buffer)
# --------------------------------------------------------------------------
def score(args) -> int:
    import torch

    from robometer.evals.eval_utils import raw_dict_to_sample
    from robometer.evals.eval_server import process_batch_helper
    from robometer.utils.save import load_model_from_hf
    from robometer.utils.setup_utils import setup_batch_collator
    from robometer_policy_learning.utils.robometer_utils import extract_success_probs_from_output

    path = os.path.join(CAL_DIR, f"{args.task}_trajs.npz")
    z = np.load(path, allow_pickle=True)
    frames, succ_step = z["frames"], z["succ_step"]
    instruction = str(z["instruction"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, tokenizer, processor, model = load_model_from_hf(model_path=args.model, device=device)
    max_frames = getattr(cfg.data, "max_frames", 16)
    # Argument order matters and is easy to get wrong: (processor, tokenizer, cfg).
    # Passing cfg first raises "'Qwen3VLProcessor' object has no attribute 'data'".
    # is_eval=True and use_multi_image=True mirror RobometerReplayBuffer exactly, so
    # the calibrated score is the same quantity training will threshold on.
    if not getattr(cfg.data, "use_multi_image", False):
        cfg.data.use_multi_image = True
    collator = setup_batch_collator(processor, tokenizer, cfg, is_eval=True)
    plt = cfg.loss.progress_loss_type.lower()
    is_discrete = plt == "discrete" or "c51" in plt

    probs, labels = [], []
    n_ep, n_steps = frames.shape[0], frames.shape[1]
    for e in range(n_ep):
        for t in range(0, n_steps, args.step_stride):
            raw = dict(frames=frames[e][: t + 1], task=instruction, id=e,
                       metadata=dict(subsequence_length=t + 1),
                       video_embeddings=None, text_embedding=None)
            sample = raw_dict_to_sample(raw_data=raw, max_frames=max_frames, sample_type="progress")
            out = process_batch_helper(
                model_type=cfg.model.model_type, model=model, tokenizer=tokenizer,
                batch_collator=collator, device=model.device, batch_data=[sample.model_dump()],
                job_id=0, is_discrete_mode=is_discrete,
                num_bins=cfg.loss.progress_discrete_bins,
            )
            probs.append(float(extract_success_probs_from_output(out)[0]))
            # positive iff GT success has ALREADY happened by step t -- the moment
            # the detector is supposed to fire.
            labels.append(int(0 <= succ_step[e] <= t))
        if (e + 1) % 5 == 0:
            print(f"  scored {e+1}/{n_ep} episodes", flush=True)

    out_path = os.path.join(CAL_DIR, f"{args.task}__{args.tag}_probs.npz")
    np.savez_compressed(out_path, probs=np.array(probs), labels=np.array(labels),
                        step_stride=args.step_stride, model=args.model)
    pos, neg = np.array(labels) == 1, np.array(labels) == 0
    print(f"\nwrote {out_path}")
    print(f"  {pos.sum()} positive steps, {neg.sum()} negative steps")
    if pos.sum():
        print(f"  mean success_prob  positives={np.array(probs)[pos].mean():.3f}  "
              f"negatives={np.array(probs)[neg].mean():.3f}")
    return 0


# --------------------------------------------------------------------------
# phase 3: pick thresholds at a matched false-positive rate
# --------------------------------------------------------------------------
def fit(args) -> int:
    files = sorted(glob.glob(os.path.join(CAL_DIR, f"{args.task}__*_probs.npz")))
    if not files:
        print(f"no scored files for {args.task} in {CAL_DIR}")
        return 2

    out = {}
    print(f"{'model':16s} {'AUROC':>7} {'thresh':>8} {'TPR':>6} {'FPR':>6}")
    for f in files:
        tag = os.path.basename(f).split("__")[1].replace("_probs.npz", "")
        z = np.load(f, allow_pickle=True)
        p, y = z["probs"], z["labels"]
        if y.sum() == 0 or (1 - y).sum() == 0:
            print(f"{tag:16s}  degenerate labels -- skipped")
            continue
        order = np.argsort(-p)
        ys = y[order]
        tpr = np.cumsum(ys) / max(1, ys.sum())
        fpr = np.cumsum(1 - ys) / max(1, (1 - ys).sum())
        auroc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
        # threshold at the matched false-positive rate: equal conservativeness
        idx = int(np.searchsorted(fpr, args.target_fpr))
        idx = min(idx, len(p) - 1)
        thr = float(p[order][idx])
        out[tag] = dict(threshold=round(thr, 4), auroc=round(auroc, 4),
                        tpr=round(float(tpr[idx]), 4), fpr=round(float(fpr[idx]), 4))
        print(f"{tag:16s} {auroc:7.3f} {thr:8.3f} {tpr[idx]:6.3f} {fpr[idx]:6.3f}")

    dest = os.path.join(CAL_DIR, "thresholds.json")
    existing = {}
    if os.path.exists(dest):
        with open(dest) as fh:
            existing = json.load(fh)
    existing.setdefault(args.task, {}).update(out)
    with open(dest, "w") as fh:
        json.dump(existing, fh, indent=2)
    print(f"\nwrote {dest}")
    print("Use per arm:  reward_model.success_detection_threshold=<value for that model+task>")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="roll out labelled trajectories (no reward model)")
    c.add_argument("--task", required=True)
    c.add_argument("--episodes", type=int, default=40, help="expert (PPO) episodes")
    c.add_argument("--random-episodes", type=int, default=20)
    c.set_defaults(func=collect)

    s = sub.add_parser("score", help="score trajectories with one model's success head")
    s.add_argument("--task", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--tag", required=True, help="short model name, e.g. run2 / base")
    s.add_argument("--step-stride", type=int, default=2, help="score every Nth step")
    s.set_defaults(func=score)

    f = sub.add_parser("fit", help="pick thresholds at a matched false-positive rate")
    f.add_argument("--task", required=True)
    f.add_argument("--target-fpr", type=float, default=0.02)
    f.set_defaults(func=fit)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.exit(main())
