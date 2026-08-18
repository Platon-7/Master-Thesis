#!/usr/bin/env python3
"""Causal success-threshold + gate calibration for ManiSkill.

Port of vlm_ibrl/jobs/diag_causal_calib.py (the protocol actually used for
MetaWorld/Robomimic, "same protocol that gave Can run2=0.80") to ManiSkill, with
three deliberate differences noted below.

Protocol
--------
Roll out episodes with the POLICY DISTRIBUTION we will actually deploy, score the
GROWING video at every step, and record per episode: the full success_prob series,
the GT outcome, and the GT-success step. Then

  (1) causal THRESHOLD sweep on per-episode MAX success_prob (TPR/FPR, Youden J);
  (2) GATE guidance: at the chosen threshold, when does success_prob FIRST cross it
      for GT-fail episodes ("fake fires") vs GT-success ones ("real fires"), and when
      do real successes actually occur -- so min_ep_steps can sit between them.

Why per-episode max rather than a pointwise ROC: the detector fires ONCE and ends the
episode. A pointwise ROC can look excellent while the detector still fires early in
most episodes, because it never asks "did it cross before the task was done".

Differences from the MetaWorld version
--------------------------------------
* No BC policy exists here (SAC trains from scratch), so trajectories come from a
  LADDER of saved SAC actor checkpoints spanning the training arc -- early ones give
  the near-misses and failures RL actually produces, late ones give clean successes.
  Calibrating on the whole arc, not just competent behaviour.
* Short clips are FRONT-PADDED to max_frames. That -- not resolution -- was the
  cause of the flat outputs: run2 scored 0.053 on true successes with 7-11 frames and
  0.369 once padded to 16, with failure scores unchanged. Once padded, 480 and 224
  give byte-identical results, so we keep 224 (what robosuite_wrapper.py uses).
* Episodes stop at the env's own boundary; the vector env auto-resets, so continuing
  would mix a fresh episode's frames into the current one.

    python scripts/causal_calib_maniskill.py \
        --task PullCube-v1 --model /path/to/ckpt --tag run2 \
        --actor-dir /scratch-shared/$USER/roboref_runs/ms_PullCube-v1_gtshift_gt_s0_*/checkpoints
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def pick_actor_checkpoints(actor_dir: str, n: int) -> list:
    """A spread across the training arc: earliest -> latest, evenly sampled."""
    steps = sorted(
        (int(os.path.basename(d)), d)
        for d in glob.glob(os.path.join(actor_dir, "*"))
        if os.path.basename(d).isdigit() and os.path.isfile(os.path.join(d, "actor.pt"))
    )
    if not steps:
        raise FileNotFoundError(f"no actor.pt under {actor_dir}")
    idx = np.unique(np.linspace(0, len(steps) - 1, min(n, len(steps))).astype(int))
    return [steps[i] for i in idx]


def rollout_and_score(args) -> int:
    from robometer.evals.eval_utils import raw_dict_to_sample
    from robometer.evals.eval_server import process_batch_helper
    from robometer.utils.save import load_model_from_hf
    from robometer.utils.setup_utils import setup_batch_collator
    from robometer_policy_learning.envs.maniskill_utils import get_task_spec
    from robometer_policy_learning.utils.env_utils import make_env
    from robometer_policy_learning.rollouts.rollout_worker import extract_info_for_env
    from robometer_policy_learning.utils.robometer_utils import (
        extract_success_probs_from_output, extract_rewards_from_output)

    spec = get_task_spec(args.task)

    # ---- reward model -------------------------------------------------------
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, tok, proc, model = load_model_from_hf(model_path=args.model, device=dev)
    if not getattr(cfg.data, "use_multi_image", False):
        cfg.data.use_multi_image = True
    coll = setup_batch_collator(proc, tok, cfg, is_eval=True)
    max_frames = int(getattr(cfg.data, "max_frames", 16))
    _plt = str(cfg.loss.progress_loss_type).lower()
    is_disc = _plt == "discrete" or "c51" in _plt
    n_bins = cfg.loss.progress_discrete_bins

    def score(frames_list):
        fr = np.stack(frames_list)
        # Front-pad to max_frames. ManiSkill episodes are 50 steps and succeed near
        # step 7, so a growing prefix never reaches max_frames=16 and the fine-tuned
        # heads collapse: run2 measured 0.053 at 7-11 frames vs 0.369 padded, with
        # failure scores unchanged. LIBERO/Robomimic never hit this (200-400 step
        # episodes), which is why calibrate_threshold.py can simply start at t=16.
        if len(fr) < max_frames:
            # sub16 semantics from reward-model-study/scripts/calibrate_threshold.py:
            # always hand the collator exactly max_frames, repeating via linspace.
            # The collator itself only reduces, so short clips would otherwise stay short.
            fr = fr[np.linspace(0, len(fr) - 1, max_frames).round().astype(int)]
        raw = dict(frames=fr, task=INSTR, id=0,
                   metadata=dict(subsequence_length=len(fr)),
                   video_embeddings=None, text_embedding=None)
        s = raw_dict_to_sample(raw_data=raw, max_frames=max_frames, sample_type="progress")
        out = process_batch_helper(model_type=cfg.model.model_type, model=model, tokenizer=tok,
                                   batch_collator=coll, device=model.device,
                                   batch_data=[s.model_dump()], job_id=0,
                                   is_discrete_mode=is_disc, num_bins=n_bins)
        return (float(extract_success_probs_from_output(out)[0]),
                float(extract_rewards_from_output(out)[0]))

    # ---- env, built the SAME way training builds it -------------------------
    # The saved actors were trained on proprio + DINOv2-base features, which the
    # VectorDinoEmbeddingWrapper adds -- and make_env only attaches that wrapper when
    # a dinov2_model is passed. Omitting it hands the actor raw 384-d proprio and it
    # dies with "mat1 and mat2 shapes cannot be multiplied (1x384 and 768x512)".
    from transformers import AutoImageProcessor, AutoModel
    _dino_name = "facebook/dinov2-base"   # configs/maniskill_online_rl.yaml
    dino_model = AutoModel.from_pretrained(_dino_name).to(dev).eval()
    dino_processor = AutoImageProcessor.from_pretrained(_dino_name)

    env, eval_env = make_env(
        env_name=f"maniskill/{args.task}", num_envs=1, chunk_size=None,
        max_episode_steps=spec.max_episode_steps, use_full_state=False,
        dinov2_model=dino_model, dinov2_processor=dino_processor, device=str(dev),
        terminate_on_success=False,
        env_kwargs={"sim_backend": "physx_cpu", "image_size": args.image_size,
                    "control_mode": spec.control_mode, "reward_mode": "normalized_dense"},
    )
    INSTR = env.get_language_instruction()

    ckpts = [(-1, None)] if args.random_policy else pick_actor_checkpoints(
        args.actor_dir, args.n_checkpoints)
    print(f"[calib] task={args.task} model={args.tag} res={args.image_size} "
          f"max_frames={max_frames}", flush=True)
    print(f"[calib] policy ladder: {[s for s, _ in ckpts]}", flush=True)

    from robometer_policy_learning.utils.training_utils import build_actor_critic_models
    from omegaconf import OmegaConf

    episodes = []
    eps_per_ckpt = max(1, args.episodes // len(ckpts))
    for step, cdir in ckpts:
        actor = None
        if cdir is not None:
            actor = torch.load(os.path.join(cdir, "actor.pt"), map_location=dev, weights_only=False)
            actor.eval()
        for e in range(eps_per_ckpt):
            obs, _ = env.reset(seed=7000 + 100 * len(episodes) + e)
            video, sps, prs, gt, gt_step = [], [], [], 0, -1
            for t in range(1, spec.max_episode_steps + 1):
                video.append(np.asarray(obs["image"]).reshape(args.image_size, args.image_size, 3)
                             .astype(np.uint8))
                if actor is None:
                    a = np.stack([env.single_action_space.sample()])
                else:
                    with torch.no_grad():
                        from robometer_policy_learning.utils.gpu_utils import convert_to_tensor, move_to_device
                        ot = move_to_device(convert_to_tensor(obs), dev)
                        a, _ = actor.act(ot, actor_state=None, deterministic=True)
                    a = a.detach().cpu().numpy()
                obs, _r, term, trunc, infos = env.step(a)
                sp, pr = score(video)
                sps.append(sp); prs.append(pr)
                info_i = extract_info_for_env(infos, 0, 1)
                if bool(info_i.get("success", False)) and gt_step < 0:
                    gt_step = t
                gt = max(gt, int(bool(info_i.get("success", False))))
                if bool(term[0]) or bool(trunc[0]):
                    break
            sps = np.array(sps)
            episodes.append(dict(gt=gt, gt_step=gt_step, sps=sps.tolist(),
                                 progress=prs, ckpt=step))
            print(f"  ep{len(episodes)-1:03d} ckpt={step:<7} GT={'S' if gt else 'F'} "
                  f"gt_step={gt_step} len={len(sps)} max_sp={sps.max():.3f}@{int(sps.argmax())+1}",
                  flush=True)

    env.close()
    if eval_env is not None and eval_env is not env:
        eval_env.close()

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.task}__{args.tag}_causal.json")
    json.dump(dict(task=args.task, tag=args.tag, image_size=args.image_size,
                   model=args.model, episodes=episodes), open(out, "w"))
    print(f"[calib] wrote {out}", flush=True)
    analyse(episodes, args.tag, args.task)
    return 0


def analyse(episodes, tag, task):
    labels = np.array([d["gt"] for d in episodes])
    maxsp = np.array([max(d["sps"]) for d in episodes])
    nS, nF = int((labels == 1).sum()), int((labels == 0).sum())
    print(f"\n=== (1) causal THRESHOLD sweep  {task} / {tag}  (n_succ={nS} n_fail={nF}) ===")
    if nS == 0 or nF == 0:
        print("  degenerate: need both successes and failures"); return

    # Sweep over observed values, not a fixed grid -- these heads operate at very
    # different scales (base ~0.9, fine-tuned ~0.15), so a 0.05..0.95 grid would
    # miss the fine-tuned models' entire operating range.
    cand = np.unique(np.round(np.concatenate([maxsp, np.linspace(0.001, 0.99, 60)]), 5))
    best = None
    for thr in cand:
        fire = maxsp > thr
        ctpr = float(fire[labels == 1].mean())
        cfpr = float(fire[labels == 0].mean())
        j = ctpr - cfpr
        if best is None or j > best[0]:
            best = (j, float(thr), ctpr, cfpr)
    j, bthr, btpr, bfpr = best
    print(f"  --> best causal threshold = {bthr:.4f}  (cTPR={btpr:.2f} cFPR={bfpr:.2f} J={j:.2f})")

    def first_cross(sps, thr):
        idx = np.where(np.array(sps) > thr)[0]
        return int(idx[0]) + 1 if len(idx) else -1

    print(f"\n=== (2) GATE guidance at thr={bthr:.4f} ===")
    fcS = [x for x in (first_cross(d["sps"], bthr) for d in episodes if d["gt"] == 1) if x > 0]
    fcF = [x for x in (first_cross(d["sps"], bthr) for d in episodes if d["gt"] == 0) if x > 0]
    gtS = [d["gt_step"] for d in episodes if d["gt"] == 1 and d["gt_step"] > 0]
    fmt = lambda v: f"min={min(v)} med={int(np.median(v))} max={max(v)}" if v else "none"
    print(f"  REAL fires  (GT-success): [{fmt(fcS)}] n={len(fcS)}/{nS}")
    print(f"  FAKE fires  (GT-fail)   : [{fmt(fcF)}] n={len(fcF)}/{nF}")
    print(f"  GT success actually at  : [{fmt(gtS)}]")
    if fcF and fcS:
        if max(fcF) < min(fcS):
            print(f"  -> gate (min_ep_steps) between {max(fcF)} (latest fake) and "
                  f"{min(fcS)} (earliest real) cleanly separates")
        else:
            print(f"  -> fake fires overlap real ones ({max(fcF)} >= {min(fcS)}); "
                  f"no gate fully separates -- raise the threshold or accept false fires")
    return dict(threshold=bthr, tpr=btpr, fpr=bfpr, youden_j=j,
                latest_fake_fire=max(fcF) if fcF else None,
                earliest_real_fire=min(fcS) if fcS else None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--actor-dir", required=True, help="a run's checkpoints/ dir")
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--n-checkpoints", type=int, default=8, help="policies spanning the arc")
    ap.add_argument("--random-policy", action="store_true",
                    help="ignore --actor-dir and act uniformly at random. The checkpoint "
                         "ladder starts at 25k, already past the random phase, so it does "
                         "NOT cover what a from-scratch RL run looks like in its first "
                         "thousands of steps. This does.")
    ap.add_argument("--image-size", type=int, default=224)  # robosuite_wrapper.py uses 224
    ap.add_argument("--out-dir", default=os.path.join(
        os.environ.get("MS_ASSET_DIR", "/tmp"), "causal_calib"))
    sys.exit(rollout_and_score(ap.parse_args()))
