#!/usr/bin/env python3
"""Does the reward model actually READ the task instruction?

Motivation: LiftPegUpright-v1 separates far worse than PullCube-v1 under the same
head (AUROC 0.798 vs 1.000; median success_prob on GT-failed episodes 0.380 vs
0.011). One hypothesis is lexical -- the model may not ground "upright".

Rollouts are instruction-independent, so this roll outs ONCE per task, caches the
frames, and re-scores the IDENTICAL frames under several phrasings. Any difference
is therefore attributable to the text alone, with sampling noise held fixed.

The decisive condition is the WRONG-TASK control, not the paraphrases:

  * paraphrases >> default        -> the specific wording is the problem
  * every phrasing scores alike,
    wrong-task control included   -> the head ignores language here; the failure is
                                     visual/geometric, and rewording cannot fix it
  * wrong-task collapses, all
    correct phrasings ~equal      -> language IS used; "upright" is not the culprit

Prefixes are subsampled (the growing-video score is monotone-ish and we only need
the per-episode max that the detector would actually see), which keeps a 7-variant
sweep to ~1.7k VLM calls instead of ~17k.

    python scripts/instruction_ablation_maniskill.py \
        --task LiftPegUpright-v1 --model /path/to/run2 --actor-dir .../checkpoints
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import glob as _glob


def pick_actor_checkpoints(actor_dir: str, n: int) -> list:
    """A spread across the training arc: earliest -> latest, evenly sampled.

    Inlined rather than imported from causal_calib_maniskill: the job scripts set
    PYTHONSAFEPATH=1 (to stop the stale in-repo `robometer/` submodule shadowing the
    real fork), which also stops scripts/ being importable as a sibling.
    """
    steps = sorted(
        (int(os.path.basename(d)), d)
        for d in _glob.glob(os.path.join(actor_dir, "*"))
        if os.path.basename(d).isdigit() and os.path.isfile(os.path.join(d, "actor.pt"))
    )
    if not steps:
        raise FileNotFoundError(f"no actor.pt under {actor_dir}")
    idx = np.unique(np.linspace(0, len(steps) - 1, min(n, len(steps))).astype(int))
    return [steps[i] for i in idx]

# Variants per task. "default" is whatever the env reports, inserted at runtime so
# this never drifts from maniskill_utils.TASK_SPECS.
VARIANTS = {
    "LiftPegUpright-v1": [
        ("drop_upright",  "Lift the peg"),
        ("vertical",      "Stand the peg up vertically"),
        ("on_its_end",    "Rotate the peg so it stands on its end"),
        ("make_vertical", "Make the peg vertical"),
        ("upright_only",  "Make the peg upright"),
        ("WRONG_TASK",    "Pull the cube to the goal region"),
        ("EMPTY",         ""),
    ],
    "PullCube-v1": [   # positive control: separation here is already perfect
        ("paraphrase",   "Drag the cube onto the target area"),
        ("WRONG_TASK",   "Lift the peg and stand it upright"),
        ("EMPTY",        ""),
    ],
    "PokeCube-v1": [
        ("paraphrase",    "Push the cube to the target using the peg"),
        ("push_onto",     "Push the cube onto the target area with the stick"),
        ("nudge",         "Nudge the cube into the goal circle using the peg"),
        ("cube_to_goal",  "Move the cube to the goal region"),
        ("WRONG_TASK",    "Lift the peg and stand it upright"),
        ("EMPTY",         ""),
    ],
}


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def dprime(pos, neg):
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    sd = math.sqrt((st.pvariance(pos) + st.pvariance(neg)) / 2) or 1e-9
    return (st.mean(pos) - st.mean(neg)) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--actor-dir", required=True)
    ap.add_argument("--tag", default="run2")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--n-checkpoints", type=int, default=6)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--stride", type=int, default=5,
                    help="score every Nth prefix (plus the final frame)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from robometer.evals.eval_utils import raw_dict_to_sample
    from robometer.evals.eval_server import process_batch_helper
    from robometer.utils.save import load_model_from_hf
    from robometer.utils.setup_utils import setup_batch_collator
    from robometer_policy_learning.envs.maniskill_utils import get_task_spec
    from robometer_policy_learning.utils.env_utils import make_env
    from robometer_policy_learning.rollouts.rollout_worker import extract_info_for_env
    from robometer_policy_learning.utils.robometer_utils import (
        extract_success_probs_from_output, extract_rewards_from_output)
    from robometer_policy_learning.utils.gpu_utils import convert_to_tensor, move_to_device

    spec = get_task_spec(args.task)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, tok, proc, model = load_model_from_hf(model_path=args.model, device=dev)
    if not getattr(cfg.data, "use_multi_image", False):
        cfg.data.use_multi_image = True
    coll = setup_batch_collator(proc, tok, cfg, is_eval=True)
    max_frames = int(getattr(cfg.data, "max_frames", 16))
    _plt = str(cfg.loss.progress_loss_type).lower()
    is_disc = _plt == "discrete" or "c51" in _plt
    n_bins = cfg.loss.progress_discrete_bins

    def score(frames_list, instr):
        """Identical padding/collation to causal_calib_maniskill.score()."""
        fr = np.stack(frames_list)
        if len(fr) < max_frames:
            fr = fr[np.linspace(0, len(fr) - 1, max_frames).round().astype(int)]
        raw = dict(frames=fr, task=instr, id=0,
                   metadata=dict(subsequence_length=len(fr)),
                   video_embeddings=None, text_embedding=None)
        s = raw_dict_to_sample(raw_data=raw, max_frames=max_frames, sample_type="progress")
        out = process_batch_helper(model_type=cfg.model.model_type, model=model, tokenizer=tok,
                                   batch_collator=coll, device=model.device,
                                   batch_data=[s.model_dump()], job_id=0,
                                   is_discrete_mode=is_disc, num_bins=n_bins)
        return (float(extract_success_probs_from_output(out)[0]),
                float(extract_rewards_from_output(out)[0]))

    from transformers import AutoImageProcessor, AutoModel
    dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
    dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    env, _ = make_env(
        env_name=f"maniskill/{args.task}", num_envs=1, chunk_size=None,
        max_episode_steps=spec.max_episode_steps, use_full_state=False,
        dinov2_model=dino_model, dinov2_processor=dino_processor, device=str(dev),
        terminate_on_success=False,
        env_kwargs={"sim_backend": "physx_cpu", "image_size": args.image_size,
                    "control_mode": spec.control_mode, "reward_mode": "normalized_dense"},
    )
    default_instr = env.get_language_instruction()
    variants = [("default", default_instr)] + VARIANTS.get(args.task, [])

    # ---- phase 1: roll out once, cache frames (no VLM in the loop) ----------
    ckpts = pick_actor_checkpoints(args.actor_dir, args.n_checkpoints)
    print(f"[abl] task={args.task} ladder={[s for s, _ in ckpts]} "
          f"default_instr={default_instr!r}", flush=True)
    cache = []
    eps_per = max(1, args.episodes // len(ckpts))
    for step, cdir in ckpts:
        actor = torch.load(os.path.join(cdir, "actor.pt"), map_location=dev, weights_only=False)
        actor.eval()
        for e in range(eps_per):
            obs, _ = env.reset(seed=7000 + 100 * len(cache) + e)
            video, gt = [], 0
            for t in range(1, spec.max_episode_steps + 1):
                video.append(np.asarray(obs["image"])
                             .reshape(args.image_size, args.image_size, 3).astype(np.uint8))
                with torch.no_grad():
                    ot = move_to_device(convert_to_tensor(obs), dev)
                    a, _ = actor.act(ot, actor_state=None, deterministic=True)
                obs, _r, term, trunc, infos = env.step(a.detach().cpu().numpy())
                info_i = extract_info_for_env(infos, 0, 1)
                gt = max(gt, int(bool(info_i.get("success", False))))
                if bool(term[0]) or bool(trunc[0]):
                    break
            cache.append(dict(video=video, gt=gt))
        print(f"[abl] ckpt {step}: cached {len(cache)} eps "
              f"(gt so far {sum(c['gt'] for c in cache)})", flush=True)

    n_sol = sum(c["gt"] for c in cache)
    print(f"[abl] cached {len(cache)} episodes, {n_sol} solved / {len(cache)-n_sol} unsolved",
          flush=True)
    if n_sol < 2 or len(cache) - n_sol < 2:
        print("[abl] ABORT: need >=2 solved and >=2 unsolved episodes to separate", flush=True)
        return 2

    # ---- phase 2: re-score the SAME frames under each phrasing --------------
    results = {}
    for name, instr in variants:
        maxes, gts = [], []
        for c in cache:
            L = len(c["video"])
            idxs = sorted(set(list(range(args.stride, L + 1, args.stride)) + [L]))
            best = max(score(c["video"][:i], instr)[0] for i in idxs)
            maxes.append(best); gts.append(c["gt"])
        sol = [m for m, g in zip(maxes, gts) if g]
        uns = [m for m, g in zip(maxes, gts) if not g]
        # Per-episode maxima are stored, not just the aggregates: every variant scores
        # the SAME cached episodes, so the comparison is PAIRED and a paired test has
        # far more power than comparing two AUROCs at this n (9 unsolved episodes gives
        # an unpaired AUROC CI of +/-0.25, which resolves almost nothing).
        results[name] = dict(instruction=instr, auroc=auroc(sol, uns), d_prime=dprime(sol, uns),
                             med_solved=st.median(sol), med_unsolved=st.median(uns),
                             n_solved=len(sol), n_unsolved=len(uns),
                             per_episode_max=maxes, per_episode_gt=gts)
        r = results[name]
        print(f"[abl] {name:14s} AUROC={r['auroc']:.3f}  d'={r['d_prime']:5.2f}  "
              f"med_sol={r['med_solved']:.3f}  med_uns={r['med_unsolved']:.3f}   {instr!r}",
              flush=True)

    print("\n================ SUMMARY: does the text matter? ================")
    base = results["default"]["auroc"]
    for name, r in results.items():
        print(f"  {name:14s} AUROC={r['auroc']:.3f}  (delta vs default {r['auroc']-base:+.3f})")
    wrong = results.get("WRONG_TASK")
    if wrong is not None:
        d = base - wrong["auroc"]
        print(f"\n  wrong-task sensitivity = {d:+.3f}")
        print("  ~0 => the head is NOT reading the instruction on this task;")
        print("        rewording cannot help and the gap is visual, not lexical.")
    out = args.out or f"/scratch/{os.environ.get('USER','')}/maniskill_assets/causal_calib/{args.task}__{args.tag}_instr_ablation.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(dict(task=args.task, tag=args.tag, model=args.model,
                       default_instruction=default_instr, results=results), f, indent=1)
    print(f"\n[abl] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
