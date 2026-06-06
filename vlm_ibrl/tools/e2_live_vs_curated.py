"""Controlled A/B — does the FT reward model score LIVE successes correctly?

Targets the doubt that the curated->live AUC collapse is a "render overfit".
Holds ICL, harness, scorer-call, and model IDENTICAL between two query sources:
  (a) curated CoffeePush keyframe clips (success + failure)
  (b) live BC-rollout clips (success + failure), collected fresh here

Every clip — curated or live — is scored by the SAME scorer call with the SAME
curated-keyframe ICL demo. The only thing that varies is the query frame source.

Readout:
  - live-success sp HIGH + live-failure sp HIGH  => task difficulty (model reads
    live frames fine, just can't catch near-miss failures). NOT a render problem.
  - live-success sp LOW                          => model is blind to unambiguous
    live successes — the surprising result that needs explaining.

Also dumps the lowest-scoring live successes as an image for eyeballing.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pyrallis
import torch
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl")
sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl/mw_main")
sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer")

from common_utils import ibrl_utils as utils
from train_rl_vlm_mw import MainConfig, Workspace

DATA = Path("/projects/prjs1958/robometer_frame_dataset/metaworld")
ARCHIVE = "metaworld_coffee_push_v3"
TASK = "push a mug under a coffee machine"
N_FRAMES = 16


def load_curated(success: bool, max_n: int) -> List[List[np.ndarray]]:
    kf = DATA / ("keyframes_success" if success else "keyframes") / ARCHIVE
    idx = json.loads((kf / "shard_index.json").read_text())
    by_shard: Dict[str, List[str]] = {}
    for eid, shard in idx.items():
        by_shard.setdefault(shard, []).append(eid)
    trajs = []
    for shard, eids in by_shard.items():
        if len(trajs) >= max_n:
            break
        with tarfile.open(kf / shard, "r") as tf:
            byep: Dict[str, list] = {}
            for m in tf.getmembers():
                if m.isfile() and m.name.endswith(".jpg"):
                    byep.setdefault(m.name.split("/")[0], []).append(m)
            for eid in eids:
                if len(trajs) >= max_n:
                    break
                if eid not in byep:
                    continue
                ms = sorted(byep[eid], key=lambda m: m.name)
                fr = [np.asarray(Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB")) for m in ms]
                if fr:
                    trajs.append(fr)
    return trajs


def subsample(frames: List[np.ndarray], n: int) -> List[np.ndarray]:
    if len(frames) <= n:
        return frames
    idx = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in idx]


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    a = np.concatenate([np.asarray(pos), np.asarray(neg)])
    order = np.argsort(a, kind="mergesort"); sa = a[order]; ranks = np.empty(len(a)); i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        ranks[i:j + 1] = (i + j + 2) / 2; i = j + 1
    inv = np.empty_like(order, dtype=float); inv[order] = ranks
    return float((inv[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--n-episodes", type=int, default=60)
    ap.add_argument("--n-curated", type=int, default=30)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--icl-source", choices=["curated", "release"], default="curated",
                    help="curated = held-out curated keyframe success demo (E2 default); "
                         "release = the release 224 demonstration frames used in actual IBRL runs")
    ap.add_argument("--release-icl-path",
                    default="release/data/metaworld/CoffeePush_frame_stack_1_224x224_end_on_success/"
                            "demonstrations/mw-coffee-push/frames")
    ap.add_argument("--policy", choices=["bc", "rl"], default="bc",
                    help="bc = behavior-cloning policy (default); rl = the IBRL-trained policy from latest.pt")
    args = ap.parse_args()

    cfg = pyrallis.load(MainConfig, open(args.run_dir / "cfg.yaml"))
    cfg.use_wb = 0
    cfg.save_dir = str(args.run_dir.parent / f"e2_{args.run_dir.name}")
    ws = Workspace(cfg)
    scorer = ws.train_env.scorer
    env = ws.train_env
    if args.policy == "rl":
        pol_path = args.run_dir / "latest.pt"
        sd = torch.load(str(pol_path), map_location="cuda", weights_only=False)
        ws.agent.load_state_dict(sd)
        actor = ws.agent
        print(f"policy = RL-trained (loaded {pol_path})")
    else:
        actor = ws.bc_policy if ws.bc_policy is not None else ws.agent
        print("policy = BC")

    # ---- fixed ICL (one of two sources; everything else identical) ----
    cur_succ_all = load_curated(success=True, max_n=args.n_curated + 1)
    if args.icl_source == "curated":
        icl = subsample(cur_succ_all[0], N_FRAMES)
        print("ICL source = curated keyframe success demo")
    else:
        # the release 224 demonstration frames used in the actual IBRL runs
        from pathlib import Path as _P
        fdir = _P(args.release_icl_path)
        avail = sorted(p for p in fdir.iterdir() if p.name.startswith("0_") and p.suffix == ".png")
        picks = np.linspace(0, len(avail) - 1, N_FRAMES).round().astype(int)
        icl = [np.asarray(Image.open(avail[i]).convert("RGB"), dtype=np.uint8) for i in picks]
        print(f"ICL source = release demo ({args.release_icl_path}), {len(avail)} frames -> {N_FRAMES}")
    cur_succ = [subsample(t, N_FRAMES) for t in cur_succ_all[1:]]
    cur_fail = [subsample(t, N_FRAMES) for t in load_curated(success=False, max_n=args.n_curated)]
    print(f"curated: {len(cur_succ)} succ + {len(cur_fail)} fail")

    # ---- collect live BC rollouts ----
    live = []  # (frames16, env_success)
    with torch.no_grad(), utils.eval_mode(actor):
        for ep in range(args.n_episodes):
            np.random.seed(1000 + ep)
            obs, image_obs = env.reset()
            terminal = False; succ = 0
            while not terminal:
                action = actor.act(obs, eval_mode=True).numpy()
                obs, reward, terminal, success, image_obs = env.step(action)
                if success:
                    succ = 1
            clip = subsample([np.asarray(f) for f in env.current_video], N_FRAMES)
            live.append((clip, succ))
            if (ep + 1) % 15 == 0:
                ns = sum(s for _, s in live)
                print(f"  rolled {ep+1}/{args.n_episodes}  live_succ={ns}")
    live_succ = [c for c, s in live if s == 1]
    live_fail = [c for c, s in live if s == 0]
    print(f"live: {len(live_succ)} succ + {len(live_fail)} fail")

    # ---- score everything with the SAME scorer + SAME ICL ----
    def score_all(clips):
        out = []
        for c in clips:
            r = scorer(c, task=TASK, icl_frames=icl)
            out.append(float(r["success_prob"]))
        return out

    sp_cs = score_all(cur_succ)
    sp_cf = score_all(cur_fail)
    sp_ls = score_all(live_succ)
    sp_lf = score_all(live_fail)

    def stat(x):
        x = [v for v in x if not np.isnan(v)]
        return f"n={len(x):>3} mean={np.mean(x):.4f} med={np.median(x):.4f}" if x else "n=0"

    print("\n==================== RESULT ====================")
    print(f"curated success : {stat(sp_cs)}")
    print(f"curated failure : {stat(sp_cf)}")
    print(f"live    success : {stat(sp_ls)}")
    print(f"live    failure : {stat(sp_lf)}")
    print(f"\nAUC curated (succ vs fail): {auc(sp_cs, sp_cf):.3f}")
    print(f"AUC live    (succ vs fail): {auc(sp_ls, sp_lf):.3f}")
    print(f"AUC cross: curated-succ vs live-fail: {auc(sp_cs, sp_lf):.3f}")
    print(f"AUC cross: live-succ    vs curated-fail: {auc(sp_ls, sp_cf):.3f}")

    # ---- save CSV + low-scoring live successes ----
    import csv as _csv
    with open(f"{args.out_prefix}.csv", "w", newline="") as f:
        w = _csv.writer(f); w.writerow(["group", "sp"])
        for v in sp_cs: w.writerow(["curated_success", v])
        for v in sp_cf: w.writerow(["curated_failure", v])
        for v in sp_ls: w.writerow(["live_success", v])
        for v in sp_lf: w.writerow(["live_failure", v])
    print(f"\nwrote {args.out_prefix}.csv")

    # dump the 4 lowest-scoring live successes
    if live_succ:
        order = np.argsort(sp_ls)
        worst = order[:min(4, len(order))]
        fig, axes = plt.subplots(len(worst), 8, figsize=(16, 2.2 * len(worst)))
        if len(worst) == 1:
            axes = axes[None, :]
        for r, wi in enumerate(worst):
            clip = live_succ[wi]; fidx = np.linspace(0, len(clip) - 1, 8, dtype=int)
            for cc, fi in enumerate(fidx):
                axes[r, cc].imshow(clip[fi]); axes[r, cc].axis("off")
            axes[r, 0].set_title(f"live SUCCESS scored sp={sp_ls[wi]:.3f}", loc="left", fontsize=10)
        fig.suptitle("Lowest-scoring LIVE SUCCESSES (env reward=1) — do they look like real successes?", fontsize=12)
        fig.tight_layout()
        fig.savefig(f"{args.out_prefix}_low_live_success.png", dpi=130, bbox_inches="tight")
        print(f"wrote {args.out_prefix}_low_live_success.png")


if __name__ == "__main__":
    main()
