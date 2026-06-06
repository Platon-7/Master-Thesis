"""Reproduce the canonical per-source success-head eval, on the CANONICAL data
(the HF arrow dataset at /projects/prjs1958/...eval_<src>_raw, frames = mp4),
NOT the keyframe tars eval_dump wrongly used. Scores the success head with the
same RobometerScorer, computes AUC + separation vs quality_label.

  python repro_canonical_eval.py --ds <hf_dataset_dir> --model <ckpt> --out <jsonl> [--limit N]
"""
import argparse, io, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl")
sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer")
from env.robometer_utils import get_robometer_4b
from datasets import load_from_disk
import imageio.v3 as iio


def load_frames(mp4_rel, ds_root, frames_dir=None):
    if frames_dir:
        # mp4_rel = "<inner_ds_name>/batch_xxxx/traj.mp4"; strip the inner name,
        # join the remainder onto frames_dir (used when the mp4 live in a sibling
        # dir, e.g. metaworld's .bak_pre_drop_metaworld_success_labels).
        parts = mp4_rel.split("/", 1)
        rel = parts[1] if len(parts) > 1 else parts[0]
        p = os.path.join(frames_dir, rel)
    else:
        p = os.path.join(ds_root, mp4_rel)
    if not os.path.isfile(p):
        return None
    try:
        return [np.asarray(f) for f in iio.imread(p, plugin="pyav")]
    except Exception:
        try:
            return [np.asarray(f) for f in iio.imread(p)]
        except Exception:
            return None


def auc(p, n):
    if not len(p) or not len(n):
        return float("nan")
    a = np.concatenate([p, n]); o = np.argsort(a, kind="mergesort"); r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_frames", type=int, default=None, help="override; default = model's training config")
    ap.add_argument("--frames_dir", default=None, help="sibling dir holding batch_*/ mp4 (for metaworld .bak)")
    args = ap.parse_args()

    ds = load_from_disk(args.ds)
    # frames paths are relative to the parent of the dataset dir
    ds_root = str(Path(args.ds).parent)
    print(f"dataset {args.ds}: {len(ds)} rows; frames root {ds_root}", flush=True)
    scorer = get_robometer_4b(model_path=args.model, max_frames=args.max_frames)
    print(f"scorer resolved max_frames = {getattr(scorer, 'max_frames', '?')}", flush=True)

    rows = ds if args.limit is None else ds.select(range(min(args.limit, len(ds))))
    sp, lab, tasks, pgs = [], [], [], []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_miss = 0
    with open(args.out, "w") as f:
        for i, r in enumerate(rows):
            frames = load_frames(r["frames"], ds_root, args.frames_dir)
            if not frames:
                n_miss += 1; continue
            out = scorer(frames, task=r.get("task", "complete the task"), icl_frames=None, detailed=True)
            s = float(out["success_prob"]); pg = float(out["progress_reward"]); y = 1 if r["quality_label"] == "successful" else 0
            t = r.get("task", "?")
            sp.append(s); lab.append(y); tasks.append(t); pgs.append(pg)
            f.write(json.dumps({"id": r["id"], "task": t, "label": y, "success_prob": s, "progress_reward": pg}) + "\n")
            if (i + 1) % 200 == 0:
                a = auc(np.array(sp)[np.array(lab) == 1], np.array(sp)[np.array(lab) == 0])
                print(f"  {i+1}/{len(rows)}  running AUC={a:.3f}  (miss {n_miss})", flush=True)
    sp = np.array(sp); lab = np.array(lab); tasks = np.array(tasks); pgs = np.array(pgs)

    def within_task(score):
        accs = []
        for t in np.unique(tasks):
            s1 = score[(tasks == t) & (lab == 1)]; s0 = score[(tasks == t) & (lab == 0)]
            if not len(s1) or not len(s0):
                continue
            gt = (s1[:, None] > s0[None, :]).sum(); ties = (s1[:, None] == s0[None, :]).sum()
            accs.append((gt + 0.5 * ties) / (len(s1) * len(s0)))
        return (float(np.mean(accs)) if accs else float("nan")), len(accs)

    a = auc(sp[lab == 1], sp[lab == 0]); sep = sp[lab == 1].mean() - sp[lab == 0].mean()
    wt_s, nt = within_task(sp)
    # PROGRESS head: this is what wandb's eval_p_rank/ranking_acc actually ranks by.
    ag = auc(pgs[lab == 1], pgs[lab == 0]); sepg = pgs[lab == 1].mean() - pgs[lab == 0].mean()
    wt_p, _ = within_task(pgs)
    print(f"\nDONE  n={len(sp)} (miss {n_miss})  successful={int(lab.sum())} failure={int((lab==0).sum())}  tasks={nt}")
    print(f"  [SUCCESS head]  global AUC={a:.4f}  within-task={wt_s:.4f}  sep={sep:+.4f}")
    print(f"  [PROGRESS head] global AUC={ag:.4f}  within-task={wt_p:.4f}  sep={sepg:+.4f}   <- compare to wandb eval_p_rank/ranking_acc")


if __name__ == "__main__":
    main()
