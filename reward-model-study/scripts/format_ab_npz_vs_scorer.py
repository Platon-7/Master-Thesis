"""Decisive format test: does the demo2reward scorer, fed the EXACT npz frames
the harness uses, reproduce the harness number? Isolates frame-SOURCE/format
from aggregation. Loads npz frames from the step2 cache, scores via the same
get_robometer_4b scorer the IBRL pipeline uses, computes within-task ranking
on BOTH heads (single-clip, no frame-steps)."""
import json, os, sys
import numpy as np
sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl")
sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer")
from env.robometer_utils import get_robometer_4b
from datasets import load_from_disk

CACHE = "/scratch-shared/pkarageorgis1/robometer_frames_hf_full_step2"

def within_task(score, lab, tasks):
    accs = []
    for t in np.unique(tasks):
        s1 = score[(tasks == t) & (lab == 1)]; s0 = score[(tasks == t) & (lab == 0)]
        if len(s1) and len(s0):
            accs.append(((s1[:, None] > s0[None, :]).sum() + 0.5*(s1[:, None] == s0[None, :]).sum())/(len(s1)*len(s0)))
    return float(np.mean(accs)) if accs else float("nan"), len(accs)

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "metaworld"
    model = sys.argv[2]
    out = sys.argv[3]
    cdir = os.path.join(CACHE, f"_projects_prjs1958_robometer_frames_hf_full_eval_{src}_raw_robometer_frames_eval_{src}")
    ds = load_from_disk(os.path.join(cdir, "processed_dataset"))
    framedir = os.path.join(cdir, "frames")
    print(f"{src}: {len(ds)} rows; frames {framedir}", flush=True)
    scorer = get_robometer_4b(model_path=model)
    print(f"max_frames={getattr(scorer,'max_frames','?')}", flush=True)
    sp, pg, lab, tasks = [], [], [], []
    miss = 0
    with open(out, "w") as f:
        for i, r in enumerate(ds):
            npzp = os.path.join(framedir, f"trajectory_{r['id']}.npz")
            if not os.path.isfile(npzp):
                miss += 1; continue
            with np.load(npzp) as d:
                frames = [np.asarray(x) for x in d["frames"]]
            o = scorer(frames, task=r["task"], icl_frames=None, detailed=True)
            s = float(o["success_prob"]); p = float(o["progress_reward"])
            y = 1 if r["quality_label"] == "successful" else 0
            sp.append(s); pg.append(p); lab.append(y); tasks.append(r["task"])
            f.write(json.dumps({"id": r["id"], "task": r["task"], "label": y, "success_prob": s, "progress_reward": p}) + "\n")
            if (i+1) % 100 == 0:
                wt, nt = within_task(np.array(pg), np.array(lab), np.array(tasks))
                print(f"  {i+1}/{len(ds)} prog within-task={wt:.3f} ({nt} tasks) miss={miss}", flush=True)
    sp, pg, lab, tasks = map(np.array, (sp, pg, lab, tasks))
    wts, _ = within_task(sp, lab, tasks); wtp, nt = within_task(pg, lab, tasks)
    print(f"\nDONE {src} n={len(sp)} miss={miss} tasks={nt}")
    print(f"  [NPZ frames -> demo2reward scorer, single-clip]")
    print(f"    SUCCESS within-task = {wts:.4f}")
    print(f"    PROGRESS within-task = {wtp:.4f}   (cf imageio 0.58, harness sum ~0.78)")

if __name__ == "__main__":
    main()
