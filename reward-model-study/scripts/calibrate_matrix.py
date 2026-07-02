"""Calibration MATRIX: {success, progress} head x {no-ICL, ICL} for one model,
across the 4 MetaWorld demo2reward tasks. Faithful to the fair-autonomous-RL sweep.

Per scorer call we get BOTH heads (progress_reward + success_prob), and we score
each frame window twice: once with icl_frames=None, once with the real demo-0 as
in-context context (attached as context_trajectory with <|demo_end|>, exactly the
training-time ICL format). The scorer subsamples to ITS OWN max_frames (FT=16,
baseline=8 -- read from each checkpoint's config), so frame count is fair to each
model's training; we do NOT pre-subsample.

Run one model per job (CALIB_MODEL=ft|4b). Fills tracker Table 3 rows.
Output: stdout + CALIB_OUT json.
"""
import os, sys, json
import numpy as np

_REPO = os.environ.get("MT_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for sub in ("MetaWorld/metaworld_repo", "MetaWorld", "vlm_ibrl_v3", "Robometer"):
    sys.path.insert(0, os.path.join(_REPO, sub))

import h5py
from env.metaworld_wrapper import MetaWorldEnv
from env.robometer_utils import get_robometer_4b
from env.vlm_prompts import METAWORLD_TASK_DESCRIPTIONS as TASK_STR
from PIL import Image
import scipy.stats as st

MODEL = os.environ.get("CALIB_MODEL", "ft")
PATHS = {
    "ft": os.environ.get("ROBOMETER_FT_PATH",
        "/shared/home/PKA4388/checkpoints/Robometer_FT_consolidated/run1_icl_ours_step4000"),
    "4b": os.environ.get("ROBOMETER_4B_PATH", "/shared/home/PKA4388/checkpoints/Robometer-4B"),
}
MODEL_PATH = PATHS.get(MODEL) or os.environ.get("ROBOMETER_FT_PATH")  # unknown model -> use FT_PATH
TASKS = os.environ.get("CALIB_TASKS", "CoffeePush,Assembly,BoxClose,StickPull").split(",")
CAM = "corner2"; RES = 224
N_SUCC = int(os.environ.get("CALIB_N_SUCC", "15"))
N_FAIL = int(os.environ.get("CALIB_N_FAIL", "15"))
MAXT = int(os.environ.get("CALIB_MAXT", "240"))
STRIDE = int(os.environ.get("CALIB_STRIDE", "10"))
MINWIN = 6                       # smallest streaming window to score
ICL_DEMO_IDX = int(os.environ.get("CALIB_ICL_DEMO_IDX", "0"))
DATA_DIR = os.path.join(_REPO, "vlm_ibrl_v3", "release", "data", "metaworld")
THRS = np.round(np.arange(0.05, 0.96, 0.05), 2)
OUT = os.environ.get("CALIB_OUT", f"/shared/home/PKA4388/vlm_ibrl_runs/calib_matrix_{MODEL}.json")

# Canonical ICL demo frames (the {demo_idx}_NNN.png dir the ROBOMETER ICL
# mechanism uses) — pre-rendered 224x224 RGB, same pipeline as the query render.
ICL_BASE = "/shared/home/PKA4388/Master-Thesis/vlm_ibrl/release/data/metaworld"
ICL_SUBDIR = {"Assembly":"mw-assembly","BoxClose":"mw-box-close",
              "CoffeePush":"mw-coffee-push","StickPull":"mw-stick-pull"}


def auc(l, s):
    l=np.asarray(l); s=np.asarray(s); P=(l==1).sum(); N=(l==0).sum()
    if P==0 or N==0: return float("nan")
    r=st.rankdata(s)
    return float((r[l==1].sum()-P*(P+1)/2)/(P*N))


def load_icl_demo(env_name):
    """Real demo-{idx} frames from the canonical {idx}_NNN.png dir (224x224 RGB).
    Pass all frames; the scorer subsamples to its own max_frames (fair per model).
    If CALIB_ICL_CORRECT_VIEW is set, use the un-zoomed correct-view frames
    (icl_correct_view/<task>) instead of the zoomed BC-demo frames."""
    import glob as _glob
    cv = os.environ.get("CALIB_ICL_CORRECT_VIEW", "")
    if cv:
        task_dir = {"CoffeePush":"coffeepush","BoxClose":"boxclose",
                    "StickPull":"stickpull","Assembly":"assembly"}[env_name]
        d = os.path.join(cv, task_dir)
    else:
        d = os.path.join(ICL_BASE, f"{env_name}_frame_stack_1_224x224_end_on_success",
                         "demonstrations", ICL_SUBDIR[env_name], "frames")
    files = sorted(_glob.glob(os.path.join(d, f"{ICL_DEMO_IDX}_*.png")))
    if not files:
        raise FileNotFoundError(f"no ICL frames {ICL_DEMO_IDX}_*.png in {d}")
    return [Image.open(f).convert("RGB") for f in files]


def rollout(env, noisy):
    env.reset(); frames=[]; succ=0; sstep=-1
    for t in range(MAXT):
        a = env.get_heuristic_action()
        if noisy: a=(a+np.random.uniform(-1,1,size=a.shape)*0.8).clip(-1,1)
        obs,r,done,info = env.step(a)
        frames.append(Image.fromarray(env.render(camera_name=CAM, width=RES, height=RES)))
        if int(info.get("success",0))==1 and sstep<0: sstep=t
        if int(info.get("success",0))==1: succ=1
        if done: break
    return frames, succ, sstep


def collect(env):
    eps=[]; ns=nf=tries=0
    while (ns<N_SUCC or nf<N_FAIL) and tries<800:
        tries+=1
        noisy = nf<N_FAIL and (ns>=N_SUCC or np.random.rand()<0.55)
        fr,sc,ss = rollout(env, noisy)
        if sc and ns>=N_SUCC: continue
        if (not sc) and nf>=N_FAIL: continue
        eps.append((fr,sc,ss)); ns+=sc; nf+=(1-sc)
    return eps, ns, nf


def sweep_block(ep_sp, ep_lab, fail_max, atsucc_max, presucc_max):
    sp=np.array(ep_sp); lab=np.array(ep_lab)
    fail_max=np.array(fail_max); atsucc_max=np.array(atsucc_max); presucc_max=np.array(presucc_max)
    a=auc(lab,sp)
    rows=[]
    for thr in THRS:
        rows.append(dict(thr=float(thr),
            ep_tpr=float((sp[lab==1]>thr).mean()) if (lab==1).any() else float("nan"),
            ep_fpr=float((sp[lab==0]>thr).mean()) if (lab==0).any() else float("nan"),
            causal_fpr=float((fail_max>thr).mean()) if len(fail_max) else float("nan"),
            causal_tpr=float((atsucc_max>thr).mean()) if len(atsucc_max) else float("nan"),
            prem_fire=float((presucc_max>thr).mean()) if len(presucc_max) else float("nan")))
    return dict(auc=a,
        succ_mean=float(sp[lab==1].mean()) if (lab==1).any() else float("nan"),
        fail_mean=float(sp[lab==0].mean()) if (lab==0).any() else float("nan"),
        sweep=rows)


def main():
    np.random.seed(0)
    print(f"[calib-matrix] model={MODEL} path={MODEL_PATH}", flush=True)
    sc = get_robometer_4b(model_path=MODEL_PATH)
    print(f"[calib-matrix] scorer.max_frames = {sc.max_frames}  (FT expect 16, baseline expect 8)", flush=True)
    # RESUMABLE: pool progress into one JSON per model so short (spot) windows
    # accumulate across requeues/resubmits instead of restarting from scratch.
    summary={"model":MODEL, "max_frames":int(sc.max_frames), "tasks":{}}
    if os.path.exists(OUT):
        try:
            prev=json.load(open(OUT))
            if prev.get("model")==MODEL: summary=prev; print(f"[resume] loaded {len(summary.get('tasks',{}))} prior tasks from {OUT}", flush=True)
        except Exception as e:
            print(f"[resume] could not read {OUT}: {e}", flush=True)

    def save_atomic(obj):
        tmp=OUT+".tmp"
        with open(tmp,"w") as f: json.dump(obj,f,indent=2)
        os.replace(tmp, OUT)

    for env_name in [t.strip() for t in TASKS]:
        TASK=TASK_STR[env_name]
        prev_t=summary["tasks"].get(env_name)
        if prev_t and len(prev_t.get("cells",{}))>=4:
            print(f"  [skip] {env_name} already calibrated ({len(prev_t['cells'])} cells)", flush=True); continue
        print(f"\n##### {env_name}  task={TASK!r} #####", flush=True)
        try:
            icl = load_icl_demo(env_name)
            print(f"  loaded ICL demo-{ICL_DEMO_IDX}: {len(icl)} frames", flush=True)
        except Exception as e:
            print(f"  !! ICL load failed: {e}", flush=True); icl=None
        env = MetaWorldEnv(env_name, camera_name=CAM, width=RES, height=RES)
        # FORMAT SANITY: confirm the ICL demo frame matches the query render
        # (channel order + vertical orientation). The static scene layout
        # dominates the vertical brightness profile, so same-orientation frames
        # correlate positively; a flip shows up as negative correlation. RGB-vs-BGR
        # shows up as swapped R/B channel means.
        if icl is not None:
            env.reset()
            qf = np.asarray(env.render(camera_name=CAM, width=RES, height=RES)).astype(float)
            df = np.asarray(icl[len(icl)//2]).astype(float)
            qrow = qf.mean(axis=(1,2)); drow = df.mean(axis=(1,2))
            corr = float(np.corrcoef(qrow, drow)[0,1])
            print(f"  [fmt] query mean RGB={qf.mean(axis=(0,1)).round(1)} demo mean RGB={df.mean(axis=(0,1)).round(1)}", flush=True)
            print(f"  [fmt] vertical-profile corr(query,demo)={corr:+.2f}  "
                  f"{'OK (same orientation)' if corr>0 else '!!! NEGATIVE — possible vertical flip mismatch'}", flush=True)
        eps, ns, nf = collect(env)
        print(f"  rollouts: {len(eps)} ({ns}s/{nf}f)", flush=True)
        if ns==0 or nf==0:
            summary["tasks"][env_name]={"error":f"degenerate {ns}s/{nf}f"}; continue

        # accumulators: [icl_setting][head] -> lists
        acc = {ic:{h:dict(ep_sp=[],ep_lab=[],fail_max=[],atsucc_max=[],presucc_max=[])
                   for h in ("success","progress")} for ic in ("noicl","icl")}
        for fr, succ, sstep in eps:
            for ic, iclf in (("noicl", None), ("icl", icl)):
                if ic=="icl" and iclf is None: continue
                # episode-level: full trajectory (scorer subsamples to max_frames)
                o = sc(fr, task=TASK, icl_frames=iclf)
                acc[ic]["success"]["ep_sp"].append(float(o["success_prob"]))
                acc[ic]["progress"]["ep_sp"].append(float(o["progress_reward"]))
                acc[ic]["success"]["ep_lab"].append(succ)
                acc[ic]["progress"]["ep_lab"].append(succ)
                # streaming
                ps={"success":0.0,"progress":0.0}; po={"success":0.0,"progress":0.0}; am={"success":0.0,"progress":0.0}
                for t in range(MINWIN, len(fr)+1, STRIDE):
                    o2 = sc(fr[:t], task=TASK, icl_frames=iclf)
                    for h,key in (("success","success_prob"),("progress","progress_reward")):
                        v=float(o2[key]); am[h]=max(am[h],v)
                        if succ and sstep>=0:
                            if t-1<sstep: ps[h]=max(ps[h],v)
                            else: po[h]=max(po[h],v)
                for h in ("success","progress"):
                    if succ: acc[ic][h]["presucc_max"].append(ps[h]); acc[ic][h]["atsucc_max"].append(po[h])
                    else: acc[ic][h]["fail_max"].append(am[h])

        summary["tasks"][env_name]={"n_succ":ns,"n_fail":nf,"cells":{}}
        for ic in ("noicl","icl"):
            for h in ("success","progress"):
                d=acc[ic][h]
                if not d["ep_sp"]: continue
                blk=sweep_block(d["ep_sp"],d["ep_lab"],d["fail_max"],d["atsucc_max"],d["presucc_max"])
                summary["tasks"][env_name]["cells"][f"{h}_{ic}"]=blk
                # print a compact op-point: best causal_tpr at causal_fpr<=0.15
                best=None
                for row in blk["sweep"]:
                    if row["causal_fpr"]<=0.15 and (best is None or row["causal_tpr"]>best["causal_tpr"]): best=row
                op = (f"thr={best['thr']:.2f} cFPR={best['causal_fpr']:.2f} cTPR={best['causal_tpr']:.2f} prem={best['prem_fire']:.2f}"
                      if best else "no point cFPR<=0.15")
                print(f"  [{env_name}/{h}/{ic}] AUC={blk['auc']:.3f} succ={blk['succ_mean']:.3f} fail={blk['fail_mean']:.3f} | op: {op}", flush=True)
        save_atomic(summary)
    print(f"\n[calib-matrix] done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
