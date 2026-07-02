"""Multi-task success-detection threshold calibration for the FT Robometer model.

Generalizes calibrate_threshold.py (CoffeePush-only) to all four MetaWorld
demo2reward tasks. Same protocol, faithful to the fair-autonomous-RL sweep:
  * reward view = un-zoomed `corner2` == the reward_camera (`corner2_default`)
    the model sees in RL (V3_CORNER2_ZOOM left unset => corner2 IS the default).
  * task string = METAWORLD_TASK_DESCRIPTIONS[env] (the simple string RL passes).
  * icl_frames = None (the sweep ran with ROBOMETER_ICL_DEMO_PATH unset).
  * oracle -> successes, noisy oracle -> failures.

For each task x FT model we report:
  EPISODE-LEVEL : linspace-16 over the whole trajectory -> one success_prob.
                  succ_mean / fail_mean / AUC + ep threshold sweep.
  STREAMING     : score growing buffer frames[0:t] every STRIDE steps (what the
                  RL detector sees). Per threshold: causal-FPR (frac failure eps
                  the detector EVER fires on), causal-TPR (frac success eps that
                  fire at/after true success), premFire (frac that fire BEFORE).

Output: stdout + a machine-readable summary at CALIB_OUT (default below).
"""
import os
import sys
import json
import numpy as np

_REPO = os.environ.get(
    "MT_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
sys.path.insert(0, os.path.join(_REPO, "MetaWorld", "metaworld_repo"))
sys.path.insert(0, os.path.join(_REPO, "MetaWorld"))
sys.path.insert(0, os.path.join(_REPO, "vlm_ibrl_v3"))
sys.path.insert(0, os.path.join(_REPO, "Robometer"))

from env.metaworld_wrapper import MetaWorldEnv
from env.robometer_utils import get_robometer_4b
from env.vlm_prompts import METAWORLD_TASK_DESCRIPTIONS as TASK_STR
from PIL import Image
import scipy.stats as st

# FT model only (baseline-4b is dead everywhere in the 288 sweep; FT is the
# recipe we're porting). Add baseline back via CALIB_MODELS=both if wanted.
FT_PATH = os.environ.get(
    "ROBOMETER_FT_PATH",
    "/shared/home/PKA4388/checkpoints/Robometer_FT_consolidated/run1_icl_ours_step4000",
)
B4_PATH = os.environ.get("ROBOMETER_4B_PATH", "/shared/home/PKA4388/checkpoints/Robometer-4B")
MODELS = {"FT_s4000": FT_PATH}
if os.environ.get("CALIB_MODELS", "ft") == "both":
    MODELS["baseline_4b"] = B4_PATH

# CoffeePush first = sanity anchor (must reproduce the known ~good numbers),
# then the three harder tasks.
TASKS = os.environ.get("CALIB_TASKS", "CoffeePush,Assembly,BoxClose,StickPull").split(",")

CAM = "corner2"   # un-zoomed corner2 == reward model's in-domain view (corner2_default)
RES = 224
N_SUCC = int(os.environ.get("CALIB_N_SUCC", "15"))
N_FAIL = int(os.environ.get("CALIB_N_FAIL", "15"))
MAXT = 240
SUB = 16
STRIDE = int(os.environ.get("CALIB_STRIDE", "10"))
THRS = np.round(np.arange(0.05, 0.96, 0.05), 2)
OUT = os.environ.get("CALIB_OUT", "/shared/home/PKA4388/vlm_ibrl_runs/calib_multitask.json")


def auc(l, s):
    l = np.asarray(l); s = np.asarray(s); P = (l == 1).sum(); N = (l == 0).sum()
    if P == 0 or N == 0:
        return float("nan")
    r = st.rankdata(s)
    return float((r[l == 1].sum() - P * (P + 1) / 2) / (P * N))


def sub16(frames):
    idx = np.linspace(0, len(frames) - 1, SUB).round().astype(int)
    return [frames[i] for i in idx]


def rollout(env, noisy):
    env.reset()
    frames, succ, succ_step = [], 0, -1
    for t in range(MAXT):
        a = env.get_heuristic_action()
        if noisy:
            a = (a + np.random.uniform(-1, 1, size=a.shape) * 0.8).clip(-1, 1)
        obs, r, done, info = env.step(a)
        frames.append(Image.fromarray(env.render(camera_name=CAM, width=RES, height=RES)))
        if int(info.get("success", 0)) == 1 and succ_step < 0:
            succ_step = t
        if int(info.get("success", 0)) == 1:
            succ = 1
        if done:
            break
    return frames, succ, succ_step


def collect(env):
    episodes, ns, nf, tries = [], 0, 0, 0
    while (ns < N_SUCC or nf < N_FAIL) and tries < 800:
        tries += 1
        noisy = nf < N_FAIL and (ns >= N_SUCC or np.random.rand() < 0.55)
        frames, succ, sstep = rollout(env, noisy)
        if succ and ns >= N_SUCC:
            continue
        if (not succ) and nf >= N_FAIL:
            continue
        episodes.append((frames, succ, sstep))
        ns += succ; nf += (1 - succ)
        if len(episodes) % 10 == 0:
            print(f"    collected {len(episodes)} ({ns}s/{nf}f) tries={tries}", flush=True)
    return episodes, ns, nf, tries


def score_task(env_name, scorers, summary):
    TASK = TASK_STR[env_name]
    print(f"\n##################### {env_name} #####################", flush=True)
    print(f"  task string: {TASK!r}", flush=True)
    env = MetaWorldEnv(env_name, camera_name=CAM, width=RES, height=RES)
    episodes, ns, nf, tries = collect(env)
    print(f"  [rollouts] {len(episodes)} eps ({ns}s/{nf}f) tries={tries}", flush=True)
    if ns == 0 or nf == 0:
        print(f"  !! degenerate ({ns}s/{nf}f) — skipping scoring for {env_name}", flush=True)
        summary[env_name] = {"error": f"degenerate {ns}s/{nf}f"}
        return

    summary[env_name] = {"n_succ": ns, "n_fail": nf, "models": {}}
    for name, sc in scorers.items():
        ep_sp, ep_lab = [], []
        fail_max, presucc_max, atsucc_max = [], [], []
        for frames, succ, sstep in episodes:
            ep_sp.append(float(sc(sub16(frames), task=TASK, icl_frames=None)["success_prob"]))
            ep_lab.append(succ)
            pres, post, allm = 0.0, 0.0, 0.0
            for t in range(SUB, len(frames) + 1, STRIDE):
                sp = float(sc(sub16(frames[:t]), task=TASK, icl_frames=None)["success_prob"])
                allm = max(allm, sp)
                if succ and sstep >= 0:
                    if t - 1 < sstep:
                        pres = max(pres, sp)
                    else:
                        post = max(post, sp)
            if succ:
                presucc_max.append(pres); atsucc_max.append(post)
            else:
                fail_max.append(allm)
        sp, lab = np.array(ep_sp), np.array(ep_lab)
        fail_max = np.array(fail_max); presucc_max = np.array(presucc_max); atsucc_max = np.array(atsucc_max)
        sm = float(sp[lab == 1].mean()); fm = float(sp[lab == 0].mean()); a = auc(lab, sp)
        print(f"\n  ===== {env_name} / {name} =====", flush=True)
        print(f"  episode-level: succ_mean={sm:.3f} fail_mean={fm:.3f} AUC={a:.3f} "
              f"(n={len(sp)}: {int(lab.sum())}s/{int((lab==0).sum())}f)", flush=True)
        print(f"  {'thr':>5} | {'epTPR':>6} {'epFPR':>6} | {'cFPR':>6} {'cTPR':>6} {'prem':>6}", flush=True)
        sweep = []
        for thr in THRS:
            ep_tpr = float((sp[lab == 1] > thr).mean())
            ep_fpr = float((sp[lab == 0] > thr).mean())
            c_fpr = float((fail_max > thr).mean()) if len(fail_max) else float("nan")
            c_tpr = float((atsucc_max > thr).mean()) if len(atsucc_max) else float("nan")
            prem = float((presucc_max > thr).mean()) if len(presucc_max) else float("nan")
            sweep.append(dict(thr=float(thr), ep_tpr=ep_tpr, ep_fpr=ep_fpr,
                              causal_fpr=c_fpr, causal_tpr=c_tpr, prem_fire=prem))
            print(f"  {thr:5.2f} | {ep_tpr:6.2f} {ep_fpr:6.2f} | {c_fpr:6.2f} {c_tpr:6.2f} {prem:6.2f}", flush=True)
        rec = {}
        for tgt in (0.05, 0.02):
            ok = [thr for thr in THRS if len(fail_max) and (fail_max > thr).mean() <= tgt]
            if ok:
                thr = float(min(ok)); c_tpr = float((atsucc_max > thr).mean())
                rec[f"fpr<={tgt}"] = dict(thr=thr, causal_tpr=c_tpr)
                print(f"  -> causalFPR<= {tgt:.0%}: thr={thr:.2f} (causalTPR={c_tpr:.2f})", flush=True)
            else:
                rec[f"fpr<={tgt}"] = None
                print(f"  -> causalFPR<= {tgt:.0%}: NONE in grid (fires on failures even at 0.95)", flush=True)
        summary[env_name]["models"][name] = dict(
            succ_mean=sm, fail_mean=fm, auc=a, sweep=sweep, recommended=rec)
        # persist after each model so a crash still leaves partial results
        with open(OUT, "w") as f:
            json.dump(summary, f, indent=2)


def main():
    np.random.seed(0)
    print(f"[calib] tasks={TASKS} models={list(MODELS)} N={N_SUCC}s/{N_FAIL}f stride={STRIDE}", flush=True)
    scorers = {}
    for name, path in MODELS.items():
        print(f"[load] {name} <- {path}", flush=True)
        scorers[name] = get_robometer_4b(model_path=path)
    summary = {}
    for env_name in TASKS:
        try:
            score_task(env_name.strip(), scorers, summary)
        except Exception as e:
            import traceback
            print(f"  !! {env_name} FAILED: {e}", flush=True)
            traceback.print_exc()
            summary[env_name] = {"error": str(e)}
            with open(OUT, "w") as f:
                json.dump(summary, f, indent=2)
    print(f"\n[calib] done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
