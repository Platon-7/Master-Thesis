"""Per-model success-detection threshold calibration for the FAIR autonomous-RL
batch (Robometer App. E-2 protocol: threshold the streaming success_prob, with a
timeout if not detected).

The paper uses success_prob > 0.6 for THEIR model; FT (~0.77/0.13) and the
off-the-shelf baseline live on different scales, so we calibrate each model HERE.

Method
------
Roll out CoffeePush in the in-domain reward view (corner2 default render, the
exact view ROBOMETER_REWARD_CAMERA=corner2_default feeds the reward model in RL),
oracle -> successes, noisy oracle -> failures. For every episode we record the
full corner2 frame list + GT success + the first GT-success step. Then we score
each model two ways:

  * EPISODE-LEVEL: linspace-16 over the whole trajectory -> one success_prob.
    Gives clean separability (succ_mean / fail_mean / AUC) and an episode-level
    threshold sweep (TPR/FPR).
  * STREAMING (causal): score the GROWING buffer frames[0:t] every STRIDE steps
    -> success_prob_t. This is what the RL detector actually sees. We measure,
    per threshold: the CAUSAL FPR (fraction of FAILURE episodes where the
    detector EVER fires) and, on successes, whether it fires at/after the true
    success (good) vs PREMATURELY (a false-positive the policy could exploit).

Output: per model, the threshold at ~5% and ~2% causal-FPR, the TPR there, and
where the paper's 0.6 sits. These thresholds feed ROBOMETER_THRESHOLD in the RL
batch.
"""
import os
import sys
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
from PIL import Image
import scipy.stats as st

TASK = "Push a mug under a coffee machine."
MODELS = {
    "FT_s4000": os.environ.get(
        "ROBOMETER_FT_PATH",
        "/shared/home/PKA4388/checkpoints/Robometer_FT_consolidated/run1_icl_ours_step4000",
    ),
    "baseline_4b": os.environ.get(
        "ROBOMETER_4B_PATH", "/shared/home/PKA4388/checkpoints/Robometer-4B"
    ),
}
CAM = "corner2"          # default (un-zoomed) corner2 == the reward's in-domain view
RES = 224                # match what vlm_envs.fetch_img feeds the reward model in RL
N_SUCC = 30
N_FAIL = 30
MAXT = 240               # paper's episode timeout; "catch the entire trajectory"
SUB = 16                 # frames per scorer call (model max_frames)
STRIDE = 8               # streaming cadence (score growing buffer every STRIDE steps)
THRS = np.round(np.arange(0.05, 0.96, 0.05), 2)


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
    """Full corner2 frame list, GT success bool, first success step (or -1)."""
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


def main():
    np.random.seed(0)
    env = MetaWorldEnv("CoffeePush", camera_name=CAM, width=RES, height=RES)
    scorers = {}
    for name, path in MODELS.items():
        print(f"[load] {name} <- {path}", flush=True)
        scorers[name] = get_robometer_4b(model_path=path)

    # ---- collect rollouts (render once) ----
    episodes = []  # (frames, succ, succ_step)
    ns = nf = tries = 0
    while (ns < N_SUCC or nf < N_FAIL) and tries < 400:
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
            print(f"  collected {len(episodes)} ({ns}s/{nf}f) tries={tries}", flush=True)
    print(f"[rollouts] {len(episodes)} episodes ({ns} succ / {nf} fail)", flush=True)

    # ---- score: episode-level + streaming, per model ----
    results = {}
    for name, sc in scorers.items():
        ep_sp, ep_lab = [], []
        # streaming: per episode, max sp BEFORE success (failures: whole episode);
        # and whether it fires at/after success.
        causal_fire_fail = []   # for failure eps: max streaming sp over the episode
        causal_max_presucc = [] # for success eps: max streaming sp strictly BEFORE success
        causal_sp_atsucc = []   # for success eps: max streaming sp at/after success
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
                causal_max_presucc.append(pres); causal_sp_atsucc.append(post)
            else:
                causal_fire_fail.append(allm)
        results[name] = dict(
            ep_sp=np.array(ep_sp), ep_lab=np.array(ep_lab),
            fail_max=np.array(causal_fire_fail),
            presucc_max=np.array(causal_max_presucc),
            atsucc_max=np.array(causal_sp_atsucc),
        )

    # ---- report ----
    for name, R in results.items():
        sp, lab = R["ep_sp"], R["ep_lab"]
        sm = sp[lab == 1].mean(); fm = sp[lab == 0].mean()
        print(f"\n================ {name} ================")
        print(f"episode-level: succ_mean={sm:.3f}  fail_mean={fm:.3f}  AUC={auc(lab, sp):.3f}"
              f"  (n={len(sp)}: {int(lab.sum())}s/{int((lab==0).sum())}f)")
        print(f"{'thr':>5} | {'ep_TPR':>7} {'ep_FPR':>7} | {'causalFPR':>9} {'causalTPR':>9} {'premFire':>8}")
        # causalFPR = frac failure eps whose streaming sp ever exceeds thr
        # causalTPR = frac success eps that fire AT/AFTER true success
        # premFire  = frac success eps that fire BEFORE true success (exploitable)
        for thr in THRS:
            ep_tpr = float((sp[lab == 1] > thr).mean())
            ep_fpr = float((sp[lab == 0] > thr).mean())
            c_fpr = float((R["fail_max"] > thr).mean()) if len(R["fail_max"]) else float("nan")
            c_tpr = float((R["atsucc_max"] > thr).mean()) if len(R["atsucc_max"]) else float("nan")
            prem = float((R["presucc_max"] > thr).mean()) if len(R["presucc_max"]) else float("nan")
            print(f"{thr:5.2f} | {ep_tpr:7.2f} {ep_fpr:7.2f} | {c_fpr:9.2f} {c_tpr:9.2f} {prem:8.2f}")
        # recommended operating points (lowest thr meeting causal-FPR target)
        for tgt in (0.05, 0.02):
            ok = [thr for thr in THRS
                  if len(R["fail_max"]) and (R["fail_max"] > thr).mean() <= tgt]
            if ok:
                thr = min(ok)
                c_tpr = float((R["atsucc_max"] > thr).mean())
                print(f"  -> causalFPR<= {tgt:.0%}: thr={thr:.2f}  (causalTPR={c_tpr:.2f})")
            else:
                print(f"  -> causalFPR<= {tgt:.0%}: NONE in grid (model fires on failures even at thr=0.95)")
        # where the paper's 0.6 lands
        c060 = float((R["fail_max"] > 0.6).mean()) if len(R["fail_max"]) else float("nan")
        t060 = float((R["atsucc_max"] > 0.6).mean()) if len(R["atsucc_max"]) else float("nan")
        print(f"  paper 0.6 -> causalFPR={c060:.2f}  causalTPR={t060:.2f}")


if __name__ == "__main__":
    main()
