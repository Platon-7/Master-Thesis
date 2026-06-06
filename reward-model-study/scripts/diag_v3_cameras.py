"""v3 counterpart of diag_v2_cameras.py — the decisive domain-match test.

Roll out MetaWorld **v3** CoffeePush (scripted oracle -> successes; noisy ->
failures) in the new vlm_ibrl_v3 env, render each state from the curated-data
cameras (corner2/corner3/gripperPOV) the EXACT way the reward model's training
data was rendered (gymnasium MujocoRenderer.camera_id + vertical flip), then
score each camera with Robometer-FT.

Reference numbers (same FT ckpt run1 s3000, same task string, no ICL):
  - v3 curated CoffeePush eval frames : AUC 0.97  (succ_mean 0.77, fail 0.13)
  - v2 live GT-policy rollout frames  : AUC 0.53  (succ_mean 0.25)
  - v2 zoomed-corner2 (diag_v2)       : ~0.5

Interpretation:
  - If v3 corner2/corner3 recovers AUC ~0.9 (succ_mean high) -> the on-policy
    collapse was a RENDERING-DOMAIN (v2-vs-v3) gap; rendering IBRL in v3 fixes
    it. POSITIVE result: the reward transfers on-policy when the domain matches.
  - If v3 cameras also stay ~0.5 -> the gap is NOT the render engine and the
    on-policy failure is something else (policy behaviour / distribution).

Runs render + score in ONE process under the demo2reward_v3 env (gymnasium
metaworld v3 + transformers 4.57.2 for the Qwen3-VL-based FT scorer).
"""
import os
import sys
import numpy as np

# vlm_ibrl_v3 (the v3 port) provides the env; Robometer/ provides the scorer pkg.
V3_ROOT = "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl_v3"
sys.path.insert(0, V3_ROOT)
sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer")

from env.metaworld_wrapper import MetaWorldEnv
from env.robometer_utils import get_robometer_4b
from PIL import Image
import scipy.stats as st

TASK = "Push a mug under a coffee machine."
MODEL = "/scratch-shared/pkarageorgis1/Robometer_FT_consolidated/run1_icl_ours_step3000"
# corner2 is the IN-DOMAIN camera: render_match_v3.py confirmed corner2+flip
# reproduces the curated CoffeePush success frame (MSE 415 vs >3800 for any
# other camera/orientation). corner3/gripperPOV are rendered too but the reward
# model never trained on them — they are out-of-domain references.
CAMERAS = ["corner2", "corner3", "gripperPOV"]
N_SUCC = 25
N_FAIL = 25
MAXT = 175            # raw v3 steps; scripted CoffeePush oracle succeeds well within this
SUB = 16             # frames per camera fed to the scorer (matches max_frames)
RES = 240            # match the curated v3 frame resolution exactly (240x240)
SAMPLE_DIR = "/gpfs/home3/pkarageorgis1/Master-Thesis/reward-model-study/results/v3_sample_frames"


def auc(l, s):
    l = np.asarray(l); s = np.asarray(s); P = (l == 1).sum(); N = (l == 0).sum()
    if P == 0 or N == 0:
        return float('nan')
    r = st.rankdata(s)
    return float((r[l == 1].sum() - P * (P + 1) / 2) / (P * N))


def rollout(env, noisy):
    """Returns dict cam->list-of-SUB PIL frames, and success bool."""
    buf = {c: [] for c in CAMERAS}
    env.reset()
    succ = 0
    for t in range(MAXT):
        a = env.get_heuristic_action()
        if noisy:
            a = (a + np.random.uniform(-1, 1, size=a.shape) * 0.8).clip(-1, 1)
        obs, r, done, info = env.step(a)
        if int(info.get("success", 0)) == 1:
            succ = 1
        for c in CAMERAS:
            buf[c].append(env.render(camera_name=c, width=RES, height=RES))
        if done:
            break
    idx = np.linspace(0, len(buf[CAMERAS[0]]) - 1, SUB).round().astype(int)
    return {c: [Image.fromarray(buf[c][i]) for i in idx] for c in CAMERAS}, succ


def main():
    np.random.seed(0)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    env = MetaWorldEnv("CoffeePush", camera_name="corner2", width=RES, height=RES)
    sc = get_robometer_4b(model_path=MODEL)
    print(f"scorer max_frames = {getattr(sc, 'max_frames', '?')}", flush=True)

    data = {c: [] for c in CAMERAS}
    labels = []
    ns = nf = 0
    tries = 0
    saved_sample = False
    while (ns < N_SUCC or nf < N_FAIL) and tries < 250:
        tries += 1
        noisy = nf < N_FAIL and (ns >= N_SUCC or np.random.rand() < 0.5)
        frames, succ = rollout(env, noisy)
        if succ and ns >= N_SUCC:
            continue
        if (not succ) and nf >= N_FAIL:
            continue
        # Save one successful-episode sample frame per camera (last frame) for a
        # visual domain-match check against a real curated v3 CoffeePush frame.
        if succ and not saved_sample:
            for c in CAMERAS:
                frames[c][-1].save(os.path.join(SAMPLE_DIR, f"v3_{c}_success_last.png"))
                frames[c][SUB // 2].save(os.path.join(SAMPLE_DIR, f"v3_{c}_success_mid.png"))
            saved_sample = True
            print(f"  saved sample success frames to {SAMPLE_DIR}", flush=True)
        labels.append(succ); ns += succ; nf += (1 - succ)
        for c in CAMERAS:
            data[c].append(sc(frames[c], task=TASK, icl_frames=None)["success_prob"])
        if len(labels) % 10 == 0:
            print(f"  collected {len(labels)} ({ns}s/{nf}f), tries={tries}…", flush=True)

    L = np.array(labels)
    print(f"\n==== v3 CoffeePush, {len(L)} eps ({int(L.sum())}s/{int((L == 0).sum())}f) — "
          f"AUC + success_prob means per camera ====")
    for c in CAMERAS:
        v = np.array(data[c]); a = auc(L, v)
        sm = v[L == 1].mean() if (L == 1).any() else float('nan')
        fm = v[L == 0].mean() if (L == 0).any() else float('nan')
        tag = "  <- IN-DOMAIN (corner2)" if c == "corner2" else ""
        print(f"  {c:13s}: AUC={a:.3f}  succ_mean={sm:.3f}  fail_mean={fm:.3f}{tag}")
    print("\nref: v3 curated AUC=0.97 (succ 0.77, fail 0.13); "
          "v2 live AUC=0.53 (succ 0.25); v2 zoomed-corner2 diag ~0.5")
    print("If v3 corner2/corner3 ~0.9 -> domain match recovers the reward on-policy (POSITIVE).")


if __name__ == "__main__":
    main()
