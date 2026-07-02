"""Calibrate RoboReward-4B (teetone/RoboReward-4B) the SAME way as the other
reward models, adapted to its nature: RoboReward is an END-OF-EPISODE video
scorer (discrete 1-5 -> normalized [0,1]), not a per-step success critic. So we
score each oracle rollout's FULL video once and build a success-vs-failure ROC,
then report the op-threshold by the uniform rule (max TPR at FPR<=0.15; best-
detection if degenerate). Reuses calibrate_matrix's oracle-rollout generator.

Env knobs: CALIB_TASKS (single task per job to dodge SLURM --export comma split),
CALIB_N_SUCC/CALIB_N_FAIL (default 15/15), RR_MAX_NEW_TOKENS (default 64),
CALIB_OUT (json).
"""
import os, sys, json, re
import numpy as np

_REPO = os.environ.get("MT_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for sub in ("MetaWorld/metaworld_repo", "MetaWorld", "vlm_ibrl_v3", "Robometer"):
    sys.path.insert(0, os.path.join(_REPO, sub))

from env.metaworld_wrapper import MetaWorldEnv
from env.roboreward_utils import get_roboreward_4b, prompt_roboreward
from env.vlm_prompts import METAWORLD_TASK_DESCRIPTIONS as TASK_STR, roboreward_prompt
from PIL import Image
import scipy.stats as st

TASKS = os.environ.get("CALIB_TASKS", "CoffeePush,BoxClose").split(",")
CAM = "corner2"; RES = 224
N_SUCC = int(os.environ.get("CALIB_N_SUCC", "15"))
N_FAIL = int(os.environ.get("CALIB_N_FAIL", "15"))
MAXT = int(os.environ.get("CALIB_MAXT", "200"))      # match episode_length
RR_MAX_NEW = int(os.environ.get("RR_MAX_NEW_TOKENS", "64"))
THRS = np.round(np.arange(0.0, 1.01, 0.25), 2)        # RoboReward is 1-5 -> {0,.25,.5,.75,1.0}
OUT = os.environ.get("CALIB_OUT", "/shared/home/PKA4388/vlm_ibrl_runs/calib_roboreward_4b.json")


def auc(l, s):
    l=np.asarray(l); s=np.asarray(s); P=(l==1).sum(); N=(l==0).sum()
    if P==0 or N==0: return float("nan")
    r=st.rankdata(s)
    return float((r[l==1].sum()-P*(P+1)/2)/(P*N))


def rollout(env, noisy):
    env.reset(); frames=[]; succ=0
    for t in range(MAXT):
        a = env.get_heuristic_action()
        if noisy: a=(a+np.random.uniform(-1,1,size=a.shape)*0.8).clip(-1,1)
        obs,r,done,info = env.step(a)
        frames.append(Image.fromarray(env.render(camera_name=CAM, width=RES, height=RES)))
        if int(info.get("success",0))==1: succ=1
        if done: break
    return frames, succ


def collect(env):
    eps=[]; ns=nf=tries=0
    while (ns<N_SUCC or nf<N_FAIL) and tries<800:
        tries+=1
        noisy = nf<N_FAIL and (ns>=N_SUCC or np.random.rand()<0.55)
        fr,sc = rollout(env, noisy)
        if sc and ns>=N_SUCC: continue
        if (not sc) and nf>=N_FAIL: continue
        eps.append((fr,sc)); ns+=sc; nf+=(1-sc)
    return eps, ns, nf


_PARSE = re.compile(r"ANSWER:\s*([1-5])")

RR_NFRAMES = int(os.environ.get("ROBOREWARD_NFRAMES", "16"))

def score_rollout(model, processor, task, frames, debug=False):
    """Score a rollout exactly as the RL env now does: subsample to RR_NFRAMES
    (final frame included) to avoid the full-frame resolution collapse, then feed
    as video. Returns (normalized_reward in [0,1] or None, raw_score 1-5, raw_text)."""
    if len(frames) > RR_NFRAMES:
        sel = np.linspace(0, len(frames) - 1, RR_NFRAMES).round().astype(int)
        frames = [frames[i] for i in sel]
    prompt = f"{roboreward_prompt}\n\nTask: {task}"
    content = [
        {"type": "video", "video": frames, "sample_fps": 20,
         "video_metadata": {"duration": len(frames) / 20.0}},
        {"type": "text", "text": prompt},
    ]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": roboreward_prompt}]},
        {"role": "user", "content": content},
    ]
    kw = dict(max_new_tokens=RR_MAX_NEW, do_sample=False, top_p=1.0, top_k=0, temperature=0)
    raw = prompt_roboreward(model=model, processor=processor, messages=messages,
                            debug=debug, prompt_kwargs=kw)
    m = _PARSE.search(raw or "")
    if not m:
        return None, None, raw
    s = int(m.group(1))
    return (s - 1) / 4.0, s, raw


def main():
    np.random.seed(0)
    print(f"[rr-calib] loading teetone/RoboReward-4B ...", flush=True)
    model, processor = get_roboreward_4b()
    print(f"[rr-calib] loaded. tasks={TASKS} N={N_SUCC}s/{N_FAIL}f max_new={RR_MAX_NEW}", flush=True)

    summary = {"model": "roboreward_4b", "tasks": {}}
    if os.path.exists(OUT):
        try:
            prev = json.load(open(OUT))
            if prev.get("model") == "roboreward_4b": summary = prev
            print(f"[resume] {len(summary['tasks'])} prior tasks", flush=True)
        except Exception as e:
            print(f"[resume] could not read {OUT}: {e}", flush=True)

    def save():
        tmp = OUT + ".tmp"
        with open(tmp, "w") as f: json.dump(summary, f, indent=2)
        os.replace(tmp, OUT)

    for env_name in [t.strip() for t in TASKS]:
        if env_name in summary["tasks"] and "scores" in summary["tasks"][env_name]:
            print(f"  [skip] {env_name} done", flush=True); continue
        TASK = TASK_STR[env_name]
        print(f"\n##### {env_name}  task={TASK!r} #####", flush=True)
        env = MetaWorldEnv(env_name, camera_name=CAM, width=RES, height=RES)
        eps, ns, nf = collect(env)
        print(f"  rollouts: {len(eps)} ({ns}s/{nf}f)", flush=True)
        scores = []; labels = []; raws = []; parse_fail = 0
        for i, (fr, sc) in enumerate(eps):
            norm, raw_s, raw = score_rollout(model, processor, TASK, fr, debug=(i < 2))
            if norm is None:
                parse_fail += 1
                print(f"  [parse-fail] ep{i} (succ={sc}) raw={raw!r}", flush=True); continue
            scores.append(norm); labels.append(sc); raws.append(raw_s)
            if i < 4 or i % 10 == 0:
                print(f"  ep{i:2d} succ={sc} rrscore={raw_s} norm={norm:.2f}", flush=True)
        scores = np.array(scores); labels = np.array(labels)
        a = auc(labels, scores)
        succ_mean = float(scores[labels == 1].mean()) if (labels == 1).any() else float("nan")
        fail_mean = float(scores[labels == 0].mean()) if (labels == 0).any() else float("nan")
        sweep = []
        for thr in THRS:
            tpr = float((scores[labels == 1] >= thr).mean()) if (labels == 1).any() else float("nan")
            fpr = float((scores[labels == 0] >= thr).mean()) if (labels == 0).any() else float("nan")
            sweep.append(dict(thr=float(thr), tpr=tpr, fpr=fpr))
        # op-threshold: uniform rule (max TPR at FPR<=0.15; else max TPR-FPR)
        ok = [r for r in sweep if r["fpr"] <= 0.15]
        op = (max(ok, key=lambda r: r["tpr"]) if ok
              else max(sweep, key=lambda r: r["tpr"] - r["fpr"]))
        print(f"  >>> AUC={a:.3f} succ_mean={succ_mean:.3f} fail_mean={fail_mean:.3f} "
              f"parse_fail={parse_fail}", flush=True)
        print(f"  >>> op-threshold(norm)={op['thr']} TPR={op['tpr']:.2f} FPR={op['fpr']:.2f}", flush=True)
        for r in sweep:
            print(f"      thr={r['thr']:.2f} TPR={r['tpr']:.2f} FPR={r['fpr']:.2f}", flush=True)
        summary["tasks"][env_name] = dict(
            auc=a, succ_mean=succ_mean, fail_mean=fail_mean, n_succ=int(ns), n_fail=int(nf),
            parse_fail=parse_fail, op_thr=op["thr"], op_tpr=op["tpr"], op_fpr=op["fpr"],
            sweep=sweep, scores=scores.tolist(), labels=labels.tolist(), rrscores=raws)
        save()
    print(f"\n[rr-calib] done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
