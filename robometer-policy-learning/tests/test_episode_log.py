"""Exercise the REAL RobometerReplayBuffer._add() with a stubbed scorer.

No GPU, no VLM, no SLURM. Drives the actual control flow -- the per-key scoring
loop, beta-mix, normalisation, potential shaping, the detection block and the
episode logger -- so it catches exactly what a smoke job would, in seconds.
"""
import json
import os
import shutil
import sys

import numpy as np

import tempfile
OUT = tempfile.mkdtemp(prefix="eplog_test_")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["RPL_EPISODE_LOG"] = OUT
os.environ["RPL_THRESHOLD_SOURCE"] = "unit-test"

import robometer_policy_learning.buffers.replay_buffer as rbmod
from robometer_policy_learning.buffers.robometer_replay_buffer import RobometerReplayBuffer

# Stub the base so we don't need real buffer storage; capture what SAC would get.
TRAINED = []
rbmod.ReplayBuffer._add = lambda self, **kw: TRAINED.append(dict(kw))


def make(detect, thr=0.10, min_ep=0, duration=2, beta=1.0, binarize=None,
         normalize=False, potential=False, log=True):
    o = object.__new__(RobometerReplayBuffer)
    o.reward_model = object()          # non-None -> enter the scoring branch
    o.use_eval_server = False
    o.reward_relabeling_keys = ["image"]
    o.max_frames = 16
    o.use_relative_rewards = False
    o.add_estimated_reward = False
    o.normalize_reward = normalize
    o.normalize_warmup, o._norm_buf, o._norm_lo, o._norm_hi = 10, __import__("collections").deque(maxlen=100), None, None
    o.progress_as_potential = potential
    o.potential_gamma, o.potential_scale, o._phi_prev = 0.95, 1.0, {}
    o.progress_beta = beta
    o.progress_binarize_threshold = binarize
    o.use_success_detection = detect
    o.success_detection_threshold = thr
    o.success_detection_duration = duration
    o.success_detection_min_ep_steps = min_ep
    o._ep_steps = {}
    o.success_tracker = {"image": __import__("collections").deque(maxlen=duration)}
    o.icl_demos = None
    o._eplog_path = os.path.join(OUT, "episodes.jsonl") if log else None
    o._eplog, o._eplog_n = {}, 0
    o._eplog_threshold_source = os.environ["RPL_THRESHOLD_SOURCE"]
    return o


def run_episode(o, *, n_steps, prog_fn, sp_fn, gt_fn, env_idx=0):
    """Drive n_steps of one episode through the real _add()."""
    o._compute_reward_single = lambda raw, _p=prog_fn, _s=sp_fn: (
        _p(raw["metadata"]["t"]), _s(raw["metadata"]["t"]))
    frame = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    for t in range(n_steps):
        # _compute_reward_single is stubbed, but _add still builds raw_data; smuggle t
        o._pad_to_max_frames = lambda f, _t=t: type("A", (np.ndarray,), {})  # unused
        o._pad_to_max_frames = lambda f, _t=t: np.zeros((4, 8, 8, 3), dtype=np.uint8)
        orig = o._compute_reward_single
        o._compute_reward_single = lambda raw, _t=t: (prog_fn(_t), sp_fn(_t))
        last = (t == n_steps - 1)
        o._add(
            language_instruction="pull the cube",
            video_frames={"image": frame},
            dino_embeddings=np.zeros((1, 4)),
            text_embedding=None,
            reward=0.0,
            done=False,
            truncated=last,
            is_success=gt_fn(t),
            episode_id=0,
            env_idx=env_idx,
        )
        o._compute_reward_single = orig


def load():
    p = os.path.join(OUT, "episodes.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fails.append(name)


print("=== A. DENSE, no termination (headline regime) ===")
o = make(detect=False)
# progress ramps; GT solved at steps 30..35 then undone
run_episode(o, n_steps=50, prog_fn=lambda t: 0.1 + 0.002 * t,
            sp_fn=lambda t: 0.02, gt_fn=lambda t: 30 <= t <= 35)
recs = load()
check("one record written", len(recs) == 1, f"got {len(recs)}")
r = recs[0]
check("episode_len == 50", r["episode_len"] == 50, str(r["episode_len"]))
check("score_per_step unthrottled", len(r["score_per_step"]) == 50, str(len(r["score_per_step"])))
check("gt_solved_anytime == 1", r["gt_solved_anytime"] == 1)
check("gt_first_solve_step == 30", r["gt_first_solve_step"] == 30, str(r["gt_first_solve_step"]))
exp = round(sum(0.1 + 0.002 * t for t in range(50)), 3)
check("vlm_return correct", abs(r["vlm_return"] - exp) < 1e-2, f"{r['vlm_return']} vs {exp}")
check("detector fields null when detection off",
      r["fired"] is None and r["gt_solved_at_fire"] is None and r["gate_suppressed"] is None)
check("threshold provenance present", r["threshold_source"] == "unit-test")
check("reward reached SAC unchanged",
      abs(TRAINED[-1]["reward"] - (0.1 + 0.002 * 49)) < 1e-6, str(TRAINED[-1]["reward"]))

print("=== B. DETECTOR with gate: early cross suppressed, later fire real ===")
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT, exist_ok=True)
TRAINED.clear()
o = make(detect=True, thr=0.5, min_ep=10, duration=2)
# sp crosses at t=2..3 (gated) and again at t=25..26 (real, GT true there)
run_episode(o, n_steps=40, prog_fn=lambda t: 0.3,
            sp_fn=lambda t: 0.9 if t in (2, 3, 25, 26) else 0.1,
            gt_fn=lambda t: t >= 25)
recs = load()
# A fire sets done=True, which ENDS the episode -- so 40 steps become 2 episodes.
check("fire segments the episode (2 records)", len(recs) == 2, f"got {len(recs)}")
check("segment lengths sum to 40", sum(r["episode_len"] for r in recs) == 40,
      str([r["episode_len"] for r in recs]))
check("2nd segment did not fire", recs[1]["fired"] == 0 if len(recs) > 1 else False)
if recs:
    r = recs[0]
    check("gate_suppressed recorded", r["gate_suppressed"] == 1, str(r["gate_suppressed"]))
    check("fired recorded", r["fired"] == 1, str(r["fired"]))
    check("fire_step >= min_ep_steps", (r["fire_step"] or 0) >= 10, str(r["fire_step"]))
    check("gt_solved_at_fire is instantaneous", r["gt_solved_at_fire"] == 1, str(r["gt_solved_at_fire"]))
    check("detection_enabled true", r["detection_enabled"] is True)

print("=== C. DISABLED (RPL_EPISODE_LOG unset) -> no file, rewards unchanged ===")
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT, exist_ok=True)
TRAINED.clear()
o = make(detect=False, log=False)
run_episode(o, n_steps=10, prog_fn=lambda t: 0.42, sp_fn=lambda t: 0.0, gt_fn=lambda t: False)
check("no file written when disabled", not os.path.exists(os.path.join(OUT, "episodes.jsonl")))
check("rewards still delivered", len(TRAINED) == 10 and abs(TRAINED[0]["reward"] - 0.42) < 1e-9)

print("=== D. beta-mix + binarize still applied correctly WITH logging on ===")
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT, exist_ok=True)
TRAINED.clear()
o = make(detect=False, beta=0.0, binarize=0.5)
run_episode(o, n_steps=6, prog_fn=lambda t: 0.9, sp_fn=lambda t: 0.8, gt_fn=lambda t: False)
check("beta=0 + binarize -> reward 1.0", abs(TRAINED[0]["reward"] - 1.0) < 1e-9, str(TRAINED[0]["reward"]))
r = load()[0]
check("logged score_per_step is RAW progress (0.9), not mixed",
      abs(r["score_per_step"][0] - 0.9) < 1e-9, str(r["score_per_step"][0]))
check("logged reward_per_step is POST-mix (1.0)",
      abs(r["reward_per_step"][0] - 1.0) < 1e-9, str(r["reward_per_step"][0]))

print()

# ---------------------------------------------------------------------------
# E. MULTI-ENV INTERLEAVING (num_envs=4, the real configuration).
# _ep_key keys on env_idx precisely because a previous version merged all envs
# into one accumulator (45 len=1 entries against 14 real episodes). Four
# concurrent episodes of DIFFERENT lengths must produce four clean records with
# no cross-contamination of scores.
# ---------------------------------------------------------------------------
print("=== E. 4 interleaved envs, different episode lengths ===")
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT, exist_ok=True)
TRAINED.clear()
o = make(detect=False)
LENS = {0: 20, 1: 30, 2: 15, 3: 50}
PROG = {0: 0.11, 1: 0.22, 2: 0.33, 3: 0.44}   # unique per env -> detects mixing
SOLVE = {0: None, 1: 12, 2: None, 3: 40}      # env1/env3 solve mid-episode
frame = np.zeros((1, 8, 8, 3), dtype=np.uint8)
o._pad_to_max_frames = lambda f: np.zeros((4, 8, 8, 3), dtype=np.uint8)
step_in = {e: 0 for e in LENS}
for tick in range(max(LENS.values())):
    for e in sorted(LENS):
        if step_in[e] >= LENS[e]:
            continue
        t = step_in[e]
        o._compute_reward_single = lambda raw, _e=e: (PROG[_e], 0.01)
        gt = SOLVE[e] is not None and t >= SOLVE[e]
        o._add(language_instruction="t", video_frames={"image": frame},
               dino_embeddings=np.zeros((1, 4)), text_embedding=None,
               reward=0.0, done=False, truncated=(t == LENS[e] - 1),
               is_success=gt, episode_id=tick, env_idx=e)
        step_in[e] += 1

recs = {int(r["env_key"]): r for r in load()}
check("one record per env", sorted(recs) == [0, 1, 2, 3], str(sorted(recs)))
for e, want in LENS.items():
    if e in recs:
        check(f"env{e} length == {want}", recs[e]["episode_len"] == want, str(recs[e]["episode_len"]))
        vals = set(recs[e]["score_per_step"])
        check(f"env{e} scores uncontaminated", vals == {PROG[e]}, str(sorted(vals)[:4]))
        check(f"env{e} first_solve == {SOLVE[e]}",
              recs[e]["gt_first_solve_step"] == SOLVE[e], str(recs[e]["gt_first_solve_step"]))

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILURES: {fails}")
sys.exit(1 if fails else 0)
