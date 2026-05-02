#!/usr/bin/env python3
"""
Diagnostic: does the physics in a FailgenWrapper stage-failure episode
actually advance, or is the simulation frozen?

Logs the end-effector TCP position at every env.step() call during one
FailPushCube-v1 stage-1 (reach_pose, trans_x) failure episode. If the
TCP moves, physics is live and the problem is purely in how we capture
frames. If the TCP is static, the problem is upstream (env/planner).

Run:
    export FAILSAFE_PATH=/home/pkarageorgis1/Master-Thesis/Failsafe/failsafe_repo
    python diagnose_motion.py
"""

import os
import sys
import numpy as np
from pathlib import Path

FAILSAFE_PATH = os.environ.get("FAILSAFE_PATH", "")
if FAILSAFE_PATH:
    sys.path.insert(0, FAILSAFE_PATH)

from failgen.env_wrapper import FailgenWrapper
import yaml as _yaml

TASK = "FailPushCube-v1"
FAIL_STAGE = 1
FAIL_TYPE = "trans_x"
NOISE = 0.15

# Patch save_path so FailgenWrapper doesn't crash
cfg_path = Path(FAILSAFE_PATH) / "failgen" / "configs" / f"{TASK}.yaml"
orig = cfg_path.read_text()
cfg = _yaml.safe_load(orig)
cfg["save_path"] = "/tmp/_diag"
cfg_path.write_text(_yaml.dump(cfg))

try:
    fw = FailgenWrapper(task_name=TASK, headless=True, save_video=True)
    fw._fail_plan_wrapper.set_active_type(FAIL_TYPE)
    fw._fail_plan_wrapper.set_active_stage(FAIL_STAGE)
    fw._fail_plan_wrapper._active_fail.noise = NOISE

    base_env = fw._env.unwrapped
    step_count = [0]
    tcp_log = []

    orig_step = fw._env.step
    def logging_step(action):
        result = orig_step(action)
        tcp_p = base_env.agent.tcp.pose.p
        if hasattr(tcp_p, "cpu"):
            tcp_p = tcp_p.cpu().numpy()
        tcp_p = np.asarray(tcp_p).reshape(-1)[:3]
        tcp_log.append((step_count[0], tcp_p.copy()))
        step_count[0] += 1
        return result
    fw._env.step = logging_step

    print(f"Running {TASK} stage {FAIL_STAGE} {FAIL_TYPE} noise={NOISE} ...")
    success = fw.get_failure(num_gt=0)
    print(f"success={success} total_steps={step_count[0]} stg_index={fw.stg_index}")

    if not tcp_log:
        print("NO STEPS RECORDED — planner never called env.step().")
    else:
        positions = np.array([p for _, p in tcp_log])
        print(f"\nTCP position summary over {len(positions)} steps:")
        print(f"  first: {positions[0]}")
        print(f"  last:  {positions[-1]}")
        print(f"  min:   {positions.min(axis=0)}")
        print(f"  max:   {positions.max(axis=0)}")
        print(f"  range: {positions.max(axis=0) - positions.min(axis=0)}")
        # Print every 10th step for compactness
        print("\nPer-step TCP (every 10th):")
        for i, (s, p) in enumerate(tcp_log):
            if i % 10 == 0 or i == len(tcp_log) - 1:
                print(f"  step {s:3d}: {p}")

        # Consecutive step diff summary
        diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        print(f"\nPer-step displacement: mean={diffs.mean():.6f} "
              f"max={diffs.max():.6f} zero-steps={int((diffs < 1e-6).sum())}/{len(diffs)}")
finally:
    cfg_path.write_text(orig)
