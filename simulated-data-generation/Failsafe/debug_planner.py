#!/usr/bin/env python3
"""
Diagnostic: run one episode of each task with NO failure injection,
print stg_index and final success, to verify the motion planner works at all.
"""
import sys, os
FAILSAFE_PATH = os.environ.get("FAILSAFE_PATH", "")
if FAILSAFE_PATH:
    sys.path.insert(0, FAILSAFE_PATH)

from failgen.env_wrapper import FailgenWrapper
from omegaconf import OmegaConf
import yaml, tempfile, shutil
from pathlib import Path

TASKS = ["FailPushCube-v1", "FailStackCube-v1"]   # skip FailPickCube (SO100 ≠ Panda)

for task in TASKS:
    print(f"\n{'='*50}\n{task}")
    tmp = Path(tempfile.mkdtemp())
    # Patch save_path in the YAML
    cfg_path = Path(FAILSAFE_PATH) / "failgen" / "configs" / f"{task}.yaml"
    orig = cfg_path.read_text()
    cfg = yaml.safe_load(orig)
    cfg["save_path"] = str(tmp)
    cfg_path.write_text(yaml.dump(cfg))

    try:
        fw = FailgenWrapper(task_name=task, headless=True, save_video=False)
        # No failure injection — all stages execute normally
        fw._fail_plan_wrapper.set_active_type("None")

        for seed in range(3):
            success = fw.get_failure(num_gt=seed)
            print(f"  seed={seed}  stg_index={fw.stg_index}  success={success}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
    finally:
        cfg_path.write_text(orig)
        shutil.rmtree(tmp, ignore_errors=True)

print("\nDone.")
