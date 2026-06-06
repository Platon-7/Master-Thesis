"""End-to-end import smoke test for the demo2reward env.

Verifies every heavy dependency Chris's IBRL harness needs imports cleanly
and that the C++ replay-buffer extension loaded. Run after install.sh.
"""

from __future__ import annotations

import sys


def check(label: str, fn):
    try:
        result = fn()
        print(f"  OK  {label}: {result}")
        return True
    except Exception as e:
        print(f"  ERR {label}: {type(e).__name__}: {e}")
        return False


def main():
    print(f"python: {sys.version.split()[0]}")
    print(f"exec  : {sys.executable}")
    print()

    ok = True
    ok &= check("torch (cu121 expected)", lambda: f"{__import__('torch').__version__}")
    ok &= check("torch.cuda.is_available", lambda: __import__('torch').cuda.is_available())
    ok &= check("torch.cuda.device_count", lambda: __import__('torch').cuda.device_count())
    ok &= check("transformers (HF main)", lambda: __import__('transformers').__version__)
    ok &= check("accelerate", lambda: __import__('accelerate').__version__)
    ok &= check("flash_attn", lambda: __import__('flash_attn').__version__)
    ok &= check("qwen_vl_utils", lambda: __import__('qwen_vl_utils').__name__)
    ok &= check("mujoco (DeepMind)", lambda: __import__('mujoco').__version__)
    ok &= check("mujoco_py (legacy)", lambda: __import__('mujoco_py').__version__)
    ok &= check("metaworld (v2-era pin)",
                lambda: getattr(__import__('metaworld'), '__version__', 'no __version__'))
    ok &= check("metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE",
                lambda: f"{len(__import__('metaworld.envs', fromlist=['ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE']).ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE)} envs")
    ok &= check("robosuite", lambda: __import__('robosuite').__version__)
    ok &= check("common_utils (C++ ext rela.so)",
                lambda: __import__('common_utils.rela', fromlist=['ReplayBuffer']).__name__)
    ok &= check("env.vlm_envs (Chris's MetaWorld critic wrapper)",
                lambda: __import__('env.vlm_envs', fromlist=['VALID_VLMS']).VALID_VLMS)
    ok &= check("env.robometer_utils (our new Robometer adapter)",
                lambda: __import__('env.robometer_utils', fromlist=['get_robometer_4b']).get_robometer_4b.__name__)

    print()
    if ok:
        print("ALL IMPORTS PASSED")
        sys.exit(0)
    else:
        print("AT LEAST ONE IMPORT FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
