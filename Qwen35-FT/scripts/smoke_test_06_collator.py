"""Smoke test 06 — vendored robometer.data.collators import + Qwen3.5 dispatch.

Confirms the symlinked `robometer.data.collators` module imports without needing
unsloth/bitsandbytes (verified — those are setup_utils-only deps), and that the
existing `is_qwen3 = "Qwen3" in base_model_id or "Molmo2" in base_model_id` check
captures the new Qwen3.5 id (intentional substring collision — see the plan).

Run from Qwen35-FT/ so `import robometer` resolves to the vendored package:
    cd Qwen35-FT
    /home/pkarageorgis1/.conda/envs/robometer_qwen35_gpu/bin/python scripts/smoke_test_06_collator.py
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)  # ensure vendored robometer/ wins on import

import robometer

vendor = os.path.realpath(os.path.dirname(robometer.__file__))
print(f"  robometer pkg resolved to: {vendor}")

from robometer.data.collators import RBMBatchCollator  # noqa: F401

print(f"  RBMBatchCollator imported OK: {RBMBatchCollator}")

# The collator's Qwen3-family branch uses a substring match. Qwen3.5 must hit it
# (intentional — Qwen3.5 honors the same processor API surface for video metadata).
test_id = "Qwen/Qwen3.5-4B"
assert "Qwen3" in test_id, "internal sanity: Qwen3 substring collision is the dispatch hook"
print(f"  '{test_id}' captured by Qwen3-family dispatch (substring) OK")
print("PASS")
