"""Smoke test 08 — vendored setup_utils import + sanity flags.

Confirms the lazy-import refactor for unsloth/bnb didn't introduce a regression.
The Qwen35-FT env intentionally does NOT install unsloth or bitsandbytes — they
were dead code at module level (bnb) or used only in a path we never hit (unsloth).
If a future edit re-adds a top-level `from unsloth import ...` line, this test
will fail before the SLURM job ever starts.

Also verifies:
    - `_is_qwen35` distinguishes Qwen3.5 from Qwen3 ids correctly
    - `HAS_QWEN35` is True (transformers 5.7+ exports the class)

Run from Qwen35-FT/:
    cd Qwen35-FT
    /home/pkarageorgis1/.conda/envs/robometer_qwen35_gpu/bin/python scripts/smoke_test_08_setup_utils.py
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

# Hard-fail if unsloth/bnb sneak back into the import graph: they're not installed.
import importlib.util

for forbidden in ("unsloth", "bitsandbytes"):
    spec = importlib.util.find_spec(forbidden)
    if spec is not None:
        print(f"  WARNING: {forbidden} is installed in this env — the lazy-import "
              f"guarantee is moot, but the test still runs.")

import robometer.utils.setup_utils as su

print(f"  imported: {su.__file__}")
print(f"  HAS_QWEN3  = {su.HAS_QWEN3}")
print(f"  HAS_QWEN35 = {su.HAS_QWEN35}")
print(f"  Qwen3_5ForConditionalGeneration = {su.Qwen3_5ForConditionalGeneration}")

assert su.HAS_QWEN35, "transformers 5.7 should export Qwen3_5ForConditionalGeneration"
assert su.Qwen3_5ForConditionalGeneration is not None

assert su._is_qwen35("Qwen/Qwen3.5-4B") is True
assert su._is_qwen35("qwen/qwen3.5-4b") is True
assert su._is_qwen35("Qwen/Qwen3-VL-4B-Instruct") is False
assert su._is_qwen35("Qwen/Qwen2.5-VL-3B-Instruct") is False
assert su._is_qwen35("HuggingFaceTB/SmolVLM-Instruct") is False
print("  _is_qwen35 substring trap dodged: 4/4 cases correct")

# Verify FastVisionModel is NOT bound at module level (proves the lazy refactor stuck).
assert not hasattr(su, "FastVisionModel"), (
    "FastVisionModel leaked back into the module namespace — the top-level "
    "`from unsloth import FastVisionModel` was re-added. Move it back inside "
    "_load_base_model_unsloth."
)
print("  FastVisionModel correctly NOT at module level (lazy-import preserved)")

print("PASS")
