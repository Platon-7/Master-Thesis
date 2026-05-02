"""Smoke test 00 — environment sanity.

Verifies the activated env is the new robometer_qwen35_gpu (transformers >= 5.7) and
that the Qwen3.5 unified VL class is importable. This is the gate for everything else.

Run from Qwen35-FT/:
    /home/pkarageorgis1/.conda/envs/robometer_qwen35_gpu/bin/python scripts/smoke_test_00_env.py
"""
import sys

assert sys.version_info[:2] == (3, 10), f"need python 3.10, got {sys.version_info}"

import transformers

ver = tuple(int(x) for x in transformers.__version__.split(".")[:2] if x.isdigit())
assert ver >= (5, 7), f"need transformers>=5.7, got {transformers.__version__}"
print(f"  transformers {transformers.__version__} OK")

from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401

print(f"  Qwen3_5ForConditionalGeneration imported: {Qwen3_5ForConditionalGeneration}")

# AutoConfig sanity: model_type registered.
from transformers import AutoConfig

print(f"  AutoConfig importable; model_type 'qwen3_5' registered:",
      "qwen3_5" in transformers.models.auto.configuration_auto.CONFIG_MAPPING_NAMES)

# Verify torch is reachable too (smoke 03/05 will use it).
import torch

print(f"  torch {torch.__version__} (cuda available: {torch.cuda.is_available()})")
print("PASS")
