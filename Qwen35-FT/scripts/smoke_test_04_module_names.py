"""Smoke test 04 — module-name introspection.

Walks named_modules() of the loaded Qwen3.5-4B model and dumps:
    1. Visual block submodule attribute names (which mlp.* names exist?)
    2. Text-tower decoder layer Linear-leaf names (for LoRA target_modules)
    3. The full visual block path (model.visual... vs model.model.visual...)

Writes findings to Qwen35-FT/configs/_discovered_modules.json so subsequent
configs / smoke tests can consume them without re-walking the 4B model.

Run AFTER smoke_test_03 has cached the weights:
    /home/pkarageorgis1/.conda/envs/robometer_qwen35_gpu/bin/python scripts/smoke_test_04_module_names.py
"""
import json
import os
from collections import Counter

import torch
import torch.nn as nn
from transformers import Qwen3_5ForConditionalGeneration

MODEL_ID = "Qwen/Qwen3.5-4B"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(HERE, "configs", "_discovered_modules.json")

print(f"  loading {MODEL_ID} (fp16, CPU)")
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cpu"
)

# 1. Visual block path discovery — what's the full attribute chain?
visual_paths = [n for n, _ in model.named_modules() if "visual" in n and ".blocks" in n][:6]
print("  visual block sample paths (first 6):")
for p in visual_paths:
    print(f"    {p}")

# 2. mlp.* leaves under the first visual block
candidates = [n for n, m in model.named_modules()
              if isinstance(m, nn.Linear) and "visual" in n and ".blocks.0" in n]
print(f"  visual blocks.0 Linear leaves: {candidates}")

# 3. Text-tower decoder Linear leaves — names that LoraConfig will need to target.
text_linear_basenames = Counter()
text_linear_examples = {}
for n, m in model.named_modules():
    if isinstance(m, nn.Linear) and "visual" not in n:
        base = n.rsplit(".", 1)[-1]
        text_linear_basenames[base] += 1
        text_linear_examples.setdefault(base, n)
print("  text-tower Linear basenames (count):")
for base, cnt in text_linear_basenames.most_common():
    print(f"    {base:24s} x{cnt:5d}   e.g. {text_linear_examples[base]}")

# 4. First decoder layer's mlp.* leaves specifically (most-targeted by LoRA presets).
layer0_mlp_leaves = [n for n, m in model.named_modules()
                     if isinstance(m, nn.Linear) and ".layers.0.mlp" in n and "visual" not in n]
print(f"  layer 0 mlp Linear leaves: {layer0_mlp_leaves}")

# 5. Best-guess visual probe path for _verify_checkpoint_loading
probe_candidates = [
    "model.visual.blocks.0.mlp.linear_fc1",
    "visual.blocks.0.mlp.linear_fc1",
    "model.visual.blocks.0.mlp.gate_proj",
    "visual.blocks.0.mlp.gate_proj",
    "model.visual.blocks.0.mlp.fc1",
    "visual.blocks.0.mlp.fc1",
]
visual_probe_path = None
for path in probe_candidates:
    obj = model
    ok = True
    for part in path.split("."):
        try:
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        except (AttributeError, IndexError, TypeError):
            ok = False
            break
    if ok and hasattr(obj, "weight"):
        visual_probe_path = path
        break
print(f"  resolved visual probe path: {visual_probe_path}")

findings = {
    "model_id": MODEL_ID,
    "visual_block_sample_paths": visual_paths,
    "visual_blocks0_linear_leaves": candidates,
    "text_linear_basename_counts": dict(text_linear_basenames),
    "text_linear_basename_examples": text_linear_examples,
    "layer0_mlp_linear_leaves": layer0_mlp_leaves,
    "visual_probe_path": visual_probe_path,
}
with open(OUT_PATH, "w") as f:
    json.dump(findings, f, indent=2)
print(f"  wrote findings to {OUT_PATH}")
print("PASS")
