"""Smoke test 07 — LoRA target_modules resolve on Qwen3.5-4B.

Reads Qwen35-FT/configs/_discovered_modules.json (written by smoke 04) and tries to
wrap the model with PEFT LoraConfig using both:

    a. The Robometer-LoRA default presets ("q_proj","k_proj","v_proj","o_proj",
       "gate_proj","up_proj","down_proj") — the Qwen2.5-style target list.
    b. Qwen3.5's actual text-tower decoder Linear leaf basenames discovered in 04.

Reports the adapter-module count for each, so we can pick the right list to set
in `Qwen35-FT/configs/train_lora_base.yaml` as a `peft.target_modules: [...]`
override.

Run AFTER smoke_test_03 (weights cached) and smoke_test_04 (JSON written):
    cd Qwen35-FT
    /home/pkarageorgis1/.conda/envs/robometer_qwen35_gpu/bin/python scripts/smoke_test_07_lora_targets.py
"""
import json
import os
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3_5ForConditionalGeneration

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISC_PATH = os.path.join(HERE, "configs", "_discovered_modules.json")
if not os.path.exists(DISC_PATH):
    sys.exit(f"FAIL: {DISC_PATH} missing — run smoke_test_04_module_names.py first")
with open(DISC_PATH) as f:
    disc = json.load(f)

# Targets discovered from layer 0 mlp + the common attention proj names.
discovered = list(disc.get("text_linear_basename_examples", {}).keys())
print(f"  discovered text-tower basenames: {discovered}")

ROBOMETER_DEFAULT = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _count_adapters(model_loader, targets):
    model = model_loader()
    cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=targets,
        lora_dropout=0.0,
        bias="none",
        task_type=None,
    )
    try:
        peft_model = get_peft_model(model, cfg)
    except ValueError as e:
        return None, str(e)
    n_lora = sum(1 for n, _ in peft_model.named_modules() if "lora_A" in n)
    return n_lora, None


def _load():
    return Qwen3_5ForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3.5-4B", torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cpu"
    )


print(f"  trying ROBOMETER_DEFAULT targets: {ROBOMETER_DEFAULT}")
n_default, err_default = _count_adapters(_load, ROBOMETER_DEFAULT)
print(f"    adapters wrapped = {n_default}, err = {err_default}")

print(f"  trying discovered text-tower targets: {discovered}")
n_disc, err_disc = _count_adapters(_load, discovered)
print(f"    adapters wrapped = {n_disc}, err = {err_disc}")

# Resolution heuristic: pick the larger non-None count and fail if both are None
chosen = None
if (n_default or 0) >= (n_disc or 0) and n_default:
    chosen = ROBOMETER_DEFAULT
elif n_disc:
    chosen = discovered
else:
    sys.exit("FAIL: neither preset wrapped any LoRA adapters")
print(f"  RECOMMENDED peft.target_modules: {chosen}")
print(f"  → set this in Qwen35-FT/configs/train_lora_base.yaml under peft.target_modules")
print("PASS")
