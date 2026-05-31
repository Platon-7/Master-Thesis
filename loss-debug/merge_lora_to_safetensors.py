"""Merge a Robometer LoRA checkpoint (base_layer + lora_A + lora_B stored
separately in safetensors) into a clean full safetensors model that the
vlm_ibrl `robometer_ft` scorer can load.

Input  : a directory with model-XXXXX-of-N.safetensors + model.safetensors.index.json
         containing keys like `model.language_model.base_model.model.layers.K.
         self_attn.q_proj.{base_layer.weight,lora_A.default.weight,lora_B.default.weight}`
Output : the same directory layout but with merged `*.weight` keys (and
         `base_layer.weight`, `lora_A.default.weight`, `lora_B.default.weight`
         dropped) and `base_model.model.` stripped from the key prefix.

Merge formula:
    effective_weight = base_layer.weight + scaling * (lora_B.default.weight @ lora_A.default.weight)
    where scaling = lora_alpha / lora_r
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-r", type=int, default=32)
    return p.parse_args()


def strip_base_model(k: str) -> str:
    # keys like 'model.language_model.base_model.model.layers.0.X.weight'
    # → 'model.language_model.layers.0.X.weight'
    return k.replace("base_model.model.", "")


def main():
    args = parse_args()
    scaling = args.lora_alpha / args.lora_r
    print(f"[merge] in_dir = {args.in_dir}")
    print(f"[merge] out_dir = {args.out_dir}")
    print(f"[merge] scaling = {args.lora_alpha} / {args.lora_r} = {scaling}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load all shards into a single dict
    index_path = args.in_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    print(f"[merge] loading {len(shard_files)} shards…")
    all_tensors = {}
    for sf in shard_files:
        shard = load_file(str(args.in_dir / sf))
        all_tensors.update(shard)
    print(f"[merge] loaded {len(all_tensors)} tensors")

    # 2. Find all (base_layer, lora_A, lora_B) triples
    base_layer_keys = [k for k in all_tensors if k.endswith(".base_layer.weight")]
    print(f"[merge] found {len(base_layer_keys)} LoRA-wrapped layers to merge")

    merged = {}
    consumed = set()
    for bl_key in base_layer_keys:
        # build the matching keys
        prefix = bl_key[: -len(".base_layer.weight")]
        a_key = f"{prefix}.lora_A.default.weight"
        b_key = f"{prefix}.lora_B.default.weight"
        if a_key not in all_tensors or b_key not in all_tensors:
            print(f"[merge] WARNING: missing lora A/B for {prefix}; copying base only")
            merged_w = all_tensors[bl_key]
        else:
            base_w = all_tensors[bl_key].float()
            a_w = all_tensors[a_key].float()
            b_w = all_tensors[b_key].float()
            delta = (b_w @ a_w) * scaling
            merged_w = (base_w + delta).to(all_tensors[bl_key].dtype)
            consumed.update([a_key, b_key])
        consumed.add(bl_key)

        new_key = strip_base_model(prefix + ".weight")
        merged[new_key] = merged_w.contiguous()

    # 3. Copy all non-LoRA tensors (also strip base_model.model. prefix)
    for k, v in all_tensors.items():
        if k in consumed:
            continue
        # Any leftover lora_A/lora_B keys (shouldn't be, but just in case)
        if "lora_A" in k or "lora_B" in k:
            continue
        new_key = strip_base_model(k)
        if new_key in merged:
            print(f"[merge] WARNING: collision on {new_key}, keeping merged version")
            continue
        merged[new_key] = v.contiguous()

    print(f"[merge] output total {len(merged)} tensors")

    # 4. Save as a single shard (simpler), update index
    out_shard = "model.safetensors"
    save_file(merged, str(args.out_dir / out_shard))
    new_index = {
        "metadata": index.get("metadata", {}),
        "weight_map": {k: out_shard for k in merged},
    }
    (args.out_dir / "model.safetensors.index.json").write_text(json.dumps(new_index, indent=2))

    # 5. Copy config.json, tokenizer, processor configs from the source dir
    for fname in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "processor_config.json",
        "preprocessor_config.json",
        "training_config.yaml",
        "video_preprocessor_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ]:
        src = args.in_dir / fname
        if src.exists():
            shutil.copy(src, args.out_dir / fname)
            print(f"[merge] copied {fname}")

    print(f"[merge] DONE. Output at: {args.out_dir}")


if __name__ == "__main__":
    main()
