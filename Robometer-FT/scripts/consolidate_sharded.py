"""Consolidate an FSDP SHARDED_STATE_DICT checkpoint to HF safetensors format.

Pipeline use case: Stage 1 (pretrain) writes per-rank `pytorch_model_fsdp_0/*.distcp`
shards via FSDP SHARDED_STATE_DICT. Stage 2 (finetune) wants to load via the standard
HF path (`RBM.from_pretrained` / `model.safetensors`) so that the checkpoint also
works on a single GPU for downstream RL-loop inference, paper figure generation,
HF Hub upload, etc.

This script:
  1. Reads the SHARDED checkpoint at <ckpt_dir>/pytorch_model_fsdp_0/
  2. Materializes the full state_dict on CPU (via torch.distributed.checkpoint)
  3. Writes <out_dir>/model.safetensors (or sharded safetensors if >5GB)
  4. Copies tokenizer/processor/config files from <base_model_id> + <ckpt_dir>
  5. Writes a config.json describing the RBM head architecture so RBM.from_pretrained
     can reconstruct heads on load

Usage:
    python scripts/consolidate_sharded.py \\
        --ckpt-dir /projects/prjs1958/Qwen35_FT_weights/run4_icl_ours_NNNN/qwen35_ft_run4_icl_ours/checkpoint-6500 \\
        --base-model-id Qwen/Qwen3.5-4B \\
        --out-dir /projects/prjs1958/Qwen35_FT_consolidated/run4_step6500

Runs on a single CPU node. ~5-15 min depending on model size and disk speed.
No FSDP / no GPUs needed; the load step is just reading distcp shards into CPU memory.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoProcessor, AutoTokenizer


def load_sharded_state_dict(ckpt_dir: Path) -> Dict[str, torch.Tensor]:
    """Read the SHARDED FSDP checkpoint and return a flat CPU state_dict mapping
    parameter name → tensor.

    Uses torch.distributed.checkpoint to read .distcp shards without needing
    an actual distributed process group.
    """
    fsdp_dir = ckpt_dir / "pytorch_model_fsdp_0"
    if not fsdp_dir.is_dir():
        raise FileNotFoundError(
            f"Expected SHARDED FSDP checkpoint at {fsdp_dir}; not found. "
            f"If the checkpoint is already FULL_STATE_DICT (single model.safetensors), "
            f"you don't need to consolidate."
        )

    # Use the dcp-to-torch-save helper if available (newer torch).
    raw: Dict[str, Any]
    try:
        from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
        tmp_pt = ckpt_dir / "_consolidated_tmp.pt"
        dcp_to_torch_save(str(fsdp_dir), str(tmp_pt))
        raw = torch.load(tmp_pt, map_location="cpu", weights_only=True)
        tmp_pt.unlink(missing_ok=True)
    except ImportError:
        # Fallback for older torch: FileSystemReader + load.
        import torch.distributed.checkpoint as dcp
        raw = {}
        dcp.load(
            state_dict=raw,
            storage_reader=dcp.FileSystemReader(str(fsdp_dir)),
            no_dist=True,
        )

    # `dcp_to_torch_save` (or HF Trainer's FSDP save format) often wraps the
    # actual parameter dict under a single top-level key like "model",
    # "state_dict", or "app_state". Unwrap until we have a dict whose values
    # are all torch.Tensor.
    def _flatten(obj: Any, depth: int = 0) -> Dict[str, torch.Tensor]:
        if isinstance(obj, dict):
            if not obj:
                return {}
            # If every value is already a tensor, we're done.
            if all(isinstance(v, torch.Tensor) for v in obj.values()):
                return obj
            # If single key wrapping (e.g. {"model": {...}}), descend.
            if len(obj) == 1:
                return _flatten(next(iter(obj.values())), depth + 1)
            # Multiple top-level keys, some are tensors and some dicts —
            # flatten by prefixing the dict keys.
            out: Dict[str, torch.Tensor] = {}
            for k, v in obj.items():
                if isinstance(v, torch.Tensor):
                    out[k] = v
                elif isinstance(v, dict):
                    sub = _flatten(v, depth + 1)
                    for sk, sv in sub.items():
                        out[f"{k}.{sk}"] = sv
                # else: skip non-tensor / non-dict (e.g. trainer_state metadata)
            return out
        if isinstance(obj, torch.Tensor):
            # Unusual top-level — caller passed a single tensor. Skip.
            return {}
        return {}

    flat = _flatten(raw)
    if not flat:
        raise RuntimeError(
            f"Loaded state_dict from {fsdp_dir} but couldn't find any tensors after flattening. "
            f"Top-level keys: {list(raw.keys())[:5]}"
        )
    return flat


def strip_fsdp_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip the `_orig_mod.` prefix some FSDP wrap paths add. Also drop any
    `_fsdp_wrapped_module.` prefixes (older accelerate). Keep top-level names
    that match what RBM.__init__ expects (e.g., `model.embed_tokens.weight`,
    `progress_head.0.weight`, `success_head.0.weight`).
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new_k = k
        for prefix in ("_orig_mod.", "_fsdp_wrapped_module.", "module."):
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        out[new_k] = v
    return out


def save_safetensors(state_dict: Dict[str, torch.Tensor], out_dir: Path, max_shard_bytes: int = 5_000_000_000) -> None:
    """Write the state_dict to out_dir as a single model.safetensors, or sharded
    `model-NNNNN-of-MMMMM.safetensors` if total size exceeds max_shard_bytes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = sum(t.numel() * t.element_size() for t in state_dict.values())
    if total_bytes <= max_shard_bytes:
        save_file(state_dict, str(out_dir / "model.safetensors"))
        print(f"  wrote model.safetensors  ({total_bytes / 1e9:.2f} GB)")
        return

    # Shard by greedy bin-packing on tensor size.
    items = sorted(state_dict.items(), key=lambda kv: -kv[1].numel() * kv[1].element_size())
    shards: list[Dict[str, torch.Tensor]] = []
    shard_sizes: list[int] = []
    for k, v in items:
        size = v.numel() * v.element_size()
        placed = False
        for i, sh in enumerate(shards):
            if shard_sizes[i] + size <= max_shard_bytes:
                sh[k] = v
                shard_sizes[i] += size
                placed = True
                break
        if not placed:
            shards.append({k: v})
            shard_sizes.append(size)

    n = len(shards)
    weight_map: Dict[str, str] = {}
    for i, sh in enumerate(shards):
        fname = f"model-{i+1:05d}-of-{n:05d}.safetensors"
        save_file(sh, str(out_dir / fname))
        for k in sh:
            weight_map[k] = fname
        print(f"  wrote {fname}  ({shard_sizes[i] / 1e9:.2f} GB, {len(sh)} tensors)")

    # safetensors index — HF convention.
    index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    print(f"  wrote model.safetensors.index.json  ({n} shards, {total_bytes / 1e9:.2f} GB total)")


def copy_aux_files(base_model_id: str, ckpt_dir: Path, out_dir: Path) -> None:
    """Copy / regenerate the auxiliary files needed for HF-style loading:
        - config.json   (from the base model, augmented with RBM head metadata)
        - tokenizer / processor files (from the base model, but with the 6 RBM special tokens added)
        - generation_config.json (from the base model)
        - any *.json metadata the trainer wrote alongside the FSDP shards (e.g., trainer_state.json)
    """
    # Base model config + processor — pull from HF cache or the base id directly.
    print(f"  loading auxiliary files from base model: {base_model_id}")
    base_config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)

    # The RBM class adds 6 special tokens via resize_token_embeddings. Pull the
    # processor with those tokens already added if a tokenizer/processor was
    # saved alongside the shards (newer HF trainer behavior); otherwise add them
    # explicitly so downstream RBM.from_pretrained finds them in the vocab.
    tokenizer_files_in_ckpt = list(ckpt_dir.glob("tokenizer*.json")) + list(ckpt_dir.glob("special_tokens_map.json"))
    if tokenizer_files_in_ckpt:
        for f in tokenizer_files_in_ckpt:
            shutil.copy2(f, out_dir / f.name)
        print(f"  copied {len(tokenizer_files_in_ckpt)} tokenizer files from checkpoint dir")
    else:
        # Pull from base model and add the RBM special tokens.
        processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
        tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        rbm_special_tokens = [
            "<|prog_token|>", "<|pref_token|>", "<|split_token|>",
            "<|reward_token|>", "<|sim_token|>", "<|demo_end|>",
        ]
        tok.add_special_tokens({"additional_special_tokens": rbm_special_tokens})
        processor.save_pretrained(str(out_dir))
        print(f"  saved processor with 6 RBM special tokens added (vocab size {len(tok)})")

    # Save the config with `architectures: ['RBM']` so AutoConfig knows to use the
    # vendored RBM class. The RBM head metadata (which heads are present, head dims)
    # should already be in the original training config dump — copy it if it exists.
    cfg_dict = base_config.to_dict()
    cfg_dict["architectures"] = ["RBM"]

    # If the trainer saved a config.yaml alongside the shards, include head info.
    # Robometer's load_model_from_hf (utils/save.py:903) requires `config.yaml`
    # in the consolidated dir, NOT `training_config.yaml`. Write both names so
    # the consolidated checkpoint is directly loadable while keeping the alias
    # for any downstream code that expected the old name.
    training_config_yaml = ckpt_dir.parent / "config.yaml"
    if training_config_yaml.is_file():
        shutil.copy2(training_config_yaml, out_dir / "training_config.yaml")
        shutil.copy2(training_config_yaml, out_dir / "config.yaml")
        print(f"  copied training config.yaml (also as config.yaml so "
              f"load_model_from_hf can find it)")

    (out_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2))
    print(f"  wrote config.json  (architectures=['RBM'])")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", type=Path, required=True,
                    help="Path to FSDP checkpoint dir (contains pytorch_model_fsdp_0/, optimizer_0/, etc.)")
    ap.add_argument("--base-model-id", type=str, required=True,
                    help="HF id or local path to base model for aux files (e.g., Qwen/Qwen3.5-4B or Qwen/Qwen3-VL-4B-Instruct)")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output dir for consolidated HF-format checkpoint")
    ap.add_argument("--shard-bytes", type=int, default=5_000_000_000,
                    help="Max bytes per safetensors shard (default 5 GB)")
    args = ap.parse_args()

    print(f"=" * 60)
    print(f"Consolidating SHARDED checkpoint")
    print(f"  ckpt_dir   : {args.ckpt_dir}")
    print(f"  base_model : {args.base_model_id}")
    print(f"  out_dir    : {args.out_dir}")
    print(f"=" * 60)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] reading sharded state_dict from .distcp shards …")
    state_dict = load_sharded_state_dict(args.ckpt_dir)
    print(f"  loaded {len(state_dict)} tensors")

    print("\n[2/3] stripping FSDP wrappers + writing safetensors …")
    state_dict = strip_fsdp_prefix(state_dict)
    save_safetensors(state_dict, args.out_dir, max_shard_bytes=args.shard_bytes)

    print("\n[3/3] copying aux files (config, tokenizer, processor) …")
    copy_aux_files(args.base_model_id, args.ckpt_dir, args.out_dir)

    print(f"\n{'=' * 60}")
    print(f"DONE → {args.out_dir}")
    print(f"  Load downstream with:")
    print(f"    from robometer.models.rbm import RBM")
    print(f"    model = RBM.from_pretrained('{args.out_dir}')")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
