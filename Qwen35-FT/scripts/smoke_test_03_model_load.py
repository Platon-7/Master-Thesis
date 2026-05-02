"""Smoke test 03 — Qwen3.5-4B weight load + token-embedding resize.

Heavy: downloads ~8GB of fp16 weights (cached after first run) and instantiates
Qwen3_5ForConditionalGeneration on CPU. Then adds the 6 RBM specials and confirms
resize_token_embeddings grows the vocab without explosion.

Time budget: 5–10 minutes on first run (download), <1 minute thereafter (cache hit).
Memory budget: ~16 GB RAM on CPU. If your login node OOMs, run via:
    sbatch jobs/smoke_tests.job

Run:
    /home/pkarageorgis1/.conda/envs/robometer_qwen35_gpu/bin/python scripts/smoke_test_03_model_load.py
"""
import os
import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

MODEL_ID = "Qwen/Qwen3.5-4B"

print(f"  loading {MODEL_ID} weights (fp16, CPU; first run downloads ~8GB)...")
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="cpu",
)
print(f"  loaded. dtype={model.dtype}, num params={sum(p.numel() for p in model.parameters()):,}")

processor = AutoProcessor.from_pretrained(MODEL_ID, do_sample_frames=False, padding_side="right")
tok = processor.tokenizer
specials = ["<|split_token|>", "<|reward_token|>", "<|pref_token|>", "<|sim_token|>", "<|prog_token|>", "<|demo_end|>"]
n_added = tok.add_special_tokens({"additional_special_tokens": specials})
print(f"  added {n_added} specials; new vocab size = {len(tok)}")

embed_layer = model.get_input_embeddings()
print(f"  embedding layer: {type(embed_layer).__name__}, weight.shape = {tuple(embed_layer.weight.shape)}")

before_n_embed = embed_layer.weight.shape[0]
model.resize_token_embeddings(len(tok))
after_n_embed = model.get_input_embeddings().weight.shape[0]
print(f"  embeddings resized: {before_n_embed} → {after_n_embed}")
assert after_n_embed == len(tok), f"resize mismatch: {after_n_embed} != {len(tok)}"

# Sanity: forward pass through the embedding layer for the new tokens shouldn't NaN.
ids = torch.tensor([tok.convert_tokens_to_ids(s) for s in specials], dtype=torch.long).unsqueeze(0)
with torch.no_grad():
    embed = model.get_input_embeddings()(ids)
assert torch.isfinite(embed).all(), "non-finite values in embeddings of new specials"
print(f"  new-special embeddings finite (shape {tuple(embed.shape)}) OK")
print("PASS")
