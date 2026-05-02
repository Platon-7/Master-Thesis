"""Smoke test 01 — Qwen3.5-4B AutoConfig load.

Pulls just the config.json from HF (no weights). Confirms the architectural facts the
plan was anchored on (verified 2026-05-01 from the live config.json):

    model_type             = qwen3_5
    text_config.hidden_size = 2560
    vision_config.hidden_size = 1024
    text_config.num_hidden_layers = 32
    vision_config.depth    = 24
    layer_types alternation: 3 linear_attention + 1 full_attention per group of 4.

Run:
    /home/pkarageorgis1/.conda/envs/robometer_qwen35_gpu/bin/python scripts/smoke_test_01_config.py
"""
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=False)

print(f"  model_type = {cfg.model_type}")
print(f"  text_config.hidden_size = {cfg.text_config.hidden_size}")
print(f"  text_config.num_hidden_layers = {cfg.text_config.num_hidden_layers}")
print(f"  vision_config.hidden_size = {cfg.vision_config.hidden_size}")
print(f"  vision_config.depth = {cfg.vision_config.depth}")

assert cfg.model_type == "qwen3_5", f"unexpected model_type: {cfg.model_type}"
assert cfg.text_config.hidden_size == 2560
assert cfg.text_config.num_hidden_layers == 32
assert cfg.vision_config.hidden_size == 1024
assert cfg.vision_config.depth == 24

# Hybrid attention layout: 32 entries, 3 linear + 1 full per 4-block group.
lt = cfg.text_config.layer_types
assert len(lt) == 32, f"layer_types length {len(lt)} != 32"
assert lt.count("linear_attention") == 24 and lt.count("full_attention") == 8, (
    f"unexpected layer_types distribution: {lt}"
)
print(f"  layer_types: 24 linear + 8 full (every 4th is full) OK")

# Special token ids the model already reserves — these must NOT collide with our
# 6 RBM specials (verified empirically in smoke_test_02).
print(f"  image_token_id  = {cfg.image_token_id}")
print(f"  video_token_id  = {cfg.video_token_id}")
print(f"  vision_start_id = {cfg.vision_start_token_id}")
print(f"  vision_end_id   = {cfg.vision_end_token_id}")

print("PASS")
