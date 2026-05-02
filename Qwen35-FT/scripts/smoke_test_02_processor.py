"""Smoke test 02 — AutoProcessor load + RBM special tokens.

Loads the Qwen3.5-4B processor and adds the 6 Robometer specials. Verifies each
gets a unique non-unk id and that none collide with the reserved vision delimiters.

Run:
    /home/pkarageorgis1/.conda/envs/robometer_qwen35_gpu/bin/python scripts/smoke_test_02_processor.py
"""
from transformers import AutoProcessor, AutoConfig

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen3.5-4B",
    trust_remote_code=False,
    do_sample_frames=False,
    padding_side="right",
)
print(f"  processor: {type(processor).__name__}")
tok = processor.tokenizer
print(f"  tokenizer: {type(tok).__name__}, vocab_size before adds = {len(tok)}")

specials = [
    "<|split_token|>",
    "<|reward_token|>",
    "<|pref_token|>",
    "<|sim_token|>",
    "<|prog_token|>",
    "<|demo_end|>",
]
added = tok.add_special_tokens({"additional_special_tokens": specials})
print(f"  added {added} new specials (some may already be reserved)")
print(f"  vocab_size after adds  = {len(tok)}")

unk_id = tok.unk_token_id
ids = {s: tok.convert_tokens_to_ids(s) for s in specials}
print(f"  ids: {ids}")

# Each token resolves to a real id (not unk) and they're all distinct.
seen = set()
for s, i in ids.items():
    assert i is not None and (unk_id is None or i != unk_id), f"{s} mapped to unk"
    assert i not in seen, f"duplicate id for {s}: {i}"
    seen.add(i)

cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-4B")
reserved = {cfg.image_token_id, cfg.video_token_id, cfg.vision_start_token_id, cfg.vision_end_token_id}
overlap = reserved & seen
assert not overlap, f"RBM specials collide with reserved vision ids: {overlap}"
print(f"  no overlap with reserved vision/image ids {reserved} OK")

# Round-trip sanity: tokenizing a string with one of our specials yields exactly 1 token.
for s in specials:
    rt = tok.encode(s, add_special_tokens=False)
    assert rt == [ids[s]], f"{s} not single-token after add: {rt}"
print(f"  single-token round-trip OK for all 6 specials")
print("PASS")
