"""Smoke test 05 — synthetic forward pass.

Builds a tiny [1, 10] input_ids tensor (text-only — no vision yet) and runs forward
through Qwen3_5ForConditionalGeneration with output_hidden_states=True. Confirms:
    - hidden_states tuple is exposed (the path the inner _forward_qwen relies on)
    - last layer hidden has shape [batch, seq, hidden_size]
    - values are finite

GPU required when `flash-linear-attention` (fla) is installed in the env — fla's
Triton kernels for the gated-DeltaNet linear-attention layers can't operate on CPU
tensors. We auto-detect CUDA; falls back to CPU only if fla is NOT importable.

Run AFTER smoke_test_03 has cached the weights:
    /home/pkarageorgis1/.conda/envs/robometer_qwen35_gpu/bin/python scripts/smoke_test_05_forward.py
"""
import sys
import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

MODEL_ID = "Qwen/Qwen3.5-4B"

# Decide device: fla forces GPU; otherwise CPU is fine.
try:
    import fla  # noqa: F401
    fla_present = True
except ImportError:
    fla_present = False

if torch.cuda.is_available():
    device_map = "cuda"
    where = "GPU (cuda)"
elif not fla_present:
    device_map = "cpu"
    where = "CPU (no fla)"
else:
    sys.exit("FAIL: fla is installed but no CUDA — smoke 05 needs GPU. Submit via SLURM.")
print(f"  device: {where}")

print(f"  loading {MODEL_ID} (fp16, {where})")
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=device_map
)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_ID, do_sample_frames=False, padding_side="right")
tok = processor.tokenizer
tok.add_special_tokens({"additional_special_tokens":
    ["<|split_token|>", "<|reward_token|>", "<|pref_token|>", "<|sim_token|>", "<|prog_token|>", "<|demo_end|>"]
})
model.resize_token_embeddings(len(tok))

prompt = "Score the progress of the following frame: <|prog_token|>"
input_ids = tok(prompt, return_tensors="pt").input_ids
if device_map == "cuda":
    input_ids = input_ids.to("cuda")
print(f"  input_ids shape = {tuple(input_ids.shape)}, device = {input_ids.device}")

with torch.no_grad():
    outputs = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)

assert hasattr(outputs, "hidden_states") and outputs.hidden_states is not None, \
    "hidden_states not in output (must be a tuple of per-layer states)"
hs = outputs.hidden_states
print(f"  hidden_states tuple len = {len(hs)} (expected 33 = embed + 32 layers)")

last = hs[-1]
print(f"  hidden_states[-1] shape = {tuple(last.shape)}, dtype = {last.dtype}, device = {last.device}")
assert last.shape[0] == 1
assert last.shape[1] == input_ids.shape[1]
assert last.shape[2] == 2560, f"expected hidden_size 2560, got {last.shape[2]}"
assert torch.isfinite(last).all(), "non-finite values in last-layer hidden states"
print("  finite values OK")
print("PASS")
