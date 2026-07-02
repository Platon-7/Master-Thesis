"""Instrument the Qwen3.5 scoring path: load via get_robometer_4b, score a real
success clip, and INSPECT the collated model input — are <|prog_token|> markers
inserted (one per frame, what the heads key off)? Pinpoints whether the breakage
is input-construction (prog tokens missing) or head extraction (present but dead).
"""
import os, sys, glob
import numpy as np

_REPO = os.environ.get("MT_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for sub in ("vlm_ibrl_v3", "Robometer"):
    sys.path.insert(0, os.path.join(_REPO, sub))
from env.robometer_utils import get_robometer_4b
from PIL import Image

CKPT = os.environ["QWEN35_FT_PATH"]
TASK = "Push a mug under a coffee machine."
FRAMES = ("/shared/home/PKA4388/Master-Thesis/vlm_ibrl/release/data/metaworld/"
          "CoffeePush_frame_stack_1_224x224_end_on_success/demonstrations/mw-coffee-push/frames")

sc = get_robometer_4b(model_path=CKPT)
print(f"[diag] max_frames={sc.max_frames}", flush=True)

# 1) collator flags actually in effect
bc = sc.batch_collator
for attr in ("use_multi_image", "use_per_frame_progress_token", "inference", "base_model_id"):
    print(f"[diag] collator.{attr} = {getattr(bc, attr, '<missing>')}", flush=True)

# 2) prog_token id
prog_id = sc.tokenizer.convert_tokens_to_ids("<|prog_token|>")
print(f"[diag] <|prog_token|> id = {prog_id}", flush=True)

# 3) capture the collated batch by wrapping the collator
import torch
captured = {}
def _scan(obj, prog_id, path=""):
    """Recursively find tensors and count prog_id occurrences; record structure."""
    hits = []
    if isinstance(obj, torch.Tensor):
        try:
            n = int((obj == prog_id).sum().item())
            captured.setdefault("tensors", []).append((path, tuple(obj.shape), n))
            if n > 0: hits.append((path, tuple(obj.shape), n))
        except Exception: pass
    elif isinstance(obj, dict) or hasattr(obj, "items"):   # dict OR BatchFeature
        try:
            for kk, vv in obj.items(): hits += _scan(vv, prog_id, f"{path}.{kk}")
        except Exception: pass
    elif isinstance(obj, (list, tuple)):
        for i, vv in enumerate(obj[:6]): hits += _scan(vv, prog_id, f"{path}[{i}]")
    return hits

_orig = sc.batch_collator
PROG_ID = sc.tokenizer.convert_tokens_to_ids("<|prog_token|>")
class _Wrap:
    def __init__(self, c): self._c = c
    def __getattr__(self, n): return getattr(self._c, n)
    def __call__(self, *a, **k):
        out = self._c(*a, **k)
        try:
            captured["prog_hits"] = _scan(out, PROG_ID)
            captured["top_keys"] = list(out.keys()) if isinstance(out, dict) else type(out).__name__
        except Exception as e: captured["err"] = str(e)
        return out
sc.batch_collator = _Wrap(_orig)

# 3b) are the head WEIGHTS actually loaded? (random/zero init -> dead head)
try:
    m = sc.model
    base = m.base_model.model if hasattr(m, "base_model") else m
    for hname in ("success_head", "progress_head"):
        head = getattr(base, hname, None) or getattr(m, hname, None)
        if head is not None:
            ws = [(tuple(p.shape), float(p.norm()), float(p.abs().mean())) for p in head.parameters()]
            print(f"[diag] {hname} param norms/means: {ws}", flush=True)
        else:
            print(f"[diag] {hname}: NOT FOUND on model", flush=True)
except Exception as e:
    print(f"[diag] head-weight check err: {e}", flush=True)

# 4) score a real success clip (first 29 frames -> through coffeepush success at ~28)
files = sorted(glob.glob(os.path.join(FRAMES, "0_*.png")))[:29]
clip = [Image.open(f).convert("RGB") for f in files]
out = sc(clip, task=TASK, icl_frames=None, detailed=True)
print(f"\n[diag] success clip ({len(clip)} frames) -> success_prob={out['success_prob']:.4f} progress={out['progress_reward']:.4f}", flush=True)
spf = out.get("success_probs_per_frame"); pgf = out.get("progress_per_frame")
print(f"[diag] per-frame success_probs: {[round(float(x),3) for x in (spf or [])]}", flush=True)
print(f"[diag] per-frame progress:      {[round(float(x),3) for x in (pgf or [])]}", flush=True)

# 5) how many prog_tokens ended up in the model input?
print(f"[diag] collator output top-level: {captured.get('top_keys')}", flush=True)
print(f"[diag] tensors seen (path, shape, #prog): {captured.get('tensors')}", flush=True)
hits = captured.get("prog_hits") or []
total_prog = sum(h[2] for h in hits)
print(f"[diag] total <|prog_token|> in model input = {total_prog}  (expect ~{sc.max_frames})", flush=True)
print(f"[diag] VERDICT: {'prog tokens MISSING/too few -> input-construction bug (scoring path does not insert per-frame prog tokens)' if total_prog < 2 else 'prog tokens present -> breakage is head extraction/features for Qwen3.5 layout, not token insertion'}", flush=True)
if captured.get("err"): print(f"[diag] capture err: {captured['err']}", flush=True)
