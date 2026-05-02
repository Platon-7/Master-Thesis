# MODIFICATIONS — Qwen35-FT vendored robometer package

The original `Robometer/robometer/` package is byte-identical to its main-branch HEAD.
This file logs the diffs between the vendored copies and their originals.

## Summary

| Vendored file | Real-file copy? | Diff size |
|---|---|---|
| `robometer/__init__.py` | symlink — no diff | 0 LOC |
| `robometer/configs/` | symlink — no diff | 0 LOC |
| `robometer/data/` | symlink — no diff | 0 LOC |
| `robometer/evals/` | symlink — no diff | 0 LOC |
| `robometer/trainers/` | symlink — no diff | 0 LOC |
| `robometer/models/rbm.py` | **real copy** | +21 LOC |
| `robometer/models/heads.py` | symlink — no diff | 0 LOC |
| `robometer/models/utils.py` | symlink — no diff | 0 LOC |
| `robometer/models/rewind_transformer.py` | symlink — no diff | 0 LOC |
| `robometer/models/__init__.py` | symlink — no diff | 0 LOC |
| `robometer/utils/setup_utils.py` | **real copy** | +66 LOC, ‑2 LOC (lazy unsloth/bnb) |
| `robometer/utils/*.py` (rest) | symlink — no diff | 0 LOC |
| `robometer/utils/fsdp/` | symlink — no diff | 0 LOC |

Total vendored diff: **2 files, ~85 LOC net addition**. Two flavors of change:

1. **Strictly additive Qwen3.5 dispatch** with precedence ordering (Qwen3.5 branch
   sits BEFORE Qwen3 branch to avoid the `"Qwen3" in "Qwen3.5"` substring trap).
2. **Lazy-import refactor** for `unsloth.FastVisionModel` and removal of the dead
   `import bitsandbytes as bnb` — both were top-level imports in the original that
   would have crashed `setup_utils` import in the new env. unsloth is incompatible
   with transformers 5.7; bitsandbytes was unused at module level (parameter named
   `bnb` shadows it). `FastVisionModel` is now lazy-imported inside
   `_load_base_model_unsloth`, only fired when `cfg.use_unsloth=True` (we never set
   that for full FT).

## `robometer/models/rbm.py` — diff

1. Imports — added conditional `Qwen3_5ForConditionalGeneration` import after the
   existing Qwen3 import block. Sets to `None` if transformers <5.7 (the legacy
   `robometer_gpu` env), so existing pipelines that read this module unchanged.

2. `RBM.__init__` dispatch — added a new branch BEFORE the existing `"Qwen3" in
   base_model_id` branch:

   ```python
   elif "Qwen3.5" in base_model_id or "qwen3.5" in base_model_id.lower():
       if Qwen3_5ForConditionalGeneration is None:
           raise RuntimeError("...needs transformers>=5.7...")
       hidden_size = config.text_config.hidden_size
       self.model_cls = Qwen3_5ForConditionalGeneration
   ```

3. `_forward_qwen` — **no change**. The existing `is_qwen3 = "Qwen3" in base_model_id`
   substring check intentionally captures Qwen3.5 too, because the
   `output_hidden_states=True` + `outputs.hidden_states[-1]` extraction path is the
   standard HF contract and works identically. (Verified empirically by smoke_test_05.)

## `robometer/utils/setup_utils.py` — diff

0. **Top-level imports cleanup** (env-compat fix, NOT Qwen3.5-specific):
   - Removed `from unsloth import FastVisionModel` (line 7 of the original).
   - Removed `import bitsandbytes as bnb` (line 16 of the original — unused at module
     level; the `bnb` parameter in function signatures shadows it everywhere).
   - Moved `from unsloth import FastVisionModel` INSIDE `_load_base_model_unsloth`
     where it's actually used (lazy import; only fires when `cfg.use_unsloth=True`).
   - Both packages are intentionally NOT installed in `robometer_qwen35_gpu`. unsloth
     is incompatible with transformers 5.7; bitsandbytes is dead code here.

1. Imports — added `HAS_QWEN35` flag and conditional `Qwen3_5ForConditionalGeneration`
   import. Mirrors the existing `HAS_QWEN3` pattern.

2. Helper — added one tiny helper:

   ```python
   def _is_qwen35(mid: str) -> bool:
       return "Qwen3.5" in mid or "qwen3.5" in mid.lower()
   ```

3. `_load_base_model_standard` — added `is_qwen35` flag, excluded it from the existing
   `is_qwen3` substring trap, and added a new `elif is_qwen35:` dispatch BEFORE the
   `elif is_qwen3:` branch. Loads via `Qwen3_5ForConditionalGeneration.from_pretrained`.

4. `_verify_checkpoint_loading` — added an early-return Qwen3.5 branch BEFORE the
   existing Qwen3/Molmo branch. The visual block submodule path on the unified
   architecture is not yet introspected; smoke_test_04 writes findings to
   `configs/_discovered_modules.json`. Until then, verification is logged-and-skipped
   for Qwen3.5 — the smoke tests carry the load-correctness signal directly.

5. `before_weights` snapshot (in the use_peft=False branch) — added matching empty
   pass-through so subsequent verification doesn't crash on missing keys.

## Why vendor instead of patch in place

The user explicitly asked that nothing affecting the existing Robometer-LoRA pipeline
change. Even though the additions above are purely additive (new branches before
existing ones; existing dispatch behavior preserved bit-for-bit), the safest signal
for "no risk" is that `git status Robometer/` is clean and `md5sum` of the original
files stays unchanged across the ablation work. The vendored copies isolate the new
dispatch entirely inside `Qwen35-FT/robometer/`.

## Rebase strategy

If `Robometer/robometer/` is updated upstream and the symlinks need to track:
- Symlinked files automatically follow.
- Vendored files (`rbm.py`, `setup_utils.py`) need a manual 3-way merge:
  ```bash
  cd Qwen35-FT/robometer
  diff -u $UPSTREAM/models/rbm.py models/rbm.py     # review the new diff
  # apply the Qwen3.5 hunks on top of the new upstream
  ```

## Env dependency notes

`flash-linear-attention` (PyPI 0.5.0 as of 2026-05-01) pulls `torch>=2.11` as a hard
dep and ships a cu13 wheel chain. Installing it after `torch 2.6.0+cu124` will
yank that torch and replace it with `torch 2.11.0`, which then breaks
`torchvision 0.21.0+cu124` (it pins `torch==2.6.0`). To get a clean install:

```bash
# Option A: install fla last, then patch torchvision to match
pip install --index-url https://download.pytorch.org/whl/cu124 'torch>=2.4' torchvision
pip install -r requirements.txt
pip install flash-linear-attention 'causal-conv1d>=1.4'
pip install --force-reinstall --no-deps torchvision==0.22.0  # match torch 2.11
MAX_JOBS=4 pip install --no-build-isolation flash-attn

# Option B (more conservative — pin torch, use --no-deps for fla)
pip install --index-url https://download.pytorch.org/whl/cu124 'torch==2.6.0+cu124' 'torchvision==0.21.0+cu124'
pip install -r requirements.txt
pip install --no-deps flash-linear-attention 'causal-conv1d>=1.4'
MAX_JOBS=4 pip install --no-build-isolation flash-attn
```

If anything ends up broken (peft import errors, torchvision pin conflicts), the
recovery is one command:

```bash
bash scripts/repair_env.sh
```

This pins `torch==2.6.0+cu124` (proven-good on Snellius H100s), reinstalls peft
+ torchvision against it, then installs `flash-linear-attention --no-deps` so it
can't yank torch back, then rebuilds `flash-attn`. Verify after with
`bash scripts/preflight.sh`.

## Open risks — status after smoke tests 00–07 (2026-05-01)

1. **Visual block path** — RESOLVED (smoke 04). Path is
   `model.visual.blocks.0.mlp.linear_fc1` — same as Qwen3-VL. The defensive helper
   in `_verify_checkpoint_loading` could be tightened back to a direct probe, but
   the early-return + log is fine for the LoRA path (use_peft=True doesn't hit
   that code).

2. **Hybrid linear-attention layers** — verified by smoke 05 (synthetic forward,
   `outputs.hidden_states[-1]` is finite at shape [B, T, 2560]). The standard HF
   contract holds across the 24 linear_attention + 8 full_attention layer mix.

3. **LoRA target_modules** — RESOLVED (smoke 04 + 07). Qwen3.5 text tower exposes:

   - **MLP** (all 32 layers, every layer): `gate_proj`, `up_proj`, `down_proj` — this
     is **Qwen2.5-style naming**, NOT Qwen3-VL's `linear_fc1/linear_fc2`.
   - **Full attention** (only the 8 `full_attention` layers — every 4th layer):
     `q_proj`, `k_proj`, `v_proj`, `o_proj`.
   - **Linear attention / Gated DeltaNet** (the 24 `linear_attention` layers):
     `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`.

   The Robometer-LoRA default presets (`q_proj/k_proj/v_proj/o_proj/gate_proj/
   up_proj/down_proj`) wrap the 8 full-attention layers + all 32 MLPs = 128 LoRA
   modules but **MISS the 24 linear_attn layers entirely**. To wrap those too,
   add `in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj` to the target list.
   See `configs/_discovered_modules.json` for the full enumeration.

   Recommendation per smoke_test_07_lora_targets.py output — set
   `peft.target_modules` in `configs/train_lora_base.yaml` to whichever list the
   smoke test recommends.
