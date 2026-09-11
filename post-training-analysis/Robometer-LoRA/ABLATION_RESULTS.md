# LoRA Bake-off — Ablation Log

Running log of every LoRA fine-tuning ablation. Each entry records the recipe, the
headline test-set numbers, and any **known caveats** that limit how much you can trust
the result. Append a new section at the top of `## Runs` for every new ablation.

---

## Setup (constant across runs unless noted)

| | |
|---|---|
| **Backbone** | `robometer/Robometer-4B` (Qwen3-VL-4B fine-tune) |
| **LoRA** | rank 32, alpha 64, dropout 0.05; q/k/v/o + gate/up/down |
| **Frozen** | backbone weights; LoRA + heads trained |
| **Precision** | bf16 forward, fp32 LoRA + heads |
| **Optimizer** | AdamW, lr 1e-4 (adapters) / 5e-5 (heads), weight decay 0.01 |
| **Schedule** | linear warmup 5% → cosine to 10% peak |
| **Steps / batch** | 7,500 steps × per-device batch 8 × grad-accum 2 |
| **Hardware** | 1× H100, ~62 h wall time per run |
| **Test set** | held-out, 3,503 trajectories (Group A only — see caveats) |

---

## Runs

### Loss 1 · asymmetric CORN  · c = 1.5
**Job**: `22244008` · **Date**: 2026-04-25 → 04-27 · **Steps**: 7,500

| Setup |  |
|---|---|
| Progress head | fresh 4-logit CORN (random-init; output dim ≠ pretrained C51's 10 bins) |
| Success head | disabled |
| Asymmetry | α_k = 1 + 1.5·(k−2) → (1, 2.5, 4, 5.5) for k = 2..5; β_k = 1 |
| Data warmup | 2,000 failure-only steps, then 50/50 stratified |
| ICL | `use_icl=true, icl_prob=0.5` — but see "Self-demo bug" caveat |

**Test-set headline** (n = 602 paired Group A trajectories, last-frame `σ(z_5)` as P(success)):

| Metric | Loss 1 |
|---|---:|
| ROC-AUC | 0.651 |
| FPR @ τ = 0.5 | 0.000 |
| ECE | 0.026 |
| Recall @ τ = 0.5 | 0% (head capped at σ ≈ 0.258) |
| Recall @ FPR = 5% | 6% |

**Read**: ranks poorly. Output range collapsed to [0.0007, 0.258] — model is over-conservative
to the point of never predicting "success". Fresh head + aggressive asymmetry + only
~7,500 LoRA steps aren't enough to push z_5 positive. AUC near chance (0.65).

---

### Loss 2 · asymmetric C51 + asymmetric BCE  · λ = 0.3
**Job**: `22244009` · **Date**: 2026-04-25 → 04-27 · **Steps**: 7,500

| Setup |  |
|---|---|
| Progress head | pretrained C51 × 10 bins (loaded from Robometer-4B release, fine-tuned) |
| Success head | pretrained binary BCE (loaded from release, fine-tuned) |
| Asymmetry (progress) | full weight on overestimation; λ on underestimation |
| Asymmetry (success) | `BCE_neg + λ · BCE_pos`, λ = 0.3 |
| Data warmup | 1,000 failure-only steps, then 50/50 stratified |
| ICL | `use_icl=true, icl_prob=0.5` — but see "Self-demo bug" caveat |

**Test-set headline** (n = 602 Group A, last-frame `σ(success_logit)` as P(success)):

| Metric | Loss 2 |
|---|---:|
| ROC-AUC | 0.783 |
| FPR @ τ = 0.5 | 0.016 |
| ECE | 0.020 |
| Recall @ τ = 0.5 | 61% |
| Recall @ FPR = 5% | 62% |

**Read**: deployable. Calibrated, conservative-but-useful. Best of the three (vs.
unsafe baseline AUC 0.855 / FPR 0.195, vs. broken Loss 1 AUC 0.651 / recall 0). Picked
as the winner of the first bake-off.

---

## Reference: pretrained baseline

For comparison context (no LoRA fine-tune; published `robometer/Robometer-4B` weights
evaluated on the same test set, same head): ROC-AUC 0.855, FPR @ τ=0.5 0.195, ECE 0.155.
Best raw ranker but unsafe at the standard threshold (19.5% false positives on real
failures).

---

## Caveats — affecting all runs above

These all postdate the bake-off. Strictly speaking, **the numbers above describe what
those specific checkpoints do, but not what the candidate losses themselves can do**.
Until we re-run with the fixes below, treat these as preliminary.

### 1. Self-demo bug (CRITICAL)
Found 2026-04-30. Every ICL training sample (and every ICL eval sample) had its **own
query trajectory** loaded as the "successful demonstration", not the partner declared
in the pair index. Root cause: pair indices have no `partner_frames_path` field, and
`_load_partner_trajectory` silently fell back to `frames_path` (the query's).

**Effect**: ICL was tautological. The model saw `[query]<|demo_end|>[query]` framed as
"Successful demonstration: …  Evaluate this trajectory: …" — same trajectory twice,
labeled success on the first half regardless of whether the query was actually a success.
For ~25% of training steps (the effective ICL rate after the drift bug below), the model
was trained on self-demos. None of the bake-off models has actually learned to use a
real partner demo.

**Fix**: `Robometer/robometer/data/samplers/base.py` now resolves `partner_frames_path`
from `pairs_unified.jsonl` at `_load_pair_index` time and requires it (no fallback).
Verified empirically — demo bytes are no longer query bytes after the patch.

### 2. HF / pair-index drift (CRITICAL)
Train HF Arrow dataset built from Apr-24 `train.jsonl`; `pairs_index_train.jsonl`
rebuilt Apr-25 with different IDs. Only ~50% of HF train rows had a matching partner
in the index. Combined with `icl_prob = 0.5`: effective ICL-on rate ≈ 25%, not 50%.

**Fix**: train HF cache rebuilt Apr-30 to match the current pair index (coverage
53.1% → 100.0%). New `data.icl_min_coverage` config field in
`Robometer/robometer/configs/experiment_configs.py` (default 0.40) hard-fails at
startup if drift returns.

### 3. Missing data sources
Group B robometer failures (3 archives, 29,509 failures) and the entire roboreward
family (29 archives, 15,860 failures + 7,313 successes) were not in the training pool
for the bake-off. They were either absent from `pairs_unified.jsonl` or absent from
the per-split index files at the time of the run.

**Fix**: `build_unified_pairs.py` and `build_shard_indices.py` extended to include
roboreward; `pairs_unified.jsonl` regenerated 2026-04-30 (628,020 → 651,193 rows);
`build_splits.py` re-run, splits now include Group B + roboreward.

### 4. Test set is stale
`pairs_index_test.jsonl` (3,503 trajectories) was built 2026-04-28 from a pre-Group-B
pairs_unified. **0 Group B / 0 roboreward in the test set.** Pure Group A. Future runs
should regenerate via `build_test_set.py` to evaluate on the full distribution.

### 5. ICL inference results are uninformative for the bake-off models
Even after fixing (1) and (2), the existing checkpoints can only be re-evaluated, not
retrained. They were trained with self-demos, so their ICL-time behavior is whatever
they extrapolate from a configuration they never saw in training. Any ICL-on test
numbers from these checkpoints reflect that policy mismatch, not the value of demos.

### 6. Eval-time failsafe sample was tiny
The `policy_ranking` eval sampler emitted ~30 failsafe trajectories per round (5
per-quality × 2 qualities × 3 tasks). Failsafe Kendall noise dominated the early-stop
signal. Bumped to ~500 starting next round (see ablation #3 below when it lands).

---

## Status of the next-round prerequisites

Before launching the next 5-LoRA round:
- [x] Self-demo bug fixed (`_load_partner_trajectory` requires `partner_frames_path`).
- [x] HF/pair-index drift caught (sampler asserts `icl_min_coverage` ≥ 0.40 at startup).
- [x] Roboreward integrated into `pairs_unified.jsonl` (now 6 families).
- [x] Per-split pair indices regenerated (Group B + roboreward propagated).
- [ ] HF Arrow caches need rebuild for the new pair indices (`SPLIT=<name> sbatch jobs/preprocess_split.job`). Otherwise the drift sniff fires.
- [ ] Test set regeneration decision (keep stable Group A test set vs. rebuild with Group B + roboreward).
- [ ] Eval-time failsafe sample bump (target ≥ 500 trajectories per round).
- [ ] Dual eval (ICL-on + ICL-off) implementation — design open.
