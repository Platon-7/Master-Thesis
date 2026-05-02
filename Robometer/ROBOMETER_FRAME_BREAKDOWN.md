# Robometer Frame Dataset — Full Breakdown

**Single source of truth** for the unified frame dataset at
`/projects/prjs1958/robometer_frame_dataset/`.

Generated from `pairs_unified.jsonl` (the master pairing file) and the per-source
manifests/scores. All counts are post-roboreward integration (Apr-30 audit, after the
ICL-debug audit found roboreward was missing from the unified pairing pipeline).

For the *upstream* archive-level audit (the 93-archive `robometer_full_dataset/` scan
that classifies humanoid / human-hand / standard-arm), see the sibling document
`Real-World-Failures/Robometer/ROBOMETER_BREAKDOWN.md`.

---

## 1. Totals

| Metric | Count |
|---|---:|
| Total rows in `pairs_unified.jsonl` | **651,193** |
| Sources | **6 distinct family-groups** |
| Total partnered rows (have a demo) | ~595k (rough — partnered episodes plus their demo-side successes) |

The dataset lives at:

```
/projects/prjs1958/robometer_frame_dataset/
  pairs_unified.jsonl           ← THIS file (master pairings, 525 MB)
  pairs_unified_report.json     ← summary stats (regenerated alongside the JSONL)
  metaworld/  failsafe/  droid/  robometer/  roboreward/  stray_files/
```

Each per-source directory contains:

```
<source>/
  keyframes/<archive>/shard-NNNNN.tar          ← failure trajectory shards
  keyframes_success/<archive>/shard-NNNNN.tar  ← success trajectory shards
  manifests/<archive>_{failures,successes}.jsonl
  scores/<archive>_scored.jsonl                ← per-frame rubric labels (1..5)
  vlm_descriptions/                            ← Stage-1 VLM caption JSONLs
  shard_index.json                             ← episode_id → shard_name lookup
                                                  (per-archive; required by build_unified_pairs)
```

DROID is special: it splits views into separate top-level dirs
(`droid/keyframes`, `droid/keyframes_ext2`, `droid/keyframes_wrist`) and its shard
indices live under `<dir>/shards/shard_index.json`.

---

## 2. Per-source breakdown

Failures get paired with same-task **success** demos (failure→success). Successes
get paired with another same-(source, task) **success** demo when at least one such
demo exists (success→success); otherwise they stay `no_pair`.

| # | Source | Total | Failures | Failures paired | Successes | Successes paired | Archives |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | **droid** | 11,366 | 5,503 | 4,889 (89%) | 5,863 | 5,782 (99%) | 13 |
| 2 | **roboreward** | 23,173 | 15,860 | 15,860 (100%) | 7,313 | 6,284 (86%) | 29 |
| 3 | **failsafe** | 1,376 | 1,301 | 1,301 (100%) | 75 | 75 (100%) | 3 |
| 4 | **metaworld** | 29,528 | 27,478 | 26,878 (98%) | 2,050 | 2,050 (100%) | 42 |
| 5 | **robometer (Group A)** | 58,011 | 39,414 | 39,351 (99.8%) | 18,597 | 18,596 (100%) | 26 |
| 6 | **robometer (Group B)** | 36,941 | 29,509 | 18,001 (61%) | 7,432 | 7,407 (100%) | 3 |
| 7 | **robometer_orphan_success** | 490,798 | 0 | n/a | 490,798 | 453,706 (92%) | 29 |
| | **TOTAL** | **651,193** | **119,065** | **106,280 (89%)** | **532,128** | **493,900 (93%)** | |

**Pairing rates after the 2026-04-30 success-pairing pass**: 92% of all rows
(600,180 / 651,193) carry a non-null `partner_episode_id`. The remaining 51,013
unpaired rows are either (a) failures whose task has no matching success anywhere
in the dataset, or (b) successes that are the only success of their (source, task)
group — singletons that can't be paired against another same-task success.

Notes:
- *robometer_orphan_success* are success-only OXE/Robo-Set/etc. archives. They get
  success→success pairings from `robometer/pairs_orphan/*_orphan_pairs.jsonl` (built
  upstream); the post-pass adds same-source same-task pairings on top for any that
  remained unpaired.
- *Group A* vs *Group B* are both `source=="robometer"` but live in different archive
  sets. Group B is the 3 split-archive failure dumps (`soar`, `roboarena_0825`,
  `roboarena_eval_debug_nowrist`). They share one source tag in `pairs_unified`;
  the per-archive table below separates them.

### Group B archives (failure-only side)

| Archive | Failures | Successes |
|---|---:|---:|
| `jesbu1_soar_rfm_soar_rfm` | 11,999 | 4,797 |
| `jesbu1_roboarena_0825_rfm_roboarena` | 10,753 | 1,626 |
| `jesbu1_roboarena_eval_debug_nowrist_roboarena_eval_debug_nowrist` | 6,757 | 1,009 |

(Successes are inside the same archives — these are mixed failure+success archives.)

---

## 3. Pairing tiers

Each partnered row carries a `tier` field describing how the demo was matched.

Tier semantics (from `build_unified_pairs.py`):
- `1_exact` — DROID/roboreward only: the partner is the exact paired success episode
  declared by the source manifest (`paired_success_id`).
- `1_same_task_fresh` — robometer-style: same task, partner not yet reused for any
  other failure (highest-fidelity match without an exact pairing).  Also used as the
  synthetic tier for the **success → success** post-pass (any same-source same-task
  success picked uniformly at random).
- `2_same_task_family_fresh` — same task family, fresh partner.
- `2_same_scene` — DROID-only: same scene as the failure (looser).
- `3_same_task` / `3_same_task_reused` — same task but the partner has been used as a
  demo for multiple failures across the dataset.
- `same_task` — metaworld/failsafe simulator matches (no fresh-vs-reused distinction;
  every failure in a sim task has an enumerated success demo).
- `no_pair` — no eligible partner; the row is in `pairs_unified.jsonl` for
  completeness but cannot be used as an ICL query.

The success → success post-pass (Apr-30) added 40,196 new `1_same_task_fresh` rows
across non-orphan sources that previously had `no_pair` on the success side.

---

## 4. Build pipeline

The unified file is regenerated end-to-end from the raw archives via three scripts at
the repo root:

```
build_shard_indices.py     # one-time per source: build  <archive>/shard_index.json
                           # — episode_id → shard_name lookup, required by next step

build_unified_pairs.py     # streams every source's manifests + scores + the external
                           # robometer pair-file, emits one row per episode with
                           # resolved frames_path and partner info → pairs_unified.jsonl

Robometer-LoRA/scripts/build_splits.py   # samples per-split pair indices for the LoRA
                                          # bake-off (train, warmup, eval_*); does NOT
                                          # touch pairs_unified.jsonl
```

When adding a new source:
1. Add it to `SIMPLE_SOURCES` (or `DROID_VIEW_DIRS`) in `build_shard_indices.py`.
2. Add a `build_<source>()` generator in `build_unified_pairs.py` and wire it into
   `main()` via `emit(build_<source>(), "<source>")`.
3. Re-run both scripts in order.
4. Re-run `build_splits.py` so the per-split index files inherit the new rows.
5. Re-run `Robometer-LoRA/jobs/preprocess_split.job` for every affected split so the
   HF Arrow caches at `/projects/prjs1958/robometer_frames_hf/<split>/` are regenerated.

If you skip step 5, the sampler's `data.icl_min_coverage` drift sniff (default 0.40)
will fire at startup because the HF dataset and pair index will reference disjoint
episode-id sets. See `Robometer/robometer/data/samplers/base.py:_assert_icl_coverage`.

---

## 5. ICL plumbing (downstream consumers)

The pair index is consumed by:

```
Robometer/robometer/data/samplers/base.py
  _load_pair_index           ← loads pairs_index_<split>.jsonl AND resolves every
                               partner_episode_id → frames_path via pairs_unified.jsonl
                               (this resolution step was added Apr-30 to fix the
                               self-demo bug; before that, _load_partner_trajectory
                               silently fell back to the query's frames_path).
  _load_partner_trajectory   ← REQUIRES partner_frames_path on the row (no fallback).
  _maybe_attach_icl_context  ← per-sample Bernoulli(icl_prob) decides whether to
                               attach the partner trajectory.
```

The training collator (`Robometer/robometer/data/collators/rbm_heads.py`) then builds
a 32-image prompt: `[16 demo frames]<|demo_end|>[16 query frames]` with a
`<|prog_token|>` after every image. Per-frame labels stay query-only (`[B, 16]`).

Eval samplers (`Robometer/robometer/data/samplers/eval/*.py`) also call
`_maybe_attach_icl_context` so eval-time ICL fires identically to training.

---

## 6. Per-split sample sizes (LoRA)

These are the **derived subsets** at `/scratch-shared/$USER/robometer_frames_splits/`,
sampled from `pairs_unified.jsonl` by `build_splits.py` and `build_test_set.py`.
They are NOT part of the master dataset and are regenerated from scratch on each
re-run.

Counts after the Apr-30 success-pairing pass (post `build_unified_pairs.py` rebuild
+ `build_splits.py` + `build_test_set.py` re-run):

| Split | Total | Failures (paired) | Successes (paired) | % paired |
|---|---:|---:|---:|---:|
| train          | 18,000 | 9,000 (8,343) | 9,000 (9,000) | 96.3% |
| warmup         | 1,500  | 1,500 (1,260) | 0 | 84.0% |
| test           | 3,492  | 2,102 (2,092) | 1,390 (1,390) | **99.7%** |
| eval_droid     | 1,125  | 565 (565)     | 560 (476)     | 92.5% |
| eval_robometer | 2,521  | 391 (336)     | 2,130 (2,127) | 97.7% |
| eval_metaworld | 573    | 466 (466)     | 107 (107)     | **100%** |
| eval_failsafe  | 552    | 477 (477)     | 75 (75)       | **100%** |

Notes:
- **eval_metaworld and eval_failsafe are now 100% paired** — required for clean
  dual-ICL eval (every sampled trajectory has a demo for the icl-on pass).
- Test split has 10 unpaired stragglers (failures whose only same-task success was
  itself unpaired-singleton at upstream-pairing time).
- Train ~3.7% unpaired = the 657 orphan-success singletons that have no other
  same-(source, task) success to pair with.

For dual-ICL eval to be apples-to-apples, every sampled query needs a partner. The
post-pass success→success pairing (Section 3) closes this gap for the success side
of every eval split.

---

## 7. Audit history

| Date | Action |
|---|---|
| 2026-04-25 | Initial `pairs_unified.jsonl` build covering 5 sources (no roboreward). |
| 2026-04-28 | `build_test_set.py` produced the held-out test split (3,503 rows, Group A only). |
| 2026-04-30 | ICL self-demo bug found: `_load_partner_trajectory` was falling back to query path because pair indices have no `partner_frames_path` field. Loader fixed to resolve via `pairs_unified.jsonl` lookup; assertion added so the fallback can't recur. |
| 2026-04-30 | Roboreward added to `build_unified_pairs.py` (was the missing 6th family). `pairs_unified.jsonl` regenerated (628,020 → 651,193 rows). |
| 2026-04-30 | Per-split pair indices regenerated via `build_splits.py`; Group B and roboreward now propagated into train/warmup/eval pools. |
| 2026-04-30 | **Success → success pairing pass** added to `build_unified_pairs.py`. For every unpartnered success row, the post-pass picks another same-(source, task) success as its demo. **40,196 new success pairings**; 38,228 singleton-task successes remain unpaired (only one success exists for that task in that source). Per-split files regenerated again to inherit the new pairings. |
| 2026-04-30 | Test split regenerated via `build_test_set.py` (was stale post-roboreward); now 99.7% paired across both quality labels. |
| 2026-04-30 | All 6 HF Arrow caches rebuilt via `preprocess_split.job` so they line up with the updated pair indices. The sampler's `data.icl_min_coverage=0.40` floor will now hold at startup. |
| 2026-04-30 | Dual-ICL eval wired into `RBMHeadsTrainer._run_custom_evaluations`. Gated by `custom_eval.policy_ranking_dual_icl` (default False; set to True in `Robometer-LoRA/configs/train_lora_base.yaml`). When active, every policy_ranking eval round runs twice — `icl_prob=1.0` (`_iclon` suffix) then `icl_prob=0.0` (`_icloff` suffix) — so wandb sees both as separate scalars per metric × dataset. |

Backup files:
- `pairs_unified.jsonl.bak_pre_roboreward` — pre-roboreward snapshot.
- `pairs_unified.jsonl.bak_pre_succ_pairing` — pre success-pairing snapshot.
- `/scratch-shared/$USER/robometer_frames_splits/_bak_pre_groupB_<DATE>/` — pre-rebuild snapshot of all per-split pair indices.
- `*.jsonl.bak_apr28` — pre-test-regen snapshots of test split.
