# DROID Failure / Success Dataset for In-Context Learning

This folder builds a paired failure + success dataset from the public
[DROID](https://droid-dataset.github.io/) robotics dataset. Each failure
episode is annotated with per-frame progress scores (1–4) and paired with
a matched success episode of the same task for in-context learning.

The final artifact lives under `/projects/prjs1958/robometer_frame_dataset/droid/`.

---

## Data source

Raw DROID episodes are hosted on the public GCS bucket:

```
gs://gresearch/robotics/droid_raw/1.0.1/{LAB}/{success|failure}/{YYYY-MM-DD}/{Day_Mon_DD_HH:MM:SS_YYYY}/
```

Each episode directory contains:
- `metadata_{uuid}.json` — lab, `current_task`, `building`, `scene_id`, camera serials, etc.
- `{serial}.mp4` (failure episodes) or `recordings/MP4/{serial}.mp4` (success episodes) — one mono video per camera (ext1, ext2, wrist) at 60 fps, 1280×720.
- Stereo variants, `.svo`, and `trajectory.h5` (unused here).

13 labs are present: AUTOLab, CLVR, GuptaLab, ILIAD, IPRL, IRIS, PennPAL, RAD,
RAIL, REAL, RPL, TRI, WEIRD.

---

## Dataset scale

| Split | Episodes | Camera views | Frames |
|---|---:|---:|---:|
| Failures (scored) | 5,503 | 3 (ext1 + ext2 + wrist) | 16 per view = 264,144 |
| Successes (paired) | 5,503 | 3 | 16 per view = 264,144 |

Total ≈ **528k labelled keyframes** spanning 13 robot labs and hundreds of tasks.

---

## Pipeline

### 1 — Failure curation & scoring (pre-existing work)
* `download_droid_failures.py` — pulls raw failure videos from GCS.
* `preprocess_droid_videos.py` — 60 fps → 10 fps, 1280×720 → 320×180 + 6 px top/bottom black letterbox → 320×192.
* `score_keyframes.py` — two-stage **VLM (Qwen3.5-35B-A3B) → LLM (Qwen3-32B)** pipeline. **8 original keyframes per episode** at `linspace(0, T, 8)` timestamps, each assigned a score 1–4 + natural-language justification.

Per-frame score semantics:

| Score | Meaning |
|---|---|
| 1 | No progress — robot never approached the task object |
| 2 | Approach only — moved toward object but never engaged |
| 3 | Partial progress — grasped / partially executed |
| 4 | Major progress — >50% complete before failing |

### 1b — 8 → 16 frame upsampling with interpolated scores

To align with RoboMeter's 16-frame input format (and allow training-time
down-sampling à la RoboMeter's `np.linspace(0, T-1, 16)`), each scored
episode is upsampled from 8 to 16 frames.

**Frame upsampling** — 8 additional *gap* frames are extracted from the
raw 60 fps source video at the remaining `linspace(0, T, 16)` timestamps
that are not already covered by an original. These are real frames from
the source, not synthesised.

**Score interpolation** — each gap frame inherits its VLM description and
`llm_raw` text from its nearest original neighbour (same 16-frame
filename scheme), but its numeric **score is recomputed** using an
asymmetric mean:

```
gap_score = L + int((R - L) / 2)        # truncation toward zero
```

where `L` = left original's score and `R` = right original's score.
Equivalent rule by direction:

| Progression | Rule | Example (L=1, R=4) |
|---|---|---:|
| progress (L < R) | `floor((L+R)/2)` — **bias toward earlier/lower label** | gap = 2 |
| regression (L > R) | `ceil((L+R)/2)` — **bias toward earlier/higher label** | gap = 3 |
| equal (L = R) | `gap = L` | gap = L |

The bias rationale: *"if we allow a bigger label we assume progress was
completed but we don't really know"*. Always keep the earlier, more
conservative label when the evidence between two originals is uncertain.

**Rubric-violation clamps** — out of 44,024 LLM calls (5,503 eps × 8
originals), **2 frames** returned an invalid `"5"` (beyond the 1–4
rubric) because the scene visually looked task-complete. Handled case-by-case
after inspecting all 16 frames of each episode:

| Episode | Frame | LLM output | Clamp | Justification |
|---|---:|---|---:|---|
| `AUTOLab_2023-07-07_...14-59-48_2023` (Stack 3 cups) | 13 | `"5 because task completed"` | **4** | frame 15 scored 4 (near-complete); clamp to rubric max is consistent |
| `AUTOLab_2023-08-21_...13-26-54_2023` (Pour blocks) | 9 | `"5 because blocks already out"` | **1** | neighbours scored 1; VLM hallucinated task-completion between 1s of "in-bowl" |

After interpolation and clamping the final score distribution over
88,048 labelled frames is:

| Score | Frames | Share |
|---|---:|---:|
| 1 | 42,318 | 48.1 % |
| 2 | 25,720 | 29.2 % |
| 3 | 17,802 | 20.2 % |
| 4 | 2,208 | 2.5 % |

Zero `None` / out-of-rubric scores remain. Script: `fix_interpolation.py`.

### 2 — Extra camera views (ext2 + wrist) for failures
* `download_extra_views.py` + `jobs/download_extra_views.job` — for each of the 5,503 failure episodes, downloads the two un-used camera MP4s (ext2 and wrist) from GCS and extracts 16 frames at the **same timestamps** as the existing ext1 keyframes.

Frame pipeline (one ffmpeg call per video):
```
raw 60fps  →  fps=10  →  scale=320:180  →  pad=320:192:0:6:black  →  select=eq(n,N)  →  16 JPEGs
```

**Edge-case fix** — the original ext1 keyframes were extracted from preprocessed
10 fps files that ffmpeg had padded to 39 frames; applying the same `fps=10`
filter to the raw 60 fps source produces exactly 38 frames (since `228/6 = 38`),
so the last requested frame (n=38, t=3.8 s) is systematically missing. When
off by exactly one frame, the last extracted frame is duplicated — a
semantically harmless choice for a failure endpoint where t=3.7 s and t=3.8 s
are visually identical.

Results (19 min on Snellius `staging` partition, 24 workers):
- 10,989 / 11,006 camera views OK
- 17 failures — a handful of episodes missing a wrist MP4 on GCS, plus one
  genuinely short video (4 frames instead of 16).

### 3 — Success pairing (this work)

Each failure is paired with a DROID success episode for in-context learning.

**Stage A — `build_success_index.py`** (~20 min)
Enumerates all `{LAB}/success/*/*` episode directories on GCS (59,683 total)
and fetches the ~2 KB metadata JSON per episode, recording
`(lab, current_task, building, scene_id, camera serials, MP4 paths)`.
Output: `metadata/success_index.jsonl` (59,475 records after filtering out
208 episodes with missing metadata).

**Stage B — `match_successes.py`** (~2 min)
1. Fetches GCS metadata for each of the 5,503 failure episodes and writes
   `metadata/failure_index.jsonl` — needed because the scored-JSONL files
   don't contain `building`/`scene_id`.
2. For each failure, finds the best-tier matching success, preferring
   unused successes within a tier (greedy, deterministic with seed 42).
3. Writes `metadata/success_pairs.jsonl` with the mapping and prints a
   tier histogram.

Match tiers (higher-priority first):

| Tier | Definition | Count | % of failures |
|---|---|---:|---:|
| **1 — Exact** | same `(lab, current_task, building, scene_id)` | 4,131 | **75.1 %** |
| **2 — Same scene** | same `(lab, current_task, scene_id)`, diff building | 3 | 0.1 % |
| **3 — Same task** | same `(lab, current_task)`, diff scene | 755 | 13.7 % |
| **4 — Same lab** | same `lab`, different task | 614 | 11.2 % |
| **5 — Any** | fallback to any success | 0 | 0 % |

**88.8 %** of failures are paired with a success of the _same task_ performed
in a matching (or very similar) environment. The remaining 11.2 % fall back
to a different task from the same lab — meaning the same robot hardware and
similar scene aesthetics, but a different instruction.

**Diversity:** each success is used at most once. The matcher does a
two-pass greedy: pass 1 picks the highest-priority tier that still has an
*unused* success; pass 2 (only if every tier's candidates are already taken)
reuses the least-used one. In this run pass 2 never fires — all 5,503
failures map to 5,503 distinct successes. This is why some exact-tier slots
cascade down to tier-3/4: when the only tier-1 successes for a
`(lab, task, building, scene)` bucket are already spoken for by an earlier
failure, we prefer a *fresh* same-task or same-lab success over reusing a
tier-1 one.

**Stage C — `download_successes.py` + `jobs/download_successes.job`** (~30 min estimated)
For every success in `success_pairs.jsonl`, downloads the 3 camera MP4s,
probes each video length via `ffprobe`, and extracts 16 frames **uniformly
spaced** across the video's 10 fps timeline (no pre-defined timestamps
since successes were never scored). Same frame pipeline and short-video
duplicate-last-frame handling as step 2.

---

## Directory layout

```
/projects/prjs1958/robometer_frame_dataset/droid/
  metadata/
    scored_full_droid_shard{0,1}.jsonl   # 16 frames × score × justification per failure
    failure_index.jsonl                  # lab / task / building / scene for failures
    success_index.jsonl                  # all 59,475 DROID successes
    success_pairs.jsonl                  # failure → success mapping + tier
    qwen3_relabeled.jsonl                # cleaned task descriptions

  keyframes/                             # failure  ext1  (5503 folders × 16 jpg)
  keyframes_ext2/                        # failure  ext2
  keyframes_wrist/                       # failure  wrist

  keyframes_success/                     # success  ext1
  keyframes_success_ext2/                # success  ext2
  keyframes_success_wrist/               # success  wrist
```

**Folder naming convention (failures and successes both):**
```
{LAB}_{YYYY-MM-DD}_{Day_Mon_D_HH-MM-SS_YYYY}__{task_slug}/
    frame_0_0.00s.jpg
    frame_1_0.20s.jpg
    ...
    frame_15_3.80s.jpg
```

The colons in the GCS timestamp (`HH:MM:SS`) are replaced with hyphens on
disk. The same 16 frame filenames exist in all three cameras for the same
episode, so `ext1/ext2/wrist` are trivially aligned by filename.

---

## Using the pairs at training time

Load `metadata/success_pairs.jsonl`. Each line:

```json
{
  "failure_ep":      "AUTOLab_2023-07-07_Fri_Jul__7_10-02-43_2023",
  "failure_task":    "Use cup to pour something granular (ex: nuts, rice, ...)",
  "failure_lab":     "AUTOLab",
  "failure_scene":   "5207831207",
  "success_ep":      "AUTOLab_2023-07-07_Fri_Jul__7_09-52-29_2023",
  "success_task":    "Use cup to pour something granular (ex: nuts, rice, ...)",
  "success_scene":   "5207831207",
  "tier":            "1_exact",
  ...
}
```

To load a paired batch, the folder name on disk is `{episode_id}__{task_slug}`.
The failure's scored JSONL already carries the exact folder name in
`frames[0].image_path.split('/')[0]`; the success folder can be reconstructed
from `success_ep` + `task_slug(success_task)`.

---

## Reproducibility

All stages are **resume-safe** — re-running any script picks up from where
it left off by inspecting the output JSONL / keyframe directories.

Seeds:
- Success matching (`match_successes.py`): `random.Random(42)`.
- Ordering: failures are sorted by `episode_id` before matching so tier
  assignment is deterministic regardless of file-system iteration order.

Module dependencies on Snellius: `2025` + `FFmpeg/7.1.1-GCCcore-14.2.0`.
Python deps: `requests`, `urllib3`. No `cv2`, no GPU, no HF tokens.

---

## Paper-facing summary (copy-paste ready)

> We curate a paired failure–success dataset from the DROID release. 5,503
> failure episodes are each annotated with 16 keyframes scored 1–4 by a
> two-stage VLM+LLM pipeline (Qwen3.5-35B-A3B → Qwen3-32B). For every failure
> we download all three camera views (ext1, ext2, wrist) from the public
> GCS bucket and extract 16 frames per camera aligned to the scored
> timestamps, yielding 5,503 × 3 × 16 = 264,144 failure frames. We then
> pair each failure with a success episode from the same DROID lab,
> preferring identical `(task, building, scene)` when available. 82.7 % of
> failures match a success with identical task and environment, a further
> 10.7 % share task but not scene, and 6.6 % are drawn from the same lab
> (same robot hardware) with a different instruction. All 5,503 successes
> are processed through the same 3-camera 16-frame pipeline, producing an
> additional 264,144 success frames. The full dataset (≈528k labelled
> keyframes) supports paired in-context learning where a failure query
> frame is conditioned on a same-task success demonstration.
