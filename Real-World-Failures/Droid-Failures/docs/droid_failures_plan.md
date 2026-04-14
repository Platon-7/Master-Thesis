# Plan: Enrich RoboReward with DROID Failure Episodes

## Context

The RoboReward training set (45,072 samples) has only **671 original DROID success demos** (from OXE) and **1,305 organic DROID failures** (from RoboArena). We want to download **all available failure episodes** from the raw DROID dataset on GCS, preprocess them to match RoboReward's format, and label them with progress scores (1–5) using VLMs.

**What we have locally:**
- 15 failure episodes + 50 success episodes already downloaded from GCS (`/var/scratch/pkarageo/my_failure_case/`, `/var/scratch/pkarageo/my_success_case/`) — "rearrange pillows on sofa" task
- Raw DROID format: 3 cameras per episode (ext1, ext2, wrist), **1280×720 @ 60fps**, H.264 AVC, metadata JSON with `success: false` and `current_task` field
- GCS bucket: `gs://gresearch/robotics/droid_raw/1.0.1/*/failure/`

**Target format (RoboReward-compatible):**
- 320×192 @ 10fps, H.264 MP4
- Metadata: `{"file_name": "...", "task": "...", "reward": 1-5, "gpt5_mini_check": "..."}`

**Camera perspective note:** RoboReward uses a single video per episode; the camera perspective varies by source dataset (inherited from OXE conversion). The paper does not document which camera is selected for DROID. For our DROID failures, we download **all 3 cameras** (ext1, ext2, wrist) and decide later which to use.

## Pipeline (5 Steps)

### Step 0: Enumerate & Install (~5 min)

Install gsutil in the roboreward conda env, then count available failure episodes:
```bash
pip install gsutil
gsutil ls gs://gresearch/robotics/droid_raw/1.0.1/*/failure/ | wc -l
```
This tells us the scale before committing to the full download.

### Step 1: Download Raw DROID Failures

**Script:** `Robo-Reward-FPS/scripts/download_droid_failures.py`
**SLURM job:** `job_files/download_droid_failures.job` (CPU-only, 12hrs)

Per episode, download:
- Metadata JSON (~1KB)
- **All 3 camera MP4s** (ext1, ext2, wrist — mono only, skip stereo)

Skip: SVO files, stereo MP4s, trajectory.h5

**Output structure:**
```
/var/scratch/pkarageo/droid_failures_raw/
  {episode_id}/
    metadata.json
    ext1.mp4          # 1280×720 @ 60fps
    ext2.mp4          # 1280×720 @ 60fps
    wrist.mp4         # 1280×720 @ 60fps
```

Resume-capable with download manifest. The script maps camera serial numbers to names (ext1/ext2/wrist) using the metadata JSON's `camera_type` field.

**Estimated storage:** ~N failures × 3 cameras × ~7MB = TBD from Step 0

### Step 2: Preprocess Videos

**Script:** `Robo-Reward-FPS/scripts/preprocess_droid_videos.py`
**SLURM job:** `job_files/preprocess_droid_videos.job` (CPU-only, 4hrs)

For each camera MP4:
1. Scale 1280×720 → **320×192** (scale to width 320, then pad height 180→192 with 6px black letterbox top+bottom)
2. Downsample 60fps → **10fps**
3. Re-encode H.264
4. Output: `/var/scratch/pkarageo/droid_failures_processed/{episode_id}_{camera}.mp4`

Uses ffmpeg one-liner per video, parallelized across CPUs:
```bash
ffmpeg -i input.mp4 -vf "scale=320:180,pad=320:192:0:6:black" -r 10 -c:v libx264 output.mp4
```

### Step 3: Reverse Counterfactual Relabeling (Task Description Generation)

**Script:** `Robo-Reward-FPS/droid_failures/label_droid_failures.py`, Phase 1
**Model:** Qwen3-VL-8B-Instruct (base, NOT RoboReward fine-tuned) — cached at `/var/scratch/pkarageo/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/`

**Problem:** DROID raw task labels are often too vague or generic for meaningful scoring:
- `"Do any task, and then reset the scene."` — completely meaningless
- `"Move object to a new position and orientation (ex: ...)"` — template, not episode-specific
- `"Wire harness"` — ambiguous, no clear success criterion
- Even specific labels like `"Stack cup on bowl"` get score 1 because many videos are very short (1-2s) and at fps=1.0 the VLM only sees 1-2 frames

**Why the RoboReward paper didn't have this problem:** They never directly scored raw failure episodes. Their pipeline:
1. OXE data = successful demos, auto-assigned score 5
2. Failures are generated via **counterfactual relabeling** of successes (VLM generates alternative task descriptions for which the same video would be a failure)
3. DROID data comes through RoboArena which has human-written task descriptions + human scores

**Our approach — reverse counterfactual relabeling:** Instead of generating wrong tasks for successful videos, we generate correct tasks for failure videos:
1. **Video Analysis** (Qwen3-VL-8B-Instruct): Feed each 10fps video with the paper's A.2.2 prompt to get a detailed scene description
2. **Task Generation** (Qwen3-VL-8B-Instruct): From the scene description, generate a specific imperative task command describing what the robot was attempting
3. Optionally keep original DROID label alongside for comparison

**Test script:** `Robo-Reward-FPS/droid_failures/test_relabel.py` — runs 4 test videos (2 bad descriptions, 2 good) comparing old vs. new task labels and their RoboReward scores.

### Step 4: VLM Scoring + Verification

**Script:** `Robo-Reward-FPS/droid_failures/label_droid_failures.py`, Phase 2
**SLURM job:** `job_files/label_droid_failures.job` (1 GPU)
**Model:** RoboReward-8B (fine-tuned Qwen3-VL, 4-bit quantized)

For each (video, generated_task) pair:

1. **Feed full 10fps video** to RoboReward (NOT subsampled to 1fps — many DROID failures are <5s, subsampling loses critical information)
2. **Direct scoring** against the RoboReward rubric:
   - Score 1: No goal-relevant change
   - Score 2: Minimal progress
   - Score 3: Partial completion
   - Score 4: Near completion (minor requirement missed)
   - Score 5: Full success (unlikely for failures, but possible if metadata is wrong)
3. **VLM verification**: Validate the (video, task, score) triple using the paper's verification prompt (Appendix A.2.3). Keep only validated examples (ANSWER: TRUE).

No augmentation or counterfactual relabeling — these are genuine failures scored directly.

### Step 5: Package as RoboReward JSONL

**Script:** `Robo-Reward-FPS/scripts/package_droid_failures.py`

Package validated examples as RoboReward-compatible JSONL:
```json
{
  "file_name": "droid_failures/{episode_id}_{camera}.mp4",
  "task": "Rearrange pillows on sofa.",
  "reward": 2,
  "gpt5_mini_check": "Qwen3-VL verification: The robot reaches toward a pillow but fails to grasp it..."
}
```

Merge with existing `metadata.jsonl` and copy processed videos to the RoboReward dataset directory.

## Files to Create

| File | Purpose |
|------|---------|
| `Robo-Reward-FPS/scripts/download_droid_failures.py` | GCS download (all cameras + metadata) |
| `Robo-Reward-FPS/scripts/preprocess_droid_videos.py` | Resize 1280×720→320×192, resample 60→10fps |
| `Robo-Reward-FPS/scripts/label_droid_failures.py` | Task cleanup (Qwen3-4B) + VLM scoring & verification (Qwen3-VL-8B) |
| `Robo-Reward-FPS/scripts/package_droid_failures.py` | Package as RoboReward JSONL + merge |
| `job_files/download_droid_failures.job` | SLURM: download (CPU, 12hrs) |
| `job_files/preprocess_droid_videos.job` | SLURM: ffmpeg batch (CPU, 4hrs) |
| `job_files/label_droid_failures.job` | SLURM: VLM labeling (2× GPU, 24hrs) |

## Models Used

| Model | Role | Hardware |
|-------|------|----------|
| Qwen3-4B-Instruct | Task description cleanup | 1× GPU (text-only) |
| Qwen3-VL-8B | Video analysis + scoring + verification | 1× GPU 24GB+ (4-bit quantized) |

## Storage Estimate

| Item | Estimated Size |
|------|---------------|
| Raw DROID MP4s (3 cameras per episode) | TBD (depends on failure count) |
| Processed videos (320×192 @ 10fps) | ~1/20th of raw (resolution + fps reduction) |
| Metadata + analysis text | <1 GB |

Location: `/var/scratch/pkarageo/`

## Verification

1. **Step 0:** Count failures on GCS → report total before downloading
2. **Step 2:** Spot-check 10 videos — verify 320×192, 10fps, playable
3. **Step 4:** Manually review 20 scored examples — are scores reasonable?
4. **Step 5:** Validate JSONL parses correctly, no duplicate file_name entries, scores in {1–5}
5. **Integration test:** Merge new JSONL with existing metadata.jsonl, verify total count increases

## Step 6: Download Task-Level Success Pairs (In-Context Learning)

**Goal:** For each failure episode, pair it with a successful demonstration of the **same task in the same scene** (matched by scene_id + generic_task). This enables in-context learning: concatenate a success demo with each training datapoint so the model sees "this is what success looks like" alongside the failure.

**Script:** `Robo-Reward-FPS/droid_failures/download_success_pairs_v2.py`

### Matching Strategy

Match failures to successes by **(scene_id, generic_task)** — exact string match on the original DROID task descriptions (before Qwen3 relabeling). The `generic_task` field in `qwen3_relabeled.jsonl` traces back to the original DROID label, which matches `current_task` in success metadata.

### Coverage Stats (as of 2026-03-16)

| Metric | Count |
|--------|-------|
| Total failure episodes | 5,518 |
| Unique (scene, task) failure pairs | 584 |
| Pairs with exact task-level success match | 458 (78.4%) |
| Failure episodes with ≥1 success match | 4,974 (90.1%) |
| Failure episodes WITHOUT any match | 544 (9.9%) |
| Unmatched (scene, task) pairs | 126 |
| Total success episodes to download | ~4,617 |
| Estimated download size | ~148 GB |

### Download Strategy

- For each matched (scene, task) pair, download **up to N successes** where N = number of failures for that pair. This ensures every failure can get a **unique** success demo when possible.
- 391/458 pairs have enough successes (≥ failure count); 67 pairs have fewer successes than failures (those will reuse demos via sampling with replacement).
- Max failures per pair: 299; median: 3; p90: 22.
- Success metadata already downloaded: 59,683 JSONs in `/var/scratch/pkarageo/droid_success_pairs/_metadata/`.

### Unmatched Failures → Scene-Level Fallback (544 episodes)

126 (scene, task) pairs have no exact success match. Largest: 67 failures for "hang or unhang object" in scene 84bd5053. All 544 have a scene-level match (same physical environment, different task).

**Strategy:** Use scene-level fallback — pair with any success from the same scene. Each episode in the manifest is flagged with `match_type`:
- `"task"` — exact (scene_id, generic_task) match (4,974 episodes)
- `"scene_fallback"` — same scene, different task (544 episodes)

This flag allows filtering out scene-only matches later if they hurt performance. To exclude them downstream: `[ep for ep in manifest["episodes"] if ep["match_type"] == "task"]`

### Output

```
/var/scratch/pkarageo/droid_success_pairs/
  task_matched/
    {lab}_{scene_id}_{timestamp}/
      ext1.mp4, ext2.mp4, wrist.mp4
  task_match_manifest.json   # Maps each failure episode to its success pair(s)
```

### Preprocessing

Success videos are raw DROID format (1280×720 @ 60fps). They need the same preprocessing as failures:
- Scale to 320×192 @ 10fps (Step 2 pipeline)
- Store in `/var/scratch/pkarageo/droid_success_processed/`

## Execution Order

1. **Quick:** Install gsutil, enumerate GCS failures (~5 min) → report count to user
2. **SLURM:** Download all failure MP4s (3 cameras) + metadata (12hrs)
3. **SLURM:** Preprocess with ffmpeg (4hrs)
4. **SLURM:** Run labeling pipeline — task cleanup + VLM scoring (24hrs, GPU)
5. **Local:** Package, verify, merge
6. **Login node:** Download task-matched success pairs via gsutil (~4,617 episodes, ~148 GB)
7. **SLURM:** Preprocess success videos (same as Step 3)
