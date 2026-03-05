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

### Step 3: Clean Task Descriptions

**Script:** Part of `Robo-Reward-FPS/scripts/label_droid_failures.py`, Phase 1
**Model:** Qwen3-4B-Instruct (text-only, runs locally)

Raw DROID task labels are noisy:
- `"reaarrange pillows on sofa"` → typos
- `"Do anything you like that takes multiple steps to complete."` → too vague
- `"Do any two tasks consecutively.\n\nSuggested tasks:\n* ..."` → multi-line noise

Apply the RoboReward prompt rewrite (Section A.2.1): fix grammar/spelling, normalize to clean imperative form. Filter out episodes with unusably vague instructions (e.g., "Do anything you like").

### Step 4: VLM Scoring + Verification

**Script:** `Robo-Reward-FPS/scripts/label_droid_failures.py`, Phase 2
**SLURM job:** `job_files/label_droid_failures.job` (2 GPUs: GPU 0 for Qwen3-4B, GPU 1 for Qwen3-VL-8B)
**Model:** Qwen3-VL-8B (already available, 4-bit quantized)

For each (video, cleaned_task) pair:

1. **Sample frames** at 1 FPS from the video
2. **VLM video analysis**: Describe what happens (same prompt as paper, Appendix A.2)
3. **Direct scoring** against the RoboReward rubric:
   - Score 1: No goal-relevant change
   - Score 2: Minimal progress
   - Score 3: Partial completion
   - Score 4: Near completion (minor requirement missed)
   - Score 5: Full success (unlikely for failures, but possible if metadata is wrong)
4. **VLM verification**: Validate the (video, task, score) triple using the paper's verification prompt (Appendix A.2.3). Keep only validated examples (ANSWER: TRUE).

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

## Execution Order

1. **Quick:** Install gsutil, enumerate GCS failures (~5 min) → report count to user
2. **SLURM:** Download all failure MP4s (3 cameras) + metadata (12hrs)
3. **SLURM:** Preprocess with ffmpeg (4hrs)
4. **SLURM:** Run labeling pipeline — task cleanup + VLM scoring (24hrs, GPU)
5. **Local:** Package, verify, merge
