# RoboReward Dataset Format & Augmentation Pipeline

Reference paper: [RoboReward: General-Purpose Vision-Language Reward Models for Robotics](https://arxiv.org/abs/2601.00675) (Lee et al., 2026)

## Dataset Overview

The RoboReward training set contains **45,072 video–task–reward triples** sourced from 29 real-robot datasets (28 from OXE + RoboArena). Each entry consists of:

| Field | Description |
|-------|-------------|
| `file_name` | Relative path: `{dataset}/{video_filename}.mp4` |
| `task` | Natural-language task instruction |
| `reward` | Discrete progress score in {1, 2, 3, 4, 5} |
| `gpt5_mini_check` | GPT-5 mini validation explanation |

The dataset is hosted on HuggingFace: `teetone/RoboReward`

## File Naming Convention

### Original episodes (reward = 5)

```
{dataset}/{dataset}_originalsplit_train_index_{N}.mp4
```

Example:
```
droid/droid_originalsplit_train_index_47028.mp4
```

These are unmodified successful demonstrations from OXE. All originals are assigned reward **5** (perfect completion). There are **30,593** original episodes.

### Augmented episodes (reward 1–4)

```
{dataset}/{dataset}_originalsplit_train_index_{N}_attempt_1_score_{S}.mp4
```

Example:
```
droid/droid_originalsplit_train_index_47028_attempt_1_score_2.mp4
```

- `attempt_1`: Always 1 (only one augmentation attempt per clip)
- `score_{S}`: The assigned reward, where S ∈ {1, 2, 3, 4}

There are **14,479** augmented episodes across 6,781 unique base episodes.

## Augmentation Pipeline

The paper describes two augmentation strategies for creating negative/near-miss examples from successful episodes. Both strategies **always truncate the video** — there are no augmented entries that use the full-length original video.

### Strategy 1: Temporal Clipping Only (11,116 clips)

The video is truncated to a fixed fraction. The task instruction stays the same as the original.

**What happens:**
1. Take a successful episode (video *v*, task *t*, reward 5)
2. Clip the video to a predetermined fraction based on the target score
3. Keep the original task instruction *t*
4. A VLM (GPT-5 mini) validates that the clipped video + original task + assigned score are coherent

**Example:**
```
Original:  ..._index_0.mp4  (64.0s)  task="Take the lid off the pot..."  reward=5
Clip:      ..._index_0_attempt_1_score_2.mp4  (19.4s)  task="Take the lid off the pot..."  reward=2
Clip:      ..._index_0_attempt_1_score_3.mp4  (32.2s)  task="Take the lid off the pot..."  reward=3
```

### Strategy 2: Counterfactual Relabeling + Temporal Clipping (3,228 clips)

The video is truncated AND the task instruction is replaced with a different (counterfactual) command.

**What happens:**
1. Take a successful episode (video *v*, task *t*, reward 5)
2. A VLM describes the video in detail
3. An LLM proposes alternative task instructions that the same video would only partially achieve
4. The video is clipped to the same fixed fraction based on the target score
5. A VLM validates the (clipped video, new task, assigned score) triple

**Example:**
```
Original:  ..._index_44.mp4  (76.6s)  task="Move the large bowl to the center"  reward=1
Clip:      ..._index_44_attempt_1_score_2.mp4  (23.2s)  task="Take the lid off the pot..."  reward=2
Clip:      ..._index_44_attempt_1_score_3.mp4  (38.5s)  task="Take the lid off the pot..."  reward=3
```

Note: The original video for index_44 has reward=1 for the *original* task — its base episode is a successful demonstration of "Move the large bowl..." but the original reward in metadata reflects how well it matches its own task label, not the augmented one.

### Key Observation: Paper vs. Implementation

The paper (Section 4.2) describes counterfactual relabeling as using the **same video** *v* with a new task. However, in the actual dataset, counterfactual entries are **also truncated** at the same fixed fractions as temporal clips. This was verified empirically — both augmentation types show identical truncation ratios for matching scores.

## Truncation Fractions

The truncation points are **deterministic** — every clip at a given score keeps exactly the same fraction of the original video:

| Score | Fraction of Original Kept | Interpretation |
|-------|--------------------------|----------------|
| 1 | **10%** | Very beginning only — no meaningful progress |
| 2 | **30%** | Early portion — minimal progress visible |
| 3 | **50%** | First half — partial completion |
| 4 | **70%** | Most of episode — near completion but ending removed |
| 5 | **100%** | Full original video (not an augmented entry) |

These fractions are consistent across all datasets, all base episodes, and both augmentation strategies.

**Verification example (index_0, original = 64.0s):**
```
score_1:  6.4s  →  6.4 / 64.0 = 0.10  ✓
score_2: 19.4s  → 19.4 / 64.0 = 0.30  ✓
score_3: 32.2s  → 32.2 / 64.0 = 0.50  ✓
score_4: 42.0s  → 42.0 / 60.0 = 0.70  ✓  (different episode)
```

## How to Distinguish the Two Augmentation Types

There is **no explicit field** in the metadata marking which augmentation strategy was used. To distinguish them, compare the clip's `task` field against its original video's `task`:

```python
original_filename = re.sub(r'_attempt_\d+_score_\d+\.mp4$', '.mp4', clip_filename)
```

- **Same task** → temporal clipping only (11,116 clips across 5,038 base episodes)
- **Different task** → counterfactual relabeling + temporal clipping (3,228 clips across 1,670 base episodes)

A given base episode receives **only one** strategy — no base episode has both types of augmented clips.

## Score Distribution

| Score | Temporal Only | Counterfactual + Temporal | Total |
|-------|-------------|--------------------------|-------|
| 1 | 4,251 | 1,387 | 5,698 |
| 2 | 3,134 | 915 | 4,089 |
| 3 | 1,613 | 436 | 2,067 |
| 4 | 2,118 | 490 | 2,625 |
| **Total** | **11,116** | **3,228** | **14,479** |

Not every base episode has clips at all four scores — some have 1, 2, or 3 clip variants. The number of clips per base depends on which scores passed the VLM validation step.

## Video Specifications

All videos in the dataset (both originals and clips) share these properties:
- Resolution: **320 × 192** pixels
- Frame rate: **10 fps**
- Codec: H.264 (MP4 container)
- Typical original duration: 30–80 seconds

## For Video Completion

Since **all 14,479 augmented entries are truncated**, they are all candidates for video completion (generating the missing ending). The amount of video to generate depends on the score:

| Score | % Missing | Avg seconds to generate |
|-------|----------|------------------------|
| 1 | 90% | ~28s |
| 2 | 70% | ~22s |
| 3 | 50% | ~16s |
| 4 | 30% | ~9s |
