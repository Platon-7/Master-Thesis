# RoboReward Dataset: Format and Metadata Report

**Date:** February 6, 2026
**Dataset:** RoboReward-8B Dataset (HuggingFace: `teetone/RoboReward`)
**Total Samples Analyzed:** 45,072 video trajectories

---

## 1. Dataset Overview

RoboReward is a large-scale robotic manipulation dataset created for training vision-language models (VLMs) to evaluate task success/failure in robot trajectories. It aggregates data from **29 different robotics datasets** spanning various manipulation tasks, environments, and robot platforms.

### Key Statistics

- **Total Samples:** 45,072 annotated video trajectories
- **Unique Tasks:** 4,551 distinct task descriptions
- **Video Format:** MP4 files (hosted on HuggingFace)
- **Annotations:** 5-point Likert scale (1=failure → 5=success)

---

## 2. Dataset Structure

### 2.1 Sample Metadata

Each sample in the dataset contains the following fields:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `video` | str | HuggingFace video path | `hf://datasets/teetone/RoboReward/.../video.mp4` |
| `task` | str | Natural language task description | "Take the lid off the pot, put the pot on the plate..." |
| `reward` | int | Success score (1-5) | `5` (perfect completion) |
| `gpt5_mini_check` | str | GPT-annotated verification | "The robot successfully removed the lid..." |

### 2.2 Reward Distribution

The dataset uses a 5-point scale to annotate trajectory quality:

```
Reward 1 (Complete Failure):     13,648 samples (30.3%)
Reward 2 (Minimal Progress):       8,577 samples (19.0%)
Reward 3 (Partial Completion):     7,850 samples (17.4%)
Reward 4 (Mostly Successful):      6,572 samples (14.6%)
Reward 5 (Perfect Completion):     8,425 samples (18.7%)
```

**Success/Failure Threshold:** Reward ≥ 4 is considered "success", Reward < 4 is "failure"

- **Successes:** 14,997 samples (33.3%)
- **Failures:** 30,075 samples (66.7%)

---

## 3. Source Dataset Distribution

The dataset aggregates 29 source datasets. Top 10 by sample count:

| Source Dataset | Samples | % | Description |
|----------------|---------|---|-------------|
| robo_arena | 7,337 | 16.3% | Simulated robotic arena tasks |
| fractal20220817_data | 3,988 | 8.8% | Fractal environment data |
| bridge | 3,826 | 8.5% | Berkeley BRIDGE dataset |
| **droid** | **3,071** | **6.8%** | DROID general manipulation |
| toto | 2,986 | 6.6% | TOTO dataset |
| taco_play | 2,856 | 6.3% | TACO play data |
| jaco_play | 2,428 | 5.4% | Kinova Jaco arm |
| berkeley_autolab_ur5 | 2,388 | 5.3% | Berkeley UR5 data |
| ucsd_pick_and_place | 2,384 | 5.3% | UCSD pick-and-place |
| nyu_door_opening | 2,369 | 5.3% | NYU door manipulation |

**Note:** Our experiments focus on the **droid** (3,071 samples) and **bridge** (3,826 samples) subsets as they represent real-world robotic manipulation with diverse tasks.

---

## 4. Video Properties

### 4.1 DROID Dataset Videos (Our Downloaded Subset)

From our performance diagnostic on `droid_general` videos:

- **Resolution:** 320 × 192 pixels (low resolution, optimized for speed)
- **Frame Rate:** 10 FPS
- **Codec:** H.264
- **Average Duration:** ~31 seconds per trajectory
- **Average Frames:** ~311 frames per video
- **Read Speed:** ~0.37 seconds per video (torchvision)
- **Processing Speed:** ~1,200 frames/second

### 4.2 Comparison with Other Datasets

For reference, the **pillows** task from our FPS experiments:

- **Resolution:** 1280 × 720 pixels (HD quality)
- **Frame Rate:** 60 FPS
- **Codec:** H.264
- **Average Duration:** ~13 seconds per trajectory
- **Read Speed:** ~28.7 seconds per video (77x slower than droid)

**Finding:** DROID videos are optimized for efficient VLM processing with lower resolution and frame rate, enabling faster experimentation.

---

## 5. Task Distribution Analysis

### 5.1 Task Uniqueness

Our analysis of the `droid_general` subset (40 videos: 20 success + 20 failure):

**Success Videos:**
- 20 unique tasks (100% unique)
- Each task appears exactly once
- Examples: "Close the cupboard door", "Put the coffee pod inside the coffeemaker", "Move the remote to the surface above the top shelf"

**Failure Videos:**
- 15 unique tasks (~75% unique)
- Some tasks repeated: "Push the detergent compartment" (3x), "Turn on the light" (2x), "Remove one black object" (2x)

**Key Finding:** The `droid_general` subset contains **mixed tasks** with minimal overlap between success and failure examples, making it challenging to create exact task-matched pairs.

### 5.2 Balanced Task Availability

From our dataset inspection (filtering for droid + bridge datasets):

- **Balanced tasks (≥5 success + ≥5 failure):** **0 tasks found**

This means the dataset does NOT naturally provide tasks with many examples of both success and failure outcomes. Most tasks lean heavily toward either success or failure.

---

## 6. Data Access

### 6.1 HuggingFace Streaming

The dataset is hosted on HuggingFace and supports streaming mode:

```python
from datasets import load_dataset

dataset = load_dataset(
    "teetone/RoboReward",
    split="train",
    streaming=True
)
```

### 6.2 Downloaded Subsets

Currently downloaded to `/var/scratch/pkarageo/roboreward_tasks/`:

```
droid_general/
├── success/
│   ├── *.mp4 (20 videos)
│   └── samples.json (metadata)
├── failure/
│   ├── *.mp4 (20 videos)
│   └── samples.json (metadata)
└── metadata.json (task info)
```

---

## 7. Task Taxonomy

### 7.1 Action Types (from sampled tasks)

The dataset covers diverse manipulation primitives:

| Action Type | Examples |
|-------------|----------|
| **Grasping & Placing** | "Put the coffee pod inside the coffeemaker", "Put the marker on the right side of the pot" |
| **Wiping & Cleaning** | "Use the white napkin to wipe the counter" |
| **Container Manipulation** | "Open the drawer on the left then place the green pear inside" |
| **Object Rearrangement** | "Move the styrofoam container to the left", "Slide the black container inwards" |
| **Articulated Objects** | "Close the cupboard door", "Turn on the light using the switch" |
| **Complex Multi-Step** | "Take the lid off the pot, put the pot on the plate, and use the tool to push it" |

### 7.2 Object Categories

Common objects across tasks:
- **Kitchen items:** Pots, bowls, cups, utensils, napkins, food items
- **Household objects:** Pillows, towels, remotes, containers
- **Geometric objects:** Blocks, boxes, bowls (for testing)
- **Articulated objects:** Drawers, cupboards, doors, light switches

---

## 8. GPT Annotation Quality

Each sample includes a `gpt5_mini_check` field with detailed verification:

**Example:**
```
"The robot successfully removed the lid from the pot, placed the
(now-unlidded) pot onto the plate, and then used the tool (spatula)
to push the pot forward on the table. All three core requirements
are visible and satisfied in the final frames, so the provided
score of 5 (Perfect Completion) matches the rubric.

ANSWER: TRUE"
```

**Insight:** The GPT annotations provide:
1. Step-by-step verification of task requirements
2. Reasoning about why the score is appropriate
3. Binary verification (TRUE/FALSE) of score validity

This could be useful for:
- Understanding failure modes
- Creating better task descriptions
- Identifying mislabeled samples

---

## 9. Limitations and Considerations

### 9.1 Task Imbalance

- **66.7% failure rate** across the dataset
- Most tasks do NOT have balanced success/failure examples
- The dataset is optimized for reward modeling, not contrastive learning

### 9.2 Task Diversity vs. Repetition

- **4,551 unique tasks** across 45,072 samples
- Average: ~10 samples per task
- But distribution is highly skewed (some tasks have 1 sample, others have many)

### 9.3 Video Quality Variation

- Different source datasets have different video resolutions, frame rates, and quality
- DROID: 320×192 @ 10 FPS (fast to process)
- Other datasets may have higher resolution but slower processing

---

## 10. Recommendations for In-Context Learning

### 10.1 Challenge

The dataset is NOT organized for failure-success pairing because:
1. Most tasks have only success OR failure examples, not both
2. Tasks are highly diverse (4,551 unique descriptions)
3. No pre-existing task grouping for contrastive pairs

### 10.2 Proposed Solutions

**Option A: Semantic Task Matching**
- Use text similarity (e.g., sentence embeddings) to match failures with "similar" successful tasks
- Pro: Can create pairs from existing data
- Con: Not exact task matches, may introduce noise

**Option B: Expand Data Collection**
- Identify specific tasks with both outcomes
- Download more samples from underrepresented tasks
- Pro: Exact task matches, cleaner in-context learning
- Con: Requires analyzing all 45K samples to find matchable tasks

**Option C: Relaxed Pairing Criteria**
- Match based on action type (e.g., "all grasping tasks")
- Provide general successful demonstrations for failure examples
- Pro: Easier to implement, more data available
- Con: Less specific guidance for the model

---

## 11. Files Generated

This analysis produced the following artifacts:

```
/var/scratch/pkarageo/roboreward_dataset/
├── dataset_summary.json          # High-level statistics
├── task_index.pkl                # Full task index (4.8 MB)
├── all_task_statistics.json      # Per-task success/failure counts (1.7 MB)
├── balanced_tasks.json           # Tasks with ≥5 success + ≥5 failure (empty)
├── droid_bridge_tasks.csv        # CSV of balanced tasks (empty)
└── sample_*.pkl                  # Sample metadata examples

/var/scratch/pkarageo/roboreward_tasks/droid_general/
├── success/*.mp4                 # 20 successful trajectories
├── failure/*.mp4                 # 20 failed trajectories
└── metadata.json                 # Task description
```

---

## 12. Next Steps

1. **Analyze Task Statistics:** Parse `all_task_statistics.json` to find tasks with at least 1 success + 1 failure
2. **Identify Pairable Tasks:** Determine which tasks can provide exact matches
3. **Design Matching Algorithm:** Implement semantic or exact task matching
4. **Create Pairing Pipeline:** Script to generate (failure, success) pairs for in-context learning

---

**Report Prepared By:** Claude Code (Sonnet 4.5)
**For:** Master's Thesis - VLM-based Robotic Reward Functions
