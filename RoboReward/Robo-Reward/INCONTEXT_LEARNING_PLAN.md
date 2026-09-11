# In-Context Learning: Failure-Success Pairing Plan

**Goal:** Create failure-success pairs from RoboReward dataset for in-context learning
**Approach:** For each failed trajectory, pair it with a successful demonstration to guide the VLM

---

## Problem Statement

**Current Challenge:**
- We want to improve VLM predictions on failed trajectories by showing them successful examples
- Ideal: For each failure on task "A", show a success on the same task "A"
- Reality: RoboReward dataset has **0 tasks** with ≥5 success + ≥5 failure examples

**Why This Is Hard:**
1. Dataset has 4,551 unique tasks across 45,072 samples
2. Most tasks are heavily skewed (either mostly success or mostly failure)
3. No pre-existing task grouping for pairing

---

## Proposed Solution: 3-Phase Approach

### Phase 1: Analyze Task Overlap (WITHOUT full download)

**Objective:** Find tasks that have at least 1 success AND 1 failure example

**Method:**
1. Parse `/var/scratch/pkarageo/roboreward_dataset/all_task_statistics.json` (1.7 MB)
2. Extract tasks where: `success_count ≥ 1 AND failure_count ≥ 1`
3. Rank by total samples (prefer tasks with multiple examples)
4. Filter by dataset (prioritize droid, bridge for consistency)

**Expected Output:**
```
Task: "Put the cup on the table"
  Dataset: droid
  Successes: 3 videos
  Failures: 2 videos
  → Can create 2 pairs (limited by min(success, failure))
```

**Script:** `analyze_pairable_tasks.py`

**Deliverable:** CSV with columns:
- `task_description`
- `dataset`
- `success_count`
- `failure_count`
- `max_pairs` (min of success/failure)

---

### Phase 2: Strategic Data Download

**Objective:** Download only the videos we need for pairing

**Strategy:**

#### Option A: Exact Task Matching (Preferred)
- Select top 10-20 tasks with both outcomes
- Download all success AND failure videos for those specific tasks
- Estimated: ~50-200 videos total (not the full 45K dataset!)

**Example Tasks to Target:**
```
Task: "Pick up the cup and place it on the table"
  → Download: 5 success videos + 3 failure videos
  → Can create: 3 pairs

Task: "Open the drawer"
  → Download: 4 success videos + 4 failure videos
  → Can create: 4 pairs
```

#### Option B: Action-Based Matching (Fallback)
If exact matches are sparse:
- Group tasks by action type: "grasping", "placing", "wiping", "sliding", etc.
- Pair failures with successes from the same action category
- Less specific but more data available

**Script:** `download_pairable_videos.py` (modify existing download script)

**Deliverable:**
```
/var/scratch/pkarageo/roboreward_pairs/
├── task_001/
│   ├── success/
│   │   ├── video_001.mp4
│   │   ├── video_002.mp4
│   ├── failure/
│   │   ├── video_003.mp4
│   │   ├── video_004.mp4
│   └── metadata.json
├── task_002/
...
```

---

### Phase 3: Create Pairing Pipeline

**Objective:** Generate (failure, success) pairs for in-context learning

**Pairing Strategies:**

#### Strategy 1: One-to-One Exact Match
```
For each failure video on task "A":
  → Pair with ONE random success video on task "A"

Example:
  Failure: "open_drawer_fail_001.mp4" (task: "Open the drawer")
  Success: "open_drawer_success_002.mp4" (task: "Open the drawer")
```

#### Strategy 2: One-to-Many Augmentation
```
For each failure video:
  → Pair with ALL success videos on same task (data augmentation)

Example:
  Failure: "open_drawer_fail_001.mp4"
  → Pair 1: (fail_001, success_001)
  → Pair 2: (fail_001, success_002)
  → Pair 3: (fail_001, success_003)
```

#### Strategy 3: Semantic Similarity Fallback
```
For failures without exact task match:
  → Find most similar success using sentence embeddings
  → Cosine similarity on task descriptions

Example:
  Failure task: "Put the blue cup on the table"
  No exact match found
  → Most similar success: "Place the red mug on the desk" (similarity: 0.87)
```

**Script:** `create_incontext_pairs.py`

**Output Format:**
```json
{
  "pairs": [
    {
      "id": "pair_001",
      "failure_video": "task_001/failure/video_003.mp4",
      "success_video": "task_001/success/video_001.mp4",
      "task": "Open the drawer",
      "match_type": "exact",
      "similarity_score": 1.0
    },
    {
      "id": "pair_002",
      "failure_video": "task_002/failure/video_005.mp4",
      "success_video": "task_003/success/video_010.mp4",
      "task_failure": "Put the blue cup on the table",
      "task_success": "Place the red mug on the desk",
      "match_type": "semantic",
      "similarity_score": 0.87
    }
  ]
}
```

---

## Implementation Plan

### Step 1: Analyze Existing Data (TODAY)

**Script:** `analyze_pairable_tasks.py`

```python
"""
Analyze which tasks in RoboReward have both success and failure examples
WITHOUT downloading the full dataset.
"""

import json
from typing import Dict, List
import pandas as pd

def load_task_statistics(path: str) -> Dict:
    """Load pre-computed task statistics."""
    with open(path, 'r') as f:
        return json.load(f)

def find_pairable_tasks(stats: Dict, min_success=1, min_failure=1) -> List[Dict]:
    """
    Find tasks with at least min_success AND min_failure examples.

    Returns list of dicts with:
      - task_key: "dataset__task_description"
      - dataset: source dataset name
      - task: task description
      - success_count: number of successful trajectories
      - failure_count: number of failed trajectories
      - max_pairs: min(success_count, failure_count)
    """
    pairable = []

    for task_key, task_data in stats.items():
        success = task_data.get('success_count', 0)
        failure = task_data.get('failure_count', 0)

        if success >= min_success and failure >= min_failure:
            dataset, task = task_key.split('__', 1)
            pairable.append({
                'task_key': task_key,
                'dataset': dataset,
                'task': task,
                'success_count': success,
                'failure_count': failure,
                'max_pairs': min(success, failure),
                'total_samples': success + failure
            })

    # Sort by max_pairs descending (tasks with most pairing potential)
    pairable.sort(key=lambda x: x['max_pairs'], reverse=True)
    return pairable

def main():
    stats_path = '/var/scratch/pkarageo/roboreward_dataset/all_task_statistics.json'

    print("Loading task statistics...")
    stats = load_task_statistics(stats_path)
    print(f"Loaded {len(stats)} unique tasks")

    print("\n" + "="*80)
    print("FINDING PAIRABLE TASKS")
    print("="*80)

    # Find all tasks with at least 1 success and 1 failure
    pairable = find_pairable_tasks(stats, min_success=1, min_failure=1)

    print(f"\n✅ Found {len(pairable)} tasks with both success AND failure examples")

    # Filter for droid and bridge datasets
    droid_bridge = [t for t in pairable if t['dataset'] in ['droid', 'bridge']]
    print(f"✅ {len(droid_bridge)} from droid/bridge datasets")

    # Save results
    output_csv = '/var/scratch/pkarageo/roboreward_dataset/pairable_tasks.csv'
    df = pd.DataFrame(pairable)
    df.to_csv(output_csv, index=False)
    print(f"\n💾 Saved to: {output_csv}")

    # Show top 20
    print("\n" + "="*80)
    print("TOP 20 TASKS BY PAIRING POTENTIAL")
    print("="*80)

    print(f"\n{'Dataset':<15} {'Max Pairs':<10} {'Success':<8} {'Failure':<8} Task")
    print("-" * 100)
    for task in pairable[:20]:
        print(f"{task['dataset']:<15} {task['max_pairs']:<10} "
              f"{task['success_count']:<8} {task['failure_count']:<8} "
              f"{task['task'][:60]}")

    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Total pairable tasks: {len(pairable)}")
    print(f"Total possible pairs: {sum(t['max_pairs'] for t in pairable)}")
    print(f"Average pairs per task: {sum(t['max_pairs'] for t in pairable) / len(pairable):.2f}")
    print(f"\nFrom droid/bridge only: {len(droid_bridge)} tasks")
    print(f"Possible pairs (droid/bridge): {sum(t['max_pairs'] for t in droid_bridge)}")

if __name__ == "__main__":
    main()
```

**Expected Runtime:** <1 minute (just parsing JSON, no video download)

**Expected Output:**
- List of tasks with both success/failure
- Estimate of how many pairs we can create
- Decision point: Is there enough data? Do we need semantic matching?

---

### Step 2: Download Targeted Videos (NEXT)

Once we know which tasks are pairable, modify `download_roboreward_videos.py` to:
1. Accept a list of specific task_keys to download
2. Download ALL videos for those tasks (both success and failure)
3. Organize by task for easy pairing

**Estimated Download:** 50-200 videos (vs. 45,072 full dataset)

---

### Step 3: Create Pairing Script (AFTER DOWNLOAD)

Implement `create_incontext_pairs.py` to:
1. Scan downloaded task folders
2. Create (failure, success) pairs using chosen strategy
3. Export pairing manifest (JSON) for experimentation

---

### Step 4: Integrate with FPS Experiments

Modify FPS confusion matrix experiments to:
1. Load pairing manifest
2. For each failure video, prepend the paired success video as in-context example
3. Measure if in-context learning improves VLM accuracy

**Hypothesis:** Showing a successful demonstration before the failure video will help the VLM better identify what went wrong.

---

## Alternative: Semantic Matching Without Download

If Step 1 reveals insufficient exact matches, implement semantic pairing:

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode all task descriptions
success_tasks = ["Put the cup on the table", "Open the drawer", ...]
failure_tasks = ["Pick up the remote", "Close the door", ...]

success_embeddings = model.encode(success_tasks)
failure_embeddings = model.encode(failure_tasks)

# For each failure, find most similar success
similarities = cosine_similarity(failure_embeddings, success_embeddings)

for i, fail_task in enumerate(failure_tasks):
    best_match_idx = similarities[i].argmax()
    best_score = similarities[i][best_match_idx]

    print(f"Failure: {fail_task}")
    print(f"  → Best success match: {success_tasks[best_match_idx]}")
    print(f"  → Similarity: {best_score:.3f}")
```

**Advantage:** Can work with our existing droid_general data (20 success + 20 failure)
**Disadvantage:** Not exact matches, may be noisy

---

## Timeline Estimate

| Phase | Task | Time | Output |
|-------|------|------|--------|
| 1 | Run `analyze_pairable_tasks.py` | 5 min | CSV of pairable tasks |
| 2 | Review results, select tasks | 15 min | List of tasks to download |
| 3 | Download targeted videos | 30-60 min | ~50-200 videos organized by task |
| 4 | Implement pairing script | 1-2 hours | Pairing manifest (JSON) |
| 5 | Test on small subset | 30 min | Verify pairs are correct |
| 6 | Integrate with FPS experiments | 2-3 hours | Modified experiment script |

**Total:** ~1 day of implementation + testing

---

## Success Metrics

We'll know this works if:
1. ✅ We find >50 tasks with both success and failure examples
2. ✅ We can create >100 failure-success pairs
3. ✅ In-context learning improves VLM F1 score by >5% on failure classification
4. ✅ The approach scales to more tasks without requiring full dataset download

---

## Next Action

**Run Step 1 NOW:** Execute `analyze_pairable_tasks.py` to see how many pairable tasks exist in RoboReward.

This will tell us:
- Is exact matching viable? (if yes, proceed with targeted download)
- Do we need semantic matching? (if exact matches are sparse)
- How much data do we need to download?

**Decision point:** Once we see the results, we can choose the best path forward.
