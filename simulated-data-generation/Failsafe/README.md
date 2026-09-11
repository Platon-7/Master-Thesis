# ManiSkill Failure Dataset (FailSafe-based)

Generates labeled failure trajectories for 3 ManiSkill tasks, in the same
format as `MetaWorld/generate_failures.py`. Labels use the RoboMeter 1–4
progress scoring scheme.

---

## Approach vs. MetaWorld

| | MetaWorld | ManiSkill (this folder) |
|--|--|--|
| Expert policy | Built-in scripted policies | FailSafe motion-planning solutions |
| Failure injection | Time/progress-based noise on any timestep | Stage-based: noise at a specific motion stage |
| Scoring | Per-step physics signals (`grasp_success`, `obj_to_target`) | Stage-at-failure → score (approximate, verified via HTML) |
| Tasks | 46 | 3 (PickCube, PushCube, StackCube) |
| Score 4 | All tasks | StackCube only (via align/place stages) |

FailSafe's stage-based injection is actually **cleaner** than MetaWorld's noise injection:
each failure has a semantically well-defined cause (e.g., "gripper misaligned during grasp").

---

## Stage → Score Mapping (hypothesis)

Scores are assigned based on **which motion stage** the failure was injected at.

### FailPickCube-v1
| Stage | Injected at | Target score | Expected behavior |
|-------|-------------|-------------|-------------------|
| random | — | **1** | Random actions, robot never approaches cube |
| 0 | reach_pose | **2** | Arm misses approach to grasp region |
| 1 | grasp_pose | **2** | Arm at reach but gripper misaligns |
| 3 | lift_to_goal | **3** | Cube grasped, dropped during transport |

### FailPushCube-v1
| Stage | Injected at | Target score | Expected behavior |
|-------|-------------|-------------|-------------------|
| random | — | **1** | Random actions |
| 0 | close_gripper | **2** | Gripper closes but misses cube contact |
| 1 | reach_pose | **2** | Arm at pushing position but fails to engage |
| 2 | push_to_goal | **3** | Cube contacted, pushed off-course |

### FailStackCube-v1
| Stage | Injected at | Target score | Expected behavior |
|-------|-------------|-------------|-------------------|
| random | — | **1** | Random actions |
| 0 | search_pose | **1** | Arm barely moves from home |
| 1 | reach_cube | **2** | Arm approaches cube_A, misses |
| 2 | grasp_cube | **2** | At grasp position, gripper misaligns |
| 4 | lift_cube | **3** | Cube_A grasped, dropped before alignment |
| 5 | align_cubes | **4** | Cube_A above cube_B, fails to lower cleanly |
| 6 | place_cube | **4** | Cube_A positioned, gripper release fails |

**These are hypotheses. Verify via the HTML viewer before running grand.**

---

## Setup

### 1. Clone FailSafe

```bash
git clone https://github.com/Jimntu/FailSafe_code /scratch/$USER/FailSafe_code
```

### 2. Create conda environment

```bash
conda env create -f /scratch/$USER/FailSafe_code/environment.yml -n failsafe
conda activate failsafe
pip install pyyaml pillow  # if not already in the env
```

### 3. Set environment variable

```bash
export FAILSAFE_PATH=/scratch/$USER/FailSafe_code
```

---

## Running

### Sample run (3 eps/score, all 3 tasks, ~1–2 hours):

```bash
sbatch jobs/generate_sample.job
```

This produces `logs/sample_review.html`. Open it and verify:
- Score 2 rows: does the arm visibly miss/misalign on the object?
- Score 3 rows: is the object visibly grasped before the failure?
- Score 4 rows (StackCube only): is cube_A clearly lifted above cube_B?

### Grand run (20 eps/score, ~6–12 hours):

```bash
# Only run after sample_review.html confirms the mapping is correct
sbatch jobs/generate_grand.job
```

### Filter the HTML viewer:

```bash
# Only StackCube, scores 3 and 4
python visualize_labels.py \
    --jsonl output/grand_v1/maniskill_failures.jsonl \
    --keyframes output/grand_v1/keyframes \
    --tasks FailStackCube-v1 --scores 3 4 \
    --output logs/stack_review.html

# Only specific stages
python visualize_labels.py ... --stages lift_cube align_cubes
```

---

## Output format

Matches `MetaWorld/generate_failures.py` output:

```
output/grand_v1/
  maniskill_failures.jsonl
  keyframes/
    FailPickCube-v1_score2_reach_pose_trans_x_001/
      frame_0_0.00s.jpg   ← 8 keyframes per episode
      frame_1_0.14s.jpg
      ...
      frame_7_1.00s.jpg
```

Each JSONL line:
```json
{
  "episode_id":       "FailPickCube-v1_score2_reach_pose_trans_x_001",
  "episode_base_id":  "FailPickCube-v1_score2_reach_pose_trans_x_001",
  "task":             "FailPickCube-v1",
  "task_description": "Pick up the red cube and move it to the green goal position.",
  "target_score":     2,
  "fail_stage":       0,
  "stage_name":       "reach_pose",
  "fail_type":        "trans_x",
  "total_steps":      43,
  "final_score":      2,
  "frames": [
    {"frame_idx": 0, "step": 0, "timestamp": 0.0,  "score": 1, "frame_label": 1,
     "image_path": "FailPickCube-v1_score2_reach_pose_trans_x_001/frame_0_0.00s.jpg"},
    ...
    {"frame_idx": 7, "step": 42, "timestamp": 1.0, "score": 2, "frame_label": 2, ...}
  ]
}
```

---

## Dataset scale (grand run, N=20)

| Task | Scores | Stage slots | Episodes | JSONL entries |
|------|--------|-------------|----------|---------------|
| FailPickCube-v1 | 1,2,3 | 4 stage×type combos | ~80 | ~80 |
| FailPushCube-v1 | 1,2,3 | 4 stage×type combos | ~80 | ~80 |
| FailStackCube-v1 | 1,2,3,4 | 8 stage×type combos | ~160 | ~160 |
| **Total** | | | **~320** | **~320** |

Small by design — this is supplementary data for RoboMeter finetuning.

---

## Known limitations

1. **Per-frame labels are approximate.** They use a linear interpolation based on
   the estimated injection step (proportional stage fraction), not live physics signals.
   MetaWorld used `grasp_success` / `obj_to_target` per step. Adding ManiSkill-native
   signals is a TODO if the HTML review shows label inaccuracies.

2. **3 tasks only.** FailSafe only has full motion-planning solutions for
   PickCube, PushCube, StackCube. The `sub_task_description_all.json` in FailSafe
   hints at PegInsertion, PlugCharger, PullCube, LiftPegUpright, but these have
   no YAML configs or solution files in the public repo.

3. **No Score 4 for PickCube/PushCube.** Neither task has a "near-goal-but-failed"
   stage distinct from success. StackCube's align/place stages fill this role.
