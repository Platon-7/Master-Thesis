# MetaWorld Failure Dataset Generation

## Goal

Enrich [RoboMeter](https://robometer.github.io/)'s evaluation dataset with synthetic failure
trajectories from MetaWorld simulation. RoboMeter scores robot episodes on a 1–4 scale
measuring task-completion percentage; our dataset provides labeled failure episodes covering
every point on that scale for 46 MetaWorld tasks, rendered from 3 camera viewpoints.

The resulting JSONL + image dataset is intended as training/evaluation data for a
**foundation reward model** that can assign task-completion scores to robot manipulation
episodes without task-specific supervision.

---

## Scoring Scheme

Each episode is labeled with a **target score** (1–4) matching RoboMeter's rubric:

| Score | Meaning |
|-------|---------|
| 1 | No meaningful progress — arm wanders, misses object entirely |
| 2 | Approach only — arm reaches object/handle but does not engage |
| 3 | Partial progress — object grasped/moved but task not near completion |
| 4 | Near-complete — significant progress (>40–50%) but task ultimately fails |

### Per-frame labels

Every episode stores 8 uniformly-sampled keyframes. Each frame carries two fields:

- **`score`** — raw physics-based detection at that frame (what the environment reports)
- **`frame_label`** — post-processed label following the trajectory spec:
  - Score 1 episodes: all frames labeled 1
  - Freeze/misplace modes: labels are monotonically non-decreasing (1 → 2 → 3 → 4)
  - Drop/wrong_dir modes: labels can regress after the peak (e.g. 1 → 2 → 3 → 2) but
    floor at 2 once score ≥ 3 has been reached

`frame_label` is the ground truth used for training/evaluation. `score` is kept for debugging.

---

## Task Categories

Tasks are grouped into 4 categories that determine which scores are valid and how
noise is injected:

| Category | Tasks | Valid scores | Notes |
|----------|-------|-------------|-------|
| `grasp_move` | pick-place, assembly, hammer, box-close, coffee-pull, … (13 tasks) | 1 2 3 4 | Full 4-score range; scoring uses `grasp_success` + `obj_to_target` |
| `push_sweep` | push, soccer, sweep-into, plate-slide, … (11 tasks) | 1 2 3 4 | Score uses `in_place_reward` progress; soccer skips score 3 |
| `mechanism` | doors, drawers, faucets, handles, windows, disassemble (17 tasks) | 1 2 4 | Single-action tasks; score 3 is not meaningful — skipped |
| `press` | button-press variants, coffee-button (5 tasks) | 1 2 4 | Like mechanism; threshold-based scoring on `grasp_reward` progress |

**Excluded tasks:** `reach-v3`, `reach-wall-v3`, `peg-insertion-side-v3`,
`peg-unplug-side-v3` — either no stable policy or degenerate score distributions.

**Total: 46 tasks.**

---

## Failure Generation Pipeline

### Overview

For each (task, target_score) pair the pipeline:
1. Runs the scripted expert policy until a task-specific **injection point** is reached
2. Switches to a **noise mode** that induces the desired failure type
3. Runs for `noisy_steps` more steps
4. Checks whether the final score matches the target
5. Repeats until `n_episodes` successes are collected (or the attempt budget runs out)

### Injection Triggers

Each category has a dedicated `should_inject_*` function. Injection fires when the
physics signal (IPR progress, obj_to_target distance, grasp_success) crosses a
category-specific threshold:

- **Score 1** — inject immediately (random noise from step 0)
- **Score 2** — inject once arm is near object (`gr > 0.4` or approach progress > 5%)
- **Score 3** — inject once object is grasped and meaningfully moved (progress > 15–25%)
- **Score 4** — inject deeper into the task (progress > 40–55%, task-specific)

### Noise Modes

| Mode | Behaviour | Used for |
|------|-----------|----------|
| `noise` | Fully random actions | Score 1 |
| `wrong_dir` | Strong bias away from goal | Score 1, 3 |
| `freeze` | Arm holds position (0%–90% expert force, task-dependent) | All scores |
| `drop` | Gripper forced open mid-transport | Score 3 |
| `late_drop` | Gripper forced open near goal | Score 4 |
| `misplace` | Expert + weak random offset near goal | Score 4 |
| `lift_away` | Arm lifts up, loses contact with object | Score 3 push_sweep |
| `push_sideways` | Arm pushes perpendicular to goal | Score 3–4 push_sweep |

**Freeze variants** for special tasks:
- `EASY_COMPLETE_MECHANISM_TASKS` (faucet, coffee-button): pure freeze (0% force) because
  50% expert still pushes mechanism to completion
- `SPRING_BACK_MECHANISM_TASKS` (handle-press, handle-pull-side): 75% expert force to
  counteract spring-back
- `SPRING_PULL_TASKS` (coffee-pull): 30% force to hold mug partially out of machine
- `stick-push-v3` score 4: 15% gentle push to maintain stick–box contact

### Episode Collection (`collect_episodes`)

```
mode_sequence = get_mode_sequence(category, target_score, n_episodes)
for each slot_mode in mode_sequence:
    try slot_mode up to max_per_mode=100 times
    if failed and slot_mode != "freeze":
        try "freeze" as fallback up to max_per_mode times
```

Max attempts per score per task = `n_episodes × 200` (100 primary + 100 fallback per slot).

The mode sequence for n=20 cycles through 2–3 visually distinct failure modes to ensure
diversity across the collected episodes.

---

## Multi-Camera Rendering

Each episode is rendered from 3 cameras simultaneously using the same physics trajectory:

| Camera | Description |
|--------|-------------|
| `corner2` | Standard 45° oblique view (default, always informative) |
| `corner3` | Third oblique angle — same scene, different perspective |
| `gripperPOV` | First-person / egocentric view from the gripper |

Camera switching is implemented by modifying `mujoco_renderer.camera_id` between render
calls (via `mujoco.mj_name2id`). The physics state is shared; only the viewpoint changes.

Each (episode × camera) pair becomes a **separate JSONL entry** with its own `episode_id`
suffix (e.g. `pick-place-v3_score3_004_corner2`). The `episode_base_id` field links all
three camera views back to the same physical episode. Frame labels are identical across
cameras (they are computed from physics, not images).

---

## Output Format

### Directory structure

```
output/grand_v1/
├── metaworld_failures.jsonl          # all metadata + labels
└── keyframes/
    └── {task}_{scoreN}_{idx}/
        ├── corner2/
        │   ├── frame_0_0.00s.jpg
        │   ├── frame_1_0.14s.jpg
        │   └── … (8 frames)
        ├── corner3/
        │   └── … (8 frames)
        └── gripperPOV/
            └── … (8 frames)
```

### JSONL schema

```json
{
  "episode_id":      "pick-place-v3_score3_004_corner2",
  "episode_base_id": "pick-place-v3_score3_004",
  "task":            "pick-place-v3",
  "task_description":"Pick and place a puck to a goal.",
  "task_category":   "grasp_move",
  "target_score":    3,
  "camera":          "corner2",
  "noise_mode":      "drop",
  "noise_scale":     0.5,
  "total_steps":     87,
  "final_score":     3,
  "frames": [
    {
      "frame_idx":   0,
      "step":        0,
      "timestamp":   0.0,
      "score":       1,
      "frame_label": 1,
      "image_path":  "pick-place-v3_score3_004/corner2/frame_0_0.00s.jpg"
    },
    ...
  ]
}
```

`image_path` is relative to the `keyframes/` root, making the dataset portable.

---

## Resume / Fault Tolerance

The script is **fully resumable**. On startup it reads the existing JSONL and skips any
(task, score) combination that already has enough episodes. If a job is killed mid-run,
resubmitting with the same `--output-dir` continues from where it left off:

```
Score 2 --- SKIP   (already have 20/20)
Score 4 --- RESUME (have 12, need 8 more)
```

Episode indices continue from the last collected index so no data is overwritten.

---

## Completeness Report

At the end of each run the script prints a summary of what was and was not collected:

```
COMPLETENESS SUMMARY
  Target: 20 episodes per score per task
  Fully collected: 148 task/score combinations
  Partial (3 combos):
    coffee-pull-v3 score 4: 11/20
    soccer-v3 score 4: 14/20
    faucet-open-v3 score 4: 17/20
  Missing / 0 collected (8 combos):
    handle-press-v3 score 4: 0/20
    handle-press-side-v3 score 4: 0/20
    window-close-v3 score 4: 0/20
    window-open-v3 score 4: 0/20
    stick-push-v3 score 4: 0/20
    drawer-close-v3 score 2: 0/20
    drawer-open-v3 score 4: 0/20
    disassemble-v3 score 2: 0/20
```

Missing combinations are **accepted** — they reflect physical limitations (spring-back
mechanisms that cannot be held at a partial-progress state reliably). The dataset simply
has no score-4 examples for those tasks.

---

## Dataset Scale (grand run, N=20)

| Category | Tasks | Scores | Episodes | × 3 cameras | JSONL entries |
|----------|-------|--------|----------|-------------|---------------|
| grasp_move | 13 | 4 | 13×4×20 = 1040 | × 3 | 3120 |
| push_sweep | 11 | ~3.9 avg | ~858 | × 3 | 2574 |
| mechanism | 17 | 3 | 17×3×20 = 1020 | × 3 | 3060 |
| press | 5 | 3 | 5×3×20 = 300 | × 3 | 900 |
| **Total** | **46** | | **~3218** | **× 3** | **~9660** |

Storage: approximately **2–3 GB** (8 keyframes × JPEG at 120×90 × 9660 entries).

_Actual numbers will be lower for task/score combos with 0 or partial collection._

---

## Running the Pipeline

### Trial run (25 tasks, 3 eps/score, 1 camera — already completed)
```bash
sbatch MetaWorld/jobs/generate_trial.job
# → output_sample/trial_v1/
# → logs/trial_labels_review.html
```

### Grand run (46 tasks, 20 eps/score, 3 cameras)
```bash
sbatch MetaWorld/jobs/generate_grand.job
# → output/grand_v1/
# → logs/grand_labels_review.html  (generated automatically at end)
```

**Estimated wall time: 12–16 hours** on 1 A100 GPU.  
Job time limit is set to 48 hours — safe margin for hard tasks.

### Visualise results
```bash
python MetaWorld/visualize_labels.py \
    --jsonl output/grand_v1/metaworld_failures.jsonl \
    --keyframes output/grand_v1/keyframes \
    --output logs/grand_labels_review.html

# Filter to specific tasks or scores:
python MetaWorld/visualize_labels.py \
    --jsonl output/grand_v1/metaworld_failures.jsonl \
    --keyframes output/grand_v1/keyframes \
    --tasks pick-place-v3 door-open-v3 \
    --scores 3 4 \
    --output logs/review_filtered.html
```

### Calibration parameters (fixed for all runs)
| Parameter | Value | Meaning |
|-----------|-------|---------|
| `--noise-scale` | 0.5 | Magnitude of injected noise actions |
| `--noisy-steps` | 40 | Steps run after injection (dynamic: `max(20, min(noise_start_step, 200))`) |
| `--freeze-fraction` | 0.12 | Unused legacy parameter |
| `max_per_mode` | 100 | Max attempts per episode slot before giving up |

---

## Key Files

| File | Purpose |
|------|---------|
| `generate_failures.py` | Main data collection script |
| `visualize_labels.py` | HTML visualiser for frame labels |
| `jobs/generate_grand.job` | Grand run SLURM job |
| `jobs/generate_trial.job` | 25-task trial (completed) |
| `jobs/smoke_test_cameras.job` | 2-task smoke test for camera validation |
| `output/grand_v1/metaworld_failures.jsonl` | Output metadata + labels |
| `output/grand_v1/keyframes/` | Output images |
