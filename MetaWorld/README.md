# MetaWorld Failure Dataset

## What is this?

This pipeline generates **labeled failure trajectories** for robotic manipulation tasks using [MetaWorld](https://meta-world.github.io/) — a benchmark of 50 simulated tabletop manipulation tasks (pick-and-place, drawer opening, sweeping, etc.) powered by MuJoCo physics.

The goal is to create a dataset of robot failure episodes where each episode is annotated with a **progress score from 1 to 4**, reflecting how far the robot got before failing. This data is used to finetune [RoboMeter](https://robometer.github.io/), a vision-language model that evaluates robot task progress.

---

## How failures are generated

### Expert policy

MetaWorld ships with **built-in scripted expert policies** — deterministic rule-based controllers that achieve 100% success on each task. We use these instead of training RL policies because:
- They are immediately available (no training required)
- They reliably perform each sub-step of the task (approach → grasp → transport → place)
- They produce clean, semantically meaningful behavior that degrades gracefully under noise

### Action noise injection

To produce failures, we inject **Gaussian noise** into the expert's actions at each timestep:

```
action = clip(expert_action + N(0, noise_scale), -1, 1)
```

Different noise levels produce qualitatively different failure modes:

| Noise level | Scale | Typical behavior |
|---|---|---|
| `moderate` | 0.5 | Expert mostly succeeds; ~30% failure rate |
| `heavy` | 0.8 | Fumbled grasps, partial lifts |
| `severe` | 1.0 | Erratic movement, rare grasps |
| `extreme` | 1.5 | Mostly random, occasional approach |
| `chaotic` | 2.0 | Effectively random actions |

For **score 1 specifically** (no progress), we use fully random actions (no expert) to ensure the robot never even approaches the object.

---

## Scoring (1–4)

Scores are assigned **per keyframe** based on MetaWorld's built-in task progress signals (`grasp_success`, `obj_to_target`, `grasp_reward`), which are available at every simulation step.

| Score | Meaning | Condition |
|---|---|---|
| **1** | No progress | Robot wanders, never meaningfully approaches the object |
| **2** | Approached, no grasp | Robot near object but fails to grasp; OR grasps but moves object away from goal |
| **3** | Grasped + moved toward goal | `grasp_success=1` AND `obj_to_target` decreased >10% from start |
| **4** | Major progress, not completed | `grasp_success=1` AND `obj_to_target` decreased >50% from start |

**Important**: Score is based on `obj_to_target` relative to the *initial* distance at episode start. If the robot grasps the object but moves it in the wrong direction, the score drops back to 2 (approached but no meaningful progress).

Each episode produces **8 evenly-spaced keyframes**. Scores are computed per step and the keyframe score is the step score at that timestep. Scores are enforced to be non-decreasing across keyframes (monotonicity).

---

## Tasks

| Task | Description |
|---|---|
| `pick-place-v3` | Pick up a small peg and place it at a target location |
| `drawer-open-v3` | Grasp a drawer handle and pull it open |
| `sweep-into-v3` | Push a puck across the table into a goal region |

---

## Output format

Each run produces:

```
output/
  keyframes/
    {task}_{noise_level}_{ep_idx}/
      frame_0_0.00s.jpg    ← keyframe 1 of 8
      frame_1_0.14s.jpg
      ...
      frame_7_1.00s.jpg
  metaworld_failures.jsonl ← one line per episode
```

Each line in the JSONL:
```json
{
  "episode_id": "pick-place-v3_heavy_000",
  "task": "pick-place-v3",
  "task_description": "pick place",
  "noise_level": "heavy",
  "noise_scale": 0.8,
  "total_steps": 500,
  "final_score": 3,
  "frames": [
    {"frame_idx": 0, "step": 0, "timestamp": 0.0, "score": 1},
    {"frame_idx": 1, "step": 71, "timestamp": 0.14, "score": 2},
    ...
    {"frame_idx": 7, "step": 499, "timestamp": 1.0, "score": 3}
  ]
}
```

---

## Running

### Generate sample (quick, 3 episodes per noise level):
```bash
sbatch jobs/generate_sample.job
```

### Generate full dataset:
```bash
# Edit generate_failures.py to set --episodes-per-noise 50+
sbatch jobs/generate_sample.job
```

### Generate presentation GIFs (one per score level):
```bash
sbatch jobs/make_demo_gifs.job
# Output: demo_gifs/pick_place_score{1,2,3,4}.gif
```

### Environment:
- Conda env: `metaworld`
- Cluster: DAS-5 GPU node (`gpu_a100` partition)
- Rendering: EGL headless (`MUJOCO_GL=egl`)
- Camera: `corner2` (top-down, right-side up after vertical flip)

---

## Key design decisions

- **Why scripted policies, not trained RL?** Trained SAC policies at 100K steps only learned to approach objects but not manipulate them — they got "stuck" near the object in every episode. The scripted expert reliably performs all task stages.
- **Why noise injection, not partial training?** Partially-trained policies all produce the same behavior (gripper near object, not moving) because MetaWorld's shaped reward heavily incentivizes object proximity. Noise injection on a working expert produces a true spectrum from erratic to near-success.
- **Why relative `obj_to_target` for scoring?** Distance-based scoring (absolute gripper-to-object distance) always gave score 4 to scripted expert episodes since the expert always approaches closely. Using `obj_to_target` relative to episode start captures *actual task completion progress*, not just proximity.
