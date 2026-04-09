# FPS Confusion Matrix Scripts - Usage Guide

## Two Versions

### 1. Standard Version (Single Task)
**Script**: `fps_confusion_matrix_experiment.py`
**Job File**: `fps_confusion_roboreward.job`

**Use When:**
- All videos are from the **same task** (e.g., "rearrange pillows on sofa")
- Task instruction is stored in `metadata.json` at the task directory level
- Example: Your previous experiment with pillows task

**How it Works:**
- Reads task instruction from `metadata.json`
- Uses the same task instruction for ALL videos
- Faster setup, simpler logic

**Example:**
```bash
sbatch --export=TASK_SLUG=pillows_task job_files/fps_confusion_roboreward.job
```

---

### 2. Mixed-Task Version (Multiple Tasks)
**Script**: `fps_confusion_matrix_experiment_mixed.py`
**Job File**: `fps_confusion_roboreward_mixed.job`

**Use When:**
- Videos are from **different tasks** (e.g., droid_general with mixed DROID data)
- Each video has its own task instruction
- Task instructions are stored in `samples.json` files

**How it Works:**
- Reads `success/samples.json` and `failure/samples.json`
- Extracts per-video task instructions
- Uses the **correct task instruction for each video** during inference
- More accurate for mixed-task datasets

**Example:**
```bash
sbatch --export=TASK_SLUG=droid_general job_files/fps_confusion_roboreward_mixed.job
```

---

## Key Differences

| Feature | Standard Version | Mixed-Task Version |
|---------|-----------------|-------------------|
| Task instruction source | `metadata.json` (one for all) | `samples.json` (per video) |
| Use case | Single task | Multiple tasks |
| Metadata requirement | `task_instruction` in metadata.json | `samples.json` with per-video tasks |
| Output filename | `fps_experiment_{name}_*.json` | `fps_experiment_MIXED_{name}_*.json` |
| Log header | "FPS SENSITIVITY EXPERIMENT" | "FPS SENSITIVITY EXPERIMENT (MIXED TASKS)" |

---

## Current Dataset Status

### droid_general (Mixed Tasks)
- **Use**: Mixed-task version
- **Videos**: 20 success + 20 failure from various DROID tasks
- **Samples.json**: ✓ Present with per-video task descriptions
- **Metadata**: Generic "Mixed DROID tasks (general manipulation)"
- **Command**:
  ```bash
  sbatch --export=TASK_SLUG=droid_general job_files/fps_confusion_roboreward_mixed.job
  ```

### Single-task datasets (e.g., pillows_task)
- **Use**: Standard version
- **Videos**: All from the same task
- **Metadata**: Specific task instruction in metadata.json
- **Command**:
  ```bash
  sbatch --export=TASK_SLUG=pillows_task job_files/fps_confusion_roboreward.job
  ```

---

## Output Files

### Standard Version
- Location: `/var/scratch/pkarageo/roboreward_results/`
- Format: `fps_experiment_RoboReward_{task_slug}_{timestamp}.json`
- Example: `fps_experiment_RoboReward_pillows_task_20260206_101530.json`

### Mixed-Task Version
- Location: `/var/scratch/pkarageo/roboreward_results/`
- Format: `fps_experiment_MIXED_RoboReward_{task_slug}_{timestamp}.json`
- Example: `fps_experiment_MIXED_RoboReward_droid_general_20260206_143045.json`

---

## Quick Start

### For droid_general (Mixed Tasks)
```bash
# Submit job
sbatch --export=TASK_SLUG=droid_general job_files/fps_confusion_roboreward_mixed.job

# Monitor
tail -f /home/pkarageo/master-thesis/Robo-Reward-FPS/logs/fps_confusion_roboreward_mixed_*.txt
```

### For single-task datasets
```bash
# Submit job
sbatch --export=TASK_SLUG=your_task_slug job_files/fps_confusion_roboreward.job

# Monitor
tail -f /home/pkarageo/master-thesis/Robo-Reward-FPS/logs/fps_confusion_roboreward_*.txt
```

---

## Troubleshooting

### Error: "samples.json not found"
- You're using the mixed-task version but the dataset doesn't have samples.json
- **Solution**: Use the standard version instead, or create samples.json

### Error: "task_instruction not found in metadata"
- You're using the standard version but metadata.json doesn't have task_instruction
- **Solution**: Add task_instruction to metadata.json or use the mixed-task version

### Low scores with mixed tasks
- If using standard version on mixed tasks, the model sees task mismatch
- **Solution**: Use the mixed-task version to provide correct task per video
