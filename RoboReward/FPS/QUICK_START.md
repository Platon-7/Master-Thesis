# RoboReward FPS Experiments - Quick Start Guide

## TL;DR - Complete Workflow

```bash
# Phase 1: Inspect dataset (find balanced tasks)
sbatch /home/pkarageo/master-thesis/job_files/inspect_roboreward.job

# Phase 2: Download videos for a task
python /home/pkarageo/master-thesis/Robo-Reward-FPS/scripts/download_roboreward_videos.py \
    --task "Take the lid off the pot and place it on the plate." \
    --task-slug pot_on_plate_task \
    --num-success 20 \
    --num-failure 20

# Phase 3: Match augmentations
python /home/pkarageo/master-thesis/Robo-Reward-FPS/scripts/match_augmentations.py \
    --task-dir /var/scratch/pkarageo/roboreward_tasks/pot_on_plate_task

# Phase 4: Run FPS experiment
sbatch --export=TASK_SLUG=pot_on_plate_task \
    /home/pkarageo/master-thesis/job_files/fps_confusion_roboreward.job
```

## Step-by-Step

### 1. Find Balanced Tasks

```bash
# Submit inspection job
sbatch /home/pkarageo/master-thesis/job_files/inspect_roboreward.job

# Monitor progress
tail -f /home/pkarageo/master-thesis/Robo-Reward-FPS/logs/inspect_roboreward_*.txt

# View results (once complete)
cat /var/scratch/pkarageo/roboreward_dataset/droid_bridge_tasks.csv
```

### 2. Select and Download Task

```bash
# Choose a task from the CSV and download videos
# Example task: "Take the lid off the pot and place it on the plate."

python /home/pkarageo/master-thesis/Robo-Reward-FPS/scripts/download_roboreward_videos.py \
    --task "Take the lid off the pot and place it on the plate." \
    --task-slug pot_on_plate_task \
    --num-success 20 \
    --num-failure 20 \
    --datasets droid bridge

# Verify downloads
ls /var/scratch/pkarageo/roboreward_tasks/pot_on_plate_task/success/ | wc -l
ls /var/scratch/pkarageo/roboreward_tasks/pot_on_plate_task/failure/ | wc -l
```

### 3. Run Augmentation Matching

```bash
python /home/pkarageo/master-thesis/Robo-Reward-FPS/scripts/match_augmentations.py \
    --task-dir /var/scratch/pkarageo/roboreward_tasks/pot_on_plate_task

# Review matches
cat /var/scratch/pkarageo/roboreward_tasks/pot_on_plate_task/matching_info.json | python -m json.tool
```

### 4. Run FPS Experiment

```bash
# Submit SLURM job
sbatch --export=TASK_SLUG=pot_on_plate_task \
    /home/pkarageo/master-thesis/job_files/fps_confusion_roboreward.job

# Monitor
squeue -u $USER
tail -f /home/pkarageo/master-thesis/Robo-Reward-FPS/logs/fps_confusion_roboreward_*.txt
```

### 5. Analyze Results

```bash
# Find latest results
ls -lt /var/scratch/pkarageo/roboreward_results/fps_experiment_RoboReward_*.json | head -1

# Quick summary
python3 << 'EOF'
import json
import glob

files = sorted(glob.glob('/var/scratch/pkarageo/roboreward_results/fps_experiment_RoboReward_*.json'))
if not files:
    print('No results found')
    exit()

with open(files[-1]) as f:
    data = json.load(f)

print(f"Dataset: {data['experiment_info']['dataset']}")
print(f"Task: {data['experiment_info']['task_instruction'][:60]}...")
print()
print(f"{'FPS':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print('-'*60)

for fps_label, result in sorted(data['results'].items()):
    cm = result['confusion_matrix']
    print(f"{fps_label:<10} {cm['accuracy']:>10.2%} {cm['precision']:>10.2%} {cm['recall']:>10.2%} {cm['f1_score']:>10.2%}")
EOF
```

## Common Commands

### Check Job Status
```bash
squeue -u $USER
```

### Monitor GPU Usage
```bash
nvidia-smi -l 5
```

### Check Disk Usage
```bash
du -h --max-depth=1 /var/scratch/pkarageo/
```

### List All Downloaded Tasks
```bash
ls -d /var/scratch/pkarageo/roboreward_tasks/*/
```

### List All Experiment Results
```bash
ls -lht /var/scratch/pkarageo/roboreward_results/
```

### Cancel Job
```bash
scancel <job_id>
```

## File Locations

### Scripts
- Inspection: `/home/pkarageo/master-thesis/Robo-Reward-FPS/scripts/inspect_roboreward_dataset.py`
- Download: `/home/pkarageo/master-thesis/Robo-Reward-FPS/scripts/download_roboreward_videos.py`
- Matching: `/home/pkarageo/master-thesis/Robo-Reward-FPS/scripts/match_augmentations.py`
- FPS Experiment: `/home/pkarageo/master-thesis/Robo-Reward-FPS/scripts/fps_confusion_matrix_experiment.py`

### Job Files
- Inspection: `/home/pkarageo/master-thesis/job_files/inspect_roboreward.job`
- FPS Experiment: `/home/pkarageo/master-thesis/job_files/fps_confusion_roboreward.job`

### Data Directories
- Inspection outputs: `/var/scratch/pkarageo/roboreward_dataset/`
- Downloaded videos: `/var/scratch/pkarageo/roboreward_tasks/{task_slug}/`
- Experiment results: `/var/scratch/pkarageo/roboreward_results/`
- Logs: `/home/pkarageo/master-thesis/Robo-Reward-FPS/logs/`

## Typical Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Inspection | 1-2 hours | Processes 10K samples |
| Download | 10-20 minutes | 40 videos @ ~30 sec/video |
| Matching | <1 second | In-memory processing |
| FPS Experiment | 2-4 hours | 40 videos × 8 FPS settings |

## Storage Requirements

| Component | Size | Location |
|-----------|------|----------|
| Inspection outputs | ~100MB | `/var/scratch/pkarageo/roboreward_dataset/` |
| Videos per task | ~2GB | `/var/scratch/pkarageo/roboreward_tasks/{task}/` |
| Results per task | ~5MB | `/var/scratch/pkarageo/roboreward_results/` |
| Logs | <1MB | `/home/pkarageo/master-thesis/Robo-Reward-FPS/logs/` |

## Troubleshooting

### Job failed to start
```bash
# Check SLURM output
cat /home/pkarageo/master-thesis/Robo-Reward-FPS/logs/<job_name>_<job_id>.txt
```

### Out of memory
```bash
# Check available GPUs
sinfo -o "%20N %10c %10m %25f %10G"
# Request different GPU or reduce batch size
```

### Download timeout
```bash
# Re-run download script - it will continue from where it left off
# Or increase timeout in download_roboreward_videos.py (line with requests.get)
```

### Videos won't open
```bash
# Validate with ffprobe
ffprobe /var/scratch/pkarageo/roboreward_tasks/{task}/success/{video}.mp4
```

## Getting Help

- Full workflow: `cat /home/pkarageo/master-thesis/Robo-Reward-FPS/ROBOREWARD_WORKFLOW.md`
- Implementation details: `cat /home/pkarageo/master-thesis/Robo-Reward-FPS/IMPLEMENTATION_SUMMARY.md`
- General guidance: `cat /home/pkarageo/master-thesis/CLAUDE.md`

## Example: Full Pipeline for Multiple Tasks

```bash
#!/bin/bash

# Define tasks (format: "task_instruction|task_slug")
TASKS=(
    "Take the lid off the pot and place it on the plate.|pot_on_plate_task"
    "rearrange pillows on sofa|pillows_task"
)

# Process each task
for task_entry in "${TASKS[@]}"; do
    IFS='|' read -r task slug <<< "$task_entry"

    echo "Processing: $slug"
    echo "============================================"

    # Download videos
    python /home/pkarageo/master-thesis/Robo-Reward-FPS/scripts/download_roboreward_videos.py \
        --task "$task" \
        --task-slug "$slug" \
        --num-success 20 \
        --num-failure 20 \
        --datasets droid bridge

    # Run matching
    python /home/pkarageo/master-thesis/Robo-Reward-FPS/scripts/match_augmentations.py \
        --task-dir /var/scratch/pkarageo/roboreward_tasks/$slug

    # Submit FPS experiment
    sbatch --export=TASK_SLUG=$slug \
        /home/pkarageo/master-thesis/job_files/fps_confusion_roboreward.job

    echo ""
done

echo "All tasks submitted!"
echo "Monitor with: squeue -u $USER"
```

Save this script as `/tmp/process_tasks.sh`, make it executable, and run:
```bash
chmod +x /tmp/process_tasks.sh
/tmp/process_tasks.sh
```
