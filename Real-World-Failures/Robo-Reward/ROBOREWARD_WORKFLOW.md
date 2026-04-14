# RoboReward FPS Confusion Matrix Workflow

This document describes the complete workflow for extending FPS confusion matrix experiments from DROID data to RoboReward dataset.

## Overview

The workflow consists of four main phases:

1. **Dataset Inspection** - Analyze RoboReward to find balanced tasks from DROID/Bridge
2. **Video Download** - Download task-specific videos from HuggingFace
3. **Augmentation Matching** - Create heuristic links between success/failure samples
4. **FPS Experiments** - Run confusion matrix analysis at different FPS settings

## Directory Structure

```
/var/scratch/pkarageo/
├── roboreward_dataset/          # Inspection outputs
│   ├── dataset_summary.json
│   ├── task_index.pkl
│   ├── all_task_statistics.json
│   ├── balanced_tasks.json
│   └── droid_bridge_tasks.csv   # Human-readable task list
├── roboreward_tasks/            # Downloaded videos
│   └── {task_slug}/
│       ├── metadata.json
│       ├── matching_info.json
│       ├── success/
│       │   ├── *.mp4
│       │   └── samples.json
│       └── failure/
│           ├── *.mp4
│           └── samples.json
└── roboreward_results/          # FPS experiment outputs
    └── fps_experiment_RoboReward_*.json

/home/pkarageo/master-thesis/
├── Robo-Reward-FPS/
│   ├── scripts/
│   │   ├── inspect_roboreward_dataset.py
│   │   ├── download_roboreward_videos.py
│   │   ├── match_augmentations.py
│   │   └── fps_confusion_matrix_experiment.py
│   └── logs/
└── job_files/
    ├── inspect_roboreward.job
    └── fps_confusion_roboreward.job
```

## Phase 1: Dataset Inspection

### Purpose
Analyze the RoboReward dataset to identify tasks from DROID/Bridge with sufficient success and failure samples.

### Execution

```bash
# Submit inspection job
sbatch /home/pkarageo/master-thesis/job_files/inspect_roboreward.job

# Monitor progress
tail -f /home/pkarageo/master-thesis/Robo-Reward-FPS/logs/inspect_roboreward_*.txt

# Check completion
squeue -u $USER
```

### Outputs

The script will create:
- `/var/scratch/pkarageo/roboreward_dataset/dataset_summary.json` - Overall statistics
- `/var/scratch/pkarageo/roboreward_dataset/droid_bridge_tasks.csv` - Human-readable task list
- `/var/scratch/pkarageo/roboreward_dataset/balanced_tasks.json` - Tasks with ≥20 success + ≥20 failure

### Review Results

```bash
# View CSV with balanced tasks
cat /var/scratch/pkarageo/roboreward_dataset/droid_bridge_tasks.csv

# Or view JSON for programmatic access
python3 -c "
import json
with open('/var/scratch/pkarageo/roboreward_dataset/balanced_tasks.json') as f:
    tasks = json.load(f)
for i, task in enumerate(tasks[:5], 1):
    print(f'{i}. [{task[\"dataset_source\"]}] {task[\"task\"][:60]}')
    print(f'   Success: {task[\"success_samples\"]} | Failure: {task[\"failure_samples\"]}')
    print()
"
```

### Select Tasks

From the output, select 2-3 tasks with:
- Good balance (≥20 success, ≥20 failure)
- Interesting semantics (manipulation tasks preferred)
- Mix of DROID and Bridge if possible

## Phase 2: Video Download

### Purpose
Download videos for selected tasks from HuggingFace and organize into success/failure folders.

### Execution

For each selected task:

```bash
# Example 1: Task from inspection results
python /home/pkarageo/master-thesis/Robo-Reward-FPS/scripts/download_roboreward_videos.py \
    --task "Take the lid off the pot and place it on the plate." \
    --task-slug pot_on_plate_task \
    --num-success 20 \
    --num-failure 20 \
    --datasets droid bridge

# Example 2: Another task
python /home/pkarageo/master-thesis/Robo-Reward-FPS/scripts/download_roboreward_videos.py \
    --task "rearrange pillows on sofa" \
    --task-slug pillows_task \
    --num-success 20 \
    --num-failure 20 \
    --datasets droid bridge
```

### Parameters

- `--task`: Exact task instruction from RoboReward (copy from CSV/JSON)
- `--task-slug`: Filesystem-safe slug (use underscores, no spaces)
- `--num-success`: Number of success videos (reward ≥4) to download
- `--num-failure`: Number of failure videos (reward <4) to download
- `--datasets`: Dataset sources to include (e.g., droid bridge)
- `--reward-threshold`: Success threshold (default: 4)
- `--output-base`: Base output directory (default: /var/scratch/pkarageo/roboreward_tasks)

### Verify Downloads

```bash
# Check directory structure
ls -lh /var/scratch/pkarageo/roboreward_tasks/pot_on_plate_task/

# Count videos
echo "Success: $(ls /var/scratch/pkarageo/roboreward_tasks/pot_on_plate_task/success/*.mp4 | wc -l)"
echo "Failure: $(ls /var/scratch/pkarageo/roboreward_tasks/pot_on_plate_task/failure/*.mp4 | wc -l)"

# Test video readability
python3 -c "
import cv2
cap = cv2.VideoCapture('/var/scratch/pkarageo/roboreward_tasks/pot_on_plate_task/success/droid_episode_0.mp4')
print(f'FPS: {cap.get(cv2.CAP_PROP_FPS):.1f}')
print(f'Frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}')
cap.release()
"

# Review metadata
cat /var/scratch/pkarageo/roboreward_tasks/pot_on_plate_task/metadata.json | python3 -m json.tool
```

## Phase 3: Augmentation Matching

### Purpose
Create heuristic links between failure samples and potential source demonstrations.

**Important**: RoboReward lacks explicit augmentation metadata. This script uses heuristics with confidence scores.

### Execution

For each downloaded task:

```bash
python /home/pkarageo/master-thesis/Robo-Reward-FPS/scripts/match_augmentations.py \
    --task-dir /var/scratch/pkarageo/roboreward_tasks/pot_on_plate_task
```

### Matching Strategy

The script applies three levels of heuristic matching:

1. **Level 1**: Task + Dataset matching (confidence: LOW)
   - Groups samples by dataset source
   - All samples already share the same task

2. **Level 2**: Episode proximity analysis (upgrades to MEDIUM)
   - Extracts episode numbers from IDs
   - Checks if episode ranges are close (gap ≤50)

3. **Level 3**: Temporal pattern detection (upgrades to HIGH)
   - Looks for filename patterns (e.g., episode_X vs episode_X_clip)
   - Rare in RoboReward but included for completeness

### Review Matches

```bash
# View matching results
cat /var/scratch/pkarageo/roboreward_tasks/pot_on_plate_task/matching_info.json | python3 -m json.tool

# Summary view
python3 -c "
import json
with open('/var/scratch/pkarageo/roboreward_tasks/pot_on_plate_task/matching_info.json') as f:
    data = json.load(f)
print(f'Task: {data[\"task\"][:60]}...')
print(f'Total matches: {data[\"summary\"][\"total_matches\"]}')
print(f'Confidence distribution: {data[\"summary\"][\"confidence_distribution\"]}')
print()
for match in data['matches']:
    print(f'{match[\"match_id\"]}: {match[\"confidence\"].upper()}')
    print(f'  {len(match[\"success_samples\"])} success ↔ {len(match[\"failure_samples\"])} failure')
"
```

### Interpreting Confidence Scores

- **LOW**: Only task + dataset match (baseline)
- **MEDIUM**: Episode IDs are numerically close
- **HIGH**: Clear temporal patterns detected (rare)

**Note**: Due to lack of explicit metadata, even MEDIUM confidence doesn't guarantee true augmentation relationships.

## Phase 4: FPS Confusion Matrix Experiment

### Purpose
Run the confusion matrix experiment at different FPS settings to analyze model sensitivity.

### Execution

```bash
# Submit SLURM job
sbatch --export=TASK_SLUG=pot_on_plate_task /home/pkarageo/master-thesis/job_files/fps_confusion_roboreward.job

# Or with custom FPS values (not yet implemented in job script)
python /home/pkarageo/master-thesis/Robo-Reward-FPS/scripts/fps_confusion_matrix_experiment.py \
    --task-slug pot_on_plate_task \
    --fps-values 2.0 5.0 10.0 15.0 20.0 30.0 60.0 70.0 \
    --output-dir /var/scratch/pkarageo/roboreward_results
```

### Monitor Progress

```bash
# Watch job queue
squeue -u $USER

# Monitor log (live)
tail -f /home/pkarageo/master-thesis/Robo-Reward-FPS/logs/fps_confusion_roboreward_*.txt

# Check GPU usage
nvidia-smi -l 5
```

### Expected Runtime

- Model loading: ~2-3 minutes
- Per video inference: ~10-30 seconds (depends on video length)
- Total for 40 videos × 8 FPS settings: ~2-4 hours

### Results Analysis

```bash
# Find results file
ls -lht /var/scratch/pkarageo/roboreward_results/fps_experiment_RoboReward_*.json | head -1

# View summary
python3 -c "
import json
import sys

# Get most recent file
import glob
files = sorted(glob.glob('/var/scratch/pkarageo/roboreward_results/fps_experiment_RoboReward_*.json'))
if not files:
    print('No results found')
    sys.exit(1)

with open(files[-1]) as f:
    data = json.load(f)

print(f'Dataset: {data[\"experiment_info\"][\"dataset\"]}')
print(f'Task: {data[\"experiment_info\"][\"task_instruction\"][:60]}...')
print(f'Threshold: {data[\"experiment_info\"][\"score_threshold\"]}')
print()
print(f'{'FPS':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}')
print('-'*60)

for fps_label, result in sorted(data['results'].items()):
    cm = result['confusion_matrix']
    print(f'{fps_label:<10} {cm[\"accuracy\"]:>10.2%} {cm[\"precision\"]:>10.2%} {cm[\"recall\"]:>10.2%} {cm[\"f1_score\"]:>10.2%}')
"
```

### Comparison with DROID Baseline

If you have DROID results, compare:

```bash
python3 -c "
import json
import glob

# Load RoboReward results
rr_files = sorted(glob.glob('/var/scratch/pkarageo/roboreward_results/fps_experiment_RoboReward_*.json'))
with open(rr_files[-1]) as f:
    rr_data = json.load(f)

# Load DROID results (adjust path if needed)
droid_files = sorted(glob.glob('/var/scratch/pkarageo/roboreward_results/fps_experiment_DROID_*.json'))
if droid_files:
    with open(droid_files[-1]) as f:
        droid_data = json.load(f)

    print('=== FPS COMPARISON ===')
    print(f'{'FPS':<10} {'DROID F1':>12} {'RoboReward F1':>15} {'Δ':>8}')
    print('-'*50)

    for fps_label in sorted(rr_data['results'].keys()):
        rr_f1 = rr_data['results'][fps_label]['confusion_matrix']['f1_score']
        if fps_label in droid_data['results']:
            droid_f1 = droid_data['results'][fps_label]['confusion_matrix']['f1_score']
            delta = rr_f1 - droid_f1
            print(f'{fps_label:<10} {droid_f1:>12.2%} {rr_f1:>15.2%} {delta:>+8.2%}')
else:
    print('No DROID results found for comparison')
"
```

## Running Multiple Tasks

To process multiple tasks in sequence:

```bash
# Create a list of tasks
cat > /tmp/roboreward_tasks.txt <<EOF
Take the lid off the pot and place it on the plate.|pot_on_plate_task
rearrange pillows on sofa|pillows_task
EOF

# Process each task
while IFS='|' read -r task slug; do
    echo "Processing: $slug"

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

    # Submit FPS experiment (queue multiple jobs)
    sbatch --export=TASK_SLUG=$slug /home/pkarageo/master-thesis/job_files/fps_confusion_roboreward.job

done < /tmp/roboreward_tasks.txt
```

## Troubleshooting

### Issue: Dataset inspection times out

**Solution**: Reduce `NUM_SAMPLES_TO_INSPECT` in `inspect_roboreward_dataset.py` (default: 10,000)

```python
NUM_SAMPLES_TO_INSPECT = 5000  # Faster but less coverage
```

### Issue: Video download fails with 404

**Problem**: HuggingFace URL conversion may be incorrect

**Solution**: Check the video path format in the dataset. The converter assumes format:
```
hf://datasets/teetone/RoboReward@{hash}/train/{dataset}/{episode}.mp4
```

### Issue: No balanced tasks found

**Solution**: Lower the thresholds in `inspect_roboreward_dataset.py`:

```python
MIN_SUCCESS_SAMPLES = 10  # Lower from 20
MIN_FAILURE_SAMPLES = 10  # Lower from 20
```

### Issue: OOM during FPS experiment

**Solution**:
- Ensure GPU has ≥24GB VRAM (TitanRTX, A6000, etc.)
- Model uses 4-bit quantization (~8GB)
- Check no other processes are using GPU: `nvidia-smi`

### Issue: Videos can't be read with OpenCV

**Problem**: Corrupted download or unsupported codec

**Solution**: Re-download or validate manually:
```bash
ffprobe /var/scratch/pkarageo/roboreward_tasks/pot_on_plate_task/success/droid_episode_0.mp4
```

## Storage Management

All large files are stored in `/var/scratch/pkarageo/`:

```bash
# Check disk usage
du -h --max-depth=1 /var/scratch/pkarageo/

# Typical sizes:
# - roboreward_dataset/: ~100MB (inspection outputs)
# - roboreward_tasks/{task}/: ~1-2GB per task (40 videos)
# - roboreward_results/: ~10-50MB per experiment (JSON results)
```

To clean up:

```bash
# Remove downloaded videos after experiments complete
rm -rf /var/scratch/pkarageo/roboreward_tasks/pot_on_plate_task/

# Keep only results
# Results are small JSON files (~1-5MB each)
```

## Key Limitations

1. **No explicit augmentation metadata** in RoboReward
   - Matching uses heuristics with confidence scores
   - Cannot guarantee true augmentation relationships

2. **Episode proximity threshold is arbitrary** (gap ≤50)
   - May need tuning based on dataset structure

3. **Temporal patterns may be rare**
   - Level 3 matching may not find patterns
   - Most matches will be LOW or MEDIUM confidence

4. **Reward threshold assumption** (reward ≥4 = success)
   - Based on typical VLM scoring patterns
   - May need adjustment based on task semantics

## Next Steps

After completing the workflow:

1. **Analyze FPS sensitivity patterns**
   - Compare RoboReward vs DROID results
   - Identify optimal FPS for different task types

2. **Evaluate matching quality**
   - Manual inspection of HIGH confidence matches
   - Validate heuristics with sample videos

3. **Extend to other OXE datasets**
   - Try Berkeley Autolab, CMU Play Fusion, etc.
   - Compare cross-dataset performance

4. **Implement adaptive FPS selection**
   - Use FPS that maximizes F1 score per task
   - Build task-specific FPS profiles

## Contact

For issues or questions, refer to:
- Main thesis repository: `/home/pkarageo/master-thesis/`
- CLAUDE.md for general guidance
- This file for RoboReward-specific workflow
