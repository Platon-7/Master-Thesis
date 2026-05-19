# Candidate datapoints for the main figure

8 candidate datapoints from `robometer_frame_dataset`, 6/8 real-world.

## Layout

- `icl_pairs/NN_<source>_<archive>/` — each contains `failure/` + `success/` (both with 16 keyframes + `meta.json`). These are the (failure, paired-success) pairs the reward model sees during ICL.
- `labeled_failures/NN_<source>_<archive>/` — each contains 16 keyframes + `meta.json` whose `frame_labels` is a per-keyframe 1/0 indicator (1 = at-goal-state).

## How to view

Open the JPG files in any image viewer. Frames are named `frame_00_TIMES.jpg` ... `frame_15_TIMES.jpg` so they sort in temporal order. The `summary.json` next to each datapoint has the task description, source/archive, and (for labeled failures) the per-frame labels.

## Picking the main figure

Reply with the path of the datapoint(s) you want to feature.
