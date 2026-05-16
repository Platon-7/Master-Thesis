#!/usr/bin/env python3
"""Sample 30 success + 30 failure DROID trajectories from train_raw and render
frame strips with per-frame labels + task overlay. Forces task-uniqueness in
the success sample (droid has ~5,700 unique tasks in ~155k successes; without
de-duping we'd otherwise see the same "Put the marker in the cup"-style task
repeated). Also dumps droid-specific stats analogous to the metaworld audit.

The goal here is to spot anything structurally wrong in droid training examples
— frame-content, task-description quality, labels mismatch, etc. — that would
explain why droid eval AUC is below 50% (worse than chance) and gets WORSE
with ICL on, despite the LoRA bake-off (butim4ky) once showing healthy droid
behavior.

Output: /gpfs/home3/pkarageorgis1/Master-Thesis/droid_samples/
  droid_succ##_<id>.png      (one per sampled success trajectory)
  droid_fail##_<id>.png      (one per sampled failure trajectory)
  manifest.json              (the sampled rows' metadata)
  stats.txt                  (droid label-distribution audit)

Bulk-materializes columns via PyArrow.to_pylist() to avoid the ~1000× slowdown
of per-row Arrow indexing (the same trap that killed prior runs at 15+ min).
"""
import os, sys, json, glob, random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import cv2
import matplotlib.pyplot as plt
from datasets import load_from_disk

sys.stdout.reconfigure(line_buffering=True)
random.seed(7)

OUT = Path('/gpfs/home3/pkarageorgis1/Master-Thesis/droid_samples')
OUT.mkdir(exist_ok=True)

N_SUCC = 30
N_FAIL = 30

# ---- Load HF dataset ---------------------------------------------------------
print("[1/4] Loading train_raw HF dataset …")
ds = load_from_disk('/projects/prjs1958/robometer_frames_hf_full/train_raw/robometer_frames_train')
print(f"  rows: {len(ds):,}")

print("\n  bulk-materializing 4 columns via PyArrow.to_pylist() …")
import time
table = ds.data.table
t = time.time(); all_sources = table.column("data_source").to_pylist(); print(f"    data_source ({len(all_sources):,}) in {time.time()-t:.1f}s")
t = time.time(); all_quality = table.column("quality_label").to_pylist(); print(f"    quality_label in {time.time()-t:.1f}s")
t = time.time(); all_labels  = table.column("frame_labels").to_pylist();  print(f"    frame_labels in {time.time()-t:.1f}s")
t = time.time(); all_tasks   = table.column("task").to_pylist();          print(f"    task in {time.time()-t:.1f}s")

# ---- Build per-quality droid index pools + per-task index for uniqueness -----
print("\n[2/4] Indexing droid rows by quality + task …")
droid_succ_by_task = defaultdict(list)   # task -> [global_idx, ...]
droid_fail_by_task = defaultdict(list)
n_succ_total = 0
n_fail_total = 0
for i in range(len(all_sources)):
    if all_sources[i] != "droid":
        continue
    t_ = all_tasks[i] or ""
    if all_quality[i] == "successful":
        droid_succ_by_task[t_].append(i)
        n_succ_total += 1
    else:
        droid_fail_by_task[t_].append(i)
        n_fail_total += 1
print(f"  droid_succ: {n_succ_total:,} trajectories across {len(droid_succ_by_task):,} unique tasks")
print(f"  droid_fail: {n_fail_total:,} trajectories across {len(droid_fail_by_task):,} unique tasks")

# ---- Sample N_SUCC successes from DIFFERENT tasks (one per task) -------------
# Then if we have fewer than N_SUCC unique tasks, allow more from same task.
print(f"\n[3/4] Sampling {N_SUCC} successes (task-unique) + {N_FAIL} failures (task-unique where possible) …")

succ_task_keys = list(droid_succ_by_task.keys())
random.shuffle(succ_task_keys)
sampled_succ = []
for tk in succ_task_keys:
    if len(sampled_succ) >= N_SUCC: break
    sampled_succ.append(random.choice(droid_succ_by_task[tk]))

fail_task_keys = list(droid_fail_by_task.keys())
random.shuffle(fail_task_keys)
sampled_fail = []
for tk in fail_task_keys:
    if len(sampled_fail) >= N_FAIL: break
    sampled_fail.append(random.choice(droid_fail_by_task[tk]))
# Top up if not enough task uniqueness in failures (droid_fail has fewer unique tasks)
if len(sampled_fail) < N_FAIL:
    flat = [i for vs in droid_fail_by_task.values() for i in vs if i not in sampled_fail]
    random.shuffle(flat)
    sampled_fail.extend(flat[: N_FAIL - len(sampled_fail)])

print(f"  sampled {len(sampled_succ)} success indices")
print(f"  sampled {len(sampled_fail)} failure indices")

# ---- Droid-specific stats ----------------------------------------------------
print("\n  computing droid label-distribution stats …")
succ_null = 0; succ_max_dist = Counter(); succ_term_dist = Counter()
fail_null = 0; fail_max_dist = Counter(); fail_term_dist = Counter()
fail_label_freq = Counter(); succ_label_freq = Counter()

for tk, idxs in droid_succ_by_task.items():
    for i in idxs:
        labs = all_labels[i]
        if labs is None or len(labs) == 0:
            succ_null += 1
        else:
            succ_label_freq.update(labs)
            succ_max_dist[max(labs)] += 1
            succ_term_dist[labs[-1]] += 1
for tk, idxs in droid_fail_by_task.items():
    for i in idxs:
        labs = all_labels[i]
        if labs is None or len(labs) == 0:
            fail_null += 1
        else:
            fail_label_freq.update(labs)
            fail_max_dist[max(labs)] += 1
            fail_term_dist[labs[-1]] += 1

stats = []
stats.append("Droid frame_labels audit — train_raw/robometer_frames_train")
stats.append("=" * 70)
stats.append(f"Total droid: {n_succ_total + n_fail_total:,}  ({n_succ_total:,} succ, {n_fail_total:,} fail)")
stats.append(f"Success rate: {n_succ_total/(n_succ_total+n_fail_total)*100:.1f}%")
stats.append(f"")
stats.append(f"Unique tasks: {len(droid_succ_by_task):,} success, {len(droid_fail_by_task):,} failure")
stats.append(f"  avg traj/task: succ {n_succ_total/max(len(droid_succ_by_task),1):.1f}, fail {n_fail_total/max(len(droid_fail_by_task),1):.1f}")
stats.append(f"")
stats.append(f"frame_labels=None / empty:")
stats.append(f"  successes: {succ_null:,} / {n_succ_total:,}  ({succ_null/max(n_succ_total,1)*100:.1f}%)")
stats.append(f"  failures:  {fail_null:,} / {n_fail_total:,}  ({fail_null/max(n_fail_total,1)*100:.1f}%)")
stats.append(f"")
stats.append(f"Per-frame label-value frequency:")
stats.append(f"  successes: {dict(sorted(succ_label_freq.items()))}")
stats.append(f"  failures:  {dict(sorted(fail_label_freq.items()))}")
stats.append(f"")
stats.append(f"MAX-label-per-trajectory distribution:")
stats.append(f"  successes: {dict(sorted(succ_max_dist.items()))}")
stats.append(f"  failures:  {dict(sorted(fail_max_dist.items()))}")
stats.append(f"")
stats.append(f"TERMINAL frame label distribution (last keyframe of trajectory):")
stats.append(f"  successes: {dict(sorted(succ_term_dist.items()))}")
stats.append(f"  failures:  {dict(sorted(fail_term_dist.items()))}")
stats.append(f"")
# Bug-detection flags
if succ_null < 0.95 * n_succ_total and n_succ_total > 0:
    pct = (n_succ_total - succ_null) / n_succ_total * 100
    stats.append(f"⚠️  {pct:.1f}% of droid SUCCESSES have non-None frame_labels (training routes these through failure-rubric → terminal target capped at <=0.75). Compare to metaworld where the patch dropped this to 0%.")
if fail_term_dist and succ_term_dist:
    overlap = set(fail_term_dist.keys()) & set(succ_term_dist.keys())
    if overlap:
        stats.append(f"⚠️  Terminal-label overlap between succ and fail: {sorted(overlap)} (model can't disambiguate via terminal-rubric alone if these are the dominant values).")

stats_path = OUT / 'stats.txt'
stats_path.write_text("\n".join(stats) + "\n")
print(f"  stats → {stats_path}")
print("".join("    " + s + "\n" for s in stats))

# ---- Render frame-strip PNGs --------------------------------------------------
print(f"\n[4/4] Rendering {len(sampled_succ) + len(sampled_fail)} PNGs …")
def render(idx, kind, n):
    row = ds[idx]
    mp4_path = row['frames']
    labels = row['frame_labels']
    quality = row['quality_label']
    traj_id = (row['id'] or 'no_id')[:80]
    task = (row['task'] or '').strip().replace('\n', ' / ')[:140]
    data_source = row['data_source']
    partial_succ = row.get('partial_success', None)

    if not os.path.exists(mp4_path):
        return f"SKIP {traj_id}: mp4 not found"

    cap = cv2.VideoCapture(mp4_path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return f"SKIP {traj_id}: 0 frames decoded"

    n_show = min(len(frames), 16)
    step = max(1, len(frames) // n_show)
    show_frames = frames[::step][:n_show]
    show_labels = labels[::step][:n_show] if labels else [None] * n_show

    cols = 8
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8 + 1.2))
    if rows == 1:
        axes = np.array([axes])
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c] if (rows > 1 or isinstance(axes[r], np.ndarray)) else axes[c]
            i = r * cols + c
            if i < n_show:
                ax.imshow(show_frames[i])
                lab = show_labels[i] if show_labels else None
                ax.set_title(f't={i*step}  L={lab}', fontsize=8)
            ax.axis('off')

    labels_str = labels if labels else 'None (success → t/T path)'
    suptitle = (
        f'DROID / {quality.upper()}  |  partial_success={partial_succ!r}\n'
        f'ID: {traj_id}\n'
        f'Task: {task}\n'
        f'frame_labels={labels_str}'
    )
    fig.suptitle(suptitle, fontsize=9, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    out_png = OUT / f'droid_{kind}{n+1:02d}_{traj_id}.png'
    plt.savefig(out_png, dpi=80, bbox_inches='tight')
    plt.close()
    return f"  wrote {out_png.name}"

manifest = {'succ': [], 'fail': []}
print("  --- successes ---")
for n, idx in enumerate(sampled_succ):
    msg = render(idx, 'succ', n)
    print(msg)
    manifest['succ'].append({
        'idx': idx, 'id': ds[idx]['id'], 'task': (ds[idx]['task'] or '')[:120],
        'frame_labels': ds[idx]['frame_labels'], 'partial_success': ds[idx].get('partial_success'),
    })
print("\n  --- failures ---")
for n, idx in enumerate(sampled_fail):
    msg = render(idx, 'fail', n)
    print(msg)
    manifest['fail'].append({
        'idx': idx, 'id': ds[idx]['id'], 'task': (ds[idx]['task'] or '')[:120],
        'frame_labels': ds[idx]['frame_labels'], 'partial_success': ds[idx].get('partial_success'),
    })

manifest_path = OUT / 'manifest.json'
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"\nDone. Output in {OUT}/")
print(f"  - {len(manifest['succ'])} success PNGs")
print(f"  - {len(manifest['fail'])} failure PNGs")
print(f"  - {manifest_path.name}")
print(f"  - stats.txt  ← READ THIS first")
