#!/usr/bin/env python3
"""Sample 10 success + 10 failure trajectories per data family (droid / failsafe /
roboreward / robometer / metaworld) and render frame strips with per-frame
labels overlaid. Also computes per-family label-distribution stats so we can
spot bugs analogous to the metaworld success-label issue.

For metaworld specifically: also verify the success-label patch took effect
(frame_labels=None on 100% of metaworld successes).

Output: /gpfs/home3/pkarageorgis1/Master-Thesis/metaworld_samples/
  <family>_succ##_<id>.png
  <family>_fail##_<id>.png
  manifest_per_family.json   (the sampled rows' metadata)
  stats.txt                (per-family label distribution summary)
"""
import os, sys, json, glob, random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import cv2
import matplotlib.pyplot as plt
from datasets import load_from_disk

# Force unbuffered stdout so SLURM logs reflect progress in real time.
sys.stdout.reconfigure(line_buffering=True)

random.seed(7)
OUT = Path('/gpfs/home3/pkarageorgis1/Master-Thesis/metaworld_samples')
OUT.mkdir(exist_ok=True)

# ---- Build family → set(data_source) map from manifest 'family' field --------
print("[1/4] Building family→data_source map from manifests …")
family_sources = defaultdict(set)
for fam_dir in ['droid', 'failsafe', 'metaworld', 'roboreward', 'robometer']:
    for f in sorted(glob.glob(f'/projects/prjs1958/robometer_frame_dataset/{fam_dir}/manifests/*.jsonl')):
        try:
            with open(f) as fh:
                line = fh.readline().strip()
                if line:
                    d = json.loads(line)
                    if d.get('family'):
                        family_sources[fam_dir].add(d['family'])
        except Exception:
            pass
    print(f"  {fam_dir}: {len(family_sources[fam_dir])} sources -> {sorted(family_sources[fam_dir])[:5]}{'...' if len(family_sources[fam_dir])>5 else ''}")

# ---- Load HF dataset ---------------------------------------------------------
print("\n[2/4] Loading train_raw HF dataset …")
ds = load_from_disk('/projects/prjs1958/robometer_frames_hf_full/train_raw/robometer_frames_train')
print(f"  rows: {len(ds):,}")

# CRITICAL: ds["col"] returns an Arrow column reference; per-row indexed
# access stays in PyArrow and converts to Python on each access (~50-100us
# each → 15+ min for 860k×3). Instead, bulk-materialize each column to a
# real Python list via the underlying PyArrow table's .to_pylist().
print("\n  bulk-materializing 3 columns via PyArrow.to_pylist() …")
import time
table = ds.data.table
t = time.time(); all_sources = table.column("data_source").to_pylist(); print(f"    data_source ({len(all_sources):,} entries) materialized in {time.time()-t:.1f}s")
t = time.time(); all_quality = table.column("quality_label").to_pylist(); print(f"    quality_label materialized in {time.time()-t:.1f}s")
t = time.time(); all_labels  = table.column("frame_labels").to_pylist();  print(f"    frame_labels materialized in {time.time()-t:.1f}s")

# ---- Compute per-family label stats and pick 10+10 indices per family --------
print("\n[3/4] Computing label stats and sampling indices …")
stats_lines = []
sampled_indices = defaultdict(list)

# Build a source → family reverse index so we can do ONE pass over all rows
# instead of N family-passes. 1× iteration of 860k Python lists ≈ 3-5s.
source_to_family = {s: fam for fam, srcs in family_sources.items() for s in srcs}

# Per-family accumulators
fam_data = {fam: {
    'succ_idx': [], 'fail_idx': [],
    'succ_lab': Counter(), 'fail_lab': Counter(),
    'succ_max': Counter(), 'fail_max': Counter(),
    'succ_null': 0, 'fail_null': 0,
    'succ_term': Counter(), 'fail_term': Counter(),
    'n_succ': 0, 'n_fail': 0,
} for fam in family_sources}

import time
t = time.time()
for i in range(len(all_sources)):
    s = all_sources[i]
    fam = source_to_family.get(s)
    if fam is None: continue
    q = all_quality[i]
    labs = all_labels[i]
    d = fam_data[fam]
    if q == 'successful':
        d['n_succ'] += 1
        d['succ_idx'].append(i)
        if labs is None or len(labs) == 0:
            d['succ_null'] += 1
        else:
            d['succ_lab'].update(labs)
            d['succ_max'][max(labs)] += 1
            d['succ_term'][labs[-1]] += 1
    elif q == 'failure':
        d['n_fail'] += 1
        d['fail_idx'].append(i)
        if labs is None or len(labs) == 0:
            d['fail_null'] += 1
        else:
            d['fail_lab'].update(labs)
            d['fail_max'][max(labs)] += 1
            d['fail_term'][labs[-1]] += 1
print(f"  scan done in {time.time()-t:.1f}s")

# Sample 10+10 indices per family and emit stats lines
for fam in family_sources:
    d = fam_data[fam]
    succ_idx = d['succ_idx'][:]
    fail_idx = d['fail_idx'][:]
    succ_lab_counter = d['succ_lab']
    fail_lab_counter = d['fail_lab']
    succ_max_counter = d['succ_max']
    fail_max_counter = d['fail_max']
    succ_null_labels = d['succ_null']
    fail_null_labels = d['fail_null']
    succ_terminal_label_counter = d['succ_term']
    fail_terminal_label_counter = d['fail_term']
    n_succ = d['n_succ']
    n_fail = d['n_fail']
    random.shuffle(succ_idx)
    random.shuffle(fail_idx)
    sampled_indices[fam] = (succ_idx[:10], fail_idx[:10])
    # Stats summary
    stats_lines.append(f"\n=== {fam} ===")
    stats_lines.append(f"  total: {n_succ + n_fail:,}  ({n_succ:,} success, {n_fail:,} failure)")
    stats_lines.append(f"  success rows with frame_labels=None: {succ_null_labels:,} / {n_succ:,}  ({succ_null_labels/max(n_succ,1)*100:.1f}%)")
    stats_lines.append(f"  failure rows with frame_labels=None: {fail_null_labels:,} / {n_fail:,}  ({fail_null_labels/max(n_fail,1)*100:.1f}%)")
    stats_lines.append(f"  success per-frame label frequency: {dict(sorted(succ_lab_counter.items()))}")
    stats_lines.append(f"  failure per-frame label frequency: {dict(sorted(fail_lab_counter.items()))}")
    stats_lines.append(f"  success MAX-label-per-traj distribution: {dict(sorted(succ_max_counter.items()))}")
    stats_lines.append(f"  failure MAX-label-per-traj distribution: {dict(sorted(fail_max_counter.items()))}")
    stats_lines.append(f"  success TERMINAL frame label distribution: {dict(sorted(succ_terminal_label_counter.items()))}")
    stats_lines.append(f"  failure TERMINAL frame label distribution: {dict(sorted(fail_terminal_label_counter.items()))}")
    # Sanity flags (analogous to the metaworld bug we already fixed):
    if succ_null_labels == 0 and n_succ > 0:
        stats_lines.append(f"  ⚠️  WARNING: 0/{n_succ} success rows have frame_labels=None — model will route them through failure-rubric path (target capped at rubric[max]). Same shape as the metaworld bug. Inspect.")
    if succ_max_counter and max(succ_max_counter) < 5 and succ_null_labels == 0:
        stats_lines.append(f"  ⚠️  WARNING: success max-label never reaches 5 (max seen: {max(succ_max_counter)}) AND frame_labels are populated. Discriminative gap from failures may be zero.")
    if fail_terminal_label_counter and succ_terminal_label_counter:
        # Check if there's a non-trivial gap between success and failure terminal labels
        succ_term_set = set(succ_terminal_label_counter.keys())
        fail_term_set = set(fail_terminal_label_counter.keys())
        if succ_term_set <= fail_term_set:
            stats_lines.append(f"  ⚠️  WARNING: every success-terminal-label value also appears as a failure-terminal-label. Possible discrimination issue.")

# Write stats file
stats_path = OUT / 'stats.txt'
with open(stats_path, 'w') as f:
    f.write("Per-family frame_labels audit (analogous to metaworld bug check)\n")
    f.write("=" * 70 + "\n")
    f.write("Run from: train_raw/robometer_frames_train (post-patch state)\n")
    for line in stats_lines:
        f.write(line + "\n")
print(f"\n  stats written to {stats_path}")

# ---- Render frame-strip PNGs --------------------------------------------------
print("\n[4/4] Rendering PNGs …")
def render(idx, fam, kind, n):
    """Render the trajectory's frames + labels into a PNG; n is 0-based sample number."""
    row = ds[idx]
    mp4_path = row['frames']
    labels = row['frame_labels']
    quality = row['quality_label']
    traj_id = row['id'][:80]
    task = (row['task'] or '').strip().replace('\n', ' / ')[:140]
    data_source = row['data_source']

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
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8 + 0.8))
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

    suptitle = f'[{fam} / {data_source}] {quality.upper()}  |  labels={labels}\nID: {traj_id}\nTask: {task}'
    fig.suptitle(suptitle, fontsize=9, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_png = OUT / f'{fam}_{kind}{n+1:02d}_{traj_id}.png'
    plt.savefig(out_png, dpi=80, bbox_inches='tight')
    plt.close()
    return f"  wrote {out_png.name}  labels={labels}"

per_family_manifest = {}
for fam, (succ_idx, fail_idx) in sampled_indices.items():
    print(f"\n  --- {fam} ---")
    per_family_manifest[fam] = {'succ': [], 'fail': []}
    for n, idx in enumerate(succ_idx):
        msg = render(idx, fam, 'succ', n)
        print(f"  {msg}")
        per_family_manifest[fam]['succ'].append({
            'idx': idx, 'id': ds[idx]['id'], 'data_source': ds[idx]['data_source'],
            'frame_labels': ds[idx]['frame_labels'], 'task': ds[idx]['task'][:120] if ds[idx]['task'] else None,
        })
    for n, idx in enumerate(fail_idx):
        msg = render(idx, fam, 'fail', n)
        print(f"  {msg}")
        per_family_manifest[fam]['fail'].append({
            'idx': idx, 'id': ds[idx]['id'], 'data_source': ds[idx]['data_source'],
            'frame_labels': ds[idx]['frame_labels'], 'task': ds[idx]['task'][:120] if ds[idx]['task'] else None,
        })

manifest_path = OUT / 'manifest_per_family.json'
with open(manifest_path, 'w') as f:
    json.dump(per_family_manifest, f, indent=2)

print(f"\nDone. Output in {OUT}/")
print(f"  - {sum(len(v['succ']) + len(v['fail']) for v in per_family_manifest.values())} PNGs")
print(f"  - {manifest_path.name}")
print(f"  - {stats_path.name}  ← READ THIS for bug-hunt summary")
