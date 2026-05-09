"""Stratified ICL-pair visual audit — pulls N partnered failure→success pairs from each
source family and dumps both demo and query as grid images. Used to assess whether the
upstream task labels match visual content across the dataset.

For each sample:
  * loads the QUERY frames from the per-archive keyframes tar (16-frame padded view)
  * loads the DEMO frames same way
  * also pulls the original NPZ from the raw archive when available, so we see the
    *unpadded* source trajectory rather than the padded keyframes — this is what the
    user wants to see whether the visual content actually matches the task label.

Output:
    results/icl_audit_stratified/<source>/sample_<i>__<task>/{demo_grid,query_grid}.png
    results/icl_audit_stratified/INDEX.md  ← human-readable summary

Usage: python3 scripts/audit_icl_pairs_stratified.py [--n-per-source 5]
"""
from __future__ import annotations

import argparse
import io
import json
import random
import re
import subprocess
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DATASET_ROOT  = Path('/projects/prjs1958/robometer_frame_dataset')
PAIRS_UNIFIED = DATASET_ROOT / 'pairs_unified.jsonl'
TEST_INDEX    = '/scratch-shared/pkarageorgis1/robometer_frames_splits/pairs_index_test.jsonl'
RAW_DIR       = Path('/projects/prjs1958/robometer_full_dataset/raw_archives/single')
NPZ_CACHE     = Path('/scratch-shared/pkarageorgis1/_video_pull')
OUT_BASE      = Path('/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer-LoRA/results/icl_audit_stratified')

# Group B subset of robometer (separated for analysis)
GROUP_B = {
    "jesbu1_soar_rfm_soar_rfm",
    "jesbu1_roboarena_0825_rfm_roboarena",
    "jesbu1_roboarena_eval_debug_nowrist_roboarena_eval_debug_nowrist",
}
RACER  = {"jesbu1_racer_rfm_racer_train", "jesbu1_racer_rfm_racer_val"}


def family_of(row: dict) -> str:
    src = row.get("source")
    arch = row.get("archive", "")
    if src == "robometer":
        if arch in GROUP_B:    return "robometer_groupB"
        if arch in RACER:      return "robometer_racer"
        return "robometer_other"
    return src or "unknown"


def load_keyframes(shard_str: str, ep_dir: str):
    shard = DATASET_ROOT / shard_str
    prefix = ep_dir.rstrip('/') + '/'
    out = []
    with tarfile.open(shard, 'r') as tf:
        members = sorted([m for m in tf.getmembers()
                          if m.name.startswith(prefix) and m.name.endswith('.jpg')],
                         key=lambda m: m.name)
        for m in members:
            f = tf.extractfile(m)
            if f is None: continue
            out.append(Image.open(io.BytesIO(f.read())).convert('RGB'))
    return out


def load_source_npz(archive: str, episode_id: str):
    """Try to extract the original (unpadded) trajectory NPZ from the raw archive.
    Returns ndarray [T, H, W, 3] uint8 or None if not available."""
    raw_tar = RAW_DIR / f"{archive}.tar"
    if not raw_tar.is_file():
        return None  # archive isn't a single-file tar (e.g., split-part archives)
    # Episode IDs in pairs_unified are `ep_<idx>_<uuid>`; raw NPZs are
    # `<archive>/frames/trajectory_<uuid>.npz`. uuid is everything after `ep_<idx>_`.
    parts = episode_id.split('_', 2)
    if len(parts) < 3: return None
    uuid = parts[2]
    npz_rel = f"{archive}/frames/trajectory_{uuid}.npz"
    cached = NPZ_CACHE / npz_rel
    if not cached.exists():
        NPZ_CACHE.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "-xf", str(raw_tar), "-C", str(NPZ_CACHE), npz_rel],
                       check=False, stderr=subprocess.DEVNULL)
    if not cached.exists():
        return None
    try:
        z = np.load(cached, allow_pickle=True)
        if "frames" in z:
            return z["frames"]
    except Exception:
        return None
    return None


def make_grid(frames, label: str, cell: int = 192, cols: int = 4) -> Image.Image:
    n = len(frames)
    rows = (n + cols - 1) // cols
    band = 28
    W = cols * cell
    H = rows * cell + band
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf', 13)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, W, band], fill=(46, 134, 171))
    draw.text((6, 6), label, fill='white', font=font)
    for i, f in enumerate(frames):
        if isinstance(f, np.ndarray):
            f = Image.fromarray(f)
        r, c = divmod(i, cols)
        x, y = c * cell, band + r * cell
        canvas.paste(f.resize((cell, cell)), (x, y))
        draw.text((x + 4, y + 4), f"{i:02d}", fill='yellow', font=font)
    return canvas


def slug(s: str, n: int = 50) -> str:
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-per-source', type=int, default=5)
    ap.add_argument('--seed', type=int, default=137)
    args = ap.parse_args()

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    # Resolve partner paths from pairs_unified
    print("Loading pairs_unified.jsonl …")
    unified_path = {}
    with PAIRS_UNIFIED.open() as f:
        for line in f:
            r = json.loads(line)
            unified_path[r['episode_id']] = r['frames_path']

    print("Reading test pair index …")
    rows = [json.loads(l) for l in open(TEST_INDEX) if l.strip()]
    partnered = [r for r in rows if r.get('partner_episode_id') and r.get('label') == 'failure']

    # Group by family
    by_family = {}
    for r in partnered:
        by_family.setdefault(family_of(r), []).append(r)

    rng = random.Random(args.seed)
    families = sorted(by_family.keys())
    print(f"\nFamilies seen in test partnered failures: {families}")

    index_md = ["# ICL-pair visual audit — stratified", "",
                "Each row is one paired (failure-query, success-demo) sampled from the test split.",
                "The keyframes_grid columns show what the LoRA sampler actually feeds the model",
                "(16 frames each, padded for short trajectories). The src_grid columns show the",
                "original NPZ trajectory (unpadded) when the raw archive is a single tar — these",
                "are missing for split-archive sources (Group B + DROID).", ""]

    for fam in families:
        fam_rows = by_family[fam]
        sample_rows = rng.sample(fam_rows, min(args.n_per_source, len(fam_rows)))
        fam_dir = OUT_BASE / fam
        fam_dir.mkdir(parents=True, exist_ok=True)
        index_md.append(f"## {fam}  ({len(fam_rows)} candidates in test, {len(sample_rows)} sampled)")
        index_md.append("")
        index_md.append("| # | task | tier | demo | query |")
        index_md.append("|---|------|------|------|-------|")

        for i, r in enumerate(sample_rows):
            partner_fp = unified_path.get(r['partner_episode_id'])
            if partner_fp is None: continue
            qs, qd = r['frames_path'].split('::', 1)
            ps, pd = partner_fp.split('::', 1)

            try:
                q_keyframes = load_keyframes(qs, qd)
                d_keyframes = load_keyframes(ps, pd)
            except Exception as e:
                print(f"  [{fam} #{i}] failed to load keyframes: {e}")
                continue
            if len(q_keyframes) < 1 or len(d_keyframes) < 1: continue

            sample_dir = fam_dir / f"sample_{i}__{slug(r['task'])}"
            sample_dir.mkdir(exist_ok=True)

            # Always dump the keyframes grid (what the model actually sees, padded)
            qg = make_grid(q_keyframes,
                           f"QUERY  {r['episode_id'][:30]}…  arch={r['archive']}")
            qg.save(sample_dir / "query_keyframes.png")
            dg = make_grid(d_keyframes,
                           f"DEMO   {r['partner_episode_id'][:30]}…  arch={r['partner_archive']}")
            dg.save(sample_dir / "demo_keyframes.png")

            # Also try the unpadded source NPZ for both
            d_src_path = ""
            d_src_frames = load_source_npz(r['partner_archive'], r['partner_episode_id'])
            if d_src_frames is not None and len(d_src_frames) > 0:
                make_grid(d_src_frames,
                          f"DEMO source ({len(d_src_frames)} frames, unpadded)").save(
                          sample_dir / "demo_source.png")
                d_src_path = "demo_source.png"
            q_src_path = ""
            q_src_frames = load_source_npz(r['archive'], r['episode_id'])
            if q_src_frames is not None and len(q_src_frames) > 0:
                make_grid(q_src_frames,
                          f"QUERY source ({len(q_src_frames)} frames, unpadded)").save(
                          sample_dir / "query_source.png")
                q_src_path = "query_source.png"

            (sample_dir / "info.txt").write_text(
                f"task              : {r['task']}\n"
                f"tier              : {r['tier']}\n"
                f"family            : {fam}\n"
                f"\nQUERY (failure):\n"
                f"  episode_id      : {r['episode_id']}\n"
                f"  archive         : {r['archive']}\n"
                f"  frames_path     : {r['frames_path']}\n"
                f"\nDEMO (success):\n"
                f"  partner_episode : {r['partner_episode_id']}\n"
                f"  partner_archive : {r['partner_archive']}\n"
                f"  resolved path   : {partner_fp}\n"
                f"  partner_task    : {r['partner_task']}\n"
            )
            print(f"  wrote {sample_dir.relative_to(OUT_BASE)}")

            index_md.append(
                f"| {i} | `{r['task'][:48]}` | {r['tier']} | "
                f"[demo_kf](./{fam}/{sample_dir.name}/demo_keyframes.png)"
                f"{' / [src](./'+fam+'/'+sample_dir.name+'/'+d_src_path+')' if d_src_path else ''}"
                f" | [query_kf](./{fam}/{sample_dir.name}/query_keyframes.png)"
                f"{' / [src](./'+fam+'/'+sample_dir.name+'/'+q_src_path+')' if q_src_path else ''}"
                f" |"
            )
        index_md.append("")

    (OUT_BASE / "INDEX.md").write_text("\n".join(index_md))
    print(f"\nDone. INDEX.md: {OUT_BASE / 'INDEX.md'}")


if __name__ == "__main__":
    main()
