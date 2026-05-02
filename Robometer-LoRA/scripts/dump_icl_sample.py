"""Dump 16 demo + 16 query frames from a real partnered test-set row.

Usage:
    python scripts/dump_icl_sample.py [--n SAMPLES] [--out OUT_DIR]

For each sampled row, writes:
    <out>/sample_<i>__<task>/grid.png      ← 8×4 grid: rows 0-3 = demo, rows 4-7 = query
    <out>/sample_<i>__<task>/info.txt      ← episode_ids, tier, archives, frames_paths
"""
from __future__ import annotations
import argparse, json, tarfile, io, random, re
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DATASET_ROOT  = Path('/projects/prjs1958/robometer_frame_dataset')
PAIRS_UNIFIED = DATASET_ROOT / 'pairs_unified.jsonl'
TEST_INDEX    = '/scratch-shared/pkarageorgis1/robometer_frames_splits/pairs_index_test.jsonl'
DEFAULT_OUT   = '/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer-LoRA/results/icl_visual_check'


def load_frames(shard_str: str, episode_dir: str) -> list[Image.Image]:
    shard = DATASET_ROOT / shard_str
    prefix = episode_dir.rstrip('/') + '/'
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


def make_grid(demo: list[Image.Image], query: list[Image.Image],
              demo_label: str, query_label: str,
              cell: int = 192, cols: int = 4) -> Image.Image:
    n_rows_each = (16 + cols - 1) // cols  # 4
    band = 28
    W = cols * cell
    H = n_rows_each * cell * 2 + band * 2
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf', 14)
    except Exception:
        font = ImageFont.load_default()

    # Demo band
    draw.rectangle([0, 0, W, band], fill=(46, 134, 171))
    draw.text((8, 6), demo_label, fill='white', font=font)
    for i, im in enumerate(demo[:16]):
        r, c = divmod(i, cols)
        x, y = c * cell, band + r * cell
        canvas.paste(im.resize((cell, cell)), (x, y))
        draw.text((x + 4, y + 4), f"d{i:02d}", fill='yellow', font=font)

    # Query band
    y0 = band + n_rows_each * cell
    draw.rectangle([0, y0, W, y0 + band], fill=(192, 57, 43))
    draw.text((8, y0 + 6), query_label, fill='white', font=font)
    for i, im in enumerate(query[:16]):
        r, c = divmod(i, cols)
        x, y = c * cell, y0 + band + r * cell
        canvas.paste(im.resize((cell, cell)), (x, y))
        draw.text((x + 4, y + 4), f"q{i:02d}", fill='yellow', font=font)

    return canvas


def slug(s: str, n: int = 50) -> str:
    s = re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')
    return s[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=3, help='samples to dump')
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # Build episode_id → frames_path from pairs_unified (the same lookup the loader does)
    print("Loading pairs_unified.jsonl …")
    unified = {}
    with PAIRS_UNIFIED.open() as f:
        for line in f:
            r = json.loads(line)
            unified[r['episode_id']] = r['frames_path']

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in open(TEST_INDEX) if l.strip()]
    partnered = [r for r in rows if r.get('partner_episode_id')]
    rng.shuffle(partnered)

    n_done = 0
    for r in partnered:
        if n_done >= args.n: break
        partner_fp = unified.get(r['partner_episode_id'])
        if not partner_fp: continue
        qs, qd = r['frames_path'].split('::', 1)
        ps, pd = partner_fp.split('::', 1)

        try:
            query_frames = load_frames(qs, qd)
            demo_frames  = load_frames(ps, pd)
        except Exception as e:
            print(f"  [skip] {r['episode_id']}: {e}")
            continue
        if len(query_frames) < 16 or len(demo_frames) < 16:
            print(f"  [skip] {r['episode_id']}: too few frames "
                  f"(q={len(query_frames)}, d={len(demo_frames)})")
            continue

        sample_dir = out / f"sample_{n_done}__{slug(r['task'])}"
        sample_dir.mkdir(exist_ok=True)
        demo_label  = f"DEMO  (success)  partner_id={r['partner_episode_id'][:32]}…  archive={r['partner_archive']}"
        query_label = f"QUERY ({r['label']})  episode_id={r['episode_id'][:32]}…  archive={r['archive']}"
        grid = make_grid(demo_frames, query_frames, demo_label, query_label)
        grid.save(sample_dir / 'grid.png')

        info = (
            f"task                : {r['task']}\n"
            f"tier                : {r['tier']}\n"
            f"\nQUERY ({r['label']}):\n"
            f"  episode_id        : {r['episode_id']}\n"
            f"  archive           : {r['archive']}\n"
            f"  frames_path       : {r['frames_path']}\n"
            f"  loaded frames     : {len(query_frames)}\n"
            f"\nDEMO ({r['partner_label']}):\n"
            f"  partner_episode_id: {r['partner_episode_id']}\n"
            f"  partner_archive   : {r['partner_archive']}\n"
            f"  partner_task      : {r['partner_task']}\n"
            f"  resolved path     : {partner_fp}\n"
            f"  loaded frames     : {len(demo_frames)}\n"
            f"\nbyte-equality demo==query: {demo_frames[0].tobytes() == query_frames[0].tobytes()}\n"
        )
        (sample_dir / 'info.txt').write_text(info)
        print(f"  wrote {sample_dir}")
        n_done += 1

    print(f"\nDumped {n_done} sample(s) to {out}")


if __name__ == '__main__':
    main()
