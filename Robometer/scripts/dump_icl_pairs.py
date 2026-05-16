"""Render one ICL pair (DEMO + QUERY) per source from the v2 train cache.

For each source in {robometer, robometer_orphan_success, droid, metaworld,
failsafe} pick a row where:
  * the row has a partner_episode_id (eligible as ICL query)
  * both anchor and partner are 'successful' so the success-progress label
    schedule (L1 = 5-bin CORN, L2 = linear i/15) is well-defined
  * task strings are non-empty and non-corrupt
Then render a stacked image (DEMO on top, QUERY on bottom), each as a 4x4
grid of keyframes with frame number, L1, L2 overlays — matching the
existing `icl_pair_keyframes.png` style.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datasets import Dataset, load_from_disk

CACHE = "/projects/prjs1958/robometer_frames_hf_v2/_projects_prjs1958_robometer_frames_hf_v2_train_raw_robometer_frames_train/processed_dataset"
PAIR_INDEX_TRAIN = "/scratch-shared/pkarageorgis1/robometer_frames_splits/pairs_index_train.jsonl"
PAIR_INDEX_EVAL_METAWORLD = "/scratch-shared/pkarageorgis1/robometer_frames_splits/pairs_index_eval_metaworld.jsonl"
PAIR_INDEX_EVAL_FAILSAFE = "/scratch-shared/pkarageorgis1/robometer_frames_splits/pairs_index_eval_failsafe.jsonl"
PAIRS_UNIFIED = "/projects/prjs1958/robometer_frame_dataset/pairs_unified.jsonl"
DATASET_ROOT = "/projects/prjs1958/robometer_frame_dataset"
OUT_DIR = Path("/gpfs/home3/pkarageorgis1/Master-Thesis/rbm_examples/icl_pairs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Source → (pair_index_path, "train"|"eval"). Train sources have anchors in
# the HF cache. Eval-only sources (metaworld, failsafe) load both anchor and
# partner from tar shards via pairs_unified.jsonl.
SOURCES_CFG = {
    "robometer":               (PAIR_INDEX_TRAIN, "train"),
    "robometer_orphan_success":(PAIR_INDEX_TRAIN, "train"),
    "droid":                   (PAIR_INDEX_TRAIN, "train"),
    "metaworld":               (PAIR_INDEX_EVAL_METAWORLD, "eval"),
    "failsafe":                (PAIR_INDEX_EVAL_FAILSAFE,  "eval"),
}


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for cand in (
        "/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/google-droid/DroidSansMono.ttf",
    ):
        if os.path.exists(cand):
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


# Progress label schedules (success trajectories).
def l1_schedule(num: int = 16):
    return [int((i + 2) / 4) / 4.0 for i in range(num)]


def l2_schedule(num: int = 16):
    return [i / (num - 1) for i in range(num)]


def overlay_frame(frame: np.ndarray, idx: int, l1: float, l2: float) -> Image.Image:
    """Draw frame number, L1, L2 overlays onto a single keyframe."""
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    fs = max(11, H // 22)
    f = load_font(fs)

    # Top-left: frame number on yellow strip
    n_text = f"{idx:02d}"
    n_w = draw.textlength(n_text, font=f) + 8
    draw.rectangle([(0, 0), (n_w, fs + 6)], fill=(240, 200, 40, 255))
    draw.text((4, 2), n_text, fill=(0, 0, 0, 255), font=f)

    # Top-right: "L=" stub (used in original to flag frame_labels source).
    l_w = draw.textlength("L=·", font=f) + 8
    draw.rectangle([(W - l_w, 0), (W, fs + 6)], fill=(0, 0, 0, 255))
    draw.text((W - l_w + 4, 2), "L=·", fill=(255, 255, 255, 255), font=f)

    # Bottom: two stripes — L1 (blue) and L2 (red), side-by-side.
    band_h = fs + 8
    half = W // 2
    draw.rectangle([(0, H - band_h), (half, H)], fill=(70, 130, 200, 230))
    draw.rectangle([(half, H - band_h), (W, H)], fill=(220, 70, 70, 230))
    draw.text((4, H - band_h + 2), f"L1={l1:.2f}", fill=(255, 255, 255), font=f)
    draw.text((half + 4, H - band_h + 2), f"L2={l2:.2f}", fill=(255, 255, 255), font=f)
    return img


def grid_with_overlays(frames: np.ndarray, l1: list, l2: list, cols: int = 4) -> Image.Image:
    T, H, W, _ = frames.shape
    rows = (T + cols - 1) // cols
    canvas = Image.new("RGB", (cols * W, rows * H), (0, 0, 0))
    for i in range(T):
        r, c = divmod(i, cols)
        canvas.paste(overlay_frame(frames[i], i, l1[i], l2[i]), (c * W, r * H))
    return canvas


def title_bar(width: int, text: str) -> Image.Image:
    h = 26
    bar = Image.new("RGB", (width, h), (60, 110, 170))
    d = ImageDraw.Draw(bar)
    f = load_font(14)
    d.text((6, 5), text, fill=(255, 255, 255), font=f)
    return bar


def stack(images: list[Image.Image]) -> Image.Image:
    W = max(im.size[0] for im in images)
    H = sum(im.size[1] for im in images)
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    y = 0
    for im in images:
        canvas.paste(im, (0, y))
        y += im.size[1]
    return canvas


def npz_frames(row: dict) -> np.ndarray:
    z = np.load(row["frames"])
    return z[z.files[0]]


import io
import tarfile


def tar_frames(frames_path: str, dataset_root: str = DATASET_ROOT) -> np.ndarray:
    """Decode all JPGs for an episode from a `<rel_tar>::<member_prefix>` path.
    Mirrors what the upstream sampler does at ICL-demo resolve time."""
    rel_tar, _, member_prefix = frames_path.partition("::")
    tar_abs = os.path.join(dataset_root, rel_tar)
    with tarfile.open(tar_abs, "r") as tf:
        members = sorted(
            (m for m in tf if m.name.startswith(member_prefix + "/") and m.name.endswith(".jpg")),
            key=lambda m: m.name,
        )
        frames = []
        for m in members:
            f = tf.extractfile(m)
            if f is None:
                continue
            frames.append(np.asarray(Image.open(io.BytesIO(f.read())).convert("RGB"), dtype=np.uint8))
    if not frames:
        raise FileNotFoundError(f"no JPGs found under {tar_abs}::{member_prefix}")
    # Resize to shortest_edge=240, preserve aspect (matches helpers.py default).
    out = []
    for fr in frames:
        H, W = fr.shape[:2]
        scale = 240.0 / min(H, W)
        nH, nW = int(round(H * scale)), int(round(W * scale))
        out.append(np.asarray(Image.fromarray(fr).resize((nW, nH), Image.BILINEAR), dtype=np.uint8))
    # Drop frames that ended up with non-modal shape (rare — shouldn't happen here).
    shapes = {f.shape for f in out}
    if len(shapes) > 1:
        from collections import Counter
        common = Counter(f.shape for f in out).most_common(1)[0][0]
        out = [f for f in out if f.shape == common]
    return np.stack(out, axis=0)


def lookup_partner_in_pairs_unified() -> dict[str, str]:
    """Return episode_id → frames_path for every row in pairs_unified.jsonl."""
    eid_to_path = {}
    with open(PAIRS_UNIFIED) as f:
        for line in f:
            r = json.loads(line)
            if r.get("episode_id") and r.get("frames_path"):
                eid_to_path[r["episode_id"]] = r["frames_path"]
    return eid_to_path


def short_id(eid: str) -> str:
    return eid[:32] + ("..." if len(eid) > 32 else "")


def short_task(task: str) -> str:
    t = (task or "").replace("\n", " ").strip()
    return (t[:60] + "...") if len(t) > 60 else t


def find_icl_pair_train(
    pair_index_path: str, hf_by_id: dict[str, int], ds: Dataset,
    partner_paths: dict[str, str], source: str
):
    """Train mode: anchor from HF cache, partner from tar shard."""
    candidates, fallbacks = [], []
    for line in open(pair_index_path):
        p = json.loads(line)
        if p.get("source") != source:
            continue
        anchor_eid, partner_eid = p["episode_id"], p.get("partner_episode_id")
        if not partner_eid or anchor_eid not in hf_by_id:
            continue
        partner_path = partner_paths.get(partner_eid)
        if not partner_path:
            continue
        ar = ds[hf_by_id[anchor_eid]]
        if not ar.get("task") or "pictures aren't working" in ar.get("task", ""):
            continue
        if p.get("label") == "failure" and p.get("partner_label") in ("success", "successful"):
            candidates.append((ar, partner_path, p))
            break
        else:
            fallbacks.append((ar, partner_path, p))
    chosen = candidates[0] if candidates else (fallbacks[0] if fallbacks else None)
    if chosen is None:
        return None
    ar, partner_path, p = chosen
    return {
        "query_frames": npz_frames(ar),
        "query_label": ar["quality_label"],
        "query_eid": ar["id"],
        "query_task": ar["task"],
        "query_frame_labels": ar.get("frame_labels"),
        "demo_frames": tar_frames(partner_path),
        "demo_label": p.get("partner_label"),
        "demo_eid": p.get("partner_episode_id"),
        "demo_task": p.get("partner_task", ""),
    }


def find_icl_pair_eval(pair_index_path: str, partner_paths: dict[str, str], source: str):
    """Eval mode: both anchor and partner loaded from tar shards. The 'source'
    field in eval pair indices may not be set (since eval split is by-source),
    so we don't filter by it — just take the first valid F/S pair."""
    candidates, fallbacks = [], []
    for line in open(pair_index_path):
        p = json.loads(line)
        anchor_eid, partner_eid = p["episode_id"], p.get("partner_episode_id")
        if not partner_eid:
            continue
        anchor_path = partner_paths.get(anchor_eid)  # same lookup table works for any episode
        partner_path = partner_paths.get(partner_eid)
        if not anchor_path or not partner_path:
            continue
        if not p.get("task"):
            continue
        if p.get("label") == "failure" and p.get("partner_label") in ("success", "successful"):
            candidates.append((anchor_path, partner_path, p))
            break
        else:
            fallbacks.append((anchor_path, partner_path, p))
    chosen = candidates[0] if candidates else (fallbacks[0] if fallbacks else None)
    if chosen is None:
        return None
    anchor_path, partner_path, p = chosen
    return {
        "query_frames": tar_frames(anchor_path),
        "query_label": p.get("label"),
        "query_eid": p.get("episode_id"),
        "query_task": p.get("task", ""),
        "query_frame_labels": None,
        "demo_frames": tar_frames(partner_path),
        "demo_label": p.get("partner_label"),
        "demo_eid": p.get("partner_episode_id"),
        "demo_task": p.get("partner_task", ""),
    }


print(f"[load] HF train cache: {CACHE}")
ds = load_from_disk(CACHE)
hf_by_id = {ds[i]["id"]: i for i in range(len(ds))}
print(f"  rows: {len(ds)}")

print(f"[load] indexing pairs_unified.jsonl for episode frames_paths")
partner_paths = lookup_partner_in_pairs_unified()
print(f"  episode_id → frames_path entries: {len(partner_paths)}")


def labels_for_query(pair: dict, num: int = 16):
    fl = pair.get("query_frame_labels")
    if fl and len(fl) == num:
        return list(fl), l2_schedule(num)
    # success → 5-bin CORN, failure → 0 (we don't have rubric labels for tar-loaded queries)
    if pair.get("query_label") in ("success", "successful"):
        return l1_schedule(num), l2_schedule(num)
    return [0.0] * num, l2_schedule(num)


def labels_for_demo(pair: dict, num: int = 16):
    if pair.get("demo_label") in ("success", "successful"):
        return l1_schedule(num), l2_schedule(num)
    return [0.0] * num, l2_schedule(num)


for src, (pi_path, mode) in SOURCES_CFG.items():
    print(f"\n=== source: {src}  ({mode}) ===")
    if mode == "train":
        pair = find_icl_pair_train(pi_path, hf_by_id, ds, partner_paths, src)
    else:
        pair = find_icl_pair_eval(pi_path, partner_paths, src)
    if pair is None:
        print(f"  no ICL pair found for source={src}")
        continue
    qf, df = pair["query_frames"], pair["demo_frames"]
    print(f"  query ({pair['query_label']:10s}): {pair['query_eid']}  shape={qf.shape}  task={short_task(pair['query_task'])!r}")
    print(f"  demo  ({pair['demo_label']:10s}): {pair['demo_eid']}  shape={df.shape}  task={short_task(pair['demo_task'])!r}")

    ql1, ql2 = labels_for_query(pair)
    dl1, dl2 = labels_for_demo(pair)
    demo_grid = grid_with_overlays(df, dl1, dl2)
    query_grid = grid_with_overlays(qf, ql1, ql2)
    W = max(demo_grid.size[0], query_grid.size[0])
    demo_title = title_bar(
        W, f"DEMO   {pair['demo_label']}   {short_id(pair['demo_eid'])}   task={short_task(pair['demo_task'])!r}"
    )
    query_title = title_bar(
        W, f"QUERY  {pair['query_label']}   {short_id(pair['query_eid'])}   task={short_task(pair['query_task'])!r}"
    )
    if demo_grid.size[0] != W:
        demo_grid = demo_grid.resize((W, demo_grid.size[1]))
    if query_grid.size[0] != W:
        query_grid = query_grid.resize((W, query_grid.size[1]))

    out = stack([demo_title, demo_grid, query_title, query_grid])
    out_path = OUT_DIR / f"icl_pair_{src}.png"
    out.save(out_path)
    print(f"  → {out_path}  ({out.size})")
