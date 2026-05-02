#!/usr/bin/env python3
"""
Visualize FailSafe multi-camera (front/side/wrist) episodes from shard tars
in the robometer_frame_dataset format. Reads meta.json + 16 keyframe jpgs
from each shard and produces a single self-contained HTML file.

Groups rows by (archive, inst) so you see the same episode across all 3 cameras
stacked together.

Usage:
    python visualize_multicam.py \
        --root /gpfs/scratch1/shared/pkarageorgis1/failsafe_dataset/full_v1 \
        --n-success 3 --n-failures 2 \
        --output failsafe_multicam.html
"""
import argparse
import base64
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path

SCORE_COLOR = {1: "#e74c3c", 2: "#e67e22", 3: "#f1c40f", 4: "#2ecc71"}


def load_shard_episodes(tar_path):
    """Yield (episode_id, meta_dict, [(frame_idx, jpg_bytes), ...]) from a shard."""
    episodes = defaultdict(lambda: {"meta": None, "frames": []})
    with tarfile.open(tar_path, "r") as tf:
        for member in tf:
            if not member.isfile():
                continue
            parts = member.name.split("/", 1)
            if len(parts) != 2:
                continue
            ep_id, fname = parts
            data = tf.extractfile(member).read()
            if fname == "meta.json":
                episodes[ep_id]["meta"] = json.loads(data.decode())
            elif fname.startswith("frame_") and fname.endswith(".jpg"):
                f_idx = int(fname.split("_")[1])
                episodes[ep_id]["frames"].append((f_idx, data))
    for ep_id, entry in episodes.items():
        if entry["meta"] is None:
            continue
        entry["frames"].sort(key=lambda x: x[0])
        yield ep_id, entry["meta"], entry["frames"]


def parse_failsafe_id(ep_id):
    """Extract (archive, inst, camera, kind) from a failsafe episode_id.
    Success: <archive>_<slot>_success_inst<N>_<cam>
    Failure: <archive>_<slot>_inst<N>_<cam>_score_<T>
    """
    parts = ep_id.rsplit("_", 1)
    # find "inst<digits>"
    tokens = ep_id.split("_")
    inst_idx = None
    for i, t in enumerate(tokens):
        if t.startswith("inst") and t[4:].isdigit():
            inst_idx = i
            break
    if inst_idx is None:
        return None
    inst = int(tokens[inst_idx][4:])
    cam = tokens[inst_idx + 1] if inst_idx + 1 < len(tokens) else "?"
    return inst, cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root (contains keyframes/, keyframes_success/)")
    ap.add_argument("--n-success", type=int, default=3, help="Instances of successes per archive")
    ap.add_argument("--n-failures", type=int, default=0, help="Instances of failures per archive/slot")
    ap.add_argument("--output", default="failsafe_multicam.html")
    args = ap.parse_args()

    root = Path(args.root)
    sections = []  # list of (archive, kind, rows) where rows = [(inst, cam, meta, frames), ...]

    # ---------- Successes ----------
    succ_root = root / "keyframes_success"
    for archive_dir in sorted(succ_root.iterdir()) if succ_root.exists() else []:
        archive = archive_dir.name
        rows = []
        seen_inst = set()
        for shard in sorted(archive_dir.glob("shard-*.tar")):
            for ep_id, meta, frames in load_shard_episodes(shard):
                parsed = parse_failsafe_id(ep_id)
                if parsed is None:
                    continue
                inst, cam = parsed
                if inst >= args.n_success:
                    continue
                rows.append((inst, cam, ep_id, meta, frames))
                seen_inst.add(inst)
                if len(seen_inst) >= args.n_success and all(
                    sum(1 for r in rows if r[0] == i) >= 3 for i in range(args.n_success)
                ):
                    break
        rows.sort(key=lambda r: (r[0], r[1]))
        if rows:
            sections.append((archive, "SUCCESS", rows))

    # ---------- Failures (optional) ----------
    fail_root = root / "keyframes"
    if args.n_failures > 0 and fail_root.exists():
        for archive_dir in sorted(fail_root.iterdir()):
            archive = archive_dir.name
            rows = []
            # group by slot via episode_id substring between archive and "_inst"
            slot_counts = defaultdict(int)
            for shard in sorted(archive_dir.glob("shard-*.tar")):
                for ep_id, meta, frames in load_shard_episodes(shard):
                    parsed = parse_failsafe_id(ep_id)
                    if parsed is None:
                        continue
                    inst, cam = parsed
                    slot = ep_id.split("_inst")[0]
                    key = (slot, inst)
                    if slot_counts[slot] >= args.n_failures * 3 and key not in {
                        (s, i) for s, i in [(r[5], r[0]) for r in rows]
                    }:
                        continue
                    if inst >= args.n_failures:
                        continue
                    rows.append((inst, cam, ep_id, meta, frames, slot))
                    slot_counts[slot] += 1
            # compact: drop the slot field before rendering
            rows.sort(key=lambda r: (r[5], r[0], r[1]))
            rows = [(inst, cam, ep_id, meta, frames) for (inst, cam, ep_id, meta, frames, _slot) in rows]
            if rows:
                sections.append((archive, "FAILURE", rows))

    # ---------- Render HTML ----------
    parts = ["""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>FailSafe Multi-Camera Review</title>
<style>
  body  { font-family: monospace; background: #1a1a2e; color: #eee; margin: 20px; }
  h1    { color: #e0e0ff; }
  h2    { color: #a0b0ff; border-bottom: 1px solid #333; padding-bottom: 4px; margin-top: 32px; }
  h3    { color: #ffcf99; margin: 18px 0 6px 0; font-size: 14px; }
  .ep   { display: flex; align-items: flex-start; gap: 10px;
           background: #16213e; border-radius: 6px; padding: 8px 12px;
           margin-bottom: 6px; }
  .meta { min-width: 240px; font-size: 11px; color: #aaa; line-height: 1.5; }
  .meta b { color: #ddd; }
  .frames { display: flex; gap: 4px; flex-wrap: wrap; }
  .frame  { text-align: center; }
  .frame img { width: 104px; height: 78px; object-fit: cover;
                border-radius: 3px; display: block; background: #000; }
  .lbl  { font-size: 10px; margin-top: 2px; padding: 1px 2px;
           border-radius: 3px; font-weight: bold; color: #000; }
  .cam-front  { border-left: 3px solid #3498db; padding-left: 6px; }
  .cam-side   { border-left: 3px solid #9b59b6; padding-left: 6px; }
  .cam-wrist  { border-left: 3px solid #e91e63; padding-left: 6px; }
</style>
</head>
<body>
<h1>FailSafe Multi-Camera Review</h1>
<p style="color:#888">Color stripes: <span style="color:#3498db">■</span> front &nbsp; <span style="color:#9b59b6">■</span> side &nbsp; <span style="color:#e91e63">■</span> wrist</p>
"""]
    total = 0
    for archive, kind, rows in sections:
        parts.append(f'<h2>{archive} — {kind}</h2>\n')
        current_inst = None
        for inst, cam, ep_id, meta, frames in rows:
            if inst != current_inst:
                current_inst = inst
                parts.append(f'<h3>instance {inst:04d}</h3>\n')
            cam_class = f"cam-{cam}" if cam in ("front", "side", "wrist") else ""
            parts.append(f'<div class="ep {cam_class}">\n')
            fl = meta.get("frame_labels", [])
            parts.append(
                f'<div class="meta"><b>{ep_id}</b><br>'
                f'cam: <b>{cam}</b> &nbsp; label: <b>{meta.get("label", "?")}</b><br>'
                f'terminal_reward: <b>{meta.get("terminal_reward", "?")}</b><br>'
                f'n_src: {meta.get("n_source_frames", "?")}  fps: {meta.get("fps", "?")}<br>'
                f'paired: {meta.get("paired_success_id", "—")}</div>\n'
            )
            parts.append('<div class="frames">\n')
            for (f_idx, jpg), lbl in zip(frames, fl + [None] * (16 - len(fl))):
                b64 = base64.b64encode(jpg).decode()
                color = SCORE_COLOR.get(lbl, "#666") if lbl is not None else "#444"
                lbl_txt = f"L={lbl}" if lbl is not None else "—"
                parts.append(
                    f'<div class="frame"><img src="data:image/jpeg;base64,{b64}"/>'
                    f'<div class="lbl" style="background:{color}">{lbl_txt}</div></div>\n'
                )
            parts.append('</div>\n</div>\n')
            total += 1

    parts.append(f'<p style="color:#555;margin-top:24px">Total: {total} episodes</p>\n</body>\n</html>')
    out = Path(args.output)
    out.write_text("".join(parts))
    print(f"Written {total} episodes → {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
