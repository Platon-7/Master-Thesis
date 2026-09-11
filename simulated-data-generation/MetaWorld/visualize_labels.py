#!/usr/bin/env python3
"""
Generate an HTML page showing all episodes with their per-frame labels.
Each (episode, camera) pair is shown as its own row.

Usage:
    python visualize_labels.py --jsonl path/to/metaworld_failures.jsonl \
                                --keyframes path/to/keyframes \
                                --output labels_review.html
"""
import json
import base64
import argparse
from pathlib import Path
from collections import defaultdict

SCORE_COLOR = {1: "#e74c3c", 2: "#e67e22", 3: "#f1c40f", 4: "#2ecc71"}


def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--keyframes", required=True)
    parser.add_argument("--output", default="labels_review.html")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--scores", nargs="+", type=int, default=None)
    args = parser.parse_args()

    episodes = []
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))

    if args.tasks:
        episodes = [e for e in episodes if e["task"] in args.tasks]
    if args.scores:
        episodes = [e for e in episodes if e["target_score"] in args.scores]

    # Sort: task → target_score → camera → base_id
    episodes.sort(key=lambda e: (
        e["task"], e["target_score"],
        e.get("camera") or "", e["episode_id"]
    ))

    kf_root = Path(args.keyframes)

    html_parts = ["""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>MetaWorld Per-Frame Labels</title>
<style>
  body  { font-family: monospace; background: #1a1a2e; color: #eee; margin: 20px; }
  h1    { color: #e0e0ff; }
  h2    { color: #a0b0ff; border-bottom: 1px solid #333; padding-bottom: 4px; margin-top: 32px; }
  .ep   { display: flex; align-items: flex-start; gap: 10px;
           background: #16213e; border-radius: 6px; padding: 8px 12px;
           margin-bottom: 8px; }
  .meta { min-width: 260px; font-size: 11px; color: #aaa; line-height: 1.6; }
  .meta b { color: #ddd; }
  .frames { display: flex; gap: 5px; flex-wrap: wrap; }
  .frame  { text-align: center; }
  .frame img { width: 112px; height: 84px; object-fit: cover;
                border-radius: 4px; display: block; }
  .lbl  { font-size: 10px; margin-top: 3px; padding: 2px 3px;
           border-radius: 3px; font-weight: bold; color: #000; }
</style>
</head>
<body>
<h1>MetaWorld Per-Frame Labels</h1>
"""]

    current_section = None
    total = 0
    for ep in episodes:
        section = (ep["task"], ep["target_score"])
        if section != current_section:
            current_section = section
            html_parts.append(f"<h2>{ep['task']} &mdash; Target Score {ep['target_score']}</h2>\n")

        ep_id = ep["episode_id"]
        cam = ep.get("camera") or "—"
        mode = ep.get("noise_mode", "?")
        base = ep.get("episode_base_id", ep_id)

        html_parts.append('<div class="ep">\n')
        html_parts.append(
            f'<div class="meta"><b>{base}</b><br>'
            f'cam: <b>{cam}</b><br>'
            f'mode: {mode}</div>\n'
        )
        html_parts.append('<div class="frames">\n')

        for frame in ep["frames"]:
            f_idx = frame["frame_idx"]
            fl = frame["frame_label"]
            sc = frame["score"]
            t = frame["timestamp"]
            color = SCORE_COLOR.get(fl, "#999")

            img_tag = '<div style="width:112px;height:84px;background:#2a2a3e;border-radius:4px;"></div>'
            rel = frame.get("image_path")
            if rel:
                abs_path = kf_root / rel
                if abs_path.exists():
                    img_tag = f'<img src="data:image/jpeg;base64,{img_to_b64(abs_path)}" />'
            else:
                # Legacy single-camera layout
                ep_dir = kf_root / base
                hits = sorted(ep_dir.glob(f"frame_{f_idx}_*.jpg")) if ep_dir.exists() else []
                if hits:
                    img_tag = f'<img src="data:image/jpeg;base64,{img_to_b64(hits[0])}" />'

            html_parts.append(
                f'<div class="frame">{img_tag}'
                f'<div class="lbl" style="background:{color}">'
                f'L={fl} S={sc} t={t:.2f}</div></div>\n'
            )

        html_parts.append('</div>\n</div>\n')
        total += 1

    html_parts.append(
        f"<p style='color:#555;margin-top:24px'>Total: {total} entries</p>\n"
        "</body>\n</html>"
    )

    out_path = Path(args.output)
    out_path.write_text("".join(html_parts))
    print(f"Written {total} entries → {out_path}")


if __name__ == "__main__":
    main()
