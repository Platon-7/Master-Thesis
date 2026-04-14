#!/usr/bin/env python3
"""
Generate an HTML page showing all ManiSkill failure episodes with per-frame labels.

Each row = one episode. Shows:
  - Task name and target score
  - Which stage was injected (fail_stage, stage_name, fail_type)
  - 8 keyframes with per-frame label + timestamp
  - Color coding: red=1, orange=2, yellow=3, green=4

Use this to VERIFY that the stage → score mapping makes visual sense.
Key questions to check:
  Score 2 (reach/grasp failure): does the arm visibly miss the object?
  Score 3 (transport failure):   is the object visibly grasped before the failure?
  Score 4 (align failure):       is the cube clearly lifted above the base cube?

Usage:
    python visualize_labels.py --jsonl output/sample_v1/maniskill_failures.jsonl \
                                --keyframes output/sample_v1/keyframes \
                                --output logs/sample_review.html

    # Filter to one task or score:
    python visualize_labels.py ... --tasks FailStackCube-v1 --scores 3 4
    python visualize_labels.py ... --stages lift_cube align_cubes
"""

import json
import base64
import argparse
from pathlib import Path
from collections import defaultdict

SCORE_COLOR = {1: "#e74c3c", 2: "#e67e22", 3: "#f1c40f", 4: "#2ecc71"}
SCORE_LABEL = {1: "No progress", 2: "Approached", 3: "Grasped/contacted", 4: "Near completion"}


def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl",     required=True)
    parser.add_argument("--keyframes", required=True)
    parser.add_argument("--output",    default="labels_review.html")
    parser.add_argument("--tasks",     nargs="+", default=None)
    parser.add_argument("--scores",    nargs="+", type=int, default=None)
    parser.add_argument("--stages",    nargs="+", default=None,
                        help="Filter by stage_name (e.g. lift_cube align_cubes)")
    parser.add_argument("--max-per-group", type=int, default=20,
                        help="Max episodes shown per (task, score, stage) group")
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
    if args.stages:
        episodes = [e for e in episodes if e.get("stage_name") in args.stages]

    # Sort: task → score → stage → episode_id
    episodes.sort(key=lambda e: (
        e["task"], e["target_score"],
        e.get("stage_name") or "zz",
        e["episode_id"]
    ))

    kf_root = Path(args.keyframes)

    # ------------------------------------------------------------------ #
    # Build stats for the header
    # ------------------------------------------------------------------ #
    stats = defaultdict(lambda: defaultdict(int))
    for ep in episodes:
        stats[ep["task"]][ep["target_score"]] += 1

    # ------------------------------------------------------------------ #
    # HTML boilerplate
    # ------------------------------------------------------------------ #
    html_parts = ["""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ManiSkill Failure Labels — Stage→Score Verification</title>
<style>
  body  { font-family: monospace; background: #1a1a2e; color: #eee; margin: 20px; }
  h1    { color: #e0e0ff; margin-bottom: 4px; }
  .subtitle { color: #778; font-size: 13px; margin-bottom: 28px; }
  h2    { color: #a0b0ff; border-bottom: 1px solid #333;
          padding-bottom: 4px; margin-top: 36px; }
  h3    { color: #88aadd; margin-top: 18px; margin-bottom: 6px; font-size: 13px; }
  .stats-table { border-collapse: collapse; margin-bottom: 24px; }
  .stats-table td, .stats-table th {
    border: 1px solid #333; padding: 4px 10px; font-size: 12px; }
  .stats-table th { background: #1e2a40; color: #aac; }
  .ep   { display: flex; align-items: flex-start; gap: 10px;
           background: #16213e; border-radius: 6px; padding: 8px 12px;
           margin-bottom: 8px; }
  .meta { min-width: 240px; font-size: 11px; color: #aaa; line-height: 1.8; }
  .meta b  { color: #ddd; }
  .meta .score-badge {
    display: inline-block; padding: 1px 7px; border-radius: 3px;
    font-weight: bold; color: #000; font-size: 11px; }
  .frames { display: flex; gap: 5px; flex-wrap: wrap; }
  .frame  { text-align: center; }
  .frame img { width: 112px; height: 84px; object-fit: cover;
                border-radius: 4px; display: block; }
  .lbl  { font-size: 10px; margin-top: 3px; padding: 2px 3px;
           border-radius: 3px; font-weight: bold; color: #000; }
  .placeholder { width: 112px; height: 84px; background: #2a2a3e;
                  border-radius: 4px; display: flex; align-items: center;
                  justify-content: center; font-size: 10px; color: #556; }
  .warn { color: #e67e22; font-size: 11px; }
</style>
</head>
<body>
<h1>ManiSkill Failure Labels — Stage&rarr;Score Verification</h1>
<div class="subtitle">
  Use this page to verify that each stage injection produces visually correct progress scores.<br>
  <b>Score 2</b>: arm visibly misses object &nbsp;|&nbsp;
  <b>Score 3</b>: object is grasped/contacted before failure &nbsp;|&nbsp;
  <b>Score 4</b>: object is above/near goal before failure
</div>
"""]

    # ------------------------------------------------------------------ #
    # Summary table
    # ------------------------------------------------------------------ #
    html_parts.append("<table class='stats-table'>\n")
    html_parts.append("<tr><th>Task</th><th>Score 1</th><th>Score 2</th>"
                      "<th>Score 3</th><th>Score 4</th></tr>\n")
    for task in sorted(stats.keys()):
        row = f"<tr><td>{task}</td>"
        for s in [1, 2, 3, 4]:
            n = stats[task].get(s, 0)
            color = SCORE_COLOR.get(s, "#555")
            row += f"<td style='background:{color}22;color:{color}'>{n}</td>"
        row += "</tr>\n"
        html_parts.append(row)
    html_parts.append("</table>\n")

    # ------------------------------------------------------------------ #
    # Episode rows, grouped by (task, score, stage_name)
    # ------------------------------------------------------------------ #
    current_task_score = None
    current_stage      = None
    group_count        = 0
    total_shown        = 0

    for ep in episodes:
        task_score = (ep["task"], ep["target_score"])
        stage_name = ep.get("stage_name") or "—"

        if task_score != current_task_score:
            current_task_score = task_score
            current_stage      = None
            score_color = SCORE_COLOR.get(ep["target_score"], "#999")
            score_desc  = SCORE_LABEL.get(ep["target_score"], "")
            html_parts.append(
                f"<h2>{ep['task']} &mdash; "
                f"<span style='color:{score_color}'>Score {ep['target_score']}: {score_desc}</span>"
                f"</h2>\n"
            )

        if stage_name != current_stage:
            current_stage = stage_name
            group_count = 0
            notes = ep.get("fail_type", "")
            html_parts.append(f"<h3>Stage: <b>{stage_name}</b></h3>\n")

        if group_count >= args.max_per_group:
            continue
        group_count += 1

        # Episode metadata
        ep_id     = ep["episode_id"]
        fail_type = ep.get("fail_type", "—")
        n_steps   = ep.get("total_steps", "?")
        score_color = SCORE_COLOR.get(ep["target_score"], "#999")

        html_parts.append('<div class="ep">\n')
        html_parts.append(
            f'<div class="meta">'
            f'<b>{ep_id}</b><br>'
            f'fail_type: <b>{fail_type}</b><br>'
            f'steps: {n_steps}<br>'
            f'target: <span class="score-badge" style="background:{score_color}">'
            f'Score {ep["target_score"]}</span>'
            f'</div>\n'
        )

        html_parts.append('<div class="frames">\n')
        for frame in ep["frames"]:
            f_idx = frame["frame_idx"]
            fl    = frame["frame_label"]
            sc    = frame["score"]
            t     = frame["timestamp"]
            color = SCORE_COLOR.get(fl, "#999")

            # Try to load image
            rel = frame.get("image_path")
            img_tag = f'<div class="placeholder">frame {f_idx}</div>'
            if rel:
                abs_path = kf_root / rel
                if abs_path.exists():
                    img_tag = f'<img src="data:image/jpeg;base64,{img_to_b64(abs_path)}" />'

            html_parts.append(
                f'<div class="frame">{img_tag}'
                f'<div class="lbl" style="background:{color}">'
                f'L={fl} S={sc} t={t:.2f}</div></div>\n'
            )

        html_parts.append('</div>\n</div>\n')
        total_shown += 1

    html_parts.append(
        f"<p style='color:#445;margin-top:28px'>Showing {total_shown} episodes "
        f"(max {args.max_per_group} per group)</p>\n"
        "</body>\n</html>"
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(html_parts))
    print(f"Written {total_shown} episodes → {out_path}")


if __name__ == "__main__":
    main()
