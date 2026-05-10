#!/usr/bin/env python3
"""
extract_frame_labels.py — post-process the scoring output into the
`frame_labels` array shape used by the rest of robometer_frame_dataset.

The scorer (score_robometer_failures.py) writes one entry per episode with:
    {
      "episode_id": "...",
      "archive": "...",
      "task": "...",
      "score": [int, "reasoning..."],          # trajectory-level (= last frame's score)
      "frames": [
        {"frame_idx": 0, "score": [int, "..."], ...},
        {"frame_idx": 1, "score": [int, "..."], ...},
        ...                                     # 16 entries
      ]
    }

Existing dataset convention (e.g. robometer/scores/<archive>_scored.jsonl):
    {
      "episode_id": "...",
      "archive": "...",
      "task": "...",
      "terminal_reward": int,                   # = score of last frame
      "frame_labels": [int, int, ..., int],     # length 16, per-frame ordinals
      "paired_success_id": null
    }

This script reads the scorer output and rewrites it in the existing convention,
so pack_scored_output.py + downstream training read the labels correctly.

Usage:
    python extract_frame_labels.py \
        --scored-dir /projects/prjs1958/robometer_frame_eval_dataset/staging/scores \
        --out-suffix _final
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-dir", required=True, type=Path)
    ap.add_argument("--in-suffix", default="_failures_scored.jsonl")
    ap.add_argument("--out-suffix", default="_failures_final.jsonl",
                    help="output filename suffix; replaces --in-suffix")
    args = ap.parse_args()

    n_files = 0
    n_episodes_total = 0
    for fp in sorted(args.scored_dir.glob(f"*{args.in_suffix}")):
        archive = fp.name[:-len(args.in_suffix)]
        out_path = fp.with_name(archive + args.out_suffix)
        n_eps = 0
        with fp.open() as fin, out_path.open("w") as fout:
            for line in fin:
                d = json.loads(line)
                frames = d.get("frames", [])
                # score is [int, "reasoning"] — extract just the int
                def _to_int(s):
                    if isinstance(s, list) and s:
                        return int(s[0])
                    if isinstance(s, int):
                        return s
                    return 1  # fallback if missing/malformed
                frame_labels = [_to_int(f.get("score")) for f in frames]
                terminal_reward = _to_int(d.get("score"))
                out = {
                    "episode_id": d["episode_id"],
                    "archive": d["archive"],
                    "task": d["task"],
                    "terminal_reward": terminal_reward,
                    "frame_labels": frame_labels,
                    "paired_success_id": None,
                }
                fout.write(json.dumps(out) + "\n")
                n_eps += 1
        print(f"  {archive}: {n_eps} episodes → {out_path.name}")
        n_files += 1
        n_episodes_total += n_eps

    print(f"\nDone: {n_episodes_total} episodes across {n_files} archives")


if __name__ == "__main__":
    main()
