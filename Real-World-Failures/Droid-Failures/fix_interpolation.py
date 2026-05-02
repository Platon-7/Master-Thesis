"""
Fix gap-frame scores in the DROID 16-frame scored JSONL using the asymmetric rule:
    gap_score = L_score + int((R_score - L_score) / 2)

Originals are the 8 VLM-scored frames at linspace(0, T, 8) timestamps.
Gaps are the other 8 frames whose scores were previously set by nearest-neighbor.
"""
import json
import shutil
from pathlib import Path
import numpy as np

META_DIR = Path("/projects/prjs1958/robometer_frame_dataset/droid/metadata")
SHARDS = ["scored_full_droid_shard0.jsonl", "scored_full_droid_shard1.jsonl"]
BACKUP_SUFFIX = ".pre_interp_fix"


def find_originals(frames):
    """Return sorted list of 8 frame indices that are the linspace-8 originals."""
    T = frames[-1]["timestamp"]
    targets = np.linspace(0, T, 8)
    timestamps = [fr["timestamp"] for fr in frames]
    used = set()
    originals = []
    for tt in targets:
        # pick the frame idx (not yet used) closest to tt
        best_i = None
        best_d = float("inf")
        for i, ts in enumerate(timestamps):
            if i in used:
                continue
            d = abs(ts - tt)
            if d < best_d:
                best_d = d
                best_i = i
        used.add(best_i)
        originals.append(best_i)
    return sorted(originals)


def fix_episode(ep):
    frames = ep["frames"]
    if len(frames) != 16:
        return ep, 0
    originals = find_originals(frames)
    orig_set = set(originals)
    gap_indices = [i for i in range(16) if i not in orig_set]

    changed = 0
    for g in gap_indices:
        # find left and right originals by frame index
        left = max((o for o in originals if o < g), default=None)
        right = min((o for o in originals if o > g), default=None)
        if left is None:
            left = right
        if right is None:
            right = left
        L = frames[left]["score"]
        R = frames[right]["score"]
        # Handle None (LLM parse failures) gracefully
        if L is None and R is None:
            continue
        if L is None:
            L = R
        if R is None:
            R = L
        new_score = L + int((R - L) / 2)
        if frames[g]["score"] != new_score:
            changed += 1
        frames[g]["score"] = new_score

    return ep, changed


def main():
    total_eps = 0
    total_changed_frames = 0
    total_skipped = 0
    for shard in SHARDS:
        src = META_DIR / shard
        bak = src.with_suffix(src.suffix + BACKUP_SUFFIX)
        if not bak.exists():
            shutil.copy2(src, bak)
            print(f"backed up -> {bak.name}")

        with open(bak) as f:
            lines = f.readlines()

        out_lines = []
        for line in lines:
            ep = json.loads(line)
            if len(ep.get("frames", [])) != 16:
                total_skipped += 1
                out_lines.append(line)
                continue
            ep_fixed, changed = fix_episode(ep)
            total_eps += 1
            total_changed_frames += changed
            out_lines.append(json.dumps(ep_fixed) + "\n")

        with open(src, "w") as f:
            f.writelines(out_lines)
        print(f"wrote {src.name}: {len(out_lines)} lines")

    print(f"\nFixed {total_eps} episodes, updated {total_changed_frames} gap frame scores")
    print(f"Skipped {total_skipped} episodes (non-16 frames)")


if __name__ == "__main__":
    main()
