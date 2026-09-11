#!/usr/bin/env python3
"""
Re-clamp rubric violations using the DROID neighbour-aware heuristic.

Group A integration blanket-clamped every out-of-range VLM score (ANSWER: 5)
to 1, because I couldn't tell at extraction time whether the VLM was seeing
real near-task-completion or hallucinating. Post-hoc rule:

    if max(other labels in the same episode) >= 3   → clamp to 4
    else                                             → keep at 1 (hallucination)

matches the spirit of the two worked examples in
`Real-World-Failures/Droid-Failures/README.md` §1b (rubric-violation clamps).

Updates:
  1. /projects/prjs1958/.../scores/<archive>_scored.jsonl — in-place rewrite
     (frame_labels + terminal_reward for affected episodes)
  2. /projects/prjs1958/.../keyframes/<archive>/shard-NNNNN.tar — rewrite the
     meta.json entries of affected episodes, keep jpgs untouched.

Only touches archives that have a rubric_violations sidecar. Idempotent-ish:
second run is a no-op because the violation sidecar is left on disk but the
scored labels already agree with the rule.
"""

from __future__ import annotations

import io
import json
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

DST = Path("/projects/prjs1958/robometer_frame_dataset/robometer")
EPS_PER_SHARD = 512


def reclamp_episode_labels(labels: list[int], violation_positions: list[int]) -> tuple[list[int], list[int]]:
    """Return (new_labels, changed_positions). Leaves untouched positions alone."""
    others = [v for i, v in enumerate(labels) if i not in violation_positions]
    high_exists = any(v >= 3 for v in others)
    target = 4 if high_exists else 1
    new = list(labels)
    changed = []
    for p in violation_positions:
        if new[p] != target:
            new[p] = target
            changed.append(p)
    return new, changed


def main():
    viol_files = sorted((DST / "scores").glob("*_rubric_violations.jsonl"))
    if not viol_files:
        print("no rubric violation sidecars — nothing to do")
        return

    # (archive -> ep_id -> sorted list of violation positions parsed from `where`)
    by_arch: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for vf in viol_files:
        for line in vf.read_text().splitlines():
            d = json.loads(line)
            where = d["where"]
            if not where.startswith(("frame_", "gap_")):
                continue
            pos = int(where.split("_", 1)[1])
            by_arch[d["archive"]][d["episode_id"]].append(pos)

    print(f"archives with violations: {list(by_arch)}")
    for arch, eps in by_arch.items():
        # Collapse each episode's positions to a set
        eps_dedup = {e: sorted(set(ps)) for e, ps in eps.items()}
        print(f"\n=== {arch}  ({len(eps_dedup)} affected episodes) ===")

        # Step 1 — rewrite scored.jsonl in place
        scored_path = DST / "scores" / f"{arch}_scored.jsonl"
        orig_lines = scored_path.read_text().splitlines()
        new_lines: list[str] = []
        updated_ep: dict[str, tuple[list[int], int]] = {}     # ep_id -> (new_labels, new_term)
        stats = {"to_1": 0, "to_4": 0, "ep_touched": 0}
        for line in orig_lines:
            d = json.loads(line)
            ep = d["episode_id"]
            if ep in eps_dedup:
                positions = eps_dedup[ep]
                old = d["frame_labels"]
                new, changed = reclamp_episode_labels(old, positions)
                if changed:
                    d["frame_labels"]    = new
                    d["terminal_reward"] = max(new)
                    stats["ep_touched"] += 1
                    stats["to_4"] += sum(1 for p in changed if new[p] == 4)
                    stats["to_1"] += sum(1 for p in changed if new[p] == 1)
                    updated_ep[ep] = (new, max(new))
            new_lines.append(json.dumps(d))
        scored_path.write_text("\n".join(new_lines) + "\n")
        print(f"  scored.jsonl: touched {stats['ep_touched']} episodes "
              f"({stats['to_4']} frames→4, {stats['to_1']} frames kept at 1)")

        if not updated_ep:
            print("  nothing changed (already consistent); skipping shard rewrite")
            continue

        # Step 2 — rewrite affected shards' meta.json entries
        kf_dir = DST / "keyframes" / arch
        shards = sorted(kf_dir.glob("shard-*.tar"))
        # Map ep_id -> shard index (order matches scored.jsonl per integration script)
        ep_order = {json.loads(l)["episode_id"]: i for i, l in enumerate(orig_lines)}
        shards_to_rewrite = sorted({ep_order[e] // EPS_PER_SHARD for e in updated_ep})
        print(f"  shards to rewrite: {shards_to_rewrite}")

        for sidx in shards_to_rewrite:
            shard_path = kf_dir / f"shard-{sidx:05d}.tar"
            affected_in_this_shard = {
                e: updated_ep[e] for e in updated_ep
                if ep_order[e] // EPS_PER_SHARD == sidx
            }
            tmp_path = shard_path.with_suffix(".tar.tmp")
            with tarfile.open(shard_path, "r") as src, tarfile.open(tmp_path, "w") as dst:
                for m in src.getmembers():
                    # Is this a meta.json of an affected episode?
                    parts = m.name.split("/", 1)
                    if len(parts) == 2 and parts[1] == "meta.json" and parts[0] in affected_in_this_shard:
                        meta = json.loads(src.extractfile(m).read())
                        new_labels, new_term = affected_in_this_shard[parts[0]]
                        meta["frame_labels"]    = new_labels
                        meta["terminal_reward"] = new_term
                        meta["notes"] = (meta.get("notes", "") +
                                         " re-clamped via neighbour heuristic (reclamp_rubric_violations.py).").strip()
                        mb = json.dumps(meta, indent=2).encode()
                        ti = tarfile.TarInfo(m.name); ti.size = len(mb); ti.mtime = 0
                        dst.addfile(ti, io.BytesIO(mb))
                    else:
                        f = src.extractfile(m)
                        data = f.read() if f is not None else b""
                        ti = tarfile.TarInfo(m.name); ti.size = len(data); ti.mtime = 0
                        dst.addfile(ti, io.BytesIO(data))
            tmp_path.replace(shard_path)
            print(f"    rewrote {shard_path.name} ({len(affected_in_this_shard)} eps touched)")


if __name__ == "__main__":
    main()
