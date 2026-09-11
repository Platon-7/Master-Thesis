#!/usr/bin/env python3
"""
Integrate Leonardo's Group A output into the canonical
robometer_frame_dataset format (16-frame sharded tars + compact scores).

Per archive:
  1. Read Leonardo's scores/<archive>_scored.jsonl (8 VLM-graded frames / ep)
  2. Open raw tar at /projects/prjs1958/.../raw_archives/single/<archive>.tar
  3. For each episode:
       - parse uuid from episode_id (= 'ep_<idx>_<uuid>')
       - load trajectory_<uuid>.npz, read frames array
       - compute 16 uniform source-frame indices, extract 16 jpgs
       - map the 8 Leonardo frames to their nearest slot in the 16 grid
         (unique-assignment, same algorithm as droid/fix_interpolation.py)
       - fill the 8 gap frame scores via
             gap = L + int((R - L) / 2)
         (asymmetric-mean rule from Droid-Failures/README.md §1b)
       - clamp any score outside {1,2,3,4} to 4 and append to
         rubric_violations.jsonl
  4. Pack 512 episodes/shard into shard-NNNNN.tar (matches failsafe/metaworld)
  5. Emit scores/<archive>_scored.jsonl with the compact schema:
       {episode_id, archive, task, terminal_reward, frame_labels,
        paired_success_id}

Trossen is special-cased: no Leonardo scored.jsonl, 6 hardcoded manual grades.

Usage:
    python integrate_group_a_to_frame_dataset.py --archive <name>
    python integrate_group_a_to_frame_dataset.py --archive all       # sequential
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

LEONARDO_OUT = Path("/scratch-shared/pkarageorgis1/leonardo_groupA/unzipped/robometer_out")
RAW_SINGLE   = Path("/projects/prjs1958/robometer_full_dataset/raw_archives/single")
DST_ROOT     = Path("/projects/prjs1958/robometer_frame_dataset/robometer")

N_KEYFRAMES    = 16
EPS_PER_SHARD  = 512
FPS_FALLBACK   = 30  # used only for timestamp in filename; doesn't affect sampling
ALLOWED_SCORES = {1, 2, 3, 4}

# ---------------------------------------------------------------------------
# Manual trossen grades (Leonardo never ran this archive)
# ---------------------------------------------------------------------------

TROSSEN_ARCHIVE = "ykorkmaz_usc_trossen_rfm_usc_trossen"

TROSSEN_MANUAL = [
    # (npz_base, task, 8-frame scores, description)
    ("aloha_open_bottle_failure_ep_000000", "Open the bottle",
     [1, 1, 1, 2, 2, 3, 1, 1],
     "Robot knocked over the bottle after grasping it."),
    ("aloha_open_bottle_failure_ep_000001", "Open the bottle",
     [1, 1, 1, 1, 2, 2, 3, 1],
     "Robot knocked over the bottle after grasping it."),
    ("aloha_open_drawer_failure_ep_000000", "Open the red drawer",
     [1, 1, 1, 2, 3, 2, 3, 2],
     "Robot manages to grasp the drawer handle twice but fails to open it."),
    ("aloha_open_drawer_failure_ep_000001", "Open the red drawer",
     [1, 1, 1, 1, 2, 2, 2, 2],
     "Robot never manages to grasp the drawer's handle."),
    ("aloha_unzip_case_failure_ep_000000", "Unzip the pencil case",
     [1, 1, 2, 3, 3, 3, 3, 3],
     "Robot grasps the case but never manages to grasp the zipper and unzip it."),
    ("aloha_unzip_case_failure_ep_000001", "Unzip the pencil case",
     [1, 1, 2, 2, 3, 3, 3, 3],
     "Robot grasps the case but never manages to grasp the zipper and unzip it."),
]

# Stable UUIDs for the 6 manual trossen episodes (generated once, reused)
TROSSEN_MANUAL_UUIDS = [
    "08a4ad3b-b7ef-4d97-9edc-107e0bd02145",
    "7c8e9baa-824d-4339-8889-07b75124280e",
    "a22e9e76-7cda-418f-aca7-82c6e066e9c9",
    "191c5137-5d2e-4acb-9660-645a8d98c148",
    "26066d31-d37e-4682-9ae4-6ac645fad97f",
    "15110ba1-ff26-4451-b4c0-7ca6636a62b2",
]

# ---------------------------------------------------------------------------
# Canonical robometer family map (coarse). Fallback = "standard".
# ---------------------------------------------------------------------------

FAMILY = {
    "jesbu1_racer_rfm_racer_train":                           "racer",
    "jesbu1_racer_rfm_racer_val":                             "racer",
    "jesbu1_auto_eval_rfm_auto_eval_rfm":                     "auto_eval",
    "ykorkmaz_libero_failure_rfm_libero_90_failure":          "libero",
    "ykorkmaz_libero_failure_rfm_libero_10_failure":          "libero",
    "ykorkmaz_libero_failure_rfm_libero_object_failure":      "libero",
    "ykorkmaz_libero_failure_rfm_libero_spatial_failure":     "libero",
    "ykorkmaz_libero_failure_rfm_libero_goal_failure":        "libero",
    "jesbu1_rfm_new_mit_franka_rfm_rfm_new_mit_franka_rfm":   "mit_franka",
    "jesbu1_usc_koch_p_ranking_rfm_usc_koch_p_ranking_all":   "usc_koch",
    "jesbu1_rfm_new_mit_franka_rfm_nowrist_rfm_new_mit_franka_rfm_nowrist": "mit_franka",
    "aliangdw_utd_so101_policy_ranking_utd_so101_policy_ranking": "so101",
    "aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking":   "usc_xarm",
    "jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top":   "so101",
    "jesbu1_utd_so101_clean_policy_ranking_wrist_utd_so101_clean_policy_ranking_wrist": "so101",
    "aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking": "usc_franka",
    TROSSEN_ARCHIVE:                                          "trossen",
    "jesbu1_mit_franka_p-rank_rfm_mit_franka_p-rank_rfm":     "mit_franka",
}

GROUP_A_ARCHIVES = list(FAMILY.keys())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def uniform_indices(total_steps: int, n: int) -> list[int]:
    if total_steps < n:
        return list(range(total_steps)) + [total_steps - 1] * (n - total_steps)
    return [round(i * (total_steps - 1) / (n - 1)) for i in range(n)]


def find_originals_indices(idx_big: list[int], target_idxs: list[int]) -> list[int]:
    """Find which positions in `idx_big` correspond to each `target_idxs` value,
    via nearest-distance unique-assignment. Same algorithm as
    droid/fix_interpolation.py::find_originals."""
    used: set[int] = set()
    originals: list[int] = []
    for tgt in target_idxs:
        best_i, best_d = None, float("inf")
        for i, v in enumerate(idx_big):
            if i in used:
                continue
            d = abs(v - tgt)
            if d < best_d:
                best_d, best_i = d, i
        used.add(best_i)
        originals.append(best_i)
    return sorted(originals)


def asymmetric_interp(labels16: list[int | None], originals: list[int]) -> list[int]:
    """Fill gap labels (positions not in `originals`) via
       gap = L + int((R - L) / 2)        (truncation toward zero)
    from Real-World-Failures/Droid-Failures/README.md §1b."""
    orig_set = set(originals)
    out = list(labels16)
    for g in range(len(out)):
        if g in orig_set:
            continue
        left  = max((o for o in originals if o < g), default=None)
        right = min((o for o in originals if o > g), default=None)
        if left  is None: left  = right
        if right is None: right = left
        L, R = out[left], out[right]
        if L is None and R is None:
            continue
        if L is None: L = R
        if R is None: R = L
        out[g] = L + int((R - L) / 2)
    return out


def clamp_and_log_violation(val, *, where: str, ep_id: str, archive: str, raw) -> int:
    """If `val` is outside {1,2,3,4}, clamp to 4 and emit a violation dict."""
    if val in ALLOWED_SCORES:
        return val, None
    clamped = 4 if (isinstance(val, int) and val > 4) else 1
    viol = {"archive": archive, "episode_id": ep_id, "where": where,
            "original": val, "clamped": clamped, "llm_raw": raw}
    return clamped, viol


# ---------------------------------------------------------------------------
# Input source abstraction
# ---------------------------------------------------------------------------

UUID_RE = re.compile(r"ep_\d+_([0-9a-f\-]+)$")


def episodes_from_leonardo(archive: str):
    """Yield (episode_id, uuid, task, frame_scores_8) for each of Leonardo's
    scored episodes in this archive."""
    scored = LEONARDO_OUT / "scores" / f"{archive}_scored.jsonl"
    with scored.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ep_id = d["episode_id"]
            m = UUID_RE.match(ep_id)
            if not m:
                print(f"  SKIP bad id: {ep_id}", flush=True)
                continue
            uuid = m.group(1)
            task = d["task"]
            fr = sorted(d["frames"], key=lambda x: x["frame_idx"])
            scores8 = [fr_i["score"] for fr_i in fr]
            raw8    = [fr_i.get("llm_raw") for fr_i in fr]
            ep_top_raw = d.get("llm_raw")
            yield ep_id, uuid, task, scores8, raw8, ep_top_raw


def episodes_from_trossen_manual():
    """Yield (episode_id, uuid, task, frame_scores_8) for the 6 manual trossen
    failures — Leonardo never ran this archive."""
    for (npz_base, task, scores8, desc), uuid in zip(TROSSEN_MANUAL, TROSSEN_MANUAL_UUIDS):
        ep_id = f"ep_{TROSSEN_MANUAL.index((npz_base, task, scores8, desc)):06d}_{uuid}"
        raw8 = [f"[MANUAL GRADE] ANSWER: {s} because {desc}" for s in scores8]
        yield ep_id, npz_base, task, scores8, raw8, raw8[0]


# ---------------------------------------------------------------------------
# Core integration
# ---------------------------------------------------------------------------

def process_archive(archive: str, *, dry_run: bool = False) -> dict:
    """Process one archive end-to-end. Returns a small stats dict."""
    family = FAMILY.get(archive, "standard")
    raw_tar = RAW_SINGLE / f"{archive}.tar"
    assert raw_tar.exists(), f"missing raw tar: {raw_tar}"

    out_kf_dir  = DST_ROOT / "keyframes" / archive
    out_sc_path = DST_ROOT / "scores"    / f"{archive}_scored.jsonl"
    viol_path   = DST_ROOT / "scores"    / f"{archive}_rubric_violations.jsonl"

    if not dry_run:
        out_kf_dir.mkdir(parents=True, exist_ok=True)
        out_sc_path.parent.mkdir(parents=True, exist_ok=True)

    # Build uuid → raw-tar-member-name map in one pass
    print(f"[{archive}] indexing raw tar...", flush=True)
    uuid_to_member: dict[str, str] = {}
    with tarfile.open(raw_tar, "r") as rt:
        trossen_map: dict[str, str] = {}   # npz_base -> member name (for trossen)
        for m in rt.getnames():
            if not m.endswith(".npz"):
                continue
            base = m.split("/")[-1][:-4]  # strip .npz
            # Leonardo's pattern: trajectory_<uuid>.npz
            um = re.match(r"trajectory_([0-9a-f\-]+)$", base)
            if um:
                uuid_to_member[um.group(1)] = m
            # Trossen pattern: trajectory_ykorkmaz__<task>_<failure|success|...>_ep_<NNNNNN>
            trossen_map[base] = m
    print(f"[{archive}] {len(uuid_to_member)} uuid-named npz,  {len(trossen_map)} total npz", flush=True)

    # Episode source + label iterator
    if archive == TROSSEN_ARCHIVE:
        ep_iter = list(episodes_from_trossen_manual())
    else:
        ep_iter = list(episodes_from_leonardo(archive))
    print(f"[{archive}] {len(ep_iter)} episodes to integrate", flush=True)

    compact_rows: list[str] = []
    violations:  list[dict] = []
    stats = {"archive": archive, "family": family,
             "expected": len(ep_iter), "written": 0,
             "violations": 0, "missing_npz": 0}

    # Pack episodes into shards (512/shard)
    current_shard_idx = 0
    current_shard_tf: tarfile.TarFile | None = None
    current_count = 0

    def open_shard(idx: int):
        path = out_kf_dir / f"shard-{idx:05d}.tar"
        if dry_run:
            return None
        return tarfile.open(path, "w")

    def close_shard(tf):
        if tf is not None:
            tf.close()

    with tarfile.open(raw_tar, "r") as src_tar:
        for ep_id, key, task, scores8, raw8, ep_top_raw in ep_iter:
            # Resolve source npz
            if archive == TROSSEN_ARCHIVE:
                member = trossen_map.get(f"trajectory_ykorkmaz__{key}")
            else:
                member = uuid_to_member.get(key)
            if member is None:
                print(f"  [{ep_id}] MISSING npz (key={key[:12]}..) SKIP", flush=True)
                stats["missing_npz"] += 1
                continue

            f = src_tar.extractfile(member)
            if f is None:
                print(f"  [{ep_id}] can't extract {member}", flush=True)
                stats["missing_npz"] += 1
                continue
            npz = np.load(io.BytesIO(f.read()))
            frames_arr = npz["frames"]   # (N,H,W,3)
            N = int(frames_arr.shape[0])

            idx16 = uniform_indices(N, N_KEYFRAMES)
            idx8  = uniform_indices(N, 8)
            originals = find_originals_indices(idx16, idx8)

            # Seed 16-label array from the 8 originals, clamping out-of-range
            labels16: list[int | None] = [None] * N_KEYFRAMES
            for pos, score, raw in zip(originals, scores8, raw8):
                clamped, v = clamp_and_log_violation(score, where=f"frame_{pos}",
                                                    ep_id=ep_id, archive=archive,
                                                    raw=raw)
                labels16[pos] = clamped
                if v: violations.append(v)

            labels16 = asymmetric_interp(labels16, originals)

            # Also clamp any interpolated gap that's outside range (shouldn't
            # happen unless originals had extreme values, but defense-in-depth)
            for i, v in enumerate(labels16):
                if v not in ALLOWED_SCORES:
                    c, vio = clamp_and_log_violation(v, where=f"gap_{i}",
                                                     ep_id=ep_id, archive=archive,
                                                     raw=None)
                    labels16[i] = c
                    if vio: violations.append(vio)

            # Emit 16 jpgs into the current shard
            if current_shard_tf is None:
                current_shard_tf = open_shard(current_shard_idx)

            for k, raw_i in enumerate(idx16):
                img = Image.fromarray(frames_arr[raw_i])
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                b = buf.getvalue()
                t_s = raw_i / FPS_FALLBACK
                name = f"{ep_id}/frame_{k:02d}_{t_s:.2f}s.jpg"
                if current_shard_tf is not None:
                    ti = tarfile.TarInfo(name); ti.size = len(b); ti.mtime = 0
                    current_shard_tf.addfile(ti, io.BytesIO(b))

            # meta.json (matches failsafe/metaworld schema)
            terminal_reward = max(labels16)
            meta = {
                "episode_id":        ep_id,
                "archive":           archive,
                "family":            family,
                "task":              task,
                "label":             "failure",
                "terminal_reward":   terminal_reward,
                "n_source_frames":   N,
                "n_keyframes":       N_KEYFRAMES,
                "keyframes_dir":     f"{archive}/{ep_id}",
                "paired_success_id": None,
                "frame_labels":      labels16,
            }
            if archive == TROSSEN_ARCHIVE:
                meta["notes"] = "MANUAL GRADE: 8 frames human-graded; 8 gaps via asymmetric interp."
            mbytes = json.dumps(meta, indent=2).encode()
            if current_shard_tf is not None:
                ti = tarfile.TarInfo(f"{ep_id}/meta.json"); ti.size = len(mbytes); ti.mtime = 0
                current_shard_tf.addfile(ti, io.BytesIO(mbytes))

            # Compact scores row
            compact_rows.append(json.dumps({
                "episode_id":        ep_id,
                "archive":           archive,
                "task":              task,
                "terminal_reward":   terminal_reward,
                "frame_labels":      labels16,
                "paired_success_id": None,
            }))

            stats["written"] += 1
            current_count += 1

            # Roll over to next shard
            if current_count >= EPS_PER_SHARD:
                close_shard(current_shard_tf)
                current_shard_tf = None
                current_count = 0
                current_shard_idx += 1

    # Close last shard if any eps remained
    close_shard(current_shard_tf)

    if not dry_run:
        out_sc_path.write_text("\n".join(compact_rows) + "\n")
        if violations:
            viol_path.write_text("\n".join(json.dumps(v) for v in violations) + "\n")

    stats["violations"] = len(violations)
    print(f"[{archive}] DONE  written={stats['written']}/{stats['expected']}  "
          f"violations={stats['violations']}  missing_npz={stats['missing_npz']}",
          flush=True)
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True,
                    help=f"one of {GROUP_A_ARCHIVES} or 'all'")
    ap.add_argument("--dry-run", action="store_true",
                    help="skip file writes, just log what would be done")
    args = ap.parse_args()

    if args.archive == "all":
        archives = GROUP_A_ARCHIVES
    else:
        if args.archive not in FAMILY:
            print(f"ERROR unknown archive: {args.archive}", file=sys.stderr)
            sys.exit(2)
        archives = [args.archive]

    t0 = time.time()
    all_stats = []
    for arch in archives:
        t = time.time()
        stats = process_archive(arch, dry_run=args.dry_run)
        stats["elapsed_s"] = round(time.time() - t, 1)
        all_stats.append(stats)

    print("\n=== SUMMARY ===")
    print(f"{'archive':65s}  {'exp':>6s}  {'wrote':>6s}  {'viol':>5s}  {'miss':>5s}  {'sec':>6s}")
    for s in all_stats:
        print(f"{s['archive']:65s}  {s['expected']:6d}  {s['written']:6d}  "
              f"{s['violations']:5d}  {s['missing_npz']:5d}  {s['elapsed_s']:6.1f}")
    print(f"total wall: {round(time.time()-t0,1)}s")


if __name__ == "__main__":
    main()
