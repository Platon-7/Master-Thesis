"""Pick 8 representative datapoints from robometer_frame_dataset for Chris to review.

Output: Master-Thesis/chris_figure_candidates/
  icl_pairs/<NN>_<source>_<archive>/
      failure/   (16 keyframes + meta.json)
      success/   (16 keyframes + meta.json)
      README.md  (task + summary)
  labeled_failures/<NN>_<source>_<archive>/
      (16 keyframes + meta.json + README.md)

Mix: 6 real-world (droid, robometer, roboreward) + 2 simulated (metaworld, failsafe).
Half (4) are ICL pairs; half (4) are single labeled failures.
"""
from __future__ import annotations

import json
import re
import shutil
import tarfile
from pathlib import Path
from typing import Optional

DATASET_ROOT = Path("/projects/prjs1958/robometer_frame_dataset")
PAIRS_INDEX  = DATASET_ROOT / "pairs_unified.jsonl"
OUT_ROOT     = Path("/gpfs/home3/pkarageorgis1/Master-Thesis/chris_figure_candidates")

# Curated picks — one per family for ICL pairs, one per family for labeled failures.
# All have partner_episode_id non-null + same_task tier where applicable.
# (source, slot) where slot ∈ {"icl", "fail"}. Chosen to span diverse tasks.
TARGET_SOURCES_ICL  = ["droid", "robometer", "roboreward", "metaworld"]
TARGET_SOURCES_FAIL = ["droid", "robometer", "roboreward", "failsafe"]


def stream_pairs():
    with open(PAIRS_INDEX) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def select_examples():
    """Walk pairs_unified.jsonl once, pick the FIRST viable row per source per slot."""
    icl_picks: dict[str, dict] = {}
    fail_picks: dict[str, dict] = {}
    seen_archives_icl: set[str] = set()
    seen_archives_fail: set[str] = set()
    for row in stream_pairs():
        src = row.get("source", "")
        label = row.get("label", "")
        partner_id = row.get("partner_episode_id")
        tier = row.get("tier", "")
        archive = row.get("archive", "")

        # ICL pair: failure + paired success, with any "good pair" tier.
        # Real-world sources use {1_exact, 1_same_task_fresh, 2_*, 3_*}; sim uses "same_task".
        # All tiers represent failure-success pairs where partner_label=="success".
        GOOD_TIERS = {
            "same_task",
            "1_exact", "1_same_task_fresh",
            "2_same_scene", "2_same_task_family_fresh",
            "3_same_task", "3_same_task_reused",
        }
        if (src in TARGET_SOURCES_ICL
                and src not in icl_picks
                and label == "failure"
                and partner_id
                and tier in GOOD_TIERS
                and row.get("partner_label") == "success"
                and archive not in seen_archives_icl):
            icl_picks[src] = row
            seen_archives_icl.add(archive)

        # Labeled failure: failure with frame labels available (in meta.json)
        if (src in TARGET_SOURCES_FAIL
                and src not in fail_picks
                and label == "failure"
                and archive not in seen_archives_fail
                # don't reuse the same archive as ICL pair for that source
                and archive not in {r["archive"] for r in icl_picks.values()}):
            fail_picks[src] = row
            seen_archives_fail.add(archive)

        if len(icl_picks) == len(TARGET_SOURCES_ICL) and len(fail_picks) == len(TARGET_SOURCES_FAIL):
            break

    return icl_picks, fail_picks


# ---------------------------------------------------------------------------
# Extraction from tars
# ---------------------------------------------------------------------------

# frames_path format:  <relpath>/shard-NNNNN.tar::<member_prefix>
TAR_RE = re.compile(r"^(.+\.tar)::(.+)$")


def parse_frames_path(frames_path: str) -> tuple[Path, str]:
    m = TAR_RE.match(frames_path)
    if not m:
        raise ValueError(f"unexpected frames_path format: {frames_path}")
    tar_rel, member_prefix = m.group(1), m.group(2)
    return DATASET_ROOT / tar_rel, member_prefix


def extract_episode(tar_path: Path, member_prefix: str, out_dir: Path) -> dict:
    """Extract every member under <prefix>/ and return parsed meta.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = None
    needle = member_prefix.rstrip("/") + "/"
    with tarfile.open(tar_path, "r") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            if not m.name.startswith(needle):
                continue
            tail = m.name[len(needle):]
            # Skip the directory entry itself; flatten frames + meta.json to out_dir
            if not tail:
                continue
            target = out_dir / tail
            target.parent.mkdir(parents=True, exist_ok=True)
            ef = tar.extractfile(m)
            if ef is None:
                continue
            with open(target, "wb") as g:
                shutil.copyfileobj(ef, g)
            if tail == "meta.json":
                target.seek if False else None
                with open(target) as g:
                    meta = json.load(g)
    if meta is None:
        raise RuntimeError(f"no meta.json found for {member_prefix} in {tar_path}")
    return meta


def find_partner_frames_path(partner_episode_id: str, partner_source: str, partner_archive: str) -> Optional[str]:
    """Locate partner row in pairs_unified.jsonl by episode_id (may have view suffix)."""
    # Partner_episode_id stored in pairs row is the BASE; the actual episode rows may carry
    # view suffixes (e.g. "_corner2"). Try exact match first, then prefix match.
    candidates_exact = []
    candidates_prefix = []
    for row in stream_pairs():
        if row.get("source") != partner_source:
            continue
        if row.get("archive") != partner_archive:
            continue
        eid = row.get("episode_id", "")
        if eid == partner_episode_id:
            candidates_exact.append(row)
        elif eid.startswith(partner_episode_id + "_") or eid.startswith(partner_episode_id):
            candidates_prefix.append(row)
    pick = candidates_exact[0] if candidates_exact else (candidates_prefix[0] if candidates_prefix else None)
    return pick.get("frames_path") if pick else None


# ---------------------------------------------------------------------------
# Build folder
# ---------------------------------------------------------------------------

def write_readme(path: Path, body: str) -> None:
    with open(path, "w") as f:
        f.write(body)


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40]


def build_icl_pair(slot: int, src: str, row: dict, out_dir: Path) -> dict:
    pair_dir = out_dir / f"{slot:02d}_{src}_{slugify(row['archive'])}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    fail_tar, fail_prefix = parse_frames_path(row["frames_path"])
    fail_meta = extract_episode(fail_tar, fail_prefix, pair_dir / "failure")

    partner_path = find_partner_frames_path(
        row["partner_episode_id"], row["partner_source"], row["partner_archive"]
    )
    if not partner_path:
        raise RuntimeError(f"partner not found for {row['episode_id']}")
    succ_tar, succ_prefix = parse_frames_path(partner_path)
    succ_meta = extract_episode(succ_tar, succ_prefix, pair_dir / "success")

    summary = {
        "kind": "icl_pair",
        "source": src,
        "archive": row["archive"],
        "task": row["task"],
        "failure_episode_id": row["episode_id"],
        "success_episode_id": succ_meta.get("episode_id"),
        "tier": row.get("tier"),
        "n_keyframes_each": fail_meta.get("n_keyframes"),
        "is_real_world": src in {"droid", "robometer", "roboreward"},
    }
    with open(pair_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    write_readme(pair_dir / "README.md", (
        f"# ICL Pair — {src}/{row['archive']}\n\n"
        f"**Task:** {row['task']}\n\n"
        f"**Source:** {src} ({'real-world' if summary['is_real_world'] else 'simulation'})\n\n"
        f"`failure/`  →  Failure trajectory ({row['episode_id']})\n\n"
        f"`success/`  →  Paired-success trajectory ({succ_meta.get('episode_id')})\n\n"
        f"Both have {fail_meta.get('n_keyframes')} keyframes (`frame_00...frame_15.jpg`) and a `meta.json`.\n"
    ))
    return summary


def build_labeled_failure(slot: int, src: str, row: dict, out_dir: Path) -> dict:
    fail_dir = out_dir / f"{slot:02d}_{src}_{slugify(row['archive'])}"
    fail_dir.mkdir(parents=True, exist_ok=True)

    fail_tar, fail_prefix = parse_frames_path(row["frames_path"])
    meta = extract_episode(fail_tar, fail_prefix, fail_dir)

    frame_labels = meta.get("frame_labels", [])
    summary = {
        "kind": "labeled_failure",
        "source": src,
        "archive": row["archive"],
        "task": row["task"],
        "episode_id": row["episode_id"],
        "n_keyframes": meta.get("n_keyframes"),
        "frame_labels": frame_labels,
        "is_real_world": src in {"droid", "robometer", "roboreward"},
    }
    with open(fail_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    lbl_str = " ".join(str(int(x)) for x in frame_labels) if frame_labels else "(none)"
    write_readme(fail_dir / "README.md", (
        f"# Labeled Failure — {src}/{row['archive']}\n\n"
        f"**Task:** {row['task']}\n\n"
        f"**Source:** {src} ({'real-world' if summary['is_real_world'] else 'simulation'})\n\n"
        f"**Per-keyframe labels** (1 = at-goal-state, 0 = not):\n\n"
        f"`{lbl_str}`\n\n"
        f"{meta.get('n_keyframes')} keyframes (`frame_00...frame_15.jpg`) + full `meta.json`.\n"
    ))
    return summary


def main():
    if OUT_ROOT.exists():
        print(f"output dir {OUT_ROOT} already exists — wiping")
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True)

    print("Selecting candidates from pairs_unified.jsonl ...")
    icl_picks, fail_picks = select_examples()
    print(f"  picked ICL pairs for: {list(icl_picks.keys())}")
    print(f"  picked labeled failures for: {list(fail_picks.keys())}")

    icl_dir = OUT_ROOT / "icl_pairs"
    fail_dir = OUT_ROOT / "labeled_failures"
    icl_dir.mkdir()
    fail_dir.mkdir()

    summaries = {"icl_pairs": [], "labeled_failures": []}

    for i, src in enumerate(TARGET_SOURCES_ICL, start=1):
        if src not in icl_picks:
            print(f"  WARNING: no ICL pair found for {src}")
            continue
        print(f"  building ICL pair #{i}: {src} / {icl_picks[src]['archive']}")
        s = build_icl_pair(i, src, icl_picks[src], icl_dir)
        summaries["icl_pairs"].append(s)

    for i, src in enumerate(TARGET_SOURCES_FAIL, start=1):
        if src not in fail_picks:
            print(f"  WARNING: no labeled failure found for {src}")
            continue
        print(f"  building labeled failure #{i}: {src} / {fail_picks[src]['archive']}")
        s = build_labeled_failure(i, src, fail_picks[src], fail_dir)
        summaries["labeled_failures"].append(s)

    with open(OUT_ROOT / "all_summaries.json", "w") as f:
        json.dump(summaries, f, indent=2)

    n_real = sum(1 for s in summaries["icl_pairs"] + summaries["labeled_failures"] if s["is_real_world"])
    n_total = len(summaries["icl_pairs"]) + len(summaries["labeled_failures"])
    write_readme(OUT_ROOT / "README.md", (
        "# Candidate datapoints for the main figure\n\n"
        f"{n_total} candidate datapoints from `robometer_frame_dataset`, "
        f"{n_real}/{n_total} real-world.\n\n"
        "## Layout\n\n"
        "- `icl_pairs/NN_<source>_<archive>/` — each contains `failure/` + `success/` "
        "(both with 16 keyframes + `meta.json`). These are the (failure, paired-success) "
        "pairs the reward model sees during ICL.\n"
        "- `labeled_failures/NN_<source>_<archive>/` — each contains 16 keyframes + "
        "`meta.json` whose `frame_labels` is a per-keyframe 1/0 indicator (1 = at-goal-state).\n\n"
        "## How to view\n\n"
        "Open the JPG files in any image viewer. Frames are named `frame_00_TIMES.jpg` ... "
        "`frame_15_TIMES.jpg` so they sort in temporal order. The `summary.json` next to each "
        "datapoint has the task description, source/archive, and (for labeled failures) the "
        "per-frame labels.\n\n"
        "## Picking the main figure\n\n"
        "Reply with the path of the datapoint(s) you want to feature.\n"
    ))

    print(f"\nDone. Output at: {OUT_ROOT}")
    print(f"  real-world: {n_real} / {n_total}")


if __name__ == "__main__":
    main()
