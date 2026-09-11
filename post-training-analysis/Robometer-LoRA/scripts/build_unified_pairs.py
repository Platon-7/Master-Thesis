#!/usr/bin/env python3
"""
build_unified_pairs.py — Emit a single JSONL with one row per episode across
every subset of /projects/prjs1958/robometer_frame_dataset/, with same-task
partner info (or `tier="no_pair"` when the partner isn't same-task) and
resolved shard-aware `frames_path` + multi-view `views` list.

Prereq: run build_shard_indices.py first.

Sources (unified `source` tag):
  metaworld                  — multi-view (corner2, corner3)
  failsafe                   — multi-view (front, side, wrist)
  droid                      — multi-view (ext1, ext2, wrist)
  robometer                  — single view (Group A + Group B failures + their successes)
  roboreward                 — single view (RBM-1M neural-annotated subset; failure manifests
                               carry inline `paired_success_id`, so we mirror build_robometer
                               but skip the external pair-file step)
  robometer_orphan_success   — single view

Per-row schema:
  episode_id, source, archive, family, task, label,
  partner_episode_id, partner_source, partner_archive, partner_task, partner_label, tier,
  frames_path                — canonical path ("<rel>/shard-NNNNN.tar::<inner_dir>")
  views                      — [{"view": str, "frames_path": str}, ...]
                               len(views) == 1 for single-view sources.
                               Loader samples a view at train time.

Multi-view sources (metaworld, failsafe) COLLAPSE per-view manifest rows into one
row per base episode. Partner references are mapped to base episode ids too
(the view-specific partner id is stripped to base — same base, different view).

Output:
  pairs_unified.jsonl
  pairs_unified_report.json
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/projects/prjs1958/robometer_frame_dataset")
OUT_PATH = ROOT / "pairs_unified.jsonl"
REPORT_PATH = ROOT / "pairs_unified_report.json"
ROBOMETER_PAIRS_DIR = Path("/projects/prjs1958/robometer_full_dataset/pairs")

DROID_SAME_TASK_TIERS = {"1_exact", "2_same_scene", "3_same_task", "3_same_task_reused"}
ROBOMETER_SAME_TASK_TIERS = {"1_same_task_fresh", "2_same_task_family_fresh", "3_same_task_reused"}

METAWORLD_VIEWS = ("corner2", "corner3")
FAILSAFE_VIEWS = ("front", "side", "wrist")
DROID_VIEWS = (
    ("ext1",  "keyframes",              "keyframes_success"),
    ("ext2",  "keyframes_ext2",         "keyframes_success_ext2"),
    ("wrist", "keyframes_wrist",        "keyframes_success_wrist"),
)

_EP_IDX_RE = re.compile(r"^ep_(\d+)_")


def parse_ep_idx(ep_id):
    m = _EP_IDX_RE.match(ep_id or "")
    return int(m.group(1)) if m else None


def load_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def norm_label(lbl):
    if lbl in ("success", "successful"):
        return "success"
    return lbl


def load_shard_index(archive_dir):
    p = Path(archive_dir) / "shard_index.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def strip_view(ep_id, views):
    """Strip a `_<view>` token from an episode_id. Returns (base_id, view) or (ep_id, None)."""
    for v in views:
        tok = f"_{v}"
        idx = ep_id.find(tok)
        if idx >= 0:
            # boundary: must be followed by `_` or end of string
            end = idx + len(tok)
            if end == len(ep_id) or ep_id[end] == "_":
                return ep_id[:idx] + ep_id[end:], v
    return ep_id, None


def rel_path(source_dir, shard, inner):
    return f"{source_dir}/{shard}::{inner}"


# --------------------------------------------------------------------------- #
# MetaWorld & Failsafe (collapse per-view rows into base rows)
# --------------------------------------------------------------------------- #
def build_mw_fs(source, manifest_dir, keyframes_sub, keyframes_succ_sub, views):
    # 1. Build shard indices per archive, cached on demand.
    shard_idx_cache = {}

    def idx_for(archive, label):
        sub = keyframes_succ_sub if label == "success" else keyframes_sub
        key = (archive, sub)
        if key not in shard_idx_cache:
            shard_idx_cache[key] = load_shard_index(ROOT / source / sub / archive)
        return shard_idx_cache[key], f"{source}/{sub}/{archive}"

    # 2. First pass: collect all rows, group by base_id.
    manifests = sorted(manifest_dir.glob(f"{source}_*_failures.jsonl")) + \
                sorted(manifest_dir.glob(f"{source}_*_successes.jsonl"))

    # base_id -> {base meta + views list}
    bases = {}
    # view_ep_id -> base_id  (for partner lookup)
    view_to_base = {}

    for p in manifests:
        for rec in load_jsonl(p):
            ep_id = rec["episode_id"]
            label = norm_label(rec.get("label"))
            base_id, view = strip_view(ep_id, views)
            view_to_base[ep_id] = base_id

            idx, rel_dir = idx_for(rec["archive"], label)
            shard = idx.get(ep_id)
            if shard is None:
                continue  # skip episodes with no extracted frames
            fpath = rel_path(rel_dir, shard, ep_id)

            if base_id not in bases:
                bases[base_id] = {
                    "episode_id": base_id,
                    "source": source,
                    "archive": rec.get("archive"),
                    "family": rec.get("family"),
                    "task": rec.get("task"),
                    "label": label,
                    "paired_success_id": rec.get("paired_success_id"),
                    "views": [],
                }
            bases[base_id]["views"].append({"view": view or "default",
                                            "frames_path": fpath})

    # 2.5  Build a per-task index of available successes to enable success-reuse
    #      fallback pairing (per user instruction: failsafe/metaworld have many
    #      more failure modes than success modes; reuse same success across
    #      multiple failures rather than leaving them no_pair).
    successes_by_task = defaultdict(list)
    for base_id_iter, b_iter in bases.items():
        if b_iter["label"] == "success" and b_iter.get("views"):
            successes_by_task[b_iter["task"]].append(base_id_iter)
    # Round-robin counter so reuse spreads evenly across available successes
    # instead of always picking the first one.
    success_use_idx = defaultdict(int)

    # 3. Resolve partners & emit unified rows.
    for base_id, b in bases.items():
        partner_view_id = b.pop("paired_success_id", None)
        partner_base = view_to_base.get(partner_view_id) if partner_view_id else None
        partner = bases.get(partner_base) if partner_base else None
        if partner and partner["task"] == b["task"] and partner["views"]:
            tier = "same_task"
            out_partner = {
                "partner_episode_id": partner["episode_id"],
                "partner_source": source,
                "partner_archive": partner["archive"],
                "partner_task": partner["task"],
                "partner_label": partner["label"],
            }
        else:
            tier = "no_pair"
            out_partner = {
                "partner_episode_id": None,
                "partner_source": None,
                "partner_archive": None,
                "partner_task": None,
                "partner_label": None,
            }

            # Fallback: if this is a failure that didn't get an explicit pair,
            # pick any same-task success (round-robin to spread reuse evenly).
            # User confirmed it's OK to reuse successes across multiple failures
            # since failure modes outnumber success modes in failsafe/metaworld.
            if b["label"] == "failure":
                candidates = successes_by_task.get(b["task"], [])
                if candidates:
                    pick_idx = success_use_idx[b["task"]] % len(candidates)
                    success_use_idx[b["task"]] += 1
                    fallback = bases[candidates[pick_idx]]
                    tier = "3_same_task_reused"
                    out_partner = {
                        "partner_episode_id": fallback["episode_id"],
                        "partner_source": source,
                        "partner_archive": fallback["archive"],
                        "partner_task": fallback["task"],
                        "partner_label": fallback["label"],
                    }

        views_sorted = sorted(b["views"], key=lambda v: v["view"])
        yield {
            **{k: b[k] for k in ["episode_id", "source", "archive", "family", "task", "label"]},
            **out_partner,
            "tier": tier,
            "frames_path": views_sorted[0]["frames_path"],
            "views": views_sorted,
        }


# --------------------------------------------------------------------------- #
# Droid (already 1 row per episode; 3 parallel view dirs)
# --------------------------------------------------------------------------- #
def build_droid():
    meta = ROOT / "droid" / "metadata"

    # Success lookup: episode_id -> record
    succ_lookup = {r["episode_id"]: r for r in load_jsonl(meta / "success_index.jsonl")}
    # Pair lookup: failure_ep -> pair record
    pair_lookup = {r["failure_ep"]: r for r in load_jsonl(meta / "success_pairs.jsonl")}

    # View indices: label -> view -> ep_id -> {shard, inner_dir}
    view_idx = {"failure": {}, "success": {}}
    for view, fail_sub, succ_sub in DROID_VIEWS:
        view_idx["failure"][view] = load_shard_index(ROOT / "droid" / fail_sub / "shards")
        view_idx["success"][view] = load_shard_index(ROOT / "droid" / succ_sub / "shards")

    def views_for(ep_id, label):
        out = []
        for view, fail_sub, succ_sub in DROID_VIEWS:
            sub = succ_sub if label == "success" else fail_sub
            entry = view_idx[label][view].get(ep_id)
            if not entry:
                continue
            out.append({
                "view": view,
                "frames_path": rel_path(f"droid/{sub}/shards",
                                        entry["shard"], entry["inner_dir"]),
            })
        return out

    seen = set()

    # Failures first
    for rec in load_jsonl(meta / "failure_index.jsonl"):
        ep_id = rec["episode_id"]
        if ep_id in seen:
            continue
        seen.add(ep_id)
        v = views_for(ep_id, "failure")
        if not v:
            continue  # no frames on disk

        pair = pair_lookup.get(ep_id)
        tier_raw = pair.get("tier") if pair else None
        if pair and tier_raw in DROID_SAME_TASK_TIERS:
            s_id = pair["success_ep"]
            s_rec = succ_lookup.get(s_id, {})
            out_partner = {
                "partner_episode_id": s_id,
                "partner_source": "droid",
                "partner_archive": s_rec.get("lab") or pair.get("failure_lab"),
                "partner_task": pair.get("success_task"),
                "partner_label": "success",
            }
            tier = tier_raw
        else:
            out_partner = {
                "partner_episode_id": None, "partner_source": None,
                "partner_archive": None, "partner_task": None, "partner_label": None,
            }
            tier = "no_pair"
        yield {
            "episode_id": ep_id, "source": "droid",
            "archive": rec.get("lab"), "family": "droid",
            "task": rec.get("current_task"), "label": "failure",
            **out_partner, "tier": tier,
            "frames_path": v[0]["frames_path"], "views": v,
        }

    # Successes
    for rec in load_jsonl(meta / "success_index.jsonl"):
        ep_id = rec["episode_id"]
        if ep_id in seen:
            continue
        seen.add(ep_id)
        v = views_for(ep_id, "success")
        if not v:
            continue
        yield {
            "episode_id": ep_id, "source": "droid",
            "archive": rec.get("lab"), "family": "droid",
            "task": rec.get("current_task"), "label": "success",
            "partner_episode_id": None, "partner_source": None,
            "partner_archive": None, "partner_task": None, "partner_label": None,
            "tier": "no_pair",
            "frames_path": v[0]["frames_path"], "views": v,
        }


# --------------------------------------------------------------------------- #
# Robometer failures + non-orphan successes (single view)
# --------------------------------------------------------------------------- #
def build_robometer():
    rm = ROOT / "robometer"
    scores_dir = rm / "scores"
    man_dir = rm / "manifests"

    fail_idx_cache = {}
    succ_idx_cache = {}

    def fail_idx(archive):
        if archive not in fail_idx_cache:
            fail_idx_cache[archive] = load_shard_index(rm / "keyframes" / archive)
        return fail_idx_cache[archive]

    def succ_idx(archive):
        if archive not in succ_idx_cache:
            succ_idx_cache[archive] = load_shard_index(rm / "keyframes_success" / archive)
        return succ_idx_cache[archive]

    # (archive, idx) -> episode_id  (from non-orphan success manifests)
    succ_by_aidx = {}
    succ_rec_by_id = {}
    for p in sorted(man_dir.glob("*_successes.jsonl")):
        if p.name.endswith("_orphan_successes.jsonl"):
            continue
        for rec in load_jsonl(p):
            i = parse_ep_idx(rec["episode_id"])
            if i is not None:
                succ_by_aidx[(rec["archive"], i)] = rec["episode_id"]
            succ_rec_by_id[rec["episode_id"]] = rec

    # (archive, failure_idx) -> pair record
    pair_lookup = {}
    for p in sorted(ROBOMETER_PAIRS_DIR.glob("*_pairs.jsonl")):
        for rec in load_jsonl(p):
            pair_lookup[(rec["archive"], rec["failure_idx"])] = rec

    def succ_frames_path(archive, ep_id):
        shard = succ_idx(archive).get(ep_id)
        if not shard:
            return None
        return rel_path(f"robometer/keyframes_success/{archive}", shard, ep_id)

    # Failures
    for p in sorted(scores_dir.glob("*_scored.jsonl")):
        for rec in load_jsonl(p):
            ep_id = rec["episode_id"]
            archive = rec["archive"]
            shard = fail_idx(archive).get(ep_id)
            if not shard:
                continue
            fpath = rel_path(f"robometer/keyframes/{archive}", shard, ep_id)

            f_idx = parse_ep_idx(ep_id)
            pair = pair_lookup.get((archive, f_idx)) if f_idx is not None else None
            tier_raw = pair.get("tier") if pair else None
            family = pair.get("family") if pair else None

            if pair and tier_raw in ROBOMETER_SAME_TASK_TIERS:
                s_arc = pair["success_archive"]
                s_idx = pair["success_idx"]
                s_ep = succ_by_aidx.get((s_arc, s_idx))
                if s_ep and succ_frames_path(s_arc, s_ep):
                    out_partner = {
                        "partner_episode_id": s_ep,
                        "partner_source": "robometer",
                        "partner_archive": s_arc,
                        "partner_task": pair.get("success_task") or
                                         succ_rec_by_id.get(s_ep, {}).get("task"),
                        "partner_label": "success",
                    }
                    tier = tier_raw
                else:
                    out_partner = {
                        "partner_episode_id": None, "partner_source": None,
                        "partner_archive": None, "partner_task": None, "partner_label": None,
                    }
                    tier = "no_pair"
            else:
                out_partner = {
                    "partner_episode_id": None, "partner_source": None,
                    "partner_archive": None, "partner_task": None, "partner_label": None,
                }
                tier = "no_pair"

            yield {
                "episode_id": ep_id, "source": "robometer",
                "archive": archive, "family": family,
                "task": rec.get("task"), "label": "failure",
                **out_partner, "tier": tier,
                "frames_path": fpath,
                "views": [{"view": "default", "frames_path": fpath}],
            }

    # Non-orphan successes
    for p in sorted(man_dir.glob("*_successes.jsonl")):
        if p.name.endswith("_orphan_successes.jsonl"):
            continue
        for rec in load_jsonl(p):
            ep_id = rec["episode_id"]
            archive = rec["archive"]
            fpath = succ_frames_path(archive, ep_id)
            if not fpath:
                continue
            yield {
                "episode_id": ep_id, "source": "robometer",
                "archive": archive, "family": rec.get("family"),
                "task": rec.get("task"), "label": norm_label(rec.get("label")),
                "partner_episode_id": None, "partner_source": None,
                "partner_archive": None, "partner_task": None, "partner_label": None,
                "tier": "no_pair",
                "frames_path": fpath,
                "views": [{"view": "default", "frames_path": fpath}],
            }


# --------------------------------------------------------------------------- #
# Robometer orphan successes (single view; shard in manifest)
# --------------------------------------------------------------------------- #
def build_roboreward():
    """Roboreward (RBM-1M's neural-annotated subset). Mirrors build_robometer() but
    uses the failure manifest's inline `paired_success_id` field instead of an external
    pairs file — every roboreward failure already names its success demo directly.
    """
    rb = ROOT / "roboreward"
    man_dir = rb / "manifests"

    fail_idx_cache, succ_idx_cache = {}, {}

    def fail_idx(archive):
        if archive not in fail_idx_cache:
            fail_idx_cache[archive] = load_shard_index(rb / "keyframes" / archive)
        return fail_idx_cache[archive]

    def succ_idx(archive):
        if archive not in succ_idx_cache:
            succ_idx_cache[archive] = load_shard_index(rb / "keyframes_success" / archive)
        return succ_idx_cache[archive]

    # Build success-record lookup so failures can resolve partner_task / partner_archive.
    succ_rec_by_id = {}
    for p in sorted(man_dir.glob("*_successes.jsonl")):
        for rec in load_jsonl(p):
            succ_rec_by_id[rec["episode_id"]] = rec

    def succ_frames_path(archive, ep_id):
        shard = succ_idx(archive).get(ep_id)
        if not shard:
            return None
        return rel_path(f"roboreward/keyframes_success/{archive}", shard, ep_id)

    # Failures
    for p in sorted(man_dir.glob("*_failures.jsonl")):
        for rec in load_jsonl(p):
            ep_id = rec["episode_id"]
            archive = rec["archive"]
            shard = fail_idx(archive).get(ep_id)
            if not shard:
                continue
            fpath = rel_path(f"roboreward/keyframes/{archive}", shard, ep_id)

            partner_id = rec.get("paired_success_id")
            partner = succ_rec_by_id.get(partner_id) if partner_id else None
            partner_fpath = (
                succ_frames_path(partner["archive"], partner_id)
                if partner else None
            )
            if partner and partner_fpath:
                out_partner = {
                    "partner_episode_id": partner_id,
                    "partner_source": "roboreward",
                    "partner_archive": partner["archive"],
                    "partner_task": partner.get("task"),
                    "partner_label": "success",
                }
                tier = "1_exact"   # paired by manifest, treated as the strict tier
            else:
                out_partner = {
                    "partner_episode_id": None, "partner_source": None,
                    "partner_archive": None, "partner_task": None, "partner_label": None,
                }
                tier = "no_pair"

            yield {
                "episode_id": ep_id, "source": "roboreward",
                "archive": archive, "family": rec.get("family"),
                "task": rec.get("task"), "label": "failure",
                **out_partner, "tier": tier,
                "frames_path": fpath,
                "views": [{"view": "default", "frames_path": fpath}],
            }

    # Successes
    for p in sorted(man_dir.glob("*_successes.jsonl")):
        for rec in load_jsonl(p):
            ep_id = rec["episode_id"]
            archive = rec["archive"]
            fpath = succ_frames_path(archive, ep_id)
            if not fpath:
                continue
            yield {
                "episode_id": ep_id, "source": "roboreward",
                "archive": archive, "family": rec.get("family"),
                "task": rec.get("task"), "label": norm_label(rec.get("label")),
                "partner_episode_id": None, "partner_source": None,
                "partner_archive": None, "partner_task": None, "partner_label": None,
                "tier": "no_pair",
                "frames_path": fpath,
                "views": [{"view": "default", "frames_path": fpath}],
            }


def _build_orphan_subtree(subdir, source_tag):
    """Common builder for any orphan-success subtree (robometer/, humanoid/,
    human_hand/). Schema and tier semantics are identical across all three —
    only the source_tag (= subdir name + '_orphan_success') and the on-disk
    frames_path prefix differ. Yields one row per episode."""
    rm = ROOT / subdir
    pairs_dir = rm / "pairs_orphan"
    man_dir = rm / "manifests"

    # ep_id -> manifest rec (for shard + label)
    by_id = {}
    for p in sorted(man_dir.glob("*_orphan_successes.jsonl")):
        for rec in load_jsonl(p):
            by_id[rec["episode_id"]] = rec

    def frames_path_for(rec):
        archive = rec["archive"]
        shard = rec.get("shard")
        if not shard:
            return None
        return rel_path(f"{subdir}/keyframes_orphan_success/{archive}",
                        shard, rec["episode_id"])

    for p in sorted(pairs_dir.glob("*_orphan_pairs.jsonl")):
        for rec in load_jsonl(p):
            ep_id = rec["episode_id"]
            src_rec = by_id.get(ep_id)
            if not src_rec:
                continue
            fpath = frames_path_for(src_rec)
            if not fpath:
                continue

            partner_id = rec.get("partner_episode_id")
            partner_rec = by_id.get(partner_id) if partner_id else None
            partner_fpath = frames_path_for(partner_rec) if partner_rec else None

            if partner_id and partner_fpath:
                out_partner = {
                    "partner_episode_id": partner_id,
                    "partner_source": source_tag,
                    "partner_archive": rec.get("partner_archive"),
                    "partner_task": rec.get("partner_task"),
                    "partner_label": "success",
                }
                tier = rec.get("tier", "no_pair")
            else:
                out_partner = {
                    "partner_episode_id": None, "partner_source": None,
                    "partner_archive": None, "partner_task": None, "partner_label": None,
                }
                tier = "no_pair"

            yield {
                "episode_id": ep_id, "source": source_tag,
                "archive": rec.get("archive"), "family": rec.get("family"),
                "task": rec.get("task"), "label": norm_label(src_rec.get("label", "success")),
                **out_partner, "tier": tier,
                "frames_path": fpath,
                "views": [{"view": "default", "frames_path": fpath}],
            }


def build_robometer_orphan():
    yield from _build_orphan_subtree("robometer", "robometer_orphan_success")


def build_humanoid_orphan():
    # Skip cleanly if the subtree hasn't been created yet (e.g., extraction
    # still running or this build is being run on an older snapshot).
    if not (ROOT / "humanoid" / "manifests").exists():
        return
    yield from _build_orphan_subtree("humanoid", "humanoid_orphan_success")


def build_human_hand_orphan():
    if not (ROOT / "human_hand" / "manifests").exists():
        return
    yield from _build_orphan_subtree("human_hand", "human_hand_orphan_success")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _pair_unpartnered_successes_inplace(out_path, seed: int = 42) -> dict:
    """Post-pass on pairs_unified.jsonl: for every success row that still has
    partner_episode_id=None, pick another success of the same (source, task) and set it
    as the demo partner. Skips orphan-success rows that were already paired upstream
    (via robometer/pairs_orphan/) and skips any (source, task) cell with only one
    success (cannot pair). Rewrites the file in place.

    Why: failure→success pairings were already populated by each per-source builder.
    But policy_ranking eval samples both quality labels per task, so unless successes
    also carry a demo, the ICL-on pass falls back to ICL-off on the success side and
    breaks apples-to-apples comparison. This pass adds same-task success→success
    pairings so every success can serve as an ICL query.

    Returns counters for the build report.
    """
    import random as _r
    rng = _r.Random(seed)

    print("\n[post] adding success → success pairings for unpartnered successes …", flush=True)
    rows = [json.loads(l) for l in open(out_path) if l.strip()]

    # Index unpartnered successes by (source, task)
    by_st = defaultdict(list)  # (source, task) → [row_index]
    for idx, r in enumerate(rows):
        if r.get("label") not in ("success", "successful"):
            continue
        if r.get("partner_episode_id"):
            continue  # already paired (orphan_success → orphan_success, etc.)
        task = r.get("task")
        if not task:
            continue
        by_st[(r["source"], task)].append(idx)

    n_paired, n_singleton = 0, 0
    for (source, task), idxs in by_st.items():
        if len(idxs) < 2:
            n_singleton += len(idxs)
            continue
        for i in idxs:
            partner_i = rng.choice([j for j in idxs if j != i])
            p = rows[partner_i]
            r = rows[i]
            r["partner_episode_id"] = p["episode_id"]
            r["partner_source"]     = p["source"]
            r["partner_archive"]    = p["archive"]
            r["partner_task"]       = p["task"]
            r["partner_label"]      = "success"
            r["tier"]               = "1_same_task_fresh"  # synthetic, mirrors the strict tier
            n_paired += 1

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"[post] paired {n_paired} success queries; {n_singleton} singleton-task successes left unpartnered", flush=True)
    return {"success_pairings_added": n_paired, "singleton_successes": n_singleton}


def main():
    counts = defaultdict(lambda: {"total": 0, "tier": Counter(),
                                  "label": Counter(), "views_dist": Counter()})
    seen = set()
    duplicates = 0

    with open(OUT_PATH, "w") as out:
        def emit(gen, label):
            nonlocal duplicates
            n = 0
            for row in gen:
                # Dedup keyed on (source, episode_id) so a shared id across sources still survives
                key = (row["source"], row["episode_id"])
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                out.write(json.dumps(row) + "\n")
                src = row["source"]
                counts[src]["total"] += 1
                counts[src]["tier"][row["tier"]] += 1
                counts[src]["label"][row["label"]] += 1
                counts[src]["views_dist"][len(row.get("views", []))] += 1
                n += 1
            print(f"[{label}] wrote {n} rows", flush=True)

        emit(build_mw_fs("metaworld", ROOT / "metaworld" / "manifests",
                         "keyframes", "keyframes_success", METAWORLD_VIEWS), "metaworld")
        emit(build_mw_fs("failsafe",  ROOT / "failsafe"  / "manifests",
                         "keyframes", "keyframes_success", FAILSAFE_VIEWS),  "failsafe")
        emit(build_droid(), "droid")
        emit(build_robometer(), "robometer")
        emit(build_roboreward(), "roboreward")
        emit(build_robometer_orphan(), "robometer_orphan_success")
        emit(build_humanoid_orphan(),  "humanoid_orphan_success")
        emit(build_human_hand_orphan(),"human_hand_orphan_success")

    # Post-pass: pair unpartnered successes with another same-task same-source success.
    # This makes pairs_unified.jsonl carry success→success demos in addition to the
    # failure→success demos already populated above. Run AFTER the main emit loop so
    # all rows are visible (a success in source X can only be paired against another
    # success in source X — we don't cross-source pair).
    post_counts = _pair_unpartnered_successes_inplace(OUT_PATH, seed=42)

    report = {
        "output": str(OUT_PATH),
        "per_source": {
            src: {
                "total": stats["total"],
                "tier": dict(stats["tier"]),
                "label": dict(stats["label"]),
                "views_distribution": dict(stats["views_dist"]),
            }
            for src, stats in counts.items()
        },
        "grand_total": sum(s["total"] for s in counts.values()),
        "duplicates_skipped": duplicates,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nGrand total: {report['grand_total']} rows")
    print(f"Duplicates skipped: {duplicates}")
    print(f"Output: {OUT_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
