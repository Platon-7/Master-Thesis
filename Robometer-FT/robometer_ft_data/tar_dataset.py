"""Direct-from-tar dataset for Robometer-FT — bypasses HF + MP4 + NPZ.

Source-of-truth (set by the dataset team — `100% consistent` as of 2026-05-08):

    /projects/prjs1958/robometer_frame_dataset/
      pairs_unified.jsonl                       ← master list of all 651k rows
      <source>/manifests/<archive>_*.jsonl      ← per-episode metadata + frame_labels

Cut-line: only `_load_all_datasets` is overridden in `RBMDataset`. Everything
downstream (samplers, collators, trainer, losses) stays exactly as upstream.

Per-row dict shape returned to samplers:
    id, data_source, quality_label, task, is_robot, partial_success,
    frame_labels, frames (lazy ndarray), embeddings_path, lang_vector,
    text_embedding
"""
from __future__ import annotations

import collections
import io
import json
import os
import tarfile
from glob import glob
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from robometer.data.datasets.rbm_data import RBMDataset


# ---------------------------------------------------------------------------
# Frame loading: tar member → ndarray (T, H, W, 3) uint8
# ---------------------------------------------------------------------------


def _decode_episode_frames(tar_path: str, member_prefix: str) -> np.ndarray:
    """Open a tar, find this episode's JPGs, decode in frame order."""
    with tarfile.open(tar_path, "r") as tf:
        candidates: List[Tuple[str, tarfile.TarInfo]] = []
        for m in tf:
            if m.name.startswith(member_prefix + "/") and m.name.endswith(".jpg"):
                candidates.append((m.name, m))
        candidates.sort(key=lambda x: x[0])  # frame_NN_*.jpg sort -> chronological

        if not candidates:
            raise FileNotFoundError(
                f"No JPGs for prefix {member_prefix!r} in {tar_path}"
            )

        frames: List[np.ndarray] = []
        for _, m in candidates:
            f = tf.extractfile(m)
            if f is None:
                continue
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
            frames.append(np.asarray(img, dtype=np.uint8))

    if not frames:
        raise FileNotFoundError(f"All JPGs unreadable for {tar_path}::{member_prefix}")

    # Drop frames with non-modal shape (rare).
    shapes = [f.shape for f in frames]
    if len(set(shapes)) > 1:
        common = collections.Counter(shapes).most_common(1)[0][0]
        frames = [f for f in frames if f.shape == common]
    return np.stack(frames, axis=0)


# ---------------------------------------------------------------------------
# HF-Dataset-API mimic
# ---------------------------------------------------------------------------


class TarKeyframeIndex:
    """In-memory row table that quacks like an HF Dataset.

    Supports the exact surface BaseDataset / RBMDataset / samplers use:
      * len(self), self[idx] → row dict (frames lazy-decoded)
      * self["colname"]      → list of values
      * self.column_names    → list[str]
      * self.select(idx)     → subset
      * self.filter(pred)    → subset
    """

    # Required scalar fields — checked at build time. `tar_path` /
    # `member_prefix` are internal (used by __getitem__ for lazy decode); not
    # exposed via column_names.
    _REQUIRED_COLS = (
        "id", "data_source", "quality_label", "task", "is_robot",
        "partial_success", "frame_labels", "frames_shape",
    )

    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows
        for r in rows[:5]:
            missing = [c for c in self._REQUIRED_COLS if c not in r]
            if missing:
                raise ValueError(f"row missing required cols {missing}: {r}")
            if "tar_path" not in r or "member_prefix" not in r:
                raise ValueError(f"row missing tar_path/member_prefix: {r}")
        # column_names = union of all keys ever set on a row (incl. those
        # added by `.map`), minus the internal tar_path/member_prefix keys,
        # plus the lazy `frames` column.
        self._extra_cols: List[str] = []  # populated by .map() output

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "frames":
                return [None] * len(self._rows)
            # Return list of values for any column actually present on rows.
            return [r.get(key) for r in self._rows]
        if isinstance(key, slice):
            return TarKeyframeIndex(self._rows[key])
        idx = int(key)
        r = self._rows[idx]
        out = {k: v for k, v in r.items() if k not in ("tar_path", "member_prefix")}
        out["frames"] = _decode_episode_frames(r["tar_path"], r["member_prefix"])
        return out

    @property
    def column_names(self) -> List[str]:
        if not self._rows:
            return list(self._REQUIRED_COLS) + ["frames"]
        # Use first row's actual keys (post-map updates included).
        keys = [k for k in self._rows[0].keys() if k not in ("tar_path", "member_prefix")]
        return keys + ["frames"]

    def select(self, indices: Sequence[int]) -> "TarKeyframeIndex":
        return TarKeyframeIndex([self._rows[i] for i in indices])

    def filter(self, predicate, **_kwargs) -> "TarKeyframeIndex":
        return TarKeyframeIndex([r for r in self._rows if predicate(r)])

    def map(self, fn, batched: bool = False, num_proc=None, desc=None, **_kwargs) -> "TarKeyframeIndex":
        """HF-Dataset-compatible `map`: apply `fn` to rows, merge its output
        dict into each row. We ignore `num_proc` (single-process is fine for
        flag-computation-style use; the actual heavy work in our pipeline is
        per-step JPG decoding inside the DataLoader workers, not here)."""
        new_rows = [dict(r) for r in self._rows]
        if batched:
            # Build column-wise batch dict from every key any row carries.
            all_keys = set()
            for r in new_rows:
                all_keys.update(r.keys())
            all_keys.discard("tar_path")
            all_keys.discard("member_prefix")
            batch = {k: [r.get(k) for r in new_rows] for k in all_keys}
            result = fn(batch)
            for col, vals in result.items():
                if len(vals) != len(new_rows):
                    raise ValueError(
                        f".map() output column {col!r} length {len(vals)} != "
                        f"row count {len(new_rows)}"
                    )
                for i, v in enumerate(vals):
                    new_rows[i][col] = v
        else:
            for r in new_rows:
                update = fn(r)
                if update:
                    r.update(update)
        return TarKeyframeIndex(new_rows)


# ---------------------------------------------------------------------------
# Build rows from pairs_unified.jsonl (joined with manifest frame_labels)
# ---------------------------------------------------------------------------


def _split_frames_path(frames_path: str) -> Tuple[str, str]:
    """`<rel_tar>::<member_prefix>` → (rel_tar, member_prefix)."""
    rel_tar, _, member_prefix = frames_path.partition("::")
    return rel_tar, member_prefix


def _resolve_droid_inner_dir(inner_dir: str, known_eids: set) -> Optional[str]:
    """For droid, inner_dir is `<episode_id>__<task_slug>`. Some episode_ids
    contain `__` themselves (e.g. `Fri_Jul__7_10-44-04`), so we can't just
    rsplit. Instead try every `__` split point from RIGHTMOST first; the
    correct split is the longest left-prefix that's a known episode_id.

    Mirrors the algorithm in
    Robometer-LoRA/scripts/build_shard_indices.py:resolve_droid_dir.
    """
    candidates = []
    i = 0
    while True:
        j = inner_dir.find("__", i)
        if j < 0:
            break
        candidates.append(j)
        i = j + 2
    for pos in reversed(candidates):
        left = inner_dir[:pos]
        if left in known_eids:
            return left
    return None


def _load_frame_labels_from_manifests(
    frame_dataset_root: str,
) -> Tuple[Dict[str, List[int]], set]:
    """Read every manifest JSONL; collect {episode_id: frame_labels}.

    Manifests are keyed by their own `episode_id` field. For non-droid sources
    that field equals pairs_unified's `frames_path::inner_dir`. For droid the
    inner_dir has a trailing `__<task_slug>` which we strip via
    `_resolve_droid_inner_dir` at lookup time. Both droid and non-droid go
    through the same lookup table; the only family-specific logic is the
    droid resolver path.

    Returns (lookup, known_eids) — known_eids is the set of all manifest
    episode_ids, used by the droid resolver.
    """
    lookup: Dict[str, List[int]] = {}
    known_eids: set = set()
    n_total_rows = 0
    n_with_fl = 0

    for fam_dir in sorted(glob(os.path.join(frame_dataset_root, "*"))):
        manifests_dir = os.path.join(fam_dir, "manifests")
        if not os.path.isdir(manifests_dir):
            continue
        for mpath in sorted(glob(os.path.join(manifests_dir, "*.jsonl"))):
            with open(mpath) as f:
                for line in f:
                    n_total_rows += 1
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    eid = r.get("episode_id")
                    if not eid:
                        continue
                    known_eids.add(eid)
                    fl = r.get("frame_labels")
                    if fl:
                        lookup[eid] = list(fl)
                        n_with_fl += 1

    print(f"[tar-dataset] manifest scan: {n_total_rows} rows seen, "
          f"{n_with_fl} with inline frame_labels, lookup size={len(lookup)}, "
          f"known_eids={len(known_eids)}")
    return lookup, known_eids


def _load_scope_eids(scope_paths: Sequence[str]) -> Optional[set]:
    """Load episode_ids from one or more *.jsonl files. Each line is a JSON
    object with at least an `episode_id` field. Returns the union as a set,
    or None if scope_paths is empty (== no filtering).
    """
    if not scope_paths:
        return None
    eids: set = set()
    for p in scope_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"scope file missing: {p}")
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    eids.add(json.loads(line)["episode_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return eids


def build_index_from_pairs_unified(
    frame_dataset_root: str,
    pairs_unified_path: Optional[str] = None,
    scope_eids: Optional[set] = None,
) -> TarKeyframeIndex:
    """Master entry point. Builds a TarKeyframeIndex from pairs_unified.jsonl
    plus per-source manifest frame_labels.

    Args:
        frame_dataset_root: e.g. /projects/prjs1958/robometer_frame_dataset/
        pairs_unified_path: defaults to <frame_dataset_root>/pairs_unified.jsonl
        scope_eids: optional allow-list of episode_ids. When provided, rows whose
            episode_id is not in this set are dropped before any other processing.
            Used by TAR_SPLITS_DIR-driven train/eval scoping (see
            TarKeyframeRBMDataset._load_all_datasets).
    """
    if pairs_unified_path is None:
        pairs_unified_path = os.path.join(frame_dataset_root, "pairs_unified.jsonl")

    frame_labels_lookup, known_eids = _load_frame_labels_from_manifests(frame_dataset_root)

    rows: List[Dict[str, Any]] = []
    n_total = 0
    n_dropped_label = 0
    n_dropped_no_fl_failure = 0
    n_dropped_out_of_scope = 0
    n_no_family = 0
    n_no_task = 0

    with open(pairs_unified_path) as f:
        for line in f:
            n_total += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            if scope_eids is not None and r.get("episode_id") not in scope_eids:
                n_dropped_out_of_scope += 1
                continue

            label = r.get("label")
            if label not in ("success", "failure"):
                n_dropped_label += 1
                continue

            source = r.get("source")
            if source is None:
                n_no_family += 1
                continue

            task = r.get("task") or ""
            if not task:
                n_no_task += 1
                continue

            frames_path = r.get("frames_path") or ""
            rel_tar, member_prefix = _split_frames_path(frames_path)
            if not rel_tar or not member_prefix:
                continue
            tar_path = os.path.join(frame_dataset_root, rel_tar)

            # Frame labels: universal join via inner_dir.
            #
            # pairs_unified.episode_id is intentionally the BASE id (view
            # collapsed by build_unified_pairs.py:strip_view), so it doesn't
            # match per-view manifest keys for failsafe/metaworld. The
            # PER-VIEW manifest key is `inner_dir = frames_path.split("::")[1]`.
            # That key works directly for failsafe/metaworld/robometer/
            # roboreward. For droid, inner_dir = `<eid>__<task_slug>`; resolve
            # to the bare eid via _resolve_droid_inner_dir.
            fl = frame_labels_lookup.get(member_prefix)
            if fl is None and source == "droid":
                resolved = _resolve_droid_inner_dir(member_prefix, known_eids)
                if resolved is not None:
                    fl = frame_labels_lookup.get(resolved)
            if label == "failure" and not fl:
                n_dropped_no_fl_failure += 1
                continue

            quality_label = "successful" if label == "success" else "failure"

            # frames_shape: number of keyframes. For failures with curated
            # frame_labels, equals len(fl). For successes (no fl), default to
            # 16 (the dataset's keyframe budget — virtually every trajectory).
            # Used by upstream's min_frames filter (passes anything > min_frames,
            # which is typically <= 8). Constant default is fine because the
            # downstream samplers + RBMDataset's skip-and-retry handle any
            # actually-too-short episodes via SampleSkipRequest.
            frames_shape = len(fl) if fl else 16

            rows.append({
                "id":              r["episode_id"],
                "data_source":     f"robometer_frames_{source}",
                "quality_label":   quality_label,
                "task":            task,
                "is_robot":        True,
                "partial_success": False,
                "frame_labels":    fl,             # list[int] for failure; None for success
                "frames_shape":    frames_shape,
                "embeddings_path": None,
                "lang_vector":     None,
                "text_embedding":  None,
                "tar_path":        tar_path,
                "member_prefix":   member_prefix,
            })

    print(f"[tar-dataset] pairs_unified rows: {n_total} total | kept {len(rows)} | "
          f"dropped {n_dropped_out_of_scope} (out of scope) + "
          f"{n_dropped_label} (non-binary label) + "
          f"{n_no_family} (no source) + {n_no_task} (no task) + "
          f"{n_dropped_no_fl_failure} (failure with missing frame_labels)")

    if rows:
        # Quick sanity log of label distribution kept.
        ql_counts = collections.Counter(r["quality_label"] for r in rows)
        print(f"[tar-dataset] kept rows by quality_label: {dict(ql_counts)}")

    return TarKeyframeIndex(rows)


# ---------------------------------------------------------------------------
# RBMDataset subclass — overrides only `_load_all_datasets`.
# ---------------------------------------------------------------------------


class TarKeyframeRBMDataset(RBMDataset):
    """Drop-in RBMDataset that reads rows from pairs_unified.jsonl + manifest
    frame_labels instead of HF arrow.

    Honors these new config attrs (set via Hydra `++data.<name>=<val>`):
        data.tar_dataset_root      — root containing pairs_unified.jsonl
        data.pairs_unified_path    — explicit override for the master file path
                                     (default: <tar_dataset_root>/pairs_unified.jsonl)
    """

    def _load_all_datasets(self) -> Tuple[TarKeyframeIndex, Dict[str, Any]]:
        # Source paths come from env vars (NOT cfg.data) because upstream
        # DataConfig is a strict dataclass that rejects unknown fields.
        # The SLURM job exports TAR_DATASET_ROOT (and optionally
        # TAR_PAIRS_UNIFIED_PATH); both have sensible defaults.
        frame_dataset_root = os.environ.get(
            "TAR_DATASET_ROOT", "/projects/prjs1958/robometer_frame_dataset"
        )
        pairs_unified_path = os.environ.get("TAR_PAIRS_UNIFIED_PATH") or None

        # Train/eval scoping: TAR_SPLITS_DIR points at a dir containing
        # train.jsonl + eval_*.jsonl files (one episode_id per line). When set,
        # we filter pairs_unified rows down to the relevant scope. Without it,
        # the loader falls back to the legacy "all 651k rows" behavior.
        splits_dir = os.environ.get("TAR_SPLITS_DIR") or None
        scope_eids = None
        if splits_dir:
            from glob import glob
            if self.is_evaluation:
                scope_paths = sorted(glob(os.path.join(splits_dir, "eval_*.jsonl")))
                if not scope_paths:
                    raise FileNotFoundError(
                        f"TAR_SPLITS_DIR={splits_dir} has no eval_*.jsonl files"
                    )
            else:
                scope_paths = [os.path.join(splits_dir, "train.jsonl")]
            scope_eids = _load_scope_eids(scope_paths)
            print(f"[tar-dataset] scope: is_eval={self.is_evaluation}, "
                  f"files={[os.path.basename(p) for p in scope_paths]}, "
                  f"unique_eids={len(scope_eids)}")

        index = build_index_from_pairs_unified(
            frame_dataset_root=frame_dataset_root,
            pairs_unified_path=pairs_unified_path,
            scope_eids=scope_eids,
        )

        # combined_indices schema (from upstream BaseDataset._build_indices):
        #   robot_trajectories: list[int]                — all rows (we have only robot data)
        #   human_trajectories: list[int]                — empty
        #   optimal_by_task:    dict[task, list[int]]    — successful rows by task
        #   suboptimal_by_task: dict[task, list[int]]    — failure rows by task
        #   quality_indices:    dict[label, list[int]]
        #   task_indices:       dict[task, list[int]]
        #   source_indices:     dict[data_source, list[int]]
        #   partial_success_indices: dict[task, list[int]]   — empty (no partials)
        #   tasks_with_multiple_quality_labels: list[str]
        #   paired_human_robot_by_task: dict[task, ...]      — empty (no humans)
        ci: Dict[str, Any] = {
            "robot_trajectories": [],
            "human_trajectories": [],
            "optimal_by_task": {},
            "suboptimal_by_task": {},
            "quality_indices": {},
            "task_indices": {},
            "source_indices": {},
            "partial_success_indices": {},
            "paired_human_robot_by_task": {},
        }
        for i, r in enumerate(index._rows):
            ci["robot_trajectories"].append(i)
            ql = r["quality_label"]
            task = r["task"]
            src = r["data_source"]
            ci["quality_indices"].setdefault(ql, []).append(i)
            ci["task_indices"].setdefault(task, []).append(i)
            ci["source_indices"].setdefault(src, []).append(i)
            if ql == "successful":
                ci["optimal_by_task"].setdefault(task, []).append(i)
            else:
                ci["suboptimal_by_task"].setdefault(task, []).append(i)
        ci["tasks_with_multiple_quality_labels"] = list(
            set(ci["optimal_by_task"]) & set(ci["suboptimal_by_task"])
        )
        return index, ci
