"""Loader for the preprocessed keyframe dataset at
/projects/prjs1958/robometer_frame_dataset/.

Emits trajectory dicts in the format expected by
Robometer/dataset_upload/generate_hf_dataset.py → create_hf_trajectory:

    {
        "id": str,                       # episode_id
        "task": str,                     # natural-language instruction
        "frames": Callable[[], np.ndarray],  # lazy (T, H, W, 3) uint8
        "is_robot": True,
        "quality_label": "successful" | "failure",
        "data_source": str,              # "robometer_frames_<family>"
        "preference_group_id": str|None, # paired_success_id (links fail→success)
        "preference_rank": int|None,
        "frame_labels": List[int] | None,# per-frame ordinals {1..4} for failures, None for successes.
                                         # Consumed by the sampler to build target_progress via rubric.
        "partial_success": None,         # explicitly unused here — see frame_labels instead
    }

Not yet wired into generate_hf_dataset.py. Wiring is a 6-line `elif` branch in
its main(); see MODIFICATIONS.md §A.2.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image

DATASET_ROOT = Path("/projects/prjs1958/robometer_frame_dataset")

FAMILY_DIRS = {
    "metaworld": "metaworld",
    "failsafe": "failsafe",
    "droid": "droid",
    "robometer": "robometer",
    "roboreward": "roboreward",
}


class TarKeyframeLoader:
    """Pickle-able lazy loader: reads 16 JPGs from a tar shard on demand."""

    def __init__(self, shard_path: str, episode_dir: str) -> None:
        self.shard_path = shard_path
        self.episode_dir = episode_dir.rstrip("/") + "/"

    def __call__(self) -> np.ndarray:
        frames: List[np.ndarray] = []
        with tarfile.open(self.shard_path, "r") as tf:
            members = [
                m for m in tf.getmembers()
                if m.name.startswith(self.episode_dir)
                and m.name.endswith(".jpg")
            ]
            members.sort(key=lambda m: m.name)
            for m in members:
                f = tf.extractfile(m)
                if f is None:
                    continue
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
                frames.append(np.asarray(img, dtype=np.uint8))
        if not frames:
            return np.empty((0, 0, 0, 3), dtype=np.uint8)
        return np.stack(frames, axis=0)


def _iter_manifest(manifest_path: Path) -> Iterable[Dict[str, Any]]:
    with manifest_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _index_shards(keyframes_root: Path) -> Dict[str, Path]:
    """Scan every archive's shard-*.tar once and index episode_dir → shard path.

    Expected layout:
        <keyframes_root>/<archive>/shard-00000.tar
        <keyframes_root>/<archive>/shard-00001.tar
        ...
    Each tar contains top-level directories named after episode_id.
    """
    index: Dict[str, Path] = {}
    for archive_dir in sorted(keyframes_root.iterdir()):
        if not archive_dir.is_dir():
            continue
        for shard in sorted(archive_dir.glob("shard-*.tar")):
            with tarfile.open(shard, "r") as tf:
                # Pull top-level dir names cheaply from the member list.
                for name in tf.getnames():
                    top = name.split("/", 1)[0]
                    if top and top not in index:
                        index[top] = shard
    return index


def _build_manifest_index(dataset_root: Path, families: Iterable[str] = ("metaworld", "failsafe", "robometer", "roboreward", "droid")) -> Dict[str, Dict[str, Any]]:
    """Cross-family episode_id → manifest_row index.

    Manifest rows carry the per-frame `frame_labels` list that the failure rubric path needs
    (rows in pairs_unified.jsonl don't include frame_labels). Built once at loader startup.
    All families now have a manifests/ dir after the May-3 frame_labels-fix rearrangement:
      * robometer/manifests/  — added <archive>_failures.jsonl synthesised from scores/
      * roboreward/manifests/ — augmented existing _failures.jsonl with frame_labels from scores/
      * droid/manifests/      — newly synthesised droid_failures.jsonl from metadata/scored shards
    """
    index: Dict[str, Dict[str, Any]] = {}
    for fam in families:
        manifests_dir = dataset_root / fam / "manifests"
        if not manifests_dir.is_dir():
            continue
        for manifest in sorted(manifests_dir.glob("*.jsonl")):
            for row in _iter_manifest(manifest):
                eid = row.get("episode_id")
                if eid:
                    index[eid] = row
    return index


def load_robometer_frames_split(
    base_path: str,
    pairs_index_path: str,
    data_source_override: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Loader for a build_splits.py-produced pairs_index file (train or any eval slice).

    Each pairs_index row has the full metadata from pairs_unified.jsonl — including
    `frames_path` ("<family>/keyframes/.../shard-NNNNN.tar::episode_dir"), so we don't have
    to scan tar shards to find episodes. For per-frame `frame_labels` (only on failure rows
    of failsafe/metaworld/robometer/roboreward), we look up the matching manifest row via a
    cross-family index built once at startup.

    When `data_source_override` is set, every emitted trajectory carries that string as its
    `data_source` instead of the per-family default. Used by the warmup split so the
    StratifiedWarmupBatchSampler (which keys on the substring 'warmup') can identify it.
    """
    base_root = Path(base_path)
    pairs_index = Path(pairs_index_path)
    if not pairs_index.is_file():
        raise FileNotFoundError(f"pairs_index file not found: {pairs_index_path}")

    print(f"[robometer_frames_loader] reading pairs_index {pairs_index_path}")
    print(f"[robometer_frames_loader] building cross-family manifest index for frame_labels …")
    manifest_index = _build_manifest_index(base_root)
    print(f"[robometer_frames_loader] manifest index: {len(manifest_index):,} entries")

    task_data: Dict[str, List[Dict[str, Any]]] = {}
    n_loaded = 0
    n_missing_shard = 0
    for row in _iter_manifest(pairs_index):
        episode_id = row["episode_id"]
        frames_path = row.get("frames_path") or ""
        if "::" not in frames_path:
            n_missing_shard += 1
            continue
        shard_rel, episode_dir = frames_path.split("::", 1)
        shard_path = base_root / shard_rel
        if not shard_path.is_file():
            n_missing_shard += 1
            continue

        # frame_labels for failures (and for successes that happen to have curated labels too)
        # come from the manifest. Lookup priority: episode_dir (view-specific id from
        # frames_path, e.g. `failsafe_push_..._front_score_2`) → episode_id (view-agnostic
        # id from pairs_unified, e.g. `failsafe_push_..._score_2`). The metaworld/failsafe
        # manifests are keyed by view-specific ids while pairs_unified strips the view; the
        # episode_dir-first lookup recovers the eval splits' frame_labels coverage from
        # 0% (failsafe) / 54% (metaworld) up to ~100%. robometer/roboreward/droid have no
        # view multiplicity, so episode_id == episode_dir for them and the order doesn't
        # matter.
        manifest_row = manifest_index.get(episode_dir) or manifest_index.get(episode_id) or {}
        frame_labels = manifest_row.get("frame_labels")

        family = row.get("family") or "unknown"
        data_source = data_source_override or f"robometer_frames_{family}"
        traj: Dict[str, Any] = {
            "id": episode_id,
            "task": row.get("task"),
            "frames": TarKeyframeLoader(str(shard_path), episode_dir),
            "is_robot": True,
            "quality_label": "successful" if row.get("label") == "success" else "failure",
            "data_source": data_source,
            "preference_group_id": row.get("partner_episode_id"),
            "preference_rank": None,
            "frame_labels": frame_labels,
            "partial_success": None,
        }
        task_data.setdefault(traj["task"] or "_unnamed_task_", []).append(traj)
        n_loaded += 1

    n_traj = sum(len(v) for v in task_data.values())
    print(f"[robometer_frames_loader] loaded {n_traj:,} trajectories across {len(task_data):,} tasks "
          f"({n_missing_shard} skipped — missing shard / malformed path)")
    return task_data


def _load_episode_ids_filter(filter_path: Optional[str]) -> Optional[set]:
    """Read a JSONL file (one row per line, each containing 'episode_id') into a set.

    Used to restrict the loader to a pre-built train/eval split (see scripts/build_splits.py).
    Returns None when no filter file is configured — the loader emits every episode it finds.
    """
    if not filter_path:
        return None
    p = Path(filter_path)
    if not p.is_file():
        raise FileNotFoundError(f"episode_ids_filter file not found: {filter_path}")
    ids = set()
    with p.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            eid = row.get("episode_id")
            if eid:
                ids.add(eid)
    return ids


def load_robometer_frames_dataset(
    base_path: str,
    family: str,
    include_successes: bool = True,
    include_failures: bool = True,
    max_episodes_per_archive: Optional[int] = None,
    episode_ids_filter: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return {task_name: [trajectory_dict, ...]} for one family.

    Args:
        episode_ids_filter: Optional path to a JSONL file (one {"episode_id": ...} per line)
            produced by scripts/build_splits.py. When set, only episodes whose id is in the
            file are loaded — the standard mechanism for materialising train/eval splits.
    """
    if family not in FAMILY_DIRS:
        raise ValueError(f"unknown family {family!r}; expected one of {list(FAMILY_DIRS)}")

    root = Path(base_path) / FAMILY_DIRS[family]
    manifests_dir = root / "manifests"
    keyframes_dir = root / "keyframes"

    if not manifests_dir.is_dir() or not keyframes_dir.is_dir():
        raise FileNotFoundError(f"missing manifests/ or keyframes/ under {root}")

    allowed_ids = _load_episode_ids_filter(episode_ids_filter)
    if allowed_ids is not None:
        print(f"[robometer_frames_loader] filtering to {len(allowed_ids):,} episode ids "
              f"from {episode_ids_filter}")

    print(f"[robometer_frames_loader] indexing shards under {keyframes_dir} ...")
    shard_index = _index_shards(keyframes_dir)
    print(f"[robometer_frames_loader] indexed {len(shard_index)} episodes")

    task_data: Dict[str, List[Dict[str, Any]]] = {}
    seen_per_archive: Dict[str, int] = {}

    for manifest in sorted(manifests_dir.glob("*.jsonl")):
        is_success_manifest = manifest.stem.endswith("_successes")
        if is_success_manifest and not include_successes:
            continue
        if not is_success_manifest and not include_failures:
            continue

        for row in _iter_manifest(manifest):
            archive = row["archive"]
            episode_id = row["episode_id"]

            # Apply the optional split filter.
            if allowed_ids is not None and episode_id not in allowed_ids:
                continue

            if max_episodes_per_archive is not None:
                n = seen_per_archive.get(archive, 0)
                if n >= max_episodes_per_archive:
                    continue
                seen_per_archive[archive] = n + 1

            shard_path = shard_index.get(episode_id)
            if shard_path is None:
                # episode listed in manifest but missing from shards — skip
                continue

            traj: Dict[str, Any] = {
                "id": episode_id,
                "task": row["task"],
                "frames": TarKeyframeLoader(str(shard_path), episode_id),
                "is_robot": True,
                "quality_label": "successful" if row["label"] == "success" else "failure",
                "data_source": f"robometer_frames_{family}",
                "preference_group_id": row.get("paired_success_id"),
                "preference_rank": None,
                # Dedicated per-frame ordinal labels ({1..4} for failures, None for successes).
                # Kept separate from Robometer's scalar `partial_success` field to avoid type
                # overloading — successes get their target_progress from the default t/T path.
                "frame_labels": row.get("frame_labels"),
                "partial_success": None,
            }
            task_data.setdefault(row["task"], []).append(traj)

    n_traj = sum(len(v) for v in task_data.values())
    print(f"[robometer_frames_loader] loaded {n_traj} trajectories across {len(task_data)} tasks")
    return task_data
