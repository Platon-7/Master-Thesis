"""One-shot mapping from episode_id → tar shard path.

The source dataset has tars laid out heterogeneously:
    droid:      droid/keyframes/shards/shard-NNNNN.tar
    failsafe:   failsafe/keyframes_success/<task>/shard-NNNNN.tar
    metaworld:  metaworld/keyframes_success/<task>/shard-NNNNN.tar
    robometer:  robometer/keyframes/<source>/shard-NNNNN.tar
    roboreward: roboreward/keyframes/<source>/shard-NNNNN.tar

Each tar contains members shaped:
    <episode_id>__<task_words>/frame_NN_<seconds>s.jpg

We walk every tar once, record episode_id → tar_path, and persist as JSON.
Subsequent runs `load_shard_index(...)` and skip the walk.

The walk is parallelised across CPU cores with multiprocessing.

Cost: ~3 minutes on 16 cores for the full dataset (~2.6k tars).
Cache size: ~80 MB JSON for ~600k episodes.
"""
from __future__ import annotations

import json
import os
import re
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from typing import Dict, Iterable, List, Optional, Tuple

# Member-name pattern produced by the upstream JPG packer.
#   "<episode_id>__<task_words>/frame_NN_<seconds>s.jpg"
_MEMBER_RE = re.compile(r"^(?P<eid>[^/]+?)__(?P<task>[^/]+)/frame_\d+_[\d.]+s\.jpg$")


def _episode_ids_in_tar(tar_path: str) -> List[Tuple[str, str]]:
    """Return (episode_id, member_prefix) tuples found in a single tar.

    member_prefix is the directory inside the tar where this episode's JPGs
    live (everything before "/frame_NN_*.jpg"); we save it to skip a second
    listing at load time.
    """
    out: List[Tuple[str, str]] = []
    seen_eids: set = set()
    try:
        with tarfile.open(tar_path, "r") as tf:
            for m in tf:
                match = _MEMBER_RE.match(m.name)
                if match is None:
                    continue
                eid = match.group("eid")
                if eid in seen_eids:
                    continue
                seen_eids.add(eid)
                # member_prefix = "<eid>__<task>"
                member_prefix = m.name.split("/", 1)[0]
                out.append((eid, member_prefix))
    except Exception as e:
        # Keep going through other tars; report at end.
        print(f"[warn] failed to read {tar_path}: {e}")
    return out


def _scan_one(tar_path: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Worker that returns (tar_path, [(episode_id, member_prefix), ...])."""
    return tar_path, _episode_ids_in_tar(tar_path)


def _enumerate_tars(frame_dataset_root: str, families: List[str], views: List[str]) -> List[str]:
    """Find all tar shards for the requested families and views.

    `views` is a list of dir names like ["keyframes", "keyframes_success",
    "keyframes_orphan_success"]. Each family has its own subset:
       droid:      keyframes/ (failures) + keyframes_success/
       failsafe:   keyframes/ (failures) + keyframes_success/
       metaworld:  keyframes/ (failures) + keyframes_success/
       robometer:  keyframes/ (failures) + keyframes_success/ + keyframes_orphan_success/

    Layout INSIDE a view dir is heterogeneous:
       droid:      <view>/shards/shard-*.tar       (single flat shard pool)
       others:     <view>/<source>/shard-*.tar     (sharded per source/task)
    """
    tars: List[str] = []
    for fam in families:
        for view in views:
            fam_root = os.path.join(frame_dataset_root, fam, view)
            if not os.path.isdir(fam_root):
                continue  # not all view×family combinations exist
            # Both layouts:
            tars.extend(sorted(glob(os.path.join(fam_root, "shards", "*.tar"))))
            tars.extend(sorted(glob(os.path.join(fam_root, "*", "shards", "*.tar"))))
            tars.extend(sorted(glob(os.path.join(fam_root, "*", "*.tar"))))
    return sorted(set(tars))


def build_shard_index(
    frame_dataset_root: str = "/projects/prjs1958/robometer_frame_dataset",
    families: Optional[List[str]] = None,
    view: str = "keyframes",
    cache_path: Optional[str] = None,
    num_workers: int = 16,
) -> Dict[str, Dict[str, str]]:
    """Build episode_id → {tar_path, member_prefix, family} index.

    Args:
        frame_dataset_root: top-level dir that contains family subdirs.
        families: which families to scan. Default = all but roboreward (matches
            the LoRA bake-off composition). Pass an explicit list to override.
        view: which view to index. "keyframes" is the default everyone trains on.
        cache_path: where to persist the JSON. If None, returns without saving.
        num_workers: parallelism for tar listing.

    Returns:
        A dict keyed by episode_id with entries:
            {"tar_path": str, "member_prefix": str, "family": str}
    """
    if families is None:
        # Default: same composition as the LoRA bake-off (no roboreward).
        families = ["droid", "failsafe", "metaworld", "robometer"]

    print(f"[shard-index] scanning families={families} view={view}")
    tars = _enumerate_tars(frame_dataset_root, families, view)
    print(f"[shard-index] found {len(tars)} tar shards to scan")

    # Track which family each tar belongs to for the index entries.
    tar_to_family: Dict[str, str] = {}
    for t in tars:
        rel = os.path.relpath(t, frame_dataset_root)
        tar_to_family[t] = rel.split(os.sep, 1)[0]

    index: Dict[str, Dict[str, str]] = {}
    n_dup_collisions = 0

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(_scan_one, t): t for t in tars}
        for done_count, fut in enumerate(as_completed(futures), 1):
            tar_path, eid_list = fut.result()
            family = tar_to_family[tar_path]
            for eid, member_prefix in eid_list:
                if eid in index:
                    n_dup_collisions += 1
                    # Keep the first occurrence; warn if family conflict.
                    if index[eid]["family"] != family:
                        print(
                            f"[warn] episode_id collision across families: "
                            f"{eid} in {index[eid]['family']} and {family}"
                        )
                    continue
                index[eid] = {
                    "tar_path": tar_path,
                    "member_prefix": member_prefix,
                    "family": family,
                }
            if done_count % 200 == 0:
                print(f"[shard-index] scanned {done_count}/{len(tars)} tars; "
                      f"{len(index)} unique episodes so far")

    print(
        f"[shard-index] done: {len(index)} unique episodes from {len(tars)} tars; "
        f"{n_dup_collisions} duplicate-id collisions skipped"
    )

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(index, f)
        print(f"[shard-index] cached to {cache_path}")

    return index


def load_shard_index(cache_path: str) -> Dict[str, Dict[str, str]]:
    """Load a previously built index. Raises if not found."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"shard index not found at {cache_path}. "
            f"Run scripts/build_shard_index.py to create it."
        )
    with open(cache_path) as f:
        return json.load(f)
