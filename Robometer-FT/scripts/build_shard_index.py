"""One-shot: scan all tar shards under the source dataset, build the
episode_id → tar_path index, persist as JSON.

Run once before the first training; re-run only when shards are added
or moved on disk.

Usage (login node, ~3 min on 16 cores):
    cd Robometer-FT
    /home/pkarageorgis1/.conda/envs/robometer_gpu/bin/python scripts/build_shard_index.py

Optional env vars:
    FRAME_DATASET_ROOT   default /projects/prjs1958/robometer_frame_dataset
    SHARD_INDEX_PATH     default /scratch-shared/$USER/robometer_shard_index.json
    SHARD_INDEX_VIEW     default keyframes
    SHARD_INDEX_FAMILIES default droid,failsafe,metaworld,robometer
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from robometer_ft_data.tar_index import build_shard_index


def main() -> None:
    frame_dataset_root = os.environ.get(
        "FRAME_DATASET_ROOT", "/projects/prjs1958/robometer_frame_dataset"
    )
    cache_path = os.environ.get(
        "SHARD_INDEX_PATH",
        f"/scratch-shared/{os.environ['USER']}/robometer_shard_index.json",
    )
    view = os.environ.get("SHARD_INDEX_VIEW", "keyframes")
    families = os.environ.get(
        "SHARD_INDEX_FAMILIES", "droid,failsafe,metaworld,robometer"
    ).split(",")

    print(f"frame_dataset_root = {frame_dataset_root}")
    print(f"families           = {families}")
    print(f"view               = {view}")
    print(f"cache_path         = {cache_path}")

    index = build_shard_index(
        frame_dataset_root=frame_dataset_root,
        families=families,
        view=view,
        cache_path=cache_path,
        num_workers=int(os.environ.get("SHARD_INDEX_WORKERS", "16")),
    )
    print(f"\nDONE: indexed {len(index)} episodes -> {cache_path}")


if __name__ == "__main__":
    main()
