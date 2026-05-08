"""Robometer-FT data layer — bypasses HF + MP4 entirely.

Reads JPGs straight from the source tar shards under
/projects/prjs1958/robometer_frame_dataset/{family}/{view}/.../shard-NNNNN.tar.

Public surface (lazy-imported so scripts that only need the tar index can
run without pulling in the full upstream Robometer package):

    from robometer_ft_data.tar_index import build_shard_index, load_shard_index
    from robometer_ft_data.tar_dataset import TarKeyframeIndex, TarKeyframeRBMDataset

We deliberately do NOT re-export from this `__init__.py` — eager imports
here would chain through to `from robometer.data.datasets.rbm_data import
RBMDataset`, which only resolves once `Robometer/` is on sys.path. Letting
callers import the submodules directly keeps `build_shard_index` etc. usable
in any environment.
"""
