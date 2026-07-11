"""Build the 'original data' step2 cache for the loss-vs-data ablation.

Subsets the merged train_no_extras cache to the authors' own data treatment:
drop metaworld and failsafe entirely (our additions), drop droid failure rows
(ours; keep the original OXE droid successes), keep everything else including
the original archives' failures (which upstream code consumes only as
quality_label for preference pairs -- it has no frame_labels handling).

Output layout mirrors upstream's per-dataset cache exactly so unmodified
Robometer-temp code loads it via ROBOMETER_PROCESSED_DATASETS_PATH +
datasets=[robometer_origdata_train].
"""
import json, os, shutil
from datasets import load_from_disk

SRC = "/shared/home/PKA4388/robometer_frames_hf_full_step2/_fsx_PKA4388_robometer_frames_hf_full_train_no_extras_raw_robometer_frames_train_no_extras"
DST_ROOT = "/shared/home/PKA4388/robometer_frames_origdata_step2"
DST = os.path.join(DST_ROOT, "robometer_origdata_train")  # mangled == itself (no slashes)

m = json.load(open(os.path.join(SRC, "index_mappings.json")))
src_idx, q_idx = m["source_indices"], m["quality_indices"]
fail = set()
for k, v in q_idx.items():
    if k != "successful":
        fail.update(v)

drop = set(src_idx.get("metaworld", [])) | set(src_idx.get("failsafe", []))
drop |= set(src_idx.get("droid", [])) & fail
n_total = sum(len(v) for v in src_idx.values())

d = load_from_disk(os.path.join(SRC, "processed_dataset"), keep_in_memory=False)
keep = sorted(set(range(len(d))) - drop)
print(f"rows: {len(d)} total, dropping {len(drop)}, keeping {len(keep)}", flush=True)

old2new = {o: n for n, o in enumerate(keep)}

def remap(x):
    if isinstance(x, list):
        out = [old2new[i] for i in x if i in old2new]
        return out
    if isinstance(x, dict):
        return {k: remap(v) for k, v in x.items()}
    return x

new_m = remap(m)
# prune now-empty groups so upstream's samplers never draw from a dead task/source
for key in ("optimal_by_task", "suboptimal_by_task", "task_indices", "source_indices", "quality_indices"):
    if isinstance(new_m.get(key), dict):
        new_m[key] = {k: v for k, v in new_m[key].items() if v}

os.makedirs(DST, exist_ok=True)
sub = d.select(keep)
sub.save_to_disk(os.path.join(DST, "processed_dataset"))
json.dump(new_m, open(os.path.join(DST, "index_mappings.json"), "w"))
shutil.copy(os.path.join(SRC, "dataset_info.json"), os.path.join(DST, "dataset_info.json"))

# sanity report
srcs = {}
for k, v in new_m["source_indices"].items():
    srcs[k] = len(v)
assert "metaworld" not in srcs and "failsafe" not in srcs
droid_fail = set(new_m["source_indices"].get("droid", [])) & set(
    i for k, v in new_m["quality_indices"].items() if k != "successful" for i in v)
assert not droid_fail, f"droid failures leaked: {len(droid_fail)}"
n_fail = sum(len(v) for k, v in new_m["quality_indices"].items() if k != "successful")
print(f"BUILD OK: {len(sub)} rows -> {DST}", flush=True)
print(f"failures remaining (original archives only): {n_fail}", flush=True)
print("top sources:", dict(sorted(srcs.items(), key=lambda kv: -kv[1])[:8]), flush=True)
