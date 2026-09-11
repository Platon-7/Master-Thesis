"""Build a RoboRewardBench raw dataset that points at the ORIGINAL mp4s.

Our first pass went through dataset_upload.generate_hf_dataset, which re-encodes
each clip: 128x128 originals upscaled to 240, capped at 32 frames, H.264, and
then step 2 cut that to 16 and the model finally saw 8. This variant skips the
re-encode entirely so preprocess_datasets decodes the untouched source, letting
us test whether the MAE gap against the published 0.72 is input fidelity.

Same schema as the re-encoded raw, but `frames` holds an absolute path to the
original file and ids are the file stems so rows are traceable.
"""
import json, os
import numpy as np
from datasets import Dataset

SPLIT = os.environ.get("RRB_SPLIT", "test")
SRC = f"/scratch-shared/{os.environ['USER']}/RoboReward/{SPLIT}"
OUT = f"/scratch-shared/{os.environ['USER']}/roborewardbench/native_raw_{SPLIT}/roboreward_{SPLIT}"
DIM = 384

rows = []
with open(f"{SRC}/metadata.jsonl") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        fn, task, rew = e.get("file_name"), e.get("task"), e.get("reward")
        if not fn or not task or rew is None:
            continue
        p = os.path.join(SRC, fn)
        if not os.path.exists(p):
            continue
        ps = (int(rew) - 1) / 4.0
        ds = fn.split("/")[0]
        if ds == "robo_arena":
            ds = "roboarena"
        rows.append({
            "id": os.path.splitext(os.path.basename(fn))[0],
            "task": task,
            "lang_vector": [0.0] * DIM,
            "data_source": f"roboreward_{ds}",
            "frames": p,                      # absolute, original, untouched
            "is_robot": True,
            "quality_label": "successful" if ps == 1.0 else "failure",
            "partial_success": ps,
            "frame_labels": None,
        })

print(f"rows: {len(rows)}")
import collections
print("reward histogram:", collections.Counter(round(r['partial_success']*4)+1 for r in rows))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
Dataset.from_list(rows).save_to_disk(OUT)
print("wrote", OUT)
