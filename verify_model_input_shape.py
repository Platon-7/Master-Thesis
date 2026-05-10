"""Definitive verification: what shape do droid vs robometer frames have
when they actually reach the model?

Loads one droid (240x400) and one robometer (240x240) sample from the v2
HF cache, applies the EXACT collator pipeline (_resize_pil → process_vision_info
→ Qwen3-VL image_processor), and dumps the resulting pixel_values tensor
shape. No simulation — uses the real upstream code path.
"""
from __future__ import annotations

import sys
import numpy as np
from PIL import Image
from datasets import load_from_disk

sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer")
sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis")

from robometer.data.collators.rbm_heads import _resize_pil, MAX_IMAGE_SIDE, MAX_IMAGE_PIXELS
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

CACHE = "/projects/prjs1958/robometer_frames_hf_v2/_projects_prjs1958_robometer_frames_hf_v2_train_raw_robometer_frames_train/processed_dataset"

ds = load_from_disk(CACHE)
hf_by_id = {ds[i]["id"]: i for i in range(len(ds))}


def find_first(source: str):
    for i in range(len(ds)):
        if ds[i]["data_source"] == source:
            return i
    raise LookupError(source)


def npz_frames(row):
    z = np.load(row["frames"])
    return z[z.files[0]]


def trace_one(idx: int, tag: str, processor):
    row = ds[idx]
    frames = npz_frames(row)
    print(f"\n=== {tag} (idx={idx}) ===")
    print(f"  data_source       : {row['data_source']}")
    print(f"  on-disk NPZ shape : {frames.shape}  (first frame {frames.shape[1]}x{frames.shape[2]})")

    # Step 1: convert to PIL list (collator does this via convert_frames_to_pil_images)
    pils_raw = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
    print(f"  PIL (raw)         : {[p.size for p in pils_raw[:3]]}  ...  ({len(pils_raw)} frames)")

    # Step 2: _resize_pil (caps at MAX_IMAGE_SIDE=480, MAX_IMAGE_PIXELS=1MP)
    pils = [_resize_pil(p) for p in pils_raw]
    print(f"  PIL after resize  : {[p.size for p in pils[:3]]}  (caps: side={MAX_IMAGE_SIDE}, pixels={MAX_IMAGE_PIXELS})")

    # Step 3: process_vision_info → image_processor.preprocess
    # Build a minimal multi-image conversation just like the collator does.
    conv = [{
        "role": "user",
        "content": [{"type": "image", "image": p} for p in pils] + [{"type": "text", "text": "describe"}],
    }]
    pvi = process_vision_info(conv, image_patch_size=processor.image_processor.patch_size)
    if len(pvi) == 3:
        image_inputs, video_inputs, video_kwargs = pvi
    else:
        image_inputs, video_inputs = pvi
        video_kwargs = {}
    # image_inputs is a list of PIL images that the processor accepts
    print(f"  process_vision_info returned: {len(image_inputs)} images, sizes={[im.size for im in image_inputs[:3]]}")

    # Step 4: actual processor call — what tensor reaches the model
    inputs = processor(
        text=[processor.apply_chat_template(conv, add_generation_prompt=True)],
        images=image_inputs,
        return_tensors="pt",
    )
    pv = inputs.get("pixel_values")
    print(f"  processor pixel_values shape: {tuple(pv.shape)}")
    if "image_grid_thw" in inputs:
        print(f"  image_grid_thw              : {inputs['image_grid_thw'].tolist()}")


print("[load] Qwen3-VL processor (this is the actual processor LoRA training uses)")
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
print(f"  patch_size={processor.image_processor.patch_size}")
print(f"  merge_size={getattr(processor.image_processor, 'merge_size', '?')}")
print(f"  size config: {processor.image_processor.size}")

trace_one(find_first("robometer_frames_droid"), "droid", processor)
trace_one(find_first("robometer_frames_racer"), "robometer (racer)", processor)
trace_one(find_first("robometer_frames_oxe_bridge_v2"), "robometer (orphan bridge)", processor)
