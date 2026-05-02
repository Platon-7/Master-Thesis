#!/usr/bin/env python3
"""CPU-only smoke test for the ICL collator branch.

Builds two synthetic progress samples — one with a context_trajectory attached (ICL-on) and
one without (ICL-off) — pushes them through RBMBatchCollator._process_progress_batch with the
Qwen-processor step monkey-patched out, and asserts on the emitted batch structure.

Run:
    python Robometer-LoRA/scripts/smoke_test_icl.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the upstream Robometer package importable when the script is run from the repo root.
_HERE = Path(__file__).resolve()
_ROBOMETER_DIR = _HERE.parent.parent.parent / "Robometer"
if str(_ROBOMETER_DIR) not in sys.path:
    sys.path.insert(0, str(_ROBOMETER_DIR))

import numpy as np
import torch

from robometer.data.collators.rbm_heads import RBMBatchCollator
from robometer.data.dataset_types import ProgressSample, Trajectory


def _fake_trajectory(task: str, episode_id: str, num_frames: int = 16) -> Trajectory:
    frames = np.random.randint(0, 255, size=(num_frames, 64, 64, 3), dtype=np.uint8)
    return Trajectory(
        id=episode_id,
        task=task,
        frames=frames,
        frames_shape=frames.shape,
        is_robot=True,
        quality_label="successful",
        data_source="robometer_frames_metaworld",
        target_progress=[i / (num_frames - 1) for i in range(num_frames)],
        success_label=[1.0] * num_frames,
        partial_success=None,
        predict_last_frame_mask=[1.0] * num_frames,
    )


class _TestCollator(RBMBatchCollator):
    """Skip the real VLM processor — we're only validating the ICL branch's effects on the
    per-sample content list and batch metadata."""

    def __init__(self):
        class _Stub:
            pass

        stub = _Stub()
        super().__init__(
            processor=stub,
            tokenizer=stub,
            max_length=4096,
            resized_height=None,
            resized_width=None,
            base_model_id="Qwen/Qwen3-VL-4B-Instruct",
            load_embeddings=False,
            use_multi_image=True,
            prog_pref=False,
            use_per_frame_progress_token=True,
            shuffle_progress_frames=False,
            inference=False,
        )
        self.captured_messages = None

    def _process_conversation(self, conversations, add_generation_prompt: bool = False):
        # Capture for assertions, return a trivial tokenised batch.
        self.captured_messages = conversations
        batch_size = len(conversations)
        dummy_len = 8
        return {
            "input_ids": torch.zeros((batch_size, dummy_len), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, dummy_len), dtype=torch.long),
        }


def main() -> int:
    torch.manual_seed(42)
    np.random.seed(42)
    collator = _TestCollator()

    task = "pick up the red block"
    query_A = _fake_trajectory(task=task, episode_id="query_A", num_frames=16)
    demo_A = _fake_trajectory(task=task, episode_id="demo_for_A", num_frames=16)
    query_B = _fake_trajectory(task=task, episode_id="query_B", num_frames=16)

    sample_icl_on = ProgressSample(
        trajectory=query_A,
        data_gen_strategy="forward_progress",
        context_trajectory=demo_A,
    )
    sample_icl_off = ProgressSample(
        trajectory=query_B,
        data_gen_strategy="forward_progress",
    )

    batch = collator._process_progress_batch([sample_icl_on, sample_icl_off])

    # --- Batch-level assertions ---
    assert "demo_frames_count" in batch, "collator must emit demo_frames_count"
    dfc = batch["demo_frames_count"].tolist()
    assert dfc == [16, 0], f"expected demo_frames_count == [16, 0], got {dfc}"

    tp_shape = tuple(batch["target_progress"].shape)
    assert tp_shape == (2, 16), f"target_progress shape mismatch: {tp_shape}"
    pm_shape = tuple(batch["padding_mask"].shape)
    assert pm_shape == (2, 16), f"padding_mask shape mismatch: {pm_shape}"

    # --- Per-sample content-list assertions ---
    conv_A = collator.captured_messages[0][0]["content"]
    conv_B = collator.captured_messages[1][0]["content"]

    n_images_A = sum(1 for c in conv_A if c.get("type") == "image")
    n_images_B = sum(1 for c in conv_B if c.get("type") == "image")
    assert n_images_A == 32, f"ICL-on should have 16 demo + 16 query images, got {n_images_A}"
    assert n_images_B == 16, f"ICL-off should have 16 query images, got {n_images_B}"

    n_demo_end_A = sum(1 for c in conv_A if c.get("type") == "text" and c.get("text") == "<|demo_end|>")
    n_demo_end_B = sum(1 for c in conv_B if c.get("type") == "text" and c.get("text") == "<|demo_end|>")
    assert n_demo_end_A == 1, f"ICL-on should have one <|demo_end|>, got {n_demo_end_A}"
    assert n_demo_end_B == 0, f"ICL-off should have zero <|demo_end|>, got {n_demo_end_B}"

    n_prog_A = sum(1 for c in conv_A if c.get("type") == "text" and c.get("text") == "<|prog_token|>")
    n_prog_B = sum(1 for c in conv_B if c.get("type") == "text" and c.get("text") == "<|prog_token|>")
    assert n_prog_A == 32, f"ICL-on should have 32 <|prog_token|> (16 demo + 16 query), got {n_prog_A}"
    assert n_prog_B == 16, f"ICL-off should have 16 <|prog_token|>, got {n_prog_B}"

    # --- Demo-before-query ordering on the ICL-on sample ---
    first_image_idx = next(i for i, c in enumerate(conv_A) if c.get("type") == "image")
    demo_end_idx = next(i for i, c in enumerate(conv_A) if c.get("type") == "text" and c.get("text") == "<|demo_end|>")
    n_images_before_sep = sum(1 for c in conv_A[first_image_idx:demo_end_idx] if c.get("type") == "image")
    assert n_images_before_sep == 16, (
        f"ICL-on: demo should be 16 images before <|demo_end|>, got {n_images_before_sep}"
    )

    print("=" * 64)
    print("ICL smoke test: all assertions passed.")
    print(f"  demo_frames_count       = {dfc}")
    print(f"  target_progress.shape   = {tp_shape}")
    print(f"  padding_mask.shape      = {pm_shape}")
    print(f"  ICL-on:  {n_images_A} images, 1 <|demo_end|>, {n_prog_A} <|prog_token|>")
    print(f"  ICL-off: {n_images_B} images, 0 <|demo_end|>, {n_prog_B} <|prog_token|>")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
