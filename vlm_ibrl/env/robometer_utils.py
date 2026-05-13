"""Robometer-4B reward-model loader and scorer.

Unlike the generative VLM critics in this directory (Qwen3-VL via
``qwen_utils`` / ``roboreward_utils``), Robometer-4B exposes explicit
progress (C51, 10-bin) and binary success heads, so inference does not
go through ``processor.apply_chat_template`` + ``model.generate``.
Instead we call ``robometer.evals.eval_server.process_batch_helper``
directly with a single-sample batch.

The ``robometer`` package is expected to be importable at runtime; add
the Robometer/ checkout to ``PYTHONPATH``.
"""

from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np
import torch
from PIL import Image


_FrameLike = Union[Image.Image, np.ndarray]


class RobometerScorer:
    """Per-call wrapper around Robometer-4B inference.

    Constructed by :func:`get_robometer_4b`. Call with a sequence of frames
    and a task string to get back ``(progress_reward, success_prob)``,
    both floats in ``[0, 1]``.
    """

    def __init__(self, model_path: str, device: str = "cuda", max_frames: int = None):
        from robometer.utils.save import load_model_from_hf
        from robometer.utils.setup_utils import setup_batch_collator

        cfg, tokenizer, processor, model = load_model_from_hf(model_path=model_path, device=device)
        if model is None:
            raise RuntimeError(f"Failed to load Robometer model from {model_path}")
        model.eval()

        # Robometer's collator assumes use_multi_image=True (matches its own buffer).
        data_cfg = getattr(cfg, "data", None)
        if data_cfg is not None and getattr(data_cfg, "use_multi_image", True) is False:
            data_cfg.use_multi_image = True

        # Resolve discrete-progress mode from the loaded config.
        loss_cfg = getattr(cfg, "loss", None)
        self._is_discrete = getattr(loss_cfg, "progress_loss_type", None) == "discrete"
        self._num_bins = int(getattr(loss_cfg, "progress_discrete_bins", 10) or 10)

        model_cfg = getattr(cfg, "model", None)
        self._model_type = getattr(model_cfg, "model_type", None)
        if self._model_type is None:
            raise ValueError("Robometer config is missing model.model_type")

        if max_frames is None:
            max_frames = int(getattr(data_cfg, "max_frames", 16)) if data_cfg is not None else 16

        self.model_path = model_path
        self.model = model
        self.config = cfg
        self.tokenizer = tokenizer
        self.processor = processor
        self.batch_collator = setup_batch_collator(processor, tokenizer, cfg, is_eval=True)
        self.max_frames = int(max_frames)
        self.device = next(model.parameters()).device

    @staticmethod
    def _frames_to_uint8(frames: Sequence[_FrameLike]) -> np.ndarray:
        """Stack a sequence of PIL.Image / ndarray frames into ``(T, H, W, C)`` uint8."""
        if len(frames) == 0:
            raise ValueError("RobometerScorer received an empty frame sequence")
        arrs: List[np.ndarray] = []
        for f in frames:
            arr = np.asarray(f) if isinstance(f, Image.Image) else f
            if arr.ndim != 3 or arr.shape[-1] != 3:
                raise ValueError(f"Expected (H, W, 3) frame, got shape {arr.shape}")
            arrs.append(arr.astype(np.uint8, copy=False))
        return np.stack(arrs, axis=0)

    def __call__(self, frames: Sequence[_FrameLike], task: str, episode_id: int = 0) -> dict:
        from robometer.evals.eval_server import process_batch_helper
        from robometer.evals.eval_utils import (
            extract_rewards_from_output,
            extract_success_probs_from_output,
            raw_dict_to_sample,
        )

        frames_np = self._frames_to_uint8(frames)
        raw = dict(
            frames=frames_np,
            task=task,
            id=int(episode_id),
            metadata=dict(subsequence_length=int(frames_np.shape[0])),
            video_embeddings=None,
            text_embedding=None,
        )
        sample = raw_dict_to_sample(raw_data=raw, max_frames=self.max_frames, sample_type="progress")

        with torch.inference_mode():
            outputs = process_batch_helper(
                model_type=self._model_type,
                model=self.model,
                tokenizer=self.tokenizer,
                batch_collator=self.batch_collator,
                device=self.device,
                batch_data=[sample.model_dump()],
                job_id=0,
                is_discrete_mode=self._is_discrete,
                num_bins=self._num_bins,
            )

        progress = float(extract_rewards_from_output(outputs)[0])
        success = float(extract_success_probs_from_output(outputs)[0])
        return {"progress_reward": progress, "success_prob": success}

    # Convenience: behave like a torch module for the env's ``self.vlm.eval()`` call.
    def eval(self):
        self.model.eval()
        return self


def get_robometer_4b(
    model_path: str = "robometer/Robometer-4B",
    device: str = "cuda",
    max_frames: int = None,
) -> RobometerScorer:
    """Load Robometer-4B and return a callable scorer.

    Args:
        model_path: HuggingFace repo id or local checkpoint path (supports ``repo@tag``).
        device: torch device string.
        max_frames: frame cap per inference; defaults to the loaded config's
            ``data.max_frames`` (16 in the released checkpoint).
    """
    return RobometerScorer(model_path=model_path, device=device, max_frames=max_frames)
