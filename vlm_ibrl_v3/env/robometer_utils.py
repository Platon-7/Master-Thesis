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
        import sys
        import os
        import yaml

        # Disable cuDNN autotune. Some Robometer-family checkpoints (e.g.
        # Robometer-4B baseline and Robometer-FT step-5000) consistently fall
        # into a cuDNN bf16 fast-path that produces ALL-NaN success_logits.
        # Forcing deterministic (slower) kernel selection avoids this pathology.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # Qwen3.5 dispatch lives only in Qwen35-FT/robometer/utils/setup_utils.py
        # (per the "intentional divergence" rule — Robometer/ keeps the Qwen3-VL
        # codepath; Qwen35-FT/ adds Qwen3.5 on top). When the consolidated
        # checkpoint's config.yaml declares a Qwen3.5 base, prepend Qwen35-FT/ to
        # sys.path BEFORE the deferred `from robometer.* import` lines below so
        # the Qwen3.5-aware copy wins. Robometer-4B (Qwen3-VL) falls through
        # unchanged — the Qwen35-FT copy still has the Qwen3-VL branch.
        cfg_yaml = os.path.join(model_path, "config.yaml")
        if os.path.isfile(cfg_yaml):
            try:
                with open(cfg_yaml) as f:
                    bid = (yaml.safe_load(f) or {}).get("model", {}).get("base_model_id", "")
            except Exception:
                bid = ""
            if "qwen3.5" in bid.lower():
                qwen35_repo = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "Qwen35-FT",
                )
                if os.path.isdir(qwen35_repo) and qwen35_repo not in sys.path:
                    sys.path.insert(0, qwen35_repo)

        from robometer.models.rbm import RBM
        from robometer.utils import setup_utils
        from robometer.utils.save import load_model_from_hf

        # transformers >=5.x renamed ``_tied_weights_keys`` to
        # ``all_tied_weights_keys`` on PreTrainedModel and changed its shape
        # from ``list[str]`` to ``dict[str, str]`` (tied_param -> source). The
        # new code calls ``.keys()`` on it. Robometer's RBM still only defines
        # the legacy list. We don't tie weights at inference, so an empty
        # dict satisfies every call site without changing behavior.
        if not hasattr(RBM, "all_tied_weights_keys"):
            RBM.all_tied_weights_keys = {}

        # Two overrides scoped to this load:
        #
        # 1. Robometer's released checkpoint sets ``use_unsloth=True`` in its
        #    config.yaml. The Unsloth runtime requires torch 2.8 + cu128, which
        #    conflicts with this env's torch 2.4 + cu121. Inference does not
        #    actually need Unsloth: ``setup_model_and_processor`` has a non-
        #    Unsloth branch (``_load_base_model_standard`` via Qwen3VLModel)
        #    that we select by forcing ``cfg.model.use_unsloth = False`` before
        #    the model is built. The ``unsloth`` module is still imported at
        #    the top of ``setup_utils`` — satisfied here by the stub at the
        #    vlm_ibrl repo root (see ``vlm_ibrl/unsloth.py``).
        #
        # 2. ``setup_model_and_processor`` probes ``import flash_attn`` and
        #    picks ``attn_implementation='flash_attention_2'`` when it succeeds.
        #    Chris's env has flash-attn 2.7.3, but Robometer's RBM wrapper
        #    rejects FA2 (``RBM does not support Flash Attention 2 yet``). We
        #    hide flash_attn from ``sys.modules`` for the duration of the load
        #    so the probe fails and Robometer's loader falls back to ``sdpa``.
        #    Flash-attn remains available everywhere else in the process.
        _orig_setup = setup_utils.setup_model_and_processor
        _SENTINEL = object()
        _saved_fa = sys.modules.get("flash_attn", _SENTINEL)
        sys.modules["flash_attn"] = None  # makes ``import flash_attn`` raise

        def _setup_patched(model_cfg, *args, **kwargs):
            model_cfg.use_unsloth = False
            return _orig_setup(model_cfg, *args, **kwargs)

        setup_utils.setup_model_and_processor = _setup_patched
        try:
            cfg, tokenizer, processor, model = load_model_from_hf(model_path=model_path, device=device)
        finally:
            setup_utils.setup_model_and_processor = _orig_setup
            if _saved_fa is _SENTINEL:
                sys.modules.pop("flash_attn", None)
            else:
                sys.modules["flash_attn"] = _saved_fa

        from robometer.utils.setup_utils import setup_batch_collator
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

        # Force fp32 if requested (env var ROBOMETER_FORCE_FP32=1). For specific
        # checkpoints (e.g. Robometer-FT step-5000 with ICL) the bf16 kernels
        # consistently produce NaN regardless of warm-up.
        if os.environ.get("ROBOMETER_FORCE_FP32") == "1":
            self.model = self.model.to(torch.float32)
            for m in self.model.modules():
                for p in m.parameters(recurse=False):
                    p.data = p.data.float()
                for b_name, b in m.named_buffers(recurse=False):
                    if b.is_floating_point():
                        try:
                            setattr(m, b_name, b.to(torch.float32))
                        except Exception:
                            pass

        # Warm-up: run dummy forward passes on random frames until we get a
        # non-NaN output (up to a max attempt count). Without this, certain
        # asymmetric-loss-trained checkpoints produce ALL-NaN success_logits
        # on their first few forwards when loaded onto a fresh GPU — we
        # traced the failure mode to a cuDNN/bf16 fast-path that gets
        # initialized incorrectly on cold start. Looping until non-NaN
        # forces stabilization regardless of which kernel was picked.
        import math
        for attempt in range(10):
            n_frames = (8, 16, 32, 4, 24, 16, 8, 12, 20, 28)[attempt]
            if n_frames > self.max_frames * 2:
                continue
            try:
                dummy = [
                    np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    for _ in range(n_frames)
                ]
                out = self(dummy, task="warm up", episode_id=-1)
                sp = out.get("success_prob")
                if sp is not None and not (isinstance(sp, float) and math.isnan(sp)):
                    break  # got a real value — GPU stabilized
            except Exception:
                pass

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

    def __call__(self, frames: Sequence[_FrameLike], task: str, episode_id: int = 0,
                 icl_frames: Sequence[_FrameLike] | None = None, detailed: bool = False) -> dict:
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

        # If ICL frames provided, build a successful-demo context trajectory and
        # attach. The collator (when context_trajectory is not None) inserts
        # `<|demo_end|>` between demo and query — same format used at training time.
        if icl_frames is not None and len(icl_frames) > 0:
            icl_np = self._frames_to_uint8(icl_frames)
            icl_raw = dict(
                frames=icl_np,
                task=task,
                id=int(episode_id) + 1_000_000,
                metadata=dict(subsequence_length=int(icl_np.shape[0])),
                video_embeddings=None,
                text_embedding=None,
            )
            icl_sample = raw_dict_to_sample(raw_data=icl_raw, max_frames=self.max_frames, sample_type="progress")
            sample.context_trajectory = icl_sample.trajectory

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
        if not detailed:
            return {"progress_reward": progress, "success_prob": success}

        # detailed mode: also surface the per-frame arrays the model produced
        # (the extract_* helpers only return the last frame). Used by the
        # unified eval (reward-model-study) for dense-ECE / Kendall / Pearson.
        try:
            sp_per_frame = list(outputs["outputs_success"]["success_probs"][0])
        except (KeyError, IndexError, TypeError):
            sp_per_frame = [success]
        try:
            pg_per_frame = list(outputs["outputs_progress"]["progress_pred"][0])
        except (KeyError, IndexError, TypeError):
            pg_per_frame = [progress]
        return {
            "progress_reward": progress,
            "success_prob": success,
            "success_probs_per_frame": [float(x) for x in sp_per_frame],
            "progress_per_frame": [float(x) for x in pg_per_frame],
        }

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
