"""Stub of the ``unsloth`` package for inference-only use of Robometer-4B.

Robometer's ``robometer.utils.setup_utils`` does a top-level
``from unsloth import FastVisionModel``. The Unsloth real package requires
torch 2.8 + cu128, which conflicts with Chris's vlm_ibrl env (torch 2.4 +
cu121). Inference does not actually need Unsloth's fast paths — Robometer's
``setup_model_and_processor`` has a non-Unsloth branch
(``_load_base_model_standard`` via ``Qwen3VLModel.from_pretrained``) that is
selected when ``cfg.model.use_unsloth = False``.

This module satisfies the import without pulling in Unsloth. If anything
ever does call ``FastVisionModel.*``, we fail loudly so the silent
mis-selection of the Unsloth path is visible.

Lives at repo root; ``set_env.sh`` adds the repo root to PYTHONPATH so
``import unsloth`` resolves here in the demo2reward env. The real Unsloth
package is not installed.
"""


class _StubError(RuntimeError):
    pass


def _refuse(*args, **kwargs):
    raise _StubError(
        "vlm_ibrl/unsloth.py is a stub for import compatibility only. "
        "Set cfg.model.use_unsloth = False before loading Robometer-4B "
        "to avoid hitting Unsloth's code paths."
    )


class FastVisionModel:
    from_pretrained = staticmethod(_refuse)
    get_peft_model = staticmethod(_refuse)


__all__ = ["FastVisionModel"]
