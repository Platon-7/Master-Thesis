"""Failure-rehearsal KL anchor — buffer + KL math (losses.md §Loss 3).

Anchors the current model's progress predictions on past failure inputs against the
predictions the (older) model produced on those same inputs at the time they were
originally processed. Goal: prevent the model from drifting its failure-mode predictions
during long success-batch streaks.

This module is intentionally pure logic (no trainer / model / hydra dependencies) so it
can be unit-tested without a GPU or a real model. The trainer wires it in via
`FailureKLBuffer.maybe_push` / `FailureKLBuffer.maybe_consume_for_kl`.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# Keys in the progress_inputs dict that we need to re-forward through the model on the
# KL path. Stored on CPU between uses (moved to device on demand). Includes both the
# model-forward inputs AND the per-frame masks needed to compute the KL term correctly.
# Anything else (target_progress, success_labels, data_gen_strategy, etc.) is irrelevant
# to the re-forward and is intentionally NOT buffered to keep memory tight.
_BUFFERED_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "video_grid_thw",
    "second_per_grid_ts",
    "image_grids",
    "image_token_pooling",
    "image_num_crops",
    "video_token_pooling",
    "demo_frames_count",
    # masks needed to align KL frames with originally-valid positions
    "target_progress_mask",
    "predict_last_frame_mask",
)


@dataclass
class FailureKLEntry:
    """One snapshot of (failure-batch input, model's logits on that input) at the time of push.

    All tensors are stored on CPU and detached. The KL re-forward will move `inputs` back
    to the device on demand. `logits` is stored detached so it has no autograd history;
    moving it to device gives a fresh leaf tensor with no grad.
    """

    inputs: Dict[str, torch.Tensor]
    logits: torch.Tensor


class FailureKLBuffer:
    """FIFO deque with the peek-when-below-N retention rule.

    Push: every failure batch.
    Consume on success: peek the head (oldest). Pop-and-discard ONLY if buffer was at
    capacity before the consume; otherwise leave the entry alive so it can anchor
    multiple consecutive success steps when failures are scarce.
    """

    def __init__(self, maxlen: int):
        if maxlen <= 0:
            raise ValueError(f"FailureKLBuffer.maxlen must be > 0, got {maxlen}")
        self._maxlen = maxlen
        # Use a list as the underlying store so we can implement peek-then-conditionally-pop
        # cleanly. deque(maxlen=N) auto-evicts on overflow but doesn't expose "is full" the
        # way we need it for the retention rule.
        self._items: list[FailureKLEntry] = []

    def __len__(self) -> int:
        return len(self._items)

    @property
    def maxlen(self) -> int:
        return self._maxlen

    @property
    def is_full(self) -> bool:
        return len(self._items) >= self._maxlen

    def push(self, entry: FailureKLEntry) -> None:
        """Append to the tail, evict head if at capacity."""
        if len(self._items) >= self._maxlen:
            self._items.pop(0)
        self._items.append(entry)

    def peek_head(self) -> Optional[FailureKLEntry]:
        """Return the oldest entry without removing it. None if empty."""
        if not self._items:
            return None
        return self._items[0]

    def consume_head(self) -> Optional[FailureKLEntry]:
        """Peek the head, then pop iff the buffer was full BEFORE this consume.

        The 'peek-when-below-N' rule: when the buffer hasn't reached steady state, we keep
        the head alive across multiple success steps so the anchor doesn't starve. Once
        the buffer is at capacity, every success step drains one entry; new failure
        pushes refill at the tail (FIFO).
        """
        if not self._items:
            return None
        head = self._items[0]
        if self.is_full:
            self._items.pop(0)
        return head


# ---------------------------------------------------------------------------------------
# Batch-quality detection
# ---------------------------------------------------------------------------------------

def detect_batch_quality(quality_labels: Any) -> Optional[str]:
    """Return 'failure', 'successful', or None for an empty/mixed/missing-quality batch.

    Expects `quality_labels` to be either a list[str], a tuple[str], a 1-D tensor of
    integer codes (rare), or absent. Returns None when quality cannot be determined or
    when the batch mixes qualities (which would violate stratified_batch_balance — log
    a warning and skip the KL push/consume rather than corrupt the buffer).
    """
    if quality_labels is None:
        return None
    if isinstance(quality_labels, (list, tuple)):
        if not quality_labels:
            return None
        first = quality_labels[0]
        if not all(q == first for q in quality_labels):
            logger.warning(
                "FailureKL: batch contains mixed quality_labels %s — skipping KL push/consume "
                "for this step. Did data.stratified_batch_balance=True actually take effect?",
                set(quality_labels),
            )
            return None
        if first in ("failure", "failed", "suboptimal"):
            return "failure"
        if first in ("successful", "success"):
            return "successful"
        return None
    return None


# ---------------------------------------------------------------------------------------
# KL math
# ---------------------------------------------------------------------------------------

def compute_failure_kl(
    logits_old: torch.Tensor,
    logits_new: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Compute KL(P_old || P_new) on per-frame progress logits.

    Dispatches on `logits_*.shape[-1]`:
      * 4   → CORN: 4 independent Bernoulli thresholds. Sum KL across thresholds, then
              average across valid frames.
      * else → categorical (e.g. C51 with 10 bins): standard categorical KL.

    Args:
        logits_old: [..., T, K] from the buffer (assumed already detached).
        logits_new: [..., T, K] from the current model (gradient flows here).
        mask:      Optional [..., T] in {0, 1}, applied frame-wise. If None, all frames
                   are counted. The mask must be broadcastable against the leading dims.
        eps:       Numerical stability for the mask-normalization denominator.

    Returns:
        Scalar KL averaged over valid frames.
    """
    if logits_old.shape != logits_new.shape:
        raise ValueError(
            f"compute_failure_kl: shape mismatch logits_old {tuple(logits_old.shape)} "
            f"vs logits_new {tuple(logits_new.shape)}"
        )
    if logits_old.ndim < 2:
        raise ValueError(
            f"compute_failure_kl: expected at least 2-D logits [..., T, K]; got {logits_old.ndim}-D"
        )

    K = logits_old.shape[-1]
    # Defensive detach in case caller forgot — KL gradient must NOT flow through old.
    logits_old = logits_old.detach()

    if K == 4:
        kl_per_frame = _bernoulli_kl_corn(logits_old, logits_new)  # [..., T]
    else:
        kl_per_frame = _categorical_kl(logits_old, logits_new)     # [..., T]

    if mask is None:
        return kl_per_frame.mean()

    # Broadcast mask to kl_per_frame's shape. Caller is responsible for ensuring the
    # mask aligns with the frame dim.
    mask = mask.to(dtype=kl_per_frame.dtype)
    if mask.shape != kl_per_frame.shape:
        # Try to squeeze trailing 1-dims (common: target_progress_mask is [B, T, 1]).
        while mask.ndim > kl_per_frame.ndim:
            mask = mask.squeeze(-1)
        if mask.shape != kl_per_frame.shape:
            raise ValueError(
                f"compute_failure_kl: mask shape {tuple(mask.shape)} not broadcastable to "
                f"per-frame KL shape {tuple(kl_per_frame.shape)}"
            )
    masked = kl_per_frame * mask
    denom = mask.sum() + eps
    return masked.sum() / denom


def _bernoulli_kl_corn(z_old: torch.Tensor, z_new: torch.Tensor) -> torch.Tensor:
    """KL between two CORN logit fields, summed across the 4 thresholds.

    For each threshold k, P_old = σ(z_old_k) and P_new = σ(z_new_k); the KL of two
    Bernoullis is `p log(p/q) + (1-p) log((1-p)/(1-q))`. Computed in log-space via
    F.logsigmoid for numerical stability across saturation regimes.

    Returns: [..., T] tensor — sum of per-threshold KL across the last (K=4) dim.
    """
    log_p_old = F.logsigmoid(z_old)        # log σ(z_old)
    log_1mp_old = F.logsigmoid(-z_old)     # log(1 - σ(z_old))
    log_p_new = F.logsigmoid(z_new)
    log_1mp_new = F.logsigmoid(-z_new)

    # p_old, (1-p_old) are detached because z_old is.
    p_old = log_p_old.exp()
    one_mp_old = log_1mp_old.exp()

    kl_per_thresh = p_old * (log_p_old - log_p_new) + one_mp_old * (log_1mp_old - log_1mp_new)
    # Sum across the K=4 thresholds → per-frame scalar.
    return kl_per_thresh.sum(dim=-1)


def _categorical_kl(z_old: torch.Tensor, z_new: torch.Tensor) -> torch.Tensor:
    """Categorical KL between two logit fields. Returns [..., T] (sum over the K bins).

    KL(P_old || P_new) = Σ_b p_old_b · (log p_old_b - log p_new_b)
    """
    log_p_old = F.log_softmax(z_old, dim=-1)
    p_old = log_p_old.exp()
    log_p_new = F.log_softmax(z_new, dim=-1)

    kl_per_bin = p_old * (log_p_old - log_p_new)
    return kl_per_bin.sum(dim=-1)


# ---------------------------------------------------------------------------------------
# Buffer-entry construction (CPU-side, detached)
# ---------------------------------------------------------------------------------------

def build_buffer_entry(
    progress_inputs: Dict[str, torch.Tensor],
    progress_logits: torch.Tensor,
) -> FailureKLEntry:
    """Detach + move-to-CPU all relevant tensors so the buffer doesn't keep GPU live or
    autograd graph references.
    """
    cpu_inputs: Dict[str, torch.Tensor] = {}
    for k in _BUFFERED_INPUT_KEYS:
        v = progress_inputs.get(k)
        if v is None:
            continue
        if isinstance(v, torch.Tensor):
            cpu_inputs[k] = v.detach().to("cpu")
        else:
            # Some keys (e.g. demo_frames_count) might already be Python ints in some
            # callers; passthrough.
            cpu_inputs[k] = v

    return FailureKLEntry(
        inputs=cpu_inputs,
        logits=progress_logits.detach().to("cpu"),
    )


def move_entry_to_device(
    entry: FailureKLEntry, device: torch.device, dtype: Optional[torch.dtype] = None
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Move a buffer entry's tensors to `device`. Inputs preserve their stored dtype
    (they're integer ids / float pixels); logits move to `dtype` if provided so they
    match the current model's compute dtype.
    """
    on_device: Dict[str, torch.Tensor] = {}
    for k, v in entry.inputs.items():
        if isinstance(v, torch.Tensor):
            on_device[k] = v.to(device)
        else:
            on_device[k] = v
    logits = entry.logits.to(device)
    if dtype is not None and logits.dtype != dtype:
        logits = logits.to(dtype)
    return on_device, logits


# ---------------------------------------------------------------------------------------
# Concat-batch builder (concat fast path)
# ---------------------------------------------------------------------------------------

# Field-handling rules:
#   * "seq" fields are 2-D [B, S] and need sequence-length padding before concat. The
#     padded positions get attention_mask=0 so the model treats them as no-ops.
#   * "batch_concat" fields are concatenated along dim 0 with no padding (stacked images,
#     per-sample metadata, etc.). Same trailing-dim shape required on both sides.
#   * "skip" fields are not part of the concatenated forward (they're loss-time data,
#     not model-forward data — we keep them per-sample on the side).
_CONCAT_SEQ_FIELDS = ("input_ids", "attention_mask")
_CONCAT_BATCH_FIELDS = (
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "second_per_grid_ts",
    "image_grids",
    "image_token_pooling",
    "image_num_crops",
    "video_grids",
    "video_token_pooling",
    "demo_frames_count",
)


def _pad_seq(t: torch.Tensor, target_len: int, pad_value: int = 0) -> torch.Tensor:
    """Pad a [B, S, ...] tensor on the sequence dim (dim 1) to `target_len`."""
    cur_len = t.shape[1]
    if cur_len >= target_len:
        return t
    pad_shape = (t.shape[0], target_len - cur_len) + tuple(t.shape[2:])
    padding = torch.full(pad_shape, pad_value, dtype=t.dtype, device=t.device)
    return torch.cat([t, padding], dim=1)


def build_concat_batch(
    success_inputs: Dict[str, Any],
    failure_inputs: Dict[str, Any],
    pad_token_id: int = 0,
) -> Tuple[Dict[str, torch.Tensor], int, int]:
    """Concatenate the success batch and a buffered failure batch along the batch dim.

    Returns:
        combined: dict of model-forward kwargs ready for forward_model
        b_succ:   batch size of the success half (== success_inputs["input_ids"].shape[0])
        b_fail:   batch size of the failure half

    Sequence-length fields (input_ids, attention_mask) are padded to the longer of the
    two sides before concat. All other model-forward tensors are concatenated along
    dim 0 with no padding (assumes same trailing shape on both sides — true for our
    setup where every sample has 16 frames at the same resolution).

    Caller splits the model's output back into the two halves at index `b_succ`.
    """
    succ_ids = success_inputs["input_ids"]
    fail_ids = failure_inputs["input_ids"]
    b_succ = int(succ_ids.shape[0])
    b_fail = int(fail_ids.shape[0])
    s_max = int(max(succ_ids.shape[1], fail_ids.shape[1]))

    combined: Dict[str, torch.Tensor] = {}

    # Sequence-padded fields (input_ids, attention_mask)
    for k in _CONCAT_SEQ_FIELDS:
        s_t = success_inputs.get(k)
        f_t = failure_inputs.get(k)
        if s_t is None or f_t is None:
            continue
        # input_ids gets padded with pad_token_id; attention_mask is padded with 0.
        pad_value = pad_token_id if k == "input_ids" else 0
        s_padded = _pad_seq(s_t, s_max, pad_value=pad_value)
        f_padded = _pad_seq(f_t, s_max, pad_value=pad_value)
        combined[k] = torch.cat([s_padded, f_padded], dim=0)

    # Plain dim-0 concat fields
    for k in _CONCAT_BATCH_FIELDS:
        s_v = success_inputs.get(k)
        f_v = failure_inputs.get(k)
        if s_v is None and f_v is None:
            continue
        if s_v is None or f_v is None:
            # Only one side has this field — pass through whichever exists; the model
            # will see a consistent batch as long as it tolerates None for the other side.
            # In practice (Qwen-VL setup) both sides always have the same fields populated.
            combined[k] = s_v if s_v is not None else f_v
            continue
        if not isinstance(s_v, torch.Tensor) or not isinstance(f_v, torch.Tensor):
            combined[k] = s_v  # passthrough non-tensor values from the success side
            continue
        combined[k] = torch.cat([s_v, f_v], dim=0)

    return combined, b_succ, b_fail


def split_concat_logits(
    combined_logits: torch.Tensor, b_succ: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a [B_total, T, K] logits tensor into the success and failure halves at b_succ."""
    return combined_logits[:b_succ], combined_logits[b_succ:]
