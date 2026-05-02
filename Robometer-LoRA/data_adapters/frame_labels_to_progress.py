"""Convert the per-frame curated failure labels from `robometer_frame_dataset` into the
`target_progress` format Robometer's training loop expects (List[float] in [0, 1]).

Label scheme (5 ordinal levels, but failures only reach up to 4):
  1 = 0%   (pre-task / no progress)
  2 = 25%  (early-stage progress)
  3 = 50%  (mid-stage progress)
  4 = 75%  (near-terminal but unsuccessful)
  5 = 100% (terminal success — only present on success trajectories; never in failure labels)

Two downstream regimes share the same ingestion but differ in whether smoothing is applied:

  - Ordinal CORN / discrete head (Loss 1): use rubric decimals AS-IS. Robometer's
    `convert_continuous_to_discrete_bins(values, num_bins=5)` buckets them to {0..4} for the
    discrete cross-entropy / CORN loss.
  - Asymmetric C51 (Loss 2): successes flow as smooth `t/T` floats, so failures (which would
    otherwise be stair-step) get a piecewise-linear ramp between label transitions so the
    target signal has similar visual character on both sides.

Selection is driven by the config key `loss.failure_label_smoothing` ∈ {"none", "linear"}.
"""

from __future__ import annotations

from typing import List, Sequence

# Rubric: raw ordinal label → decimal progress in [0, 1].
# Keys are ints in {1,2,3,4,5}; failures only emit 1..4, successes get 5 when terminal.
_RUBRIC: dict[int, float] = {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0}


def frame_labels_to_decimals(labels: Sequence[int]) -> List[float]:
    """Map each per-frame ordinal label to its rubric decimal.

    Raises ValueError if any label is outside {1..5}.
    """
    out: List[float] = []
    for v in labels:
        iv = int(v)
        if iv not in _RUBRIC:
            raise ValueError(f"unexpected frame label {v!r}; expected int in {{1,2,3,4,5}}")
        out.append(_RUBRIC[iv])
    return out


def piecewise_linear_ramp(decimals: Sequence[float]) -> List[float]:
    """Option A smoothing: linearly interpolate between label-transition frames.

    Treats the first and last frames as anchors and every "label transition" frame in between
    as an intermediate anchor, then fills the gaps with a linear ramp. Preserves the anchor
    values exactly; only fills the flat plateaus between transitions with a gradient.

    Example (T=16):
        in : [0, 0, 0, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
        out: [0, 0.083, 0.167, 0.25, 0.31, 0.38, 0.44, 0.5, 0.625, 0.75, 0.75, ...]

    Constant sequences (failure that never progressed, e.g. all 0.0) are preserved as-is.
    """
    seq = [float(v) for v in decimals]
    n = len(seq)
    if n <= 1:
        return list(seq)

    # Anchor indices: every position where the value changes relative to its predecessor,
    # plus the first and last frames so the ramp is fully determined at both ends.
    anchors: List[int] = [0]
    for i in range(1, n):
        if seq[i] != seq[i - 1]:
            anchors.append(i)
    if anchors[-1] != n - 1:
        anchors.append(n - 1)

    out: List[float] = [0.0] * n
    # For each consecutive anchor pair, linearly interpolate the values in between.
    for a_prev, a_next in zip(anchors[:-1], anchors[1:]):
        v_prev = seq[a_prev]
        v_next = seq[a_next]
        span = a_next - a_prev
        out[a_prev] = v_prev
        out[a_next] = v_next
        if span <= 1:
            continue
        for j in range(1, span):
            t = j / span
            out[a_prev + j] = v_prev + t * (v_next - v_prev)
    return out


def apply_smoothing(decimals: Sequence[float], mode: str) -> List[float]:
    """Dispatch entry point used by the sampler.

    mode ∈ {"none", "linear"}. Unknown modes raise.
    """
    if mode == "none":
        return [float(v) for v in decimals]
    if mode == "linear":
        return piecewise_linear_ramp(decimals)
    raise ValueError(
        f"unknown failure_label_smoothing mode {mode!r}; expected one of {{'none', 'linear'}}"
    )
