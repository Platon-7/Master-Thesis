#!/usr/bin/env python3
"""Smoke test for Loss 1 (asymmetric CORN, losses.md §Loss 1).

Exercises the CORN branch math directly — bypasses the full trainer — so each component
can be hand-verified. CPU-only.

Run:
    python Robometer-LoRA/scripts/smoke_test_corn_loss.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROBOMETER_DIR = _HERE.parent.parent.parent / "Robometer"
if str(_ROBOMETER_DIR) not in sys.path:
    sys.path.insert(0, str(_ROBOMETER_DIR))

import torch
import torch.nn.functional as F

# Load the full collator package first to avoid the samplers-package circular import, same
# trick as the earlier smoke tests.
from robometer.data.collators.rbm_heads import should_compute_progress  # noqa: F401


# ---------------------------------------------------------------------------
# Verbatim re-implementation of the CORN trainer branch (see
# rbm_heads_trainer.py::_compute_progress_loss_helper for the canonical version).
# ---------------------------------------------------------------------------

def _corn_asymmetric_loss(logits: torch.Tensor, target_decimals: torch.Tensor,
                          c: float, mask: torch.Tensor | None = None):
    """Returns (masked_loss[B,T], per_frame_loss[B,T], y[B,T], y_hat[B,T], correct[B,T])."""
    if logits.dim() != 3 or logits.shape[-1] != 4:
        raise ValueError(f"logits shape must be [B,T,4]; got {logits.shape}")

    y = (target_decimals.float() * 4.0).round().long() + 1
    y = y.clamp(min=1, max=5)                         # [B, T]

    thresholds = torch.arange(2, 6, device=y.device)
    bits = (y.unsqueeze(-1) >= thresholds).float()    # [B, T, 4]

    log_sigma = F.logsigmoid(logits)                  # log σ(z_k)
    log_one_minus_sigma = F.logsigmoid(-logits)       # log(1 - σ(z_k))

    alpha_k = 1.0 + c * torch.arange(0, 4, device=logits.device, dtype=logits.dtype)
    alpha_k = alpha_k.view(1, 1, 4)

    pos_term = bits * log_sigma                       # β=1
    neg_term = alpha_k * (1.0 - bits) * log_one_minus_sigma
    per_threshold_loss = -(pos_term + neg_term)
    per_frame_loss = per_threshold_loss.sum(dim=-1)   # [B, T]

    if mask is None:
        mask = torch.ones_like(per_frame_loss)
    masked_loss = per_frame_loss * mask

    sigma = torch.sigmoid(logits)
    pred_bits = (sigma > 0.5).float()
    y_hat = 1 + pred_bits.sum(dim=-1).long()
    correct = (y_hat == y).float()

    return masked_loss, per_frame_loss, y, y_hat, correct


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ordinal_decoding_from_decimals():
    # round(4*decimal) + 1 → ordinal in {1..5}
    decimals = torch.tensor([[0.0, 0.1, 0.25, 0.40, 0.5, 0.6, 0.75, 0.9, 1.0]])
    expected = torch.tensor([[1, 1, 2, 3, 3, 3, 4, 5, 5]])
    logits = torch.zeros(1, 9, 4)
    _, _, y, _, _ = _corn_asymmetric_loss(logits, decimals, c=1.5)
    assert torch.equal(y, expected), f"ordinal decoding mismatch: got {y}, expected {expected}"


def test_cumulative_bits_encoding():
    # y=1 → (0,0,0,0); y=2 → (1,0,0,0); …; y=5 → (1,1,1,1)
    decimals = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]])  # ordinals 1..5
    logits = torch.zeros(1, 5, 4)
    _, _, y, _, _ = _corn_asymmetric_loss(logits, decimals, c=1.5)
    thresholds = torch.arange(2, 6)
    bits = (y.unsqueeze(-1) >= thresholds).float()[0]
    expected = torch.tensor([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ], dtype=torch.float32)
    assert torch.equal(bits, expected), f"cumulative bits mismatch: got {bits}, expected {expected}"


def test_c_zero_recovers_symmetric_corn():
    # With c=0, alpha_k = 1 for all k → symmetric CORN = sum of 4 per-threshold BCEs with
    # equal weighting. Manually recompute and compare.
    torch.manual_seed(42)
    B, T = 2, 16
    logits = torch.randn(B, T, 4)
    decimals = torch.rand(B, T)
    masked_loss, _, _, _, _ = _corn_asymmetric_loss(logits, decimals, c=0.0)

    # Hand-rolled symmetric reference: BCE on each threshold bit.
    y = (decimals * 4.0).round().long() + 1
    y = y.clamp(1, 5)
    bits = (y.unsqueeze(-1) >= torch.arange(2, 6)).float()
    ref = F.binary_cross_entropy_with_logits(logits, bits, reduction="none").sum(dim=-1)
    assert torch.allclose(masked_loss, ref, atol=1e-5), (
        f"c=0 should equal symmetric sum-of-BCEs: diff {((masked_loss - ref).abs().max()).item()}"
    )


def test_alpha_k_schedule_matches_doc():
    # alpha_k = 1 + c*(k-2) for k in {2,3,4,5}. With c=1.5: (1, 2.5, 4, 5.5).
    # We'll verify indirectly: set logits so σ(z)=0.5 everywhere, y=1 (all bits 0). Then
    # pos_term = 0, neg_term = alpha_k * 1 * log(0.5). Per-frame loss = sum_k alpha_k * log2.
    # For c=1.5: sum alpha_k = 1 + 2.5 + 4 + 5.5 = 13. Loss = 13 * log(2).
    logits = torch.zeros(1, 1, 4)
    decimals = torch.zeros(1, 1)  # y=1 → all bits 0
    _, per_frame_loss, _, _, _ = _corn_asymmetric_loss(logits, decimals, c=1.5)
    expected = 13.0 * math.log(2.0)
    assert abs(per_frame_loss.item() - expected) < 1e-5, (
        f"per-frame loss for all-zero bits, c=1.5 should be 13*log(2) = {expected}; got {per_frame_loss.item()}"
    )


def test_asymmetric_penalises_overprediction_more():
    # Overprediction: y=2 (bits 1,0,0,0), model predicts y=5 (σ(z_k)→1 for all k).
    # Underprediction: y=5 (bits 1,1,1,1), model predicts y=2 (σ(z_2)→1, others→0).
    # Intuition: false-positive "y≥5" is the worst error (weighted α_5=5.5 with c=1.5).
    #            false-negative on "y≥2" matters less (weighted β_2=1).
    #
    # So overpredicting should incur MORE loss than an equally-magnitude underprediction.
    c = 1.5
    big_pos = 5.0    # ≈ σ → 0.993
    big_neg = -5.0   # ≈ σ → 0.007

    # Over-prediction case: y=2 (0.25 decimal), model says y=5
    decimals_over = torch.tensor([[0.25]])
    logits_over = torch.tensor([[[big_pos, big_pos, big_pos, big_pos]]])

    # Under-prediction case: y=5 (1.0 decimal), model says y=2
    decimals_under = torch.tensor([[1.0]])
    logits_under = torch.tensor([[[big_pos, big_neg, big_neg, big_neg]]])

    _, loss_over, _, _, _ = _corn_asymmetric_loss(logits_over, decimals_over, c=c)
    _, loss_under, _, _, _ = _corn_asymmetric_loss(logits_under, decimals_under, c=c)
    assert loss_over.item() > loss_under.item(), (
        f"overprediction must be penalised more: over={loss_over.item():.4f} "
        f"under={loss_under.item():.4f}"
    )
    # Sanity: the gap should be meaningful, not just numerical noise.
    assert (loss_over - loss_under).item() > 1.0, (
        f"expected at least ~1 nat gap between over and under; got "
        f"{(loss_over - loss_under).item():.4f}"
    )


def test_accuracy_decoding():
    # Build logits for a known ordinal target and check decoded prediction matches.
    decimals = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]])
    # σ(z) > 0.5 for the first k bits, ≤ 0.5 for the rest
    # y=1 → 0 bits active → logits all < 0
    # y=3 → 2 bits active → logits[0,1] > 0, logits[2,3] < 0
    logits = torch.tensor([[
        [-5, -5, -5, -5],  # y_hat = 1
        [ 5, -5, -5, -5],  # y_hat = 2
        [ 5,  5, -5, -5],  # y_hat = 3
        [ 5,  5,  5, -5],  # y_hat = 4
        [ 5,  5,  5,  5],  # y_hat = 5
    ]], dtype=torch.float32)
    _, _, y, y_hat, correct = _corn_asymmetric_loss(logits, decimals, c=1.5)
    assert torch.equal(y_hat, torch.tensor([[1, 2, 3, 4, 5]]))
    assert torch.equal(y, y_hat)
    assert torch.all(correct == 1.0)


def test_mask_zeros_masked_frames():
    decimals = torch.rand(2, 4)
    logits = torch.randn(2, 4, 4)
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.float32)
    masked_loss, per_frame_loss, _, _, _ = _corn_asymmetric_loss(logits, decimals, c=1.5, mask=mask)
    assert torch.all(masked_loss[0, 2:] == 0.0)
    assert torch.all(masked_loss[1, :2] == 0.0)
    # Unmasked entries unchanged
    assert torch.allclose(masked_loss[0, :2], per_frame_loss[0, :2])
    assert torch.allclose(masked_loss[1, 2:], per_frame_loss[1, 2:])


def test_expected_ordinal_for_spearman():
    # The trainer uses E[y] = 1 + Σ σ(z_k) as a continuous prediction for Spearman.
    # Sanity: with logits = (5, 5, -5, -5), σ≈(1,1,0,0), E[y]≈3, mapped to decimal ≈ 0.5.
    logits = torch.tensor([[[5.0, 5.0, -5.0, -5.0]]])
    pred_decimal = torch.sigmoid(logits).sum(dim=-1) / 4.0
    assert abs(pred_decimal.item() - 0.5) < 1e-2, (
        f"E[y]-based decimal for bits-on-(2,3) should be ~0.5; got {pred_decimal.item():.4f}"
    )


def main() -> int:
    torch.manual_seed(0)
    test_ordinal_decoding_from_decimals()
    test_cumulative_bits_encoding()
    test_c_zero_recovers_symmetric_corn()
    test_alpha_k_schedule_matches_doc()
    test_asymmetric_penalises_overprediction_more()
    test_accuracy_decoding()
    test_mask_zeros_masked_frames()
    test_expected_ordinal_for_spearman()

    print("=" * 60)
    print("P4 CORN asymmetric-loss smoke test: all checks passed.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
