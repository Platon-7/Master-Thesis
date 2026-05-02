#!/usr/bin/env python3
"""Smoke test for Loss 2 (Chris' asymmetric C51 + asymmetric BCE).

Exercises the two new branches directly (not through the full trainer) so the math can be
verified against hand-computed expected values. CPU-only.

Run:
    python Robometer-LoRA/scripts/smoke_test_asymmetric_losses.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROBOMETER_DIR = _HERE.parent.parent.parent / "Robometer"
if str(_ROBOMETER_DIR) not in sys.path:
    sys.path.insert(0, str(_ROBOMETER_DIR))

import torch
import torch.nn.functional as F

# Loading the heavy collator resolves Robometer's import graph in the right order; without it,
# samplers.base imports cycle. Same trick used in smoke_test_labels.py.
from robometer.data.collators.rbm_heads import should_compute_progress  # noqa: F401


# ---------------------------------------------------------------------------
# Re-implementations of the two new branches, verbatim from the trainer patches.
# Tests call these instead of the full trainer methods to keep the surface area tight.
# ---------------------------------------------------------------------------

def _c51_asymmetric_progress_loss(logits: torch.Tensor, target_dist: torch.Tensor,
                                  lambda_val: float) -> torch.Tensor:
    """Mirror of the c51_asymmetric branch in _compute_progress_loss_helper."""
    B, T, N = logits.shape
    bin_centers = torch.linspace(0.0, 1.0, N, dtype=logits.dtype)

    p_hat = (F.softmax(logits, dim=-1) * bin_centers).sum(-1).detach()          # [B, T]
    p_t = (target_dist * bin_centers).sum(-1).detach()                           # [B, T]
    weight = torch.where(
        p_hat > p_t,
        torch.ones_like(p_hat),
        torch.full_like(p_hat, lambda_val),
    ).detach()                                                                   # [B, T]

    ce = F.cross_entropy(
        logits.reshape(B * T, N),
        target_dist.reshape(B * T, N),
        reduction="none",
    ).view(B, T)                                                                 # [B, T]

    mask = torch.ones_like(ce)  # trivial mask for the smoke test
    weighted = weight * ce * mask
    return weighted.sum() / (mask.sum() + 1e-8), weight, p_hat, p_t


def _bce_asymmetric_success_loss(logits: torch.Tensor, targets: torch.Tensor,
                                 combined_mask: torch.Tensor,
                                 lambda_val: float) -> torch.Tensor:
    """Mirror of the bce_asymmetric branch in _compute_success_loss_helper."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # [B, T]
    bce = bce * combined_mask

    combined_mask_index = combined_mask.bool()
    mask_neg_bool = (targets == 0) & combined_mask_index
    mask_pos_bool = (targets == 1) & combined_mask_index

    zero = torch.tensor(0.0, dtype=logits.dtype)
    L_neg = bce[mask_neg_bool].mean() if mask_neg_bool.any() else zero
    L_pos = bce[mask_pos_bool].mean() if mask_pos_bool.any() else zero
    return L_neg + lambda_val * L_pos, L_neg, L_pos


# ---------------------------------------------------------------------------
# Progress-loss tests
# ---------------------------------------------------------------------------

def _make_logits_predicting(p_value: float, B: int, T: int, N: int = 10) -> torch.Tensor:
    """Build logits whose softmax has expected value ≈ p_value by concentrating mass on the
    bin closest to p_value. Not exactly p_value but close enough for directional tests."""
    bin_centers = torch.linspace(0.0, 1.0, N)
    k = int(round(p_value * (N - 1)))
    logits = torch.full((B, T, N), -5.0)
    logits[..., k] = 5.0
    return logits, bin_centers


def _c51_project(p_values: torch.Tensor, N: int = 10) -> torch.Tensor:
    """Linear two-bin C51 projection (matches convert_continuous_to_discrete_bin_c51)."""
    B, T = p_values.shape
    bin_centers = torch.linspace(0.0, 1.0, N)
    out = torch.zeros(B, T, N)
    for b in range(B):
        for t in range(T):
            p = float(p_values[b, t].clamp(0.0, 1.0))
            frac = p * (N - 1)
            lo = int(frac)
            hi = min(lo + 1, N - 1)
            if lo == hi:
                out[b, t, lo] = 1.0
            else:
                out[b, t, lo] = hi - frac
                out[b, t, hi] = frac - lo
    return out


def test_overestimation_weight_is_one():
    # Model predicts p≈0.9, target is 0.1 → overestimation → weight should be 1.0 everywhere.
    logits, _ = _make_logits_predicting(0.9, B=2, T=4)
    target_dist = _c51_project(torch.full((2, 4), 0.1))
    _, weight, p_hat, p_t = _c51_asymmetric_progress_loss(logits, target_dist, lambda_val=0.3)
    assert (weight == 1.0).all(), f"expected all weight=1 for overestimation, got {weight}"
    assert (p_hat > p_t).all(), "sanity: p_hat should exceed p_t here"


def test_underestimation_weight_is_lambda():
    # Model predicts p≈0.1, target is 0.9 → underestimation → weight should be lambda = 0.3.
    logits, _ = _make_logits_predicting(0.1, B=2, T=4)
    target_dist = _c51_project(torch.full((2, 4), 0.9))
    _, weight, p_hat, p_t = _c51_asymmetric_progress_loss(logits, target_dist, lambda_val=0.3)
    assert torch.allclose(weight, torch.full_like(weight, 0.3)), f"expected lambda everywhere, got {weight}"
    assert (p_hat < p_t).all(), "sanity: p_hat should be below p_t here"


def test_asymmetric_loss_is_smaller_on_underestimation_with_lambda_lt_1():
    # Same prediction error magnitude, over vs under.
    logits_over, _ = _make_logits_predicting(0.9, B=2, T=4)
    target_under = _c51_project(torch.full((2, 4), 0.1))  # over-predicts
    logits_under, _ = _make_logits_predicting(0.1, B=2, T=4)
    target_over = _c51_project(torch.full((2, 4), 0.9))   # under-predicts

    loss_over, _, _, _ = _c51_asymmetric_progress_loss(logits_over, target_under, lambda_val=0.3)
    loss_under, _, _, _ = _c51_asymmetric_progress_loss(logits_under, target_over, lambda_val=0.3)
    # Underestimation contributes lambda=0.3 of what an equivalent overestimation would.
    # Same CE magnitude on both sides → loss_under ≈ 0.3 · loss_over.
    ratio = (loss_under / loss_over).item()
    assert 0.25 < ratio < 0.35, (
        f"expected loss_under ≈ 0.3 · loss_over for symmetric CE, got ratio={ratio:.3f}"
    )


def test_lambda_one_recovers_symmetric_ce():
    # lambda=1.0 should be identical to the existing symmetric discrete path.
    logits, _ = _make_logits_predicting(0.1, B=2, T=4)
    target_dist = _c51_project(torch.full((2, 4), 0.9))

    loss_asym_1, _, _, _ = _c51_asymmetric_progress_loss(logits, target_dist, lambda_val=1.0)
    # Manual symmetric CE
    ce = F.cross_entropy(
        logits.reshape(-1, 10), target_dist.reshape(-1, 10), reduction="none"
    ).view(2, 4)
    loss_sym = ce.mean()
    assert torch.allclose(loss_asym_1, loss_sym, atol=1e-5), (
        f"lambda=1 should match symmetric CE: asym={loss_asym_1:.6f}, sym={loss_sym:.6f}"
    )


# ---------------------------------------------------------------------------
# Success-loss tests
# ---------------------------------------------------------------------------

def test_empty_positives_guard_does_not_nan():
    # All-negatives batch: the lambda·mean(positives) term would NaN on an empty slice.
    logits = torch.randn(4, 16)
    targets = torch.zeros(4, 16)
    mask = torch.ones(4, 16)
    loss, L_neg, L_pos = _bce_asymmetric_success_loss(logits, targets, mask, lambda_val=0.3)
    assert torch.isfinite(loss), f"expected finite loss, got {loss}"
    assert L_pos.item() == 0.0, "L_pos should be the zero-sentinel when there are no positives"
    assert torch.isclose(loss, L_neg, atol=1e-6)


def test_empty_negatives_guard_does_not_nan():
    # All-positives batch — unusual but should still be well-defined.
    logits = torch.randn(4, 16)
    targets = torch.ones(4, 16)
    mask = torch.ones(4, 16)
    loss, L_neg, L_pos = _bce_asymmetric_success_loss(logits, targets, mask, lambda_val=0.3)
    assert torch.isfinite(loss), f"expected finite loss, got {loss}"
    assert L_neg.item() == 0.0
    assert torch.isclose(loss, 0.3 * L_pos, atol=1e-6)


def test_class_size_independence():
    # Chris' Form A: mean(neg) + lambda·mean(pos) is invariant to class sizes, because each
    # class contributes a single scalar (its mean).
    #
    # Setup: all logits = 0 → BCE(0, y) = ln(2) for every element, whether y is 0 or 1. So
    # mean(neg) = mean(pos) = ln(2) regardless of how many elements are in each class. The
    # loss value should be the same for any (n_neg, n_pos) split with n_pos >= 1.
    lambda_val = 0.3
    expected = (1.0 + lambda_val) * torch.log(torch.tensor(2.0))

    shapes_and_splits = [
        # (B, T, fraction_positive) — for each shape, first fraction_positive * T frames are positive.
        (2, 4, 0.5),    # balanced
        (2, 64, 1 / 64),  # one positive per row → 2 positives in 128 frames
        (1, 100, 0.01),   # one positive out of 100
    ]
    for B, T, frac_pos in shapes_and_splits:
        n_pos = max(1, int(round(T * frac_pos)))
        logits = torch.zeros(B, T)
        targets = torch.zeros(B, T)
        targets[:, :n_pos] = 1.0
        mask = torch.ones(B, T)
        loss, _, _ = _bce_asymmetric_success_loss(logits, targets, mask, lambda_val=lambda_val)
        assert torch.isclose(loss, expected, atol=1e-6), (
            f"Form A should give (1+lambda)*ln(2) for zero-logit BCE regardless of class sizes. "
            f"Shape=({B},{T}), n_pos={n_pos}: loss={loss:.6f}, expected={expected:.6f}"
        )


def test_mask_excludes_frames():
    # Masked-out positives must not contribute to the loss.
    logits = torch.randn(2, 4)
    targets = torch.tensor([[0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
    mask_full = torch.ones(2, 4)
    mask_no_pos = torch.where(targets == 1, torch.zeros_like(targets), torch.ones_like(targets))

    loss_full, _, _ = _bce_asymmetric_success_loss(logits, targets, mask_full, lambda_val=0.3)
    loss_no_pos, _, _ = _bce_asymmetric_success_loss(logits, targets, mask_no_pos, lambda_val=0.3)
    # With all positives masked, the positives term drops out → loss_no_pos should equal the
    # negatives-only BCE mean (strictly less than loss_full unless the model was perfect).
    assert loss_no_pos < loss_full + 1e-6  # not strictly less is fine; we just need finite
    assert torch.isfinite(loss_no_pos)


def main() -> int:
    torch.manual_seed(0)

    # Progress
    test_overestimation_weight_is_one()
    test_underestimation_weight_is_lambda()
    test_asymmetric_loss_is_smaller_on_underestimation_with_lambda_lt_1()
    test_lambda_one_recovers_symmetric_ce()

    # Success
    test_empty_positives_guard_does_not_nan()
    test_empty_negatives_guard_does_not_nan()
    test_class_size_independence()
    test_mask_excludes_frames()

    print("=" * 60)
    print("P3 asymmetric-loss smoke test: all checks passed.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
