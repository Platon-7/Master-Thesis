"""Smoke-test the WeightedRandomSampler failure-oversample tilt without GPU.

What this checks (in seconds, on CPU):
    1) Config field `data.failure_oversample_factor` is wired and loads.
    2) WeightedRandomSampler with the production weights yields the target failure
       batch-fraction in expectation (within statistical tolerance over a large
       draw count).
    3) DDP simulation: two ranks with rank-offset seeds draw DIFFERENT indices
       (so failure tilt × DDP gives the expected batch-size multiplier).

Run:
    /scratch-shared/$USER/envs/robometer_gpu_fa2/bin/python \\
        Robometer-LoRA/scripts/smoke_oversample.py
"""

from __future__ import annotations

import sys

import torch
from torch.utils.data import WeightedRandomSampler


def _make_weights(n_total: int, frac_failures: float, oversample: float) -> tuple[torch.Tensor, list[str]]:
    """Build (weights tensor, label list) for a synthetic dataset of n_total samples
    with frac_failures share marked 'failure', rest 'successful'. Apply oversample
    factor to failure rows (mirrors the trainer's branch logic)."""
    n_fail = int(round(n_total * frac_failures))
    labels = ["failure"] * n_fail + ["successful"] * (n_total - n_fail)
    weights = torch.tensor(
        [oversample if lbl != "successful" else 1.0 for lbl in labels],
        dtype=torch.double,
    )
    return weights, labels


def _draw_failure_fraction(weights, labels, n_draws: int, seed: int) -> float:
    gen = torch.Generator().manual_seed(seed)
    s = WeightedRandomSampler(weights=weights, num_samples=n_draws, replacement=True, generator=gen)
    fails = sum(1 for idx in s if labels[idx] != "successful")
    return fails / n_draws


def _draw_indices(weights, n_draws: int, seed: int) -> list[int]:
    gen = torch.Generator().manual_seed(seed)
    s = WeightedRandomSampler(weights=weights, num_samples=n_draws, replacement=True, generator=gen)
    return list(s)


def main() -> int:
    print("=" * 70)
    print("Failure-oversample smoke test")
    print("=" * 70)

    # --- 1) Config field is wired ---------------------------------------
    print("\n[1/3] Config field wiring")
    try:
        from robometer.configs.experiment_configs import DataConfig
        d = DataConfig()
        assert hasattr(d, "failure_oversample_factor"), "missing field"
        assert d.failure_oversample_factor == 1.0, f"default should be 1.0, got {d.failure_oversample_factor}"
        print(f"    PASS: DataConfig().failure_oversample_factor = {d.failure_oversample_factor} (default)")
    except Exception as e:
        print(f"    FAIL: {type(e).__name__}: {e}")
        return 1

    # --- 2) Expectation check -------------------------------------------
    print("\n[2/3] Failure batch-fraction matches target (expectation)")
    cases = [
        ("Robometer-FT", 0.185, 1.89, 0.300),  # finetune: 18.5% natural, w=1.89 -> 30%
        ("Qwen35-FT",    0.141, 2.61, 0.300),  # pretrain: 14.1% natural, w=2.61 -> 30%
        ("baseline",     0.185, 1.00, 0.185),  # no oversample -> recovers natural rate
    ]
    n_total = 100_000
    n_draws = 200_000
    failures = []
    for name, frac, oversample, target in cases:
        weights, labels = _make_weights(n_total, frac, oversample)
        observed = _draw_failure_fraction(weights, labels, n_draws, seed=42)
        # Tolerance: ±1.0pp covers 4 stdev at n_draws=200k for p~0.3
        err_pp = abs(observed - target) * 100
        status = "PASS" if err_pp < 1.0 else "FAIL"
        print(f"    {status}: {name:<14}  oversample={oversample:.2f}  "
              f"observed={observed:.4f}  target={target:.4f}  err={err_pp:.2f}pp")
        if status == "FAIL":
            failures.append(name)

    # --- 3) DDP-rank de-sync check --------------------------------------
    print("\n[3/3] DDP ranks draw different indices (rank-offset seeds)")
    weights, labels = _make_weights(n_total, 0.185, 1.89)
    rank0 = _draw_indices(weights, 1000, seed=42 + 0)
    rank1 = _draw_indices(weights, 1000, seed=42 + 1)
    overlap = len(set(rank0) & set(rank1)) / 1000
    # With replacement on 100k items, expected ~10% collision; > 50% would be bad
    status = "PASS" if overlap < 0.20 else "FAIL"
    print(f"    {status}: rank0 vs rank1 index-overlap = {overlap*100:.1f}%  "
          f"(expect <20%; would be 100% if seeds collided)")
    if status == "FAIL":
        failures.append("ddp_desync")

    print()
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("ALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
