#!/usr/bin/env python3
"""Smoke test for StratifiedWarmupBatchSampler (P8 / two-phase training).

Verifies:
  * Warmup phase: every batch is drawn ONLY from warmup indices.
  * Main phase: every batch is exactly half failure / half success.
  * Empty-warmup fallback uses the broader failure pool.
  * WarmupStepSyncCallback updates the sampler's current_step.
  * Sampler is deterministic given a fixed seed.

CPU-only, no real dataset required.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROBOMETER_DIR = _HERE.parent.parent.parent / "Robometer"
if str(_ROBOMETER_DIR) not in sys.path:
    sys.path.insert(0, str(_ROBOMETER_DIR))

# Trigger the upstream import graph in a safe order.
from robometer.data.collators.rbm_heads import should_compute_progress  # noqa: F401

from robometer.data.samplers.stratified_warmup import (  # noqa: E402
    StratifiedWarmupBatchSampler,
    WarmupStepSyncCallback,
)


def test_warmup_phase_uses_warmup_pool_only():
    sampler = StratifiedWarmupBatchSampler(
        warmup_indices=list(range(0, 100)),
        failure_indices=list(range(100, 200)),
        success_indices=list(range(200, 300)),
        batch_size=4,
        warmup_steps=5,
        seed=0,
    )
    sampler.current_step = 0
    it = iter(sampler)
    for _ in range(5):
        batch = next(it)
        assert all(0 <= i < 100 for i in batch), f"warmup batch leaked outside warmup pool: {batch}"


def test_main_phase_is_exact_50_50():
    sampler = StratifiedWarmupBatchSampler(
        warmup_indices=list(range(0, 100)),
        failure_indices=list(range(100, 200)),
        success_indices=list(range(200, 300)),
        batch_size=4,
        warmup_steps=0,
        seed=0,
    )
    sampler.current_step = 0  # warmup_steps=0 → immediately in main phase
    it = iter(sampler)
    for _ in range(50):
        batch = next(it)
        n_fail = sum(1 for i in batch if 100 <= i < 200)
        n_succ = sum(1 for i in batch if 200 <= i < 300)
        n_warmup = sum(1 for i in batch if 0 <= i < 100)
        assert n_fail == 2 and n_succ == 2, (
            f"main-phase batch is not 50/50: fail={n_fail} succ={n_succ} warmup={n_warmup}"
        )
        assert n_warmup == 0, f"main-phase batch leaked a warmup index: {batch}"


def test_phase_switches_at_warmup_step():
    sampler = StratifiedWarmupBatchSampler(
        warmup_indices=list(range(0, 100)),
        failure_indices=list(range(100, 200)),
        success_indices=list(range(200, 300)),
        batch_size=4,
        warmup_steps=10,
        seed=0,
    )
    it = iter(sampler)
    # Step through 0..9 (warmup) then 10..14 (main) and check composition each time.
    for step in range(15):
        sampler.current_step = step
        batch = next(it)
        if step < 10:
            assert all(0 <= i < 100 for i in batch), f"step={step}: expected warmup-only batch, got {batch}"
        else:
            n_fail = sum(1 for i in batch if 100 <= i < 200)
            n_succ = sum(1 for i in batch if 200 <= i < 300)
            assert n_fail == 2 and n_succ == 2, f"step={step}: expected 50/50, got fail={n_fail} succ={n_succ}"


def test_empty_warmup_falls_back_to_failure_pool():
    sampler = StratifiedWarmupBatchSampler(
        warmup_indices=[],  # no dedicated warmup pool
        failure_indices=list(range(100, 200)),
        success_indices=list(range(200, 300)),
        batch_size=4,
        warmup_steps=3,
        seed=0,
    )
    sampler.current_step = 0
    it = iter(sampler)
    for _ in range(3):
        batch = next(it)
        assert all(100 <= i < 200 for i in batch), f"fallback should sample from failure pool: {batch}"


def test_callback_syncs_step():
    sampler = StratifiedWarmupBatchSampler(
        warmup_indices=list(range(50)),
        failure_indices=list(range(100, 200)),
        success_indices=list(range(200, 300)),
        batch_size=4,
        warmup_steps=10,
        seed=0,
    )

    class _FakeState:
        def __init__(self, step): self.global_step = step

    cb = WarmupStepSyncCallback(sampler)
    assert sampler.current_step == 0
    cb.on_step_begin(args=None, state=_FakeState(7), control=None)
    assert sampler.current_step == 7
    cb.on_step_begin(args=None, state=_FakeState(99), control=None)
    assert sampler.current_step == 99


def test_seed_determinism():
    s1 = StratifiedWarmupBatchSampler([], list(range(100, 200)), list(range(200, 300)), 4, 0, seed=42)
    s2 = StratifiedWarmupBatchSampler([], list(range(100, 200)), list(range(200, 300)), 4, 0, seed=42)
    s1.current_step = s2.current_step = 0
    it1, it2 = iter(s1), iter(s2)
    for _ in range(10):
        assert next(it1) == next(it2), "same seed should yield identical batch sequence"


def test_in_batch_no_duplicates_when_pool_large_enough():
    sampler = StratifiedWarmupBatchSampler(
        warmup_indices=[],
        failure_indices=list(range(100, 200)),
        success_indices=list(range(200, 300)),
        batch_size=4,
        warmup_steps=0,
        seed=0,
    )
    sampler.current_step = 0
    it = iter(sampler)
    for _ in range(20):
        batch = next(it)
        assert len(set(batch)) == len(batch), f"in-batch duplicates: {batch}"


def main() -> int:
    test_warmup_phase_uses_warmup_pool_only()
    test_main_phase_is_exact_50_50()
    test_phase_switches_at_warmup_step()
    test_empty_warmup_falls_back_to_failure_pool()
    test_callback_syncs_step()
    test_seed_determinism()
    test_in_batch_no_duplicates_when_pool_large_enough()
    print("=" * 60)
    print("P8 stratified-warmup smoke test: all checks passed.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
