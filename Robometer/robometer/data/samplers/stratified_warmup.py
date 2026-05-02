"""Stratified two-phase BatchSampler for the Robometer-LoRA bake-off.

Two phases controlled by the trainer's global step:

  Phase 1 (warmup, step < warmup_steps)
      Each batch is drawn ONLY from failure samples whose data_source contains the
      "warmup" tag (or any failure if the warmup pool is empty). Pre-exposes the model
      to failure patterns before balanced training begins.

  Phase 2 (main, step >= warmup_steps)
      Each batch is built as exactly half failure / half success — strict per-batch
      balance, not balance-in-expectation.

The current step is kept in sync via WarmupStepSyncCallback, which fires at the start
of every training step in the main process (where the BatchSampler also lives, so the
mutation is visible without IPC gymnastics).
"""

from __future__ import annotations

import random
from typing import Iterable, List, Sequence

from torch.utils.data import Sampler
from transformers.trainer_callback import TrainerCallback

from robometer.utils.logger import get_logger

logger = get_logger()


class StratifiedWarmupBatchSampler(Sampler[List[int]]):
    """BatchSampler with infinite cycling — HF Trainer stops at max_steps.

    Yields a fresh `batch_size`-list of dataset indices on every `next()` call. The composition
    flips at `warmup_steps`:

    * before: random sample of `batch_size` from `warmup_indices` (with replacement; the warmup
      pool is small by design so cycling is fine).
    * after: exactly `batch_size // 2` failures + the remainder successes, sampled without
      replacement within the batch and shuffled to avoid positional patterns.
    """

    def __init__(
        self,
        warmup_indices: Sequence[int],
        failure_indices: Sequence[int],
        success_indices: Sequence[int],
        batch_size: int,
        warmup_steps: int = 0,
        seed: int = 42,
    ) -> None:
        # Note: skip Sampler.__init__ — its data_source kwarg signature has churned across
        # torch versions (deprecated in 2.x, removed in some patches). DataLoader is duck-typed
        # on __iter__/__len__, so the inheritance is purely for instanceof checks downstream.
        if batch_size < 2:
            raise ValueError(f"stratified batch sampler requires batch_size >= 2; got {batch_size}")
        if not failure_indices:
            raise ValueError("stratified batch sampler requires at least one failure index")
        if not success_indices:
            raise ValueError("stratified batch sampler requires at least one success index")
        self.warmup_indices: List[int] = list(warmup_indices)
        self.failure_indices: List[int] = list(failure_indices)
        self.success_indices: List[int] = list(success_indices)
        self.batch_size = int(batch_size)
        self.warmup_steps = int(warmup_steps)
        self.current_step: int = 0
        self._rng = random.Random(seed)
        logger.info(
            f"[StratifiedWarmupBatchSampler] init: warmup_steps={self.warmup_steps}, "
            f"batch_size={self.batch_size}, "
            f"|warmup|={len(self.warmup_indices)}, "
            f"|fail|={len(self.failure_indices)}, "
            f"|succ|={len(self.success_indices)}"
        )

    def _warmup_pool(self) -> List[int]:
        # Fall back to the broader failure pool if no warmup-tagged samples exist.
        return self.warmup_indices if self.warmup_indices else self.failure_indices

    def __iter__(self) -> Iterable[List[int]]:
        # Infinite generator. HF Trainer's loop is bounded by max_steps anyway.
        while True:
            if self.current_step < self.warmup_steps:
                pool = self._warmup_pool()
                # Sample with replacement: warmup pool is intentionally small (~1500 episodes).
                batch = [self._rng.choice(pool) for _ in range(self.batch_size)]
            else:
                n_fail = self.batch_size // 2
                n_succ = self.batch_size - n_fail
                # Sample without replacement WITHIN a batch — HF Trainer does many batches per
                # epoch so cross-batch repetition is fine but in-batch dupes waste compute.
                fails = (
                    self._rng.sample(self.failure_indices, n_fail)
                    if n_fail <= len(self.failure_indices)
                    else [self._rng.choice(self.failure_indices) for _ in range(n_fail)]
                )
                succs = (
                    self._rng.sample(self.success_indices, n_succ)
                    if n_succ <= len(self.success_indices)
                    else [self._rng.choice(self.success_indices) for _ in range(n_succ)]
                )
                batch = fails + succs
                self._rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        # HF Trainer uses this to size epoch progress bars. Return a large finite value;
        # max_steps stops training before we exhaust it.
        return 10**9


class WarmupStepSyncCallback(TrainerCallback):
    """Trainer callback that pushes `state.global_step` into a sampler's `current_step`
    attribute on every step. The sampler lives in the main process (the DataLoader feeds
    indices to workers from there), so a plain attribute assignment is enough — no IPC."""

    def __init__(self, sampler: StratifiedWarmupBatchSampler) -> None:
        self.sampler = sampler

    def on_step_begin(self, args, state, control, **kwargs):
        self.sampler.current_step = int(state.global_step)
