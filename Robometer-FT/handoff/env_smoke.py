"""Robometer-FT env smoke — verifies the conda-packed env works on Leonardo's
account without needing any Robometer code, data, or model weights.

Tests:
  1. Python from the unpacked env is callable.
  2. torch + CUDA visible from inside a SLURM allocation.
  3. The libs Robometer's setup_utils touches at import time all load.
  4. Multi-GPU NCCL gather works across 4 ranks (proves accelerate + NCCL +
     conda-unpack shebang rewrite all line up correctly).

If this prints `>>> ENV SMOKE PASSED` the env is ready for prod runs.
"""
import sys

print(f"python    : {sys.executable}")
print(f"version   : {sys.version.split()[0]}")

import torch
print(f"torch     : {torch.__version__}")
print(f"cuda built: {torch.version.cuda}")
print(f"cuda avail: {torch.cuda.is_available()}")
print(f"devices   : {torch.cuda.device_count()}")

# Heavy lifters Robometer's setup imports unconditionally.
import transformers, accelerate, peft, hydra, omegaconf, datasets
print(f"transformers: {transformers.__version__}")
print(f"accelerate  : {accelerate.__version__}")
print(f"peft        : {peft.__version__}")
print(f"hydra       : {hydra.__version__}")
print(f"datasets    : {datasets.__version__}")

# Multi-GPU NCCL gather smoke.
from accelerate import Accelerator
from accelerate.utils import gather

acc = Accelerator()
print(f"[rank {acc.process_index}/{acc.num_processes}] device={acc.device}")

t = torch.tensor([float(acc.process_index) + 1.0], device=acc.device)
acc.wait_for_everyone()
gathered = gather(t).tolist()
expected = [float(i) + 1.0 for i in range(acc.num_processes)]

if acc.is_main_process:
    if gathered == expected:
        print(f"\n>>> ENV SMOKE PASSED — gather across {acc.num_processes} ranks: {gathered}")
    else:
        print(f"\n>>> ENV SMOKE FAILED — gathered {gathered}, expected {expected}")
        sys.exit(1)
