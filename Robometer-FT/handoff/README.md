# Robometer-FT env handoff smoke

Self-contained: only the conda-packed env tarball + these 3 files. No
Robometer code, data, or model weights needed. Verifies that the env
unpacks cleanly on Leonardo's account and that multi-GPU NCCL works.

## What's here

| File           | Purpose                                                      |
| -------------- | ------------------------------------------------------------ |
| `extract.sh`   | One-time tarball extract + `conda-unpack`. Login node only.  |
| `env_smoke.job`| 4-GPU H100 SLURM job that launches the smoke script.         |
| `env_smoke.py` | Imports key libs + does a multi-GPU NCCL gather.             |

## What Leonardo runs

```bash
cd /projects/prjs1958/handoff/env_smoke

# 1. Extract the env (one-time, ~3 min, no GPU)
bash extract.sh

# 2. Submit the smoke (15 min walltime cap, ~5 min real time)
sbatch --account=<YOUR_ACCOUNT> \
       --export=ALL,ENV_PREFIX=/projects/prjs1958/envs/robometer_gpu \
       env_smoke.job
```

Pass signal in the .out file:
```
>>> ENV SMOKE PASSED — gather across 4 ranks: [1.0, 2.0, 3.0, 4.0]
```

Failure signal:
```
>>> ENV SMOKE FAILED — gathered [...], expected [...]
```
or any traceback before the gather check.

## What's actually being verified

1. The packed env extracts and `conda-unpack` rewrites paths cleanly.
2. Python from the new prefix is callable.
3. CUDA is visible from inside Leonardo's SLURM allocation.
4. transformers / accelerate / peft / hydra / omegaconf / datasets all
   import (the libs Robometer's `setup_utils` touches at module-load).
5. `accelerate launch --multi_gpu --num_processes=4` spawns 4 workers.
6. NCCL initializes and gathers tensors across the 4 ranks.

If all 6 pass, the env is ready for the prod Robometer-FT runs — the only
remaining handoff items will be code, indexes, and the HF model cache.
