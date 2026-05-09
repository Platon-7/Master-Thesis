# MULTI_NODE_CHANGES.md

A running log of every file modified to add multi-node support, plus a one-command
revert path if it fails.

## Status

**STARTED** 2026-05-02. **VALIDATED 2026-05-05.** All 3 tests pass.

| Test | Job | Result |
|---|---|---|
| #1 — single-node regression (`--nodes=1`) | 22479983 | PASS in 5:23 (loss 1.18→0.70) |
| #2 — multi-node smoke (`--nodes=2`, 8 ranks) | 22480067 | PASS in 7:35 (loss 1.15→1.01, no NCCL hang) |
| #3 — multi-node + FSDP save (`--nodes=2`) | 22480327 | PASS in 12:37 (model.safetensors=19GB, optimizer.bin=31GB) |

Single-node path is preserved by branching on `$SLURM_NNODES` — `--nodes=1` runs
the exact same `accelerate launch` command as before, no `srun` wrapper.

Backup tarball of pre-multi-node state (kept for revert):
`/scratch-shared/$USER/qwen35_ft_pre_multinode_backup/snapshot_20260505_122419.tgz`

## Files modified

### 1. `jobs/train_loss2.job`

Header — added `--nodes=1`, replaced `--gres=gpu:h100:4` with `--gpus-per-node=4`:

```diff
-#SBATCH --gres=gpu:h100:4
+#SBATCH --nodes=1
+#SBATCH --ntasks-per-node=1
+#SBATCH --gpus-per-node=4
 #SBATCH --cpus-per-task=16
-#SBATCH --ntasks-per-node=1
 #SBATCH --mem=480G
```

Body — added topology discovery + branching launch (single-node = direct accelerate;
multi-node = srun + per-node accelerate with shared rendezvous):

- Compute `NUM_NODES` from `$SLURM_NNODES`
- Compute `TOTAL_PROCESSES = NUM_NODES * GPUS_PER_NODE`
- Pick `RDZV_PORT` deterministically from `$SLURM_JOB_ID`
- Resolve `HEAD_NODE_IP` via `srun -w "$HEAD_NODE" hostname --ip-address`
- Set `NCCL_ASYNC_ERROR_HANDLING=1`, `OMP_NUM_THREADS=1`
- Branch:
  - `NUM_NODES <= 1` → call `accelerate launch` directly with `--num_machines=1`
  - `NUM_NODES > 1`  → `srun --ntasks-per-node=1` + per-node `accelerate launch` with
    `--machine_rank=$SLURM_NODEID` and shared `main_process_ip` / `main_process_port`

### 2. `jobs/train_loss1.job`

DONE — same diff as `train_loss2.job` applied. Only differences from the L2 file:
job-name (`qwen35_ft_l1`), output filenames (`train_loss1_*`), the second YAML
sed-flatten input (`loss1_corn.yaml` instead of `loss2_c51.yaml`), and the output_dir
suffix (`loss1_`).

### 3. `configs/distributed/fsdp_qwen35.yaml`

NO CHANGES. The YAML's `num_machines: 1` and `num_processes: 1` are overridden by
the `accelerate launch` CLI flags (`--num_machines=$NUM_NODES`, `--num_processes=$TOTAL`).

### 4. `README.md`

DONE — added a "Multi-node training (UNTESTED)" subsection in the "Go" section
that documents the `sbatch --nodes=N` launch pattern, the `per_device_train_batch_size`
adjustment to keep effective global batch comparable, and a revert pointer to
this file.

## Revert path

If the multi-node launch fails badly, restore single-node behavior with:

```bash
cd /gpfs/home3/pkarageorgis1/Master-Thesis/Qwen35-FT
git checkout jobs/train_loss2.job jobs/train_loss1.job  # if these were committed
# OR keep a tarball backup before each edit:
tar czf /tmp/qwen35_pre_multinode.tgz jobs/ configs/ MULTI_NODE_CHANGES.md
```

A tagged backup of the pre-multi-node single-node-working version is at:

    /scratch-shared/$USER/qwen35_ft_dryrun_save/loss2_22422191/

That run (job 22422191, 2026-05-02 18:30) used the single-node code and produced
a successful 2-step + FSDP save. Diff against current `jobs/train_loss2.job` to see
exactly what changed.

## Bugs found and fixed during testing

### 2026-05-05 12:30 — bash arithmetic syntax (test #1, job 22479963)

`RDZV_PORT=$(( 10000 + (SLURM_JOB_ID:-0) % 50000 ))` is invalid: the `:-0`
default-substitution syntax doesn't work bare inside `$(( ))`. Must wrap with
`${...}` braces. Fix:

```diff
-RDZV_PORT=$(( 10000 + (SLURM_JOB_ID:-0) % 50000 ))
+RDZV_PORT=$(( 10000 + ${SLURM_JOB_ID:-0} % 50000 ))
```

Symptom: job died at 13s with stderr `missing `)' (error token is ":-0) % 50000 ")`
followed by `RDZV_PORT: unbound variable` from the next line under `set -u`.
Applied to both `train_loss1.job` and `train_loss2.job`.

## Test plan (in order)

1. **Single-node regression** — submit identical-to-before dryrun, confirm same behavior:

   ```bash
   sbatch --nodes=1 --time=00:30:00 --job-name=qwen35_mn_test1 \
          --export=ALL,EXTRA="++training.max_steps=2 ++training.save_strategy=no \
                              ++training.eval_steps=999999 ++logging.log_to=[]" \
          jobs/train_loss2.job
   ```
   Pass = ≥1 train step logged, no errors.

2. **Multi-node smoke** — same overrides, but on 2 nodes:

   ```bash
   sbatch --nodes=2 --time=00:30:00 --job-name=qwen35_mn_test2 \
          --export=ALL,EXTRA="++training.max_steps=2 ++training.save_strategy=no \
                              ++training.eval_steps=999999 ++logging.log_to=[] \
                              ++training.per_device_train_batch_size=4" \
          jobs/train_loss2.job
   ```
   - With 2 nodes × 4 GPUs = 8 ranks; FSDP shards 4B params across all 8.
   - `per_device_train_batch_size=4` keeps total batch comparable to single-node baseline.
   - Pass = ≥1 train step logged on rank 0, no NCCL hangs, all 8 ranks active.

3. **Multi-node + save** — verify FULL_STATE_DICT save still gathers cleanly:

   ```bash
   sbatch --nodes=2 --time=00:30:00 --job-name=qwen35_mn_test3 \
          --export=ALL,EXTRA="++training.max_steps=2 ++training.save_strategy=steps \
                              ++training.save_steps=2 ++training.eval_steps=999999 \
                              ++logging.log_to=[] \
                              ++training.per_device_train_batch_size=4",WEIGHTS_DIR=/scratch-shared/$USER/qwen35_mn_save \
          jobs/train_loss2.job
   ```
   Pass = `model.safetensors` written, file size ~20 GB (same as single-node).

## Known multi-node risks (not yet tested)

1. **NCCL inter-node bring-up** — if Snellius IB requires specific `NCCL_IB_HCA=` /
   `NCCL_SOCKET_IFNAME=` settings the script doesn't set, all-gather may hang on
   first iter. Symptom: job sits at "Loading weights" or first forward forever.
   Fix: add `NCCL_DEBUG=INFO` via EXTRA env-vars and read the first 100 lines of
   stderr to see which interface NCCL chose.

2. **`scontrol show hostnames` format** — works on most SLURM clusters but not all.
   If `NODE_LIST` ends up empty, `srun -w "$HEAD_NODE"` will fail. Fix: add
   `echo "DEBUG NODE_LIST: ${NODE_LIST[*]}"` before the launch summary.

3. **GPU budget account multi-node permission** — your `gusei17535` account may not
   be authorized for `--nodes>1`. Single-node always worked. If sbatch errors with
   "Access/permission denied" → contact admins.

4. **FSDP FULL_STATE_DICT save across nodes** — gathers all 4B params to rank 0.
   With 8 ranks (2 nodes), the gather is over a slower inter-node link. Save may
   take 2-3× longer than single-node. Should still complete; just slower.

5. **DataLoader workers per rank** — at multi-node, each of 8 ranks spawns
   `cfg.training.dataloader_num_workers` worker processes. If config sets it high,
   total CPU pressure grows. Watch `--mem` per node; we set 480G which has headroom.
