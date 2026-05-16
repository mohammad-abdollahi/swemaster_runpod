# SWE-Master on Trillium (Apptainer, no Docker)

Helper scripts for getting the RL training pipeline running on the Trillium
cluster (Compute Canada / Alliance). The goal of the **smoke test** is to verify
the rllm + verl training loop fires end-to-end on this cluster and produces
per-step metric logs. Once that works we can layer on an Apptainer backend for
the actual SWE-bench environments.

All heavy artifacts (Apptainer SIFs, Python venv, HF cache, run outputs) live
on `/scratch/$USER/swemaster/` — this avoids the cramped 100 GiB `$HOME` quota
and keeps the project tree clean.

## Layout

```
/scratch/$USER/swemaster/
├── sif/         # pytorch_27_cu128.sif (PyTorch 2.7 + CUDA 12.8)
├── cache/       # apptainer cache, pip cache, HF cache, torch cache
├── venv/rllm/   # Python venv with rllm + verl + flash-attn + vllm
├── models/      # optional pre-downloaded model snapshots
├── runs/        # tensorboard event files, checkpoints
├── logs/        # SLURM stdout/stderr
└── tmp/         # apptainer build tmpdir
```

## One-time setup

```bash
# Build the Apptainer SIF via a SLURM job (login-node ulimits + NFS-hardlink
# quirks both block this on the login node).
sbatch scripts/build_sif.sbatch
squeue -u $USER

# Once the SIF is at /scratch/$USER/swemaster/sif/pytorch_27_cu128.sif:

# Build the Python env (rllm + verl + deps) inside the container.
bash scripts/00_enter_container.sh bash scripts/01_build_env.sh

# Download Qwen3-4B into the HF cache on /scratch.
bash scripts/00_enter_container.sh bash scripts/02_download_model.sh

# Register the FrozenLake dataset.
bash scripts/00_enter_container.sh bash scripts/03_prepare_frozenlake.sh
```

## Launch the smoke training

Launch under your own allocation (`salloc`, custom sbatch, etc.) — Trillium has
account-specific submit rules that vary per project, so we don't ship a generic
training sbatch. The actual training command is a regular shell script:

```bash
# Example: interactive allocation, then run inside the container.
salloc --account=def-hemmati-ac --partition=debug \
       --gpus-per-node=h100:4 --cpus-per-task=32 --time=01:00:00

bash scripts/00_enter_container.sh bash scripts/04_train_frozenlake.sh
```

Progress is written to:
- stdout — console scalar lines from verl's `Tracking`
- `/scratch/$USER/swemaster/runs/<project>/<experiment>/` — tensorboard event files

To view tensorboard scalars:

```bash
bash scripts/00_enter_container.sh tensorboard --logdir /scratch/$USER/swemaster/runs --port 6006 --bind_all
```

## Next steps (after smoke is green)

1. Build an Apptainer backend for `R2E-Gym/src/r2egym/agenthub/runtime/docker.py`
   (separate from the `docker` and `kubernetes` branches). Each SWE-bench
   instance ships as a Docker image; Apptainer can run them via
   `docker://...` but every `container.exec_run`, `put_archive`, etc., needs
   an Apptainer equivalent.
2. Convert the parquet RL dataset referenced in the paper's training script
   to a path that resolves on this cluster.
3. Download an SFT-trained model (the RL bootstrap point) from HuggingFace.
