# TRL_RL — GRPO training on this cluster without Ray

This directory contains a self-contained TRL-based RL training pipeline. It
exists because the main `DeepSWE_RL/` stack (verl + Ray + vLLM async) cannot
boot on the Trillium **login node** — Ray's `raylet` allocates a thread pool
sized to `hardware_concurrency()` (192 here) and the user has `ulimit -u 1024`
(no root, hard cap). See `scripts/README.md` at the repo root for the full
diagnostic.

TRL uses 🤗 Accelerate (no Ray daemon), so process count stays under the
ulimit. The pipeline here is intentionally minimal — its job is to **produce
per-step RL metric logs** to verify the training infrastructure works on this
cluster, not to reproduce the SWE-Master paper.

When/if you get an interactive `salloc` GPU compute node, switch back to
`DeepSWE_RL/` for the faithful pipeline.

## Layout

```
TRL_RL/
├── README.md
├── configs/
│   └── smoke.yaml           # GRPO hyperparameters for the smoke run
├── src/
│   ├── train_grpo.py        # GRPOTrainer entry point
│   ├── rewards.py           # reward function(s) — synthetic + GSM8K
│   └── data.py              # dataset builders
└── scripts/
    ├── 00_install_trl.sh    # add trl into the existing apptainer venv
    └── 01_train_smoke.sh    # launch the smoke run inside the container
```

The Apptainer container, Python venv, and HF cache are **reused** from the
DeepSWE_RL setup at `/scratch/$USER/swemaster/{sif,venv,cache}/...` — we just
add the `trl` wheel on top.

## Run order

```bash
# 1. Install TRL on top of the existing venv (one-time)
bash scripts/00_enter_container.sh bash TRL_RL/scripts/00_install_trl.sh

# 2. Launch the smoke run (writes tensorboard + console logs)
bash scripts/00_enter_container.sh bash TRL_RL/scripts/01_train_smoke.sh
```

Outputs land at `/scratch/$USER/swemaster/runs/trl-smoke/<experiment>/`.

## Smoke task

A trivial *verifiable* task: the model is given a prompt asking it to emit a
specific token (e.g. `"YES"`), and the reward function returns `1.0` if that
token appears in the completion, else `0.0`. This is enough to drive GRPO
through real rollout → reward → advantage → policy update cycles and have
TRL emit per-step `train/reward`, `train/loss`, `train/kl`, etc. lines.

## What this is NOT

- Not a SWE-bench environment. SWE rollouts require Docker/Apptainer per
  instance, which is still TODO on this cluster.
- Not GRPO-on-Qwen-2.5-Coder-32B-SFT. Different base model, different reward.
- Not a faithful reproduction of the SWE-Master paper's metrics.

For the real pipeline, see `../DeepSWE_RL/` and run from a compute node.
