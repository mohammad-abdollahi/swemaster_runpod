# SWE-Master-4B RL on 8× H100 — run guide

Branch: **`train-8-gpus`**. This is the multi-GPU port of the single-H200
Active-Test-Selection (+ agentic test-refinement) pipeline, tuned close to the
original SWE-Master paper recipe. Everything is parameterized so you can launch
with one command and lose no time on setup.

## Why 8 GPUs (and why the single-GPU run kept crashing)

On one GPU, vLLM (generation) and FSDP (training) time-share the device. At the
rollout→update hand-off, FSDP's parameter gather + backward on a long sequence
collided with vLLM's memory → `CUDA illegal memory access` / `CUBLAS…EngineDeadError`.
It was deterministic on the big-repo `verified_live` data and could not be fully
fixed on a single 80 GB GPU.

On 8× H100, FSDP **FULL_SHARD** splits the 4B model + gradients + optimizer across
all 8 devices (tiny per-GPU), so there's ample headroom at the hand-off. That lets
this config run **full context, no CPU offload, paper-scale batch** — which also
makes it far faster than the single-GPU recipe.

## Files (all at repo root unless noted)

| File | Purpose |
|---|---|
| `preflight_8xh100.sh` | Validate the box (GPUs, torch/vllm, model, data, docker, refine backend) before launching. |
| `run_8xh100.sh` | One-command launcher; selects the reward arm (`REFINE_MODE`). |
| `DeepSWE_RL/rllm/examples/swe/train_active_tests_swe_master_4b_8xh100.sh` | The actual 8-GPU verl launch (paper-scale defaults). |
| `orchestrator_8xh100.sh` | Disconnect-proof wrapper: relaunch-on-death (verl auto-resume), checkpoint prune, auto-export. |
| `export_to_hf.sh` | Merge the final FSDP checkpoint → HF model dir (now world-size-agnostic). |

## Prerequisites on the 8-GPU box

1. **Repo + submodules** checked out (this branch). `DeepSWE_RL/` and `R2E-Gym/`
   present. Set `REPO_DIR` if it's not the repo you run from.
2. **Python env** with torch (CUDA), vllm, flash-attn — the same stack as the
   original box. Activate it, or pass `PYTHON_BIN=/path/to/venv/bin/python`.
3. **Base model** locally, e.g. `SWE-Master-4B-SFT` (or `SWE-Master-4B-Agentic-refine`
   to continue from it). Point at it with `MODEL_PATH=`.
4. **Dataset** parquets (train + val). Point at them with `TRAIN_PARQUET=` /
   `VAL_PARQUET=`. Default expects `data_examples/rl_data/swerebench_verified_live/`.
5. **Docker** daemon up, user in the docker group. Your ~900 prebuilt images mean
   no Docker Hub pulls at train time (set the dataset to reference those image tags).
6. For `REFINE_MODE=agentic`: a logged-in `claude` CLI on the box. No Claude?
   use `REFINE_MODE=none` (baseline ATS-RL) or `REFINE_MODE=online` (`OPENAI_API_KEY`).

## Run it (3 steps)

```bash
cd <repo>
export PYTHON_BIN=/path/to/venv/bin/python      # or activate the venv
export MODEL_PATH=/path/to/SWE-Master-4B-SFT
export TRAIN_PARQUET=/path/to/train_00000_verl.parquet
export VAL_PARQUET=/path/to/test_00000_verl.parquet

# 1) validate the box
REFINE_MODE=agentic bash preflight_8xh100.sh

# 2) launch (disconnect-proof; survives SSH drop). REFINE_MODE: none|agentic|online
EXPERIMENT_NAME=run1 REFINE_MODE=agentic STEPS=150 \
  M_FINAL=$PWD/models/SWE-Master-4B-8xh100-run1 \
  setsid nohup bash orchestrator_8xh100.sh > orch_8xh100.log 2>&1 < /dev/null &

# 3) watch
tail -f train_8xh100.log            # training progress
tail -f orch_8xh100.log             # orchestrator (relaunch/prune/export)
```

On completion the orchestrator merges the final checkpoint to `M_FINAL`.

> Prefer to run without the orchestrator? `EXPERIMENT_NAME=run1 REFINE_MODE=agentic
> TOTAL_TRAINING_STEPS=150 bash run_8xh100.sh` runs in the foreground.

## Key knobs (env vars → sensible 8×H100 defaults)

| Var | Default | Notes |
|---|---|---|
| `N_GPUS` | 8 | GPUs per node. |
| `TP_SIZE` | 1 | vLLM tensor-parallel. 1 ⇒ 8 data-parallel replicas (best throughput for 4B). Try 2 if KV cache is tight. |
| `TRAIN_BATCH_SIZE` / `PPO_MINI_BATCH_SIZE` | 64 / 64 | Paper-scale global batch. Raise (e.g. 128) if GPUs underused. |
| `ROLLOUT_N` | 4 | Samples per prompt (group size). |
| `MAX_RESPONSE` | 24576 | Full context now fits. Can push to 32768 on H100 80 GB. |
| `AGENT_MAX_STEPS` | 150 | Paper value. **The #1 lever for learning — keep it high.** |
| `TRAJECTORY_TIMEOUT` | 5400 | Per-trajectory cap (s). |
| `LR` | 1e-6 | Paper. |
| `GPU_MEM_UTIL` | 0.60 | vLLM KV-cache fraction. Lower if you see vLLM OOM at init; raise for more rollout throughput. |
| `PARAM_OFFLOAD` / `OPTIMIZER_OFFLOAD` | False / False | OFF is safe on 8 GPUs and much faster. |
| `TOTAL_TRAINING_STEPS` / `STEPS` | — / 150 | Fixed PPO-step cap. **Save note:** verl's save check runs for steps `1..(N-1)`, so a run of `STEPS=N` does N−1 updates; set `SAVE_FREQ` to divide a step ≤ N−1 (default `SAVE_FREQ=1` is safe). |
| `TOTAL_EPOCHS` | 40 | Epoch bound if `TOTAL_TRAINING_STEPS` unset. |
| `RLLM_MAX_ROLLOUT_CONCURRENCY` | 32 | Concurrent envs/containers. Scale to your CPU/RAM. |
| `REFINE_MODE` | agentic | `none` = baseline ATS-RL; `agentic` = `claude -p` refine; `online` = OpenAI. |
| `RLLM_ACTIVE_TESTS_REFINE_AFTER` | 6 | Ticks-stalled before a test is refined (only matters when replace is on). |
| `RLLM_ACTIVE_TESTS_REFINE_REPLACE` | **0** | Master switch for the refine-and-replace behavior (`_replace_test`). **Off by default** → the refined test never enters the scored set; reward stays on the ORIGINAL benchmark tests (== baseline). `=1` enables the agentic refiner to replace the stalled scored test with its refined version. |
| `RLLM_ACTIVE_TESTS_REFINE_KEEP_ORIGINAL` | 0 | Only when replace=1: keep the original test scored *alongside* the refined test (additive; reward needs BOTH) instead of dropping it. |
| `RAY_NUM_CPUS` / `RAY_OBJECT_STORE` | `nproc` / 64G | Ray head resources. |
| `MODEL_PATH`,`TRAIN_PARQUET`,`VAL_PARQUET`,`REPO_DIR`,`PYTHON_BIN`,`M_FINAL` | see above | Per-machine paths. |

## Monitoring quick-reference

```bash
# steps completed + latest reward
grep -aoE 'critic/rewards/mean:-?[0-9.]+' train_8xh100.log | tail -10
# checkpoints saved so far
ls -d training_outputs/active_tests_swe_master_8xh100/run1/global_step_*/
# GPU use across all 8
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

Healthy signs: `critic/rewards/mean` present each step and trending up/stable
(not collapsing to ~0 or NaN); `actor/grad_norm` sane (~0.1–0.3); refine
firing (`AGENTIC REFINE replaced`) if in agentic mode.

## Troubleshooting

- **vLLM OOM at init** → lower `GPU_MEM_UTIL` (0.6→0.5) or set `TP_SIZE=2`.
- **Illegal-memory-access at the update on 8 GPUs** (shouldn't happen, but if it does)
  → lower `MAX_RESPONSE` or set `PARAM_OFFLOAD=True OPTIMIZER_OFFLOAD=True`.
- **Init looks frozen** → model shard load + FSDP + vLLM engine has legit multi-minute
  silent gaps; only treat as hung if the log mtime is frozen >30 min AND GPUs are idle.
- **Docker Hub pull errors** → you have the images locally; make sure the dataset's
  `docker_image` tags match your local image names (no pull needed).
- **Relaunch loop** → orchestrator retries up to `MAXRELAUNCH=8` then stops with
  `FATAL: exceeded` in `orch_8xh100.log`.
