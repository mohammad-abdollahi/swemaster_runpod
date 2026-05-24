# Active-Test-Selection RL — Pipeline Handoff

**Audience:** a future Claude Code session continuing this work.
**Branch:** `reward/active-test-selection` (off `setup/runpod-h200`), pushed to
`github.com:mohammad-abdollahi/swemaster_runpod`.
**Last updated:** 2026-05-24.

Read this first, then `REWARD_ACTIVE_TEST_SELECTION.md` (design), then
`SESSION_LOG_2026-05-20.md` (earlier session narrative). This file supersedes
both for *operational* detail.

---

## 0. One-paragraph summary

We replaced the SWE-Master RL pipeline's **binary terminal reward**
(`int(all FAIL_TO_PASS pass)`) with an **Active-Test-Selection (ATS)** reward:
the agent sees a bounded subset of the target tests in its prompt, and the
reward is continuous partial credit over `FAIL_TO_PASS`, minus a
`PASS_TO_PASS` regression penalty, minus a small (capped) token-usage penalty.
It's a drop-in reward swap — gated by `RLLM_ACTIVE_TESTS_ENABLED=1`, set
`=0` to get the original binary recipe back from the same code. The method is
fully implemented, unit-tested (23 tests), and validated end-to-end on
SWE-Master-4B-SFT: a paper-matched run (max_steps=150) reached 0.250 held-out
solve rate with healthy, stable gradients.

---

## 1. The reward — where it lives and what it computes

### Core module (pure Python, unit-tested)
`R2E-Gym/src/r2egym/agenthub/runtime/active_tests.py`
- `SchedulerConfig` — dataclass of every knob (see §3).
- `ActiveTestScheduler` — the state machine + reward.
  - 4 buckets over `FAIL_TO_PASS`: ACTIVE (shown in prompt), POOL (hidden),
    COOLDOWN (recently solved, masked, re-verified on expiry), RETIRED.
  - `tick(f2p_pass_map, p2p_pass_map=None)` → `SchedulerEvent` (dense per-tick
    reward; only used in v2 mid-episode mode).
  - `terminal_reward(full_pass_map, tokens_used, p2p_pass_map)` →
    `(reward, breakdown)` — the **main** reward path (v1).
  - `prompt_block()` → the `<active_tests>` string injected into the prompt.
- Tests: `R2E-Gym/src/r2egym/agenthub/runtime/test_active_tests.py` (23, all
  pass: `python -m pytest <that file> -v`).

### Reward formula (v1.1, current)
```
r_final =  w_solve   · (|FAIL_TO_PASS solved| / |FAIL_TO_PASS|)        # primary, test-based
         − w_regress · (|PASS_TO_PASS broken| / |PASS_TO_PASS|)        # regression guard
         − lam_token · min(tokens_used / token_ref, token_penalty_cap) # capped efficiency nudge
```
- Defaults: `w_solve=10` (runtime override; SchedulerConfig default is 5),
  `w_regress = w_solve/2`, `lam_token=0.1`, `token_ref=32000`,
  `token_penalty_cap=3.0`.
- The trainer halves the reward for trajectories ending in TRUNCATION /
  MAX_STEPS / TRAJ_TIMEOUT (engine `reward_half` logic), so a full solve that
  hits max_steps shows as ~+5 not +10.

### Docker runtime hooks
`R2E-Gym/src/r2egym/agenthub/runtime/docker.py`
- `get_fail_to_pass_ids()` / `get_pass_to_pass_ids()` — the test universes.
- `get_f2p_and_p2p_status(timeout)` — runs `/run_tests.sh` ONCE, returns
  `(f2p_pass_map, p2p_pass_map)`. Used by both the env's terminal reward and
  mid-episode ticks. (`get_fail_to_pass_status` is the older F2P-only variant.)

### Env wiring
`DeepSWE_RL/rllm/rllm/environments/swe/swe_arsenal.py` (env name `swe_arsenal`,
registered in `rllm/trainer/env_agent_mappings.py`):
- `_scheduler_config_from_env()` — builds `SchedulerConfig` from
  `RLLM_ACTIVE_TESTS_*` env vars.
- `SWEEnv.reset()` — builds the scheduler from the task's F2P/P2P, appends the
  `<active_tests>` block + a "don't break the N regression tests" line to the
  task instruction.
- `SWEEnv.compute_final_reward()` — runs `get_f2p_and_p2p_status` once, calls
  `scheduler.terminal_reward(...)`, prints the `!!!!!!! ATS terminal: ...` line
  (this is what all the analysis tools parse).
- `SWEEnv.step()` — per-step env reward stays 0 in v1; under
  `RLLM_ACTIVE_TESTS_MID_EPISODE=1` it runs pytest every `k_steps` and ticks
  the scheduler (v2, expensive).

The PPO engine already propagates `compute_final_reward()` into the last
step's reward and sums to the trajectory reward
(`rllm/engine/agent_execution_engine.py` ~line 527), so no trainer change was
needed.

---

## 2. Two operating modes

- **v1 (default):** prompt-injection + partial-credit terminal reward. Cost ==
  binary baseline (one `/run_tests.sh` at episode end). This is what all the
  real runs used.
- **v2 (`RLLM_ACTIVE_TESTS_MID_EPISODE=1`):** also runs pytest every `k_steps`
  mid-episode and emits dense rewards (alpha solves, beta stall-expand, delta
  cooldown-regression, gamma step, eta·1/(n+1)) + dense P2P break/recover. Adds
  ~1 `/run_tests.sh` per k_steps per trajectory — expensive; for ablations.

---

## 3. Env-var knobs (read by `_scheduler_config_from_env`)

| Var | Default | Meaning |
|---|---|---|
| `RLLM_ACTIVE_TESTS_ENABLED` | 0 (scripts set 1) | master switch (0 = binary baseline) |
| `RLLM_ACTIVE_TESTS_MID_EPISODE` | 0 | v2 mid-episode pytest |
| `RLLM_ACTIVE_TESTS_K` | 5 | tick cadence / stall window |
| `RLLM_ACTIVE_TESTS_COOLDOWN` | 10 | masked-after-solve steps |
| `RLLM_ACTIVE_TESTS_M_INIT` | 2 | initial ACTIVE subset size |
| `RLLM_ACTIVE_TESTS_M_MAX` | 6 | hard cap on |ACTIVE| |
| `RLLM_ACTIVE_TESTS_EXPAND` | 2 | tests promoted on stall |
| `RLLM_ACTIVE_TESTS_ALPHA/BETA/DELTA/GAMMA/ETA` | 1.0/0.2/0.5/0.01/0.05 | dense (v2) reward weights |
| `RLLM_ACTIVE_TESTS_W_SOLVE` | 5.0 (scripts use 10) | terminal solve credit |
| `RLLM_ACTIVE_TESTS_W_REGRESS` | w_solve/2 | terminal P2P penalty |
| `RLLM_ACTIVE_TESTS_DELTA_P2P_BREAK/RECOVER` | 0.5/0.25 | dense (v2) P2P weights |
| `RLLM_ACTIVE_TESTS_LAM_TOKEN` | 0.1 | token penalty weight |
| `RLLM_ACTIVE_TESTS_TOKEN_REF` | 32000 | token-penalty reference |
| `RLLM_ACTIVE_TESTS_TOKEN_PENALTY_CAP` | 3.0 | **caps token penalty** (added after a bug, see §7) |
| `RLLM_ACTIVE_TESTS_BLOCK_CHARS` | 4000 | prompt-block size cap |

Training-script knobs (env-overridable): `TRAIN_PARQUET`, `VAL_PARQUET`,
`TRAIN_BATCH_SIZE`, `PPO_MINI_BATCH_SIZE`, `TOTAL_EPOCHS`,
`TOTAL_TRAINING_STEPS` (hard cap), `AGENT_MAX_STEPS`, `TRAJECTORY_TIMEOUT`,
`MAX_RESPONSE`, `GPU_MEM_UTIL`, `SAVE_FREQ`, `TEST_FREQ`,
`RLLM_MAX_ROLLOUT_CONCURRENCY`, `LR` (scaled wrapper only), `EXPERIMENT_NAME`,
`PROJECT_NAME`.

---

## 4. Training scripts

- **`DeepSWE_RL/rllm/examples/swe/train_active_tests_swe_master_4b_h200.sh`** —
  the base launcher. `lr=1e-6` (paper), ATS on by default, all knobs
  env-overridable. **`SAVE_FREQ` now defaults to 1** (see §7 — was 10/5 and we
  lost a final model).
- `train_active_tests_scaled_4b_h200.sh` — wrapper that `sed`s lr→`${LR}` and
  raises some defaults.
- `run_scaled.sh` / `run_baseline.sh` — one-command launchers (ATS vs binary).

### Canonical launch (paper-matched, single H200)
```bash
TRAIN_PARQUET=.../swerebench_min_ftp5_v2/train_00000_verl.parquet \
VAL_PARQUET=.../swerebench_min_ftp5_v2/test_00000_verl.parquet \
TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=16 TOTAL_EPOCHS=5 \
AGENT_MAX_STEPS=150 TRAJECTORY_TIMEOUT=5400 MAX_RESPONSE=24576 \
RLLM_MAX_ROLLOUT_CONCURRENCY=8 RLLM_ACTIVE_TESTS_ENABLED=1 \
RLLM_ACTIVE_TESTS_W_SOLVE=10.0 SAVE_FREQ=1 TEST_FREQ=999 \
EXPERIMENT_NAME=run_$(date +%s) PROJECT_NAME=active_tests \
nohup sg docker -c "bash DeepSWE_RL/rllm/examples/swe/train_active_tests_swe_master_4b_h200.sh" \
  > /home/temp/train.log 2>&1 &
```
Launch via `sg docker -c` (socket is `srw-rw---- root docker`; do NOT chmod
666). Outputs → `/home/temp/training_outputs/<PROJECT>/<EXPERIMENT>/`.

---

## 5. Datasets

`build_comprehensive_swerebench.py` pulls `nebius/SWE-rebench` (public mirror;
`docker_image` is the public `swerebench/sweb.eval.x86_64.*`), embeds
`make_test_spec`, writes verl parquets. Env knobs: `N_TRAIN`, `N_VAL`,
`MAX_PROBE`, `MIN_FTP` (min FAIL_TO_PASS size), `OUT_DIR`, `CACHED_ONLY`
(=1 → select only locally-cached images, avoids Docker Hub pulls at train time).

Existing parquets under `data_examples/rl_data/`:
- `swerebench_comprehensive/` — 32 tasks, median |F|=1 (too easy/degenerate for partial credit)
- `swerebench_min_ftp5/` — 16 tasks, mean |F|=11.4
- **`swerebench_min_ftp5_v2/`** — 39 train + 8 val, mean |F|=24 (used for the paper-matched run)
- `swerebench_cached/` — built with CACHED_ONLY (18 tasks)

**Paper's actual RL data** (not public at exact path): SWE-rebench + SWE-Gym
OOD, difficulty-filtered to ~9k "mixed-result" instances. We only wired the
SWE-rebench source. SWE-Gym/R2E-Gym/SWE-smith loaders are TODO if closer
replication is wanted. See PAPER_DRAFT §4 + the dataset research in chat.

---

## 6. Monitoring & analysis tooling

- `training_status.sh <log>` — one-line health summary (steps, solves, errors, GPU).
- `analyze_trend.py <log>` — per-step + **cumulative** reward / solves / grad_norm trend.
- `analyze_active_tests.py <log>` — per-step ATS reward + within-step variance.
- `simulate_binary_baseline.py <log>` — counterfactual: what gradient the binary baseline would have gotten on the same trajectories.
- `analyze_behavior_drift.py <exp_dir>` — agent behavior over steps (pytest rate, first-action mix).
- `dump_metrics_csv.py <log> <csv>` — per-step metrics → CSV (for plotting).
- `plot_metrics_csv.py <csv> <png>` — 4-panel learning-curve figure.
- `plot_ats_vs_binary.py <log> <png>` — ATS vs simulated-binary variance bars.

**Metrics are saved 3 ways:** TensorBoard (`tensorboard_log/`, canonical,
continuous), the raw training log, and the CSV. The `!!!!!!! ATS terminal:`
lines in the log carry per-trajectory frac_solved / P2P / token detail.

---

## 7. HARD-WON LESSONS (read before launching)

1. **`save_freq` MUST divide the total step count — or set it to 1.** The
   paper150 run trained 9 PPO steps but `save_freq=5` only saved `global_step_5`;
   the run ended at step 9 (before the step-10 save) with no final save, and
   the step-9 weights were lost when the trainer freed GPU memory. **Default is
   now 1.** Always confirm the latest checkpoint after a run.

2. **Lowering `agent.max_steps` starves the learning signal.** At `max_steps=30`
   the agent couldn't finish fixes → ~3% solve rate → grad_norm collapsed to
   ~0.001 on no-solve steps (the only reward variance was the tiny token
   penalty). At `max_steps=150` (paper) → ~44% solve rate, grad_norm ~0.15,
   reward strongly positive. **Keep max_steps high; it's the #1 lever for
   whether the model learns.** Pair with `trajectory_timeout=5400`.

3. **Token penalty was unbounded — caused a training-destabilizing reward
   outlier.** One trajectory dumped ~317M chars of bash output → `r_token=-989`
   → reward −495 → `grad_norm` spike 5.4. Fixed with `token_penalty_cap` (§3).
   verl's default `grad_clip=1.0` contained the damage (reported grad_norm is
   pre-clip), but always cap reward terms.

4. **`train_batch_size=32` idles the GPU ~1h/step on a single machine.** The
   trainer creates ALL `batch×n` envs/containers up-front before any rollout.
   batch=32 → 128 containers → ~1h of GPU-idle setup. batch=8-16 is the sweet
   spot on 1 GPU. (Batch size is otherwise "memory-free" per SETUP_NOTES — it's
   the container-setup that scales.)

5. **Docker Hub unauthenticated pull limit (100/6h) → NO_CONTAINER failures.**
   76/128 trajectories failed once. Fix: `docker login` (auth) AND/OR pre-pull
   all task images (they then stay cached). `CACHED_ONLY=1` parquet build
   guarantees no pulls at train time. We have ~63 swerebench images cached.

6. **Per-step container accumulation slows throughput.** Killed runs leak
   running containers. Clean between runs:
   `sg docker -c "docker kill \$(docker ps -q); docker container prune -f"` and
   `ray stop --force`.

7. **Process management:** launch long jobs with `setsid nohup ... &` so they're
   isolated from shell-signal cascades (a chained `pkill` killed a watcher
   once). The auto-mode safety classifier blocks killing processes this session
   didn't create and blocks broad `docker rm -f` while other work is running.

8. **A parallel Claude session once launched a competing SWE-World run** that
   crashed our job via GPU/Ray contention. Check `pgrep -f train_agent_ppo` and
   `nvidia-smi` before launching.

---

## 8. Results (paper-matched run `paper150_20260524-050050`)

- 9 PPO steps, max_steps=150, batch=16, lr=1e-6, 39-task min_ftp5_v2 parquet.
- Per-step solve rate ~30-44%; **cumulative solves 28→186** (monotonic).
- grad_norm stable ~0.12-0.20 (one capped-bug spike at step 8).
- **Held-out validation solve rate: 0.250** (`val/test_score/swe`).
- Reward mean positive every step except the step-8 token-bug outlier.
- Metrics archived: `paper/paper150_metrics.csv`, `paper/fig6_paper150_curves.png`.

Earlier smoke/scaled runs (max_steps=20-50) are documented in
`SMOKE_RESULTS_FOR_USER.md` and `paper/PAPER_DRAFT.md` §5 (incl. the
ATS-vs-binary counterfactual: ATS gives gradient on ~100% of PPO steps vs
binary's ~50%, the headline claim).

---

## 9. Saved models

| Model | Path | Note |
|---|---|---|
| SFT base (step 0) | `/home/temp/models/SWE-Master-4B-SFT` | RL init |
| ATS-RL (step 5) | `/home/temp/models/SWE-Master-4B-ATS-RL` | 5-PPO-step (the step-9 was lost, see §7.1) |

HF export: `bash export_to_hf.sh <experiment_dir> <target_dir>` runs verl's
`scripts/model_merger.py merge --backend fsdp`. Upload:
`hf upload <user>/<name> <target_dir> .` (needs `hf auth login`).
`auto_export_on_checkpoint.sh` / `auto_export_after_training.sh` automate this
at end-of-run.

---

## 10. Next steps (if continuing)

1. **Re-run with `save_freq=1`** to get a fully-trained (10+ step) model with
   all checkpoints — the current default now does this.
2. **ATS-vs-binary head-to-head** at matched compute (`run_baseline.sh` flips
   `RLLM_ACTIVE_TESTS_ENABLED=0`) — the paper's headline experiment.
3. **Multi-source dataset** (add SWE-Gym/R2E-Gym/SWE-smith loaders) to match
   the paper's distribution.
4. **Held-out eval on SWE-bench Verified** (build a separate val parquet;
   current val is SWE-rebench-Lite).
5. **v2 ablation** (`RLLM_ACTIVE_TESTS_MID_EPISODE=1`) for the dense-reward +
   cooldown-recheck terms.
6. Paper draft is in `paper/PAPER_DRAFT.md` — fill §5 results tables.

## 11. Env / infra

Single H200 (143 GB), `/home/temp/venv` (Python 3.11, torch 2.8+cu128, vllm
0.10.2, flash-attn 2.8.3). `bash build_env.sh` rebuilds. Docker daemon must be
up + user in docker group (`sg docker -c`). See SETUP_NOTES.md for the full
machine history and the H200 full-param-PPO memory analysis (max_response cap
= 32768 on 1 GPU; peak ~128 GB).
