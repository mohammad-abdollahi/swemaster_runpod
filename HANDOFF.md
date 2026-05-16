# SWE-Master RL — Handoff / Resume Document

> Branch: `nibi-preparation`. This is a **self-contained snapshot** of the
> working state built on the Trillium **login node** (heavily constrained),
> meant to be cloned and resumed on a better server (Nibi, or any box with
> Docker / higher ulimits / multi-GPU). Written 2026-05-16.

---

## 0. TL;DR

We made the SWE-Master RL pipeline **run end-to-end and log per-step metrics**
on a Docker-less, `ulimit -u 1024`, single-GPU login node, by building a
**Ray-free / TRL-style reimplementation** under [`TRL_RL/`](TRL_RL/) that keeps
the paper's algorithm (multi-turn agent rollout → GRPO with leave-one-out
advantage). Both reward modes work:

- `proxy` — diff-similarity (fractional, gives a visible gradient; v7 reached reward 0.11→0.21, loss/grad_norm ≠ 0)
- `test` — **the paper's exact binary SWE-bench reward** (apply patch → run FAIL_TO_PASS/PASS_TO_PASS in a per-instance conda env → 1.0/0.0), verified correct (gold patch → 1.0).

The original Ray + verl + vLLM-async path **could not run on the login node**
(see §3) but is **untouched and ready** to use on Nibi if Ray works there.

---

## 1. Goal

Make the SWE-Master RL training pipeline work, train, and log metrics each
step. Cluster has **Apptainer only (no Docker)**.

## 2. Where everything lives

| Thing | Path | Notes |
|---|---|---|
| Repo / code | `/project/def-hemmati-ac/$USER/SWE-Master` | this tree |
| Apptainer sandbox | `/scratch/$USER/swemaster/sif/pytorch_27_cu128/` | torch 2.8+cu128, py3.11, **rebuild on Nibi** |
| Python venv | `/scratch/$USER/swemaster/venv/rllm/` | vllm 0.10.2, verl 0.7.1, trl 0.29, flash-attn 2.8.3 |
| HF cache | `/scratch/$USER/swemaster/cache/hf/hub/` | Qwen3-4B, **RUC-AIBOX/SWE-Master-4B-SFT** |
| Cloned SWE repos | `/scratch/$USER/swemaster/repos/<repo>` | 3 demo repos at base_commit |
| Per-instance test envs | `/scratch/$USER/swemaster/test_envs/<instance_id>/` | conda envs for the binary reward |
| Run outputs / TB | `/scratch/$USER/swemaster/runs/` | tensorboard event files + checkpoints |

**None of `/scratch` is in this git snapshot** — it must be rebuilt on Nibi
(scripts below automate it). Only code + the demo data + this doc + the chat
archive are committed.

verl is vendored at `DeepSWE_RL/rllm/verl/` (its `.git` removed; provenance in
`verl/VERL_PROVENANCE.txt` = `volcengine/verl @ 6558b6b`, which resolves to
verl 0.7.1, pinning `ray>=2.41`).

## 3. Why the original pipeline couldn't run here (READ BEFORE NIBI)

The paper's RL = `DeepSWE_RL/rllm` (rLLM) + `verl` + **Ray** + vLLM-async,
launched by `examples/swe/swe_rl_debug_hls_try_arsenal_long_step_and_long_time.sh`
(64 GPUs, Qwen2.5-Coder-32B-SFT, Docker per-instance env, binary test reward,
GRPO `adv_estimator=loop`).

Blockers on the Trillium **login node** (most **do not apply on a real
compute node / Nibi**):

1. **No Docker** — cluster has Apptainer only. The env framework is Docker/K8s-only.
2. **`ulimit -u 1024`** (hard, no root). Ray's `raylet` (C++) spawns a
   `boost::asio` thread pool sized to `hardware_concurrency()` = **192** at
   boot → `pthread_create` EAGAIN → SIGABRT. We proved `taskset` doesn't help
   (libstdc++ reads `sysconf(_SC_NPROCESSORS_ONLN)`, not affinity).
3. **`ray.init(local_mode=True)` removed** in Ray ≥ 2.32; verl pins `ray>=2.41`
   — so no in-process escape hatch.
4. **1-hour CPU ulimit** (`ulimit -t 3600`) + single shared GPU.

### KEEP RAY IN MIND FOR NIBI

**On Nibi / a compute node these blockers likely vanish**: compute nodes
usually allow ≥4096 procs (Ray boots fine), longer/no CPU-time cap, and
multi-GPU. If Nibi also has Apptainer-with-Docker-emulation or you build the
SWE-bench images as Apptainer SIFs, the **original Ray+verl path is the
faithful reproduction and should be preferred there.** Steps to try on Nibi:

```bash
# 1. Rebuild the container + venv (scripts/01_build_env.sh works as-is)
# 2. Confirm Ray boots on a compute node (NOT a login node):
python -c "import ray; ray.init(); print(ray.cluster_resources()); ray.shutdown()"
# 3. If OK, use the ORIGINAL launcher (edit paths/model/data first):
bash DeepSWE_RL/rllm/examples/swe/swe_rl_debug_hls_try_arsenal_long_step_and_long_time.sh
```

The only edit we made to original code is
`DeepSWE_RL/rllm/rllm/trainer/agent_trainer.py`: `ray.init()` now takes
`include_dashboard=False` and an `RLLM_RAY_LOCAL_MODE` env switch
(default on). **On Nibi set `RLLM_RAY_LOCAL_MODE=0`** to restore normal
distributed Ray. The diagnostic comments there explain the whole ulimit story
— left intentionally for the Nibi attempt.

## 4. What we built — `TRL_RL/` (the Ray-free pipeline)

```
TRL_RL/
├── README.md
├── src/
│   ├── data.py            # loads the paper RL parquet; conversational prompts;
│   │                      #   carries gold_patch + test_patch + make_test_spec
│   │                      #   + FAIL_TO_PASS/PASS_TO_PASS for the test reward
│   ├── rewards.py          # proxy reward: SequenceMatcher(model_diff, gold)
│   ├── swe_sandbox.py      # per-instance git sandbox + safe tools:
│   │                      #   list_files, read_file, bash(read-only whitelist,
│   │                      #   no shell metachars), apply_patch (forgiving
│   │                      #   4-strategy chain), edit_file (PATH/OLD/NEW
│   │                      #   string replace — the one that finally worked),
│   │                      #   submit
│   ├── agent_rollout.py    # minimal OpenHands-style multi-turn loop +
│   │                      #   forgiving tool-call parser
│   ├── swe_test_reward.py  # PAPER-FAITHFUL binary SWE-bench reward, Docker-free
│   ├── build_test_envs.py  # builds one conda env per instance from make_test_spec
│   ├── train_agent_grpo.py # custom multi-turn GRPO trainer (no Ray, no TRL dep):
│   │                      #   leave-one-out advantage, assistant-token-masked
│   │                      #   loss, per-trajectory backward, TB + console logs
│   └── train_grpo.py       # single-shot TRL GRPOTrainer variant (smoke)
└── scripts/
    ├── 00_install_trl.sh         # add trl into the venv
    ├── 01_train_smoke.sh         # synthetic "say YES" GRPO smoke
    ├── 02_train_swe.sh           # single-shot SWE GRPO (proxy reward)
    ├── 03_train_swe_agent.sh     # MAIN: multi-turn agent GRPO (proxy|test)
    └── 04_setup_test_envs.sh     # one-time: build per-instance conda envs
```

Container/env scaffolding is in top-level [`scripts/`](scripts/):
`00_enter_container.sh` (Apptainer wrapper — note it uses `--cleanenv`, so it
forwards a whitelist of `REWARD/MAX_STEPS/...` via `APPTAINERENV_*`),
`01_build_env.sh`, `02_download_model.sh`, plus a FrozenLake verl smoke.
See [`scripts/README.md`](scripts/README.md) for the full cluster diagnosis.

### How the training works (one paragraph)

For each step: pick B prompts, run G rollouts each. A rollout = reset the git
sandbox to `base_commit`, then loop ≤ `max_turns`: model emits a `<tool>` call
→ sandbox runs it → result appended → repeat until `submit` or budget. The
final worktree diff is the candidate patch. Reward per rollout = `proxy`
(SequenceMatcher vs gold) or `test` (binary SWE-bench). Group-relative
advantage `A_i = R_i − mean(R_{j≠i})` (leave-one-out, exactly the paper's
`adv_estimator=loop`). Loss = `−A_i · mean(log π_θ(assistant tokens))`,
per-trajectory backward, then AdamW step. Metrics → stdout + TensorBoard.

Full detail is in the chat archive (`.claude-archive/`) and `TRL_RL/README.md`.

## 5. How to reproduce on Nibi (TRL path — works even without Ray/Docker)

```bash
# 0. clone this branch
git clone -b nibi-preparation git@github.com:mohammad-abdollahi/swemaster_runpod.git
cd swemaster_runpod                    # (or wherever)

# 1. build the Apptainer container + venv  (edit USER/paths in scripts if needed)
bash scripts/00_enter_container.sh bash scripts/01_build_env.sh   # see scripts/README

# 2. models + demo repos + test envs
bash scripts/00_enter_container.sh bash scripts/02_download_model.sh   # Qwen3-4B
#   SWE-Master-4B-SFT: huggingface RUC-AIBOX/SWE-Master-4B-SFT (snapshot_download)
#   clone the 3 demo repos to /scratch/$USER/swemaster/repos (see chat archive)
bash scripts/00_enter_container.sh bash TRL_RL/scripts/04_setup_test_envs.sh

# 3a. proxy-reward run (fast, fractional, shows a gradient)
REWARD=proxy MAX_STEPS=50 bash scripts/00_enter_container.sh \
  bash TRL_RL/scripts/03_train_swe_agent.sh

# 3b. paper-faithful BINARY test reward
REWARD=test MAX_STEPS=20 bash scripts/00_enter_container.sh \
  bash TRL_RL/scripts/03_train_swe_agent.sh
```

(Default model in `03_train_swe_agent.sh` is already
`RUC-AIBOX/SWE-Master-4B-SFT`. The `REWARD`/`MAX_STEPS` etc. env vars are
auto-forwarded into the container by `00_enter_container.sh`.)

## 6. Results so far (login node, tiny scale)

| Run | Reward | Result |
|---|---|---|
| Synthetic "YES" smoke | proxy | reward 0.5→1.0, loss/grad ≠ 0 — loop verified |
| SWE single-shot (Qwen3-4B base) | proxy | mean ≈ 0.05 |
| SWE multi-turn v7 (SWE-Master-4B-SFT) | proxy | reward **0.11 → 0.21** (max 0.83), loss/grad ≠ 0 |
| SWE multi-turn (SWE-Master-4B-SFT) | **test (paper)** | R = 0.0 (model didn't fully resolve any of 3 demo issues; reward logic verified by gold→1.0). loss/grad = 0 because all-tie groups → 0 advantage (paper handles this with `rejection_sample=True` over a large pool) |

This is a **working, faithful pipeline**, not a convergence demonstration —
the latter needs paper scale (32B, 150-turn OpenHands scaffold, thousands of
steps, 64 GPUs, rejection sampling). That is the Nibi job.

## 7. Open items / next steps on Nibi

1. **Try the original Ray+verl path first** (see §3) — it's the faithful one
   and Nibi likely lifts the ulimit/Ray blockers. Set `RLLM_RAY_LOCAL_MODE=0`.
2. If Docker is available on Nibi: use the real SWE-bench images instead of our
   local conda test envs (more instances, more robust reward). Our
   `swe_test_reward.py` already mirrors `_calculate_reward_swebench`.
3. Scale the TRL path: full RL parquet (not the 3-row demo), `num_generations`
   ≥ 4, `max_turns` ≥ 50, add rejection sampling of all-tie groups, longer runs.
4. `falconry__falcon-2254` test env has an async-dep gap (gold → 0). Fix its
   conda env (pytest-asyncio / uvicorn) or drop it.
5. Optional: port the `edit_file` / forgiving-parser improvements back into the
   OpenHands scaffold used by the original pipeline.

## 8. The chat / context archive

`.claude-archive/` contains the full conversation transcript + session data so
you (or Claude) can reconstruct the entire reasoning trail on Nibi. Start a new
Claude Code session in the repo and point it at `HANDOFF.md` +
`.claude-archive/` to resume with full context. No credentials are included.
