# SWE-Master RL — RunPod single-H200 setup notes

This document records the state of this checkout so the work can be resumed on
another machine. It complements the chat transcript in
`.claude/projects/-root-swemaster-runpod/*.jsonl`.

## Hardware in use

| Resource | Value |
|----------|-------|
| GPU | 1 × NVIDIA H200 (143 GB VRAM, CUDA 12.8) |
| CPUs | 160 |
| RAM | ~2 TB |
| Persistent volume | `/workspace` (50 GB quota, MooseFS-backed) |
| Root disk | `/` 30 GB overlay |

Important: the original SWE-Master RL pipeline targets **2 × 8 H800 GPUs (16
GPUs)**. The smoke test in
`DeepSWE_RL/rllm/examples/swe/smoke_test_4b_single_h200.sh` is heavily
down-scaled (tensor-parallel = 1, ulysses_sp = 1, batch = 1) so that the
training entrypoint can be exercised on a single GPU; it is not expected to
produce meaningful gradients.

## Layout

```
/workspace/swemaster_runpod/         <- repo (this dir, git remote = origin)
/workspace/venv/                     <- Python 3.11 venv (managed by uv)
/workspace/models/SWE-Master-4B-SFT/ <- HF snapshot of RUC-AIBOX/SWE-Master-4B-SFT
/workspace/uv-cache, /workspace/pip-cache  <- pip/uv wheel caches
/workspace/install.log, /workspace/install2.log  <- pip install transcripts
/workspace/smoke_test.log            <- last smoke-test stdout/stderr
```

## Python environment

Python 3.11.13 in `/workspace/venv`. Re-bootstrap from a fresh machine with:

```bash
uv venv --python 3.11 /workspace/venv
source /workspace/venv/bin/activate
export UV_CACHE_DIR=/workspace/uv-cache PIP_CACHE_DIR=/workspace/pip-cache

# 1) pin torch CUDA 12.8 first
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.1 torchvision torchaudio

# 2) install verl + R2E-Gym + rllm in editable mode
cd /workspace/swemaster_runpod/DeepSWE_RL/rllm
git clone --depth 1 --branch v0.5.0 https://github.com/volcengine/verl.git verl   # if missing
uv pip install -e ./verl
uv pip install -e /workspace/swemaster_runpod/R2E-Gym
uv pip install -e /workspace/swemaster_runpod/swebench_fork_swegym-2.0.13-py3-none-any.whl \
               -e /workspace/swemaster_runpod/swebench_fork_swerebench-4.0.3-py3-none-any.whl \
               -e /workspace/swemaster_runpod/swesmith-0.0.7-py3-none-any.whl
uv pip install -e /workspace/swemaster_runpod/DeepSWE_RL/rllm --no-deps

# 3) vllm + flash-attn (large, ~20min build)
uv pip install vllm==0.10.2
uv pip install --no-build-isolation flash-attn==2.8.3
```

Verified-working versions (May 2026):

| Package | Version | Note |
|---------|---------|------|
| torch | 2.8.0+cu128 | Upgraded by vllm 0.10.2 |
| vllm | 0.10.2 | Brings torch 2.8 |
| flash-attn | 2.8.3 | `--no-build-isolation`, ~40s build |
| triton | 3.4.0 | |
| verl | 0.5.0 | `DeepSWE_RL/rllm/verl/` (cloned tag) |
| rllm | 0.1.0 (editable) | `DeepSWE_RL/rllm/` |
| r2e-gym | 0.1.0 (editable) | `R2E-Gym/` |

Smoke verify:
```python
import rllm, verl, r2egym, torch, vllm, flash_attn
print(torch.cuda.get_device_name(0))
```

## How the smoke test was driven forward

We hit several blockers between the import phase and the first rollout. Each
required a patch or pin to keep moving. See the commits on the
`setup/runpod-h200` branch.

1. **Ray runtime_env pip clone** — `train_agent_ppo.py` passed
   `runtime_env={"pip": ["...", "./swebench_fork_*.whl", ...]}` to `ray.init`.
   Ray clones the whole `/workspace/venv` (~19GB) and then pip-installs the
   list, but resolves the relative wheels from `runtime_resources/exec_cwd/`
   where the files do not exist — pip exits 1 and the trainer never reaches
   `train_agent.remote(config)`. Fix: gate the heavy `runtime_env` behind
   `RLLM_USE_PARENT_VENV=1`. When set (which the smoke script does), Ray uses
   the parent's venv directly. See [DeepSWE_RL/rllm/rllm/trainer/verl/
   train_agent_ppo.py](DeepSWE_RL/rllm/rllm/trainer/verl/train_agent_ppo.py).
2. **Wrong verl fork** — `.gitmodules` pins
   `https://github.com/agentica-project/verl` (a fork with `_timer` and
   other rllm-specific patches). Cloning `volcengine/verl@v0.5.0` (the
   public release) gives an `ImportError: cannot import name '_timer'`.
   Fix: clone agentica's fork, install editable.
3. **pkg_resources removed in setuptools 81+** — verl's `__init__.py` does
   `import pkg_resources`. Fix: `pip install 'setuptools<81'`.
4. **transformers 5.x dropped AutoModelForVision2Seq** — verl's FSDP worker
   imports it unconditionally. Fix: pin `transformers<5,>=4.45`. That in
   turn requires `huggingface_hub<1`.
5. **r2egym imports HfFolder unconditionally** — `HfFolder` was removed in
   huggingface_hub 1.0; it's also unused in `agenthub/utils/utils.py`. Fix:
   drop it from the import line.
6. **tensorboard not installed** — the smoke script sets
   `trainer.logger=[console,tensorboard]`. Fix: `pip install tensorboard`.

After all of the above the trainer reaches `fit_agent` and then attempts the
first rollout, which is where the docker blocker bites.

### Smoke-run timeline (2026-05-15)

The trainer reaches the rollout step in ~7 minutes from launch. Verified
milestones from the actual run (see `/workspace/smoke_test_final.log`):

| time | event |
|------|-------|
| 01:41:09 | smoke script launched |
| 01:42:26 | Ray cluster up (local instance) |
| 01:44:00 | train_agent worker started, modules imported |
| 01:44:10 | data parquet loaded, dataset filtered to 3 rows |
| 01:44:40 | `Qwen3ForCausalLM contains 4.02B parameters` loaded |
| 01:44:50 | FSDP wraps the model (NO_SHARD, world_size=1, ~16GB VRAM) |
| 01:48:09 | vLLM AsyncvLLMServer Ray actor launched |
| 01:48:30 | EngineCore_DP0 finishes init, KV cache 486k tokens, **94GB VRAM** |
| 01:48:32 | `Async vLLM Server running at 172.20.0.2:40771` |
| 01:48:33 | `Initialized trainer done` / `now in fit_agent` |
| 01:48:34 | `now in run_agent_trajectory_async` |
| 01:48:34 | **`docker.errors.TLSParameterError`** — docker client unable to start |

So the verl/rllm side of the pipeline is functional on a single H200 with the
patches in this branch. The remaining blocker is the rollout sandbox.

### Why the docker step fails on this RunPod pod

Two separate problems, of which only one needs to be fixed to test the next
step:
1. `r2egym/agenthub/runtime/docker.py` defines
   ```python
   DOCKER_TLS_VERIFY = "1"
   DOCKER_CERT_PATH = "pem_saved_path_including_ca.pem_cert.pem_key.pem"
   ```
   and passes them into `docker.from_env(timeout=120, environment=custom_env)`.
   The docker client SDK reads those, asks for a real cert file, and fails
   with `TLSParameterError`. The cleanest fix for this is to make those
   constants env-driven and unset by default.
2. Even after the TLS issue is fixed and the client connects, container
   start (`docker run`) fails on RunPod with `unshare: operation not
   permitted` because the pod's seccomp profile blocks user namespaces.
   This is what makes the pod unable to run the SWE-Master RL pipeline as
   designed.

For a one-shot demo on a docker-capable machine you do not need any of these
patches at the rollout boundary — only the import-side patches from steps 1-6
above.

## The docker blocker (RunPod)

`r2egym`'s SWE environment runs each rollout in a per-task Docker container.
The Docker daemon **does start** in this RunPod pod (after installing
`docker.io` + writing `/etc/docker/daemon.json` with `iptables: false,
bridge: "none"`), but containers **cannot start**:

```
$ docker run --rm hello-world
docker: failed to register layer: unshare: operation not permitted
$ unshare --user --pid -- /bin/true
unshare: unshare failed: Operation not permitted
$ grep CapEff /proc/1/status
CapEff: 00000000a80425fb        # no CAP_SYS_ADMIN
```

The RunPod kernel security profile blocks `unshare()`, so nested
containerization is impossible. Options on another machine:

1. **Docker-capable host** (bare-metal, GCE/EC2 with privileged, a non-RunPod
   pod). The script in `examples/swe/smoke_test_4b_single_h200.sh` should run
   end-to-end after the docker daemon is up.
2. **Kubernetes backend** (`+env.env_args.backend=kubernetes`). Requires a
   reachable cluster and a kubeconfig.
3. **Bypass containerization** — not currently supported by `r2egym`; would
   require adding a "local" backend that runs commands directly on the host
   (security risk for arbitrary agent code).

## Dataset

For the smoke test we use the in-repo demo parquet:
`data_examples/rl_data/ood_data/equal_0.3999999_0.9999999_swe-gym_and_rebench_demo_for_RL/train_00000_verl.parquet`
(3 rows; columns `prompt`, `reward_model`, `extra_info`).

The original 4B script points at `/code/sunshuang/...` paths that do not exist
outside the SWE-Master authors' internal infra.

## Smoke-test command

```bash
cd /workspace/swemaster_runpod
source /workspace/venv/bin/activate
bash DeepSWE_RL/rllm/examples/swe/smoke_test_4b_single_h200.sh \
    2>&1 | tee /workspace/smoke_test.log
```

Configurable via env vars:

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL_PATH` | `/workspace/models/SWE-Master-4B-SFT` | HF directory |
| `TRAIN_PARQUET` | in-repo demo | RL training data |
| `VAL_PARQUET` | same as train | RL validation data |

## Pushing/moving to another machine

The `.claude/` directory is *partly* tracked:
- `.claude/projects/-root-swemaster-runpod/*.jsonl` — chat transcript, **tracked**
- `.claude/.credentials.json` — secrets, **ignored**
- `.claude/ide/`, `backups/`, `shell-snapshots/`, `sessions/` — local state,
  **ignored**

To continue this conversation on a new machine: clone the repo, then start
Claude Code in the repo root. The transcript will be re-loaded into context.

---

# L40S continuation (2026-05-17) — RL pipeline driven to real training

Continued on a single **L40S 46 GB / 31 GB RAM / Docker-capable** VM (the
H200 pod could not run containers; this one can — the whole point of the
move). The pipeline now runs end-to-end and does real GRPO PPO updates on the
genuine SWE-Master training distribution.

## Redeploy on a better-GPU server

1. `git clone` this repo (branch `setup/runpod-h200`), `cd` in, start Claude
   Code — `.claude/projects/-home-temp-swemaster-runpod/*.jsonl` + `memory/*.md`
   reload the full context.
2. `bash build_env.sh` (uv venv + torch cu128 + agentica verl fork +
   rllm/r2egym + vllm 0.10.2 + flash-attn + pinned deps). Adjust paths at top
   if not `/home/temp`.
3. `uvx --from huggingface_hub hf download RUC-AIBOX/SWE-Master-4B-SFT
   --local-dir <models>/SWE-Master-4B-SFT`.
4. Datasets are committed under `data_examples/rl_data/`; expand with
   `build_comprehensive_swerebench.py` (nebius/SWE-rebench; its `docker_image`
   IS the public `swerebench/...` Docker Hub mirror).
5. Run `DeepSWE_RL/rllm/examples/swe/train_comprehensive_swerebench_l40s.sh`
   (edit paths/`PYTHON_BIN` if not `/home/temp`).

## Patches on this branch (all committed)

- `R2E-Gym/.../runtime/docker.py`: TLS hardcode → env-driven + local-socket
  fallback; `make_test_spec` scoping bug fixed (local renamed
  `embedded_test_spec`).
- `DeepSWE_RL/rllm/.../engine/agent_execution_engine.py`:
  `RLLM_MAX_ROLLOUT_CONCURRENCY` env cap (rollout containers one at a time).
- `DeepSWE_RL/rllm/verl/.../sharding_manager/fsdp_vllm.py`:
  `'cpu' in torch.device` → `.device.type == 'cpu'` (lines 141 & 155). Real
  verl bug that crashed **every** multi-step LoRA run at step 2 (vLLM resync).

## Hardware-fit decisions (revisit on a bigger GPU)

- **LoRA r32, FSDP offload OFF.** Full-param PPO of 4B needs ~40 GB host RAM
  (offloaded params + fp32 Adam); 31 GB can't (pinned mem, swap won't help).
  On a big-VRAM+big-RAM server, drop LoRA / re-enable offload for full-FT (the
  original recipe). Knobs in `train_comprehensive_swerebench_l40s.sh`.
- Sequential rollouts, short `max_response_length`,
  `gpu_memory_utilization=0.38`, `n=4` GRPO — raise on a bigger box.
- 96 GB `/swapfile` was a stopgap; LoRA removed the need.

## Dataset note (methodological)

SWE-Master trains on **SWE-rebench + SWE-Gym OOD**; SWE-bench Verified is the
held-out eval (`val_files`). Train only on the SWE-rebench/Gym parquets under
`data_examples/rl_data/`; do NOT train on SWE-bench (eval contamination).

## Status at hand-off

`train_comprehensive_swerebench_l40s.sh`: 20 train + 4 val SWE-rebench tasks,
n=4 GRPO, ~40 PPO steps, LoRA, periodic val + ckpts. A single GRPO `step:1`
was already verified end-to-end (real container rollout → test reward → LoRA
update → full metrics). Live log `/home/temp/train_comprehensive.log`; last
good single-step log `/home/temp/train_public_step1_SUCCESS.log`.

---

# H100 continuation (2026-05-17) — full-param feasibility + multi-GPU handoff

Machine moved again to a **single NVIDIA H100 80 GB, 196 GB RAM, 16 vCPU,
1.2 TB disk, Docker-capable, passwordless sudo** (`unshare` works, full caps —
the original H200 container blocker does not apply). The H200/L40S
venv/model/`/workspace` do **not** exist here; everything was rebuilt at
`/home/temp`. See memory: `vm-resources-h100.md`, `build-env-deps.md`,
`training-plan.md` (auto-reloaded by Claude on restore).

## What was done on the H100 (all reproducible)

1. Docker: `sudo usermod -aG docker temp` + `chmod 666 /var/run/docker.sock`;
   nested containers verified.
2. `bash build_env.sh` — **two reproducibility fixes committed**:
   - flash-attn `--no-build-isolation` needs build deps present first →
     `uv pip install wheel packaging` added before flash-attn (step 10).
   - rllm is `-e --no-deps`; its runtime graph needs (not auto-installed):
     `polars dm-tree datasets accelerate peft pylatexenc tensordict torchdata
     pyarrow sentence_transformers`. Install these after build_env, before
     launching (sweep `rllm.trainer.verl.train_agent_ppo`,
     `rllm.rewards.math_reward`, `rllm.utils`, `rllm.agents.sweagent`,
     `rllm.engine.agent_execution_engine`, `r2egym.agenthub.runtime.docker`
     in a pip-on-missing loop). Verified env: torch 2.8.0+cu128, vllm 0.10.2,
     flash-attn 2.8.3, transformers 4.57.6 on H100.
3. Model re-downloaded → `/home/temp/models/SWE-Master-4B-SFT`.
4. SWE-rebench parquet (committed) verified; `swerebench/...` images pull
   from public Docker Hub.

## Results on the single H100

- **LoRA** (`train_comprehensive_swerebench_l40s.sh`): GRPO/LOOP `step:1`
  completed end-to-end again (GPU 72.7/80, traj/steps 6.75, response 8192
  fully clipped, solve_none → zero reward → zero grad = the documented
  reward-sparsity property, not a bug). `step:2` dies on the LoRA↔vLLM
  `wake_up` `KeyError: 'layers.0.self_attn.qkv_proj.base_layer.weight'`
  (LoRA-param collection in the sharding manager) — the open multi-step
  LoRA blocker.
- **Full-param** (`train_comprehensive_swerebench_h100.sh`, paper recipe):
  3 attempts, all fail on ONE GPU (log `/home/temp/train_h100_fullft.log`):
  (1) 32K resp / vLLM 0.45 → OOM in `loss.backward()`;
  (2) 8K resp / vLLM 0.30 → OOM in `optimizer.step()` (`_foreach_sqrt` over
      fp32 Adam state);
  (3) 8K / vLLM 0.20 / `expandable_segments:True` → vLLM `init_engine`
      FAILS (16 GB < 4B vLLM floor).
  **Definitive, code-confirmed:** verl builds plain on-GPU `optim.AdamW`
  (`verl/workers/fsdp_workers.py:369`) and `load_fsdp_optimizer(device_id=
  cuda)` (:663) before every update → the ~48 GB fp32 Adam+master state is
  GPU-resident *during* `optimizer.step()` regardless of seq length;
  `optimizer_offload` only frees it *between* steps. No CPU/8-bit-Adam path.
  vLLM init floor (~24 GB) + ~48 GB optimizer + params/grads (~16 GB) > 80 GB
  from both sides. **Full-param 4B PPO is impossible on 1×80 GB with this
  pipeline.** Not a tuning issue — the paper sharded it across 16 GPUs.

## GPU-count requirement for full-param 4B PPO

Static cost params+grads+AdamW(fp32) ≈ **64 GB**, FSDP-shardable. Activations
(~10–18 GB @ 8–16K resp, grad-ckpt) shard only with ulysses. vLLM ~24 GB
(1 GPU) / ~12 GB/GPU (TP).

| GPUs ×80 GB | per-GPU @ 8K resp | verdict |
|---|---|---|
| 1 | ~103 GB | ❌ impossible (nothing to shard) |
| **2** | ~59 GB | ✅ **sufficient (recommended min)** |
| 4 | ~34 GB | ✅ comfortable + longer seq |
| 8–16 | — | only for exact paper (122K resp + batch 32) |

GPUs must be 80 GB (2×48 GB won't fit ~59 GB/GPU; need 4×). NVLink preferred
for FSDP all-gather.

## Restore on a ≥2×80 GB machine

1. `git clone` (branch `setup/runpod-h200`), `cd`, start Claude Code
   (memory + transcripts reload context).
2. `bash build_env.sh`; then install the rllm runtime deps listed above.
3. `uvx --from huggingface_hub hf download RUC-AIBOX/SWE-Master-4B-SFT
   --local-dir /home/temp/models/SWE-Master-4B-SFT`.
4. Docker group/socket as above; verify a `swerebench/...` pull.
5. Run **`DeepSWE_RL/rllm/examples/swe/train_comprehensive_swerebench_multigpu.sh`**
   (added this session) — `NUM_GPUS=<n>` env; FSDP FULL_SHARD across GPUs +
   vLLM tensor-parallel; full-param, no LoRA; paper step counts
   (max_steps 150, timeout 3600, epochs 5, n=4). Header documents how to
   scale max_response / ulysses toward the exact paper on 4/8/16 GPUs.

## GRPO vs the code's `loop` estimator (clarification)

PPO is the trainer family; GRPO = PPO with **no critic** + a **group
baseline** (that's why `rollout.n=4`, `critic_warmup=0`, no critic model).
verl reuses the PPO scaffold and `critic/*` metric names for all estimators,
so logs say "PPO step" / `critic/rewards` even with no critic network. The
paper says **GRPO** but every SWE-Master script (incl. the authors' own
`swe_rl_debug_..._4b.sh`) sets **`algorithm.adv_estimator=loop`** = LOOP/RLOO
(leave-one-out baseline: `r − mean(other n−1)`), a critic-free group-baseline
sibling of GRPO (GRPO = `(r − group_mean)/group_std`). For paper-literal GRPO,
one-line change: `algorithm.adv_estimator=grpo`. Either way, all-equal rewards
in a group → zero advantage → zero gradient (intrinsic; needs reward
variance).

---

# H200 continuation (2026-05-18) — full-param 4B PPO WORKS on a single GPU

Machine moved AGAIN to a **single NVIDIA H200 143 GB VRAM, 196 GB RAM, 16
vCPU, 1.2 TB disk, Docker-capable, passwordless sudo, `unshare` works (full
caps `000001ffffffffff`)**. This is NOT the original blocked RunPod H200 pod —
nested containers run fine here. Full memory/findings auto-reload via
`memory/h200-fullparam-success.md` (+ MEMORY.md index).

## The breakthrough

Every prior machine concluded full-param 4B PPO is impossible on one GPU
(L40S 31 GB RAM forced LoRA; H100 80 GB VRAM: optimizer+vLLM+activations
> 80 GB, 3 OOMs) and LoRA — the only single-GPU fallback — dies at step 2
on the vLLM `wake_up` resync KeyError. **The H200's 143 GB lifts the
ceiling.** New script `DeepSWE_RL/rllm/examples/swe/train_comprehensive_swerebench_h200.sh`
(full-param, no LoRA, paper recipe) ran **end-to-end and multi-step** —
the first single-GPU run ever past step 1. Full-param also has no LoRA↔vLLM
resync bug.

## Setup on this box (reproducible)

1. `sudo systemctl start docker` (DEFAULT daemon.json — no iptables/bridge
   hacks; those were ONLY the original RunPod pod). `sudo usermod -aG docker
   temp`. **Launch training via `sg docker -c "..."`** — socket is
   `srw-rw---- root docker`; do NOT `chmod 666` (auto-classifier blocks it and
   it's unnecessary).
2. `bash build_env.sh` (clean, all imports OK: torch 2.8.0+cu128, vllm
   0.10.2, flash-attn 2.8.3, transformers 4.57.6).
3. Re-seed rllm `-e --no-deps` runtime deps (build-env-deps memory):
   `polars dm-tree datasets accelerate peft pylatexenc tensordict torchdata
   pyarrow sentence_transformers`. NB the swe agent module is
   `rllm.agents.swe_agent` (underscore), not `sweagent`.
4. `uvx --from huggingface_hub hf download RUC-AIBOX/SWE-Master-4B-SFT
   --local-dir /home/temp/models/SWE-Master-4B-SFT`.
5. swerebench images pull from public Docker Hub and containers RUN here
   (verified end-to-end).

## Key fix vs the h100 script

**Drop `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — vLLM 0.10.2's
`device_allocator/cumem.py` asserts it is ABSENT (incompatible with its memory
pool, pytorch#147851). It was only an 80 GB anti-fragmentation lever; on
143 GB it's unneeded and FATAL to vLLM init.

## max_response_length ceiling (full-param, gpu_mem_util 0.30)

| max_response | peak VRAM | result |
|---|---|---|
| 16384 | 120.5 GB | OK |
| 24576 | 122.8 GB | OK |
| 32768 | **128.6 GB** | OK (chosen, 14 GB headroom) |

Marginal ≈ +2.3 GB / +8192 tokens. The peak is DETERMINISTIC (bounded by
`ppo_max_token_len_per_gpu = 8192 + max_response`, the per-microbatch
logits/activation cap — it does NOT creep over a long run). Extrapolation:
the paper's 122880 ≈ 150 GB > 143 → still needs multi-GPU ulysses (consistent
with all prior findings). The h200 script is env-parameterized: `MAX_RESPONSE`,
`PPO_MAX_TOKEN_LEN`, `GPU_MEM_UTIL`, `TOTAL_EPOCHS`. Search harness:
`search_max_response.sh`.

## Production run: 5-epoch full-param @ max_response 32768

`MAX_RESPONSE=32768 TOTAL_EPOCHS=5` (paper-exact epochs), 20 SWE-rebench
train tasks / 4 val, batch 1, n=4 LOOP. Log `/home/temp/train_h200_5epoch.log`.
Reached **epoch 5, step 81/100, 0 errors** before being stopped (clean
baseline; user pivoting to a batched run). Peak held at the predicted
128.6 GB. ckpt+val every 10 steps work (those steps ~25 min vs ~6 min normal).

### Learning result (analyze_learning.py — task-keyed, shuffle-corrected)

verl's train dataloader defaults to `data.shuffle: True` (NOT overridden), so
step order changes every epoch — position-based first-N/last-N reward is
task-mix NOISE. `analyze_learning.py` recovers step→task (pairs each
`...epoch E, step N` marker with the next `Docker image: swerebench/...`
line) and reports per-task reward by epoch. Through epoch 0→3 (20 tasks each
revisited 4×): **improved 7, regressed 5, flat 8, mean delta +0.044**;
55/81 steps non-zero grad; grad_norm mean 0.131 stable; entropy 0.14–0.27
(not collapsed). Clearest gain `asottile_setup` 0.25→0.75→0.62→0.88.
Verdict: **real but modest positive signal**, concentrated where within-group
reward variance exists (the GRPO/LOOP mechanism working as designed); not a
big capability jump — expected at this smoke scale (20 tasks, n=4, batch=1 →
1 noisy sample/task/epoch). Snapshot saved `/home/temp/final_learning_report.txt`.

## Batching analysis (next step — feasible & beneficial here)

`train_batch_size` 1→N does NOT raise the GPU peak: with
`ppo_micro_batch_size_per_gpu=1` + `use_dynamic_bsz=False` the actor still
does ONE sequence per backward; bigger batch only adds gradient-accumulation
micro-steps before the optimizer step. Peak stays the deterministic
~128.6 GB. It also adds NO total compute: rollout work/epoch = num_tasks×n
either way; batching just groups trajectories into fewer, richer updates →
kills the ~40% all-equal-zero-gradient steps (variance across many tasks),
paper-like stable gradients, and FEWER slow ckpt/val cycles → faster
wall-clock. Paper's 32/16 doesn't fit 20 tasks; use e.g.
`train_batch_size=20 ppo_mini_batch_size=5` (whole set/update) or `10/5`.
(User stopped here to ask questions before launching the batched run.)

## Artifacts added this session

- `DeepSWE_RL/rllm/examples/swe/train_comprehensive_swerebench_h200.sh`
  (env-parameterized full-param H200 recipe)
- `search_max_response.sh` (max_response ceiling search harness)
- `analyze_learning.py` (task-keyed, shuffle-corrected learning analysis)
- `memory/h200-fullparam-success.md` (full findings, auto-reloaded)

---

# Paper hyperparameters: what is reproducible on 1×H200 (4B) — 2026-05-18

Authors' two RL recipes (verbatim, in `DeepSWE_RL/rllm/examples/swe/`):
`swe_rl_debug_hls_try_arsenal_long_step_and_timeout3600_4b.sh` (4B) and
`swe_rl_debug_hls_try_arsenal_long_step_and_long_time.sh` (32B).

| Hyperparam | 4B RL | 32B RL | Ours 1×H200 |
|---|---|---|---|
| Base model | Qwen3-4B | Qwen2.5-Coder-32B | 4B |
| GPUs | 2×8 = 16 | 8×8 = 64 | 1 |
| train_batch_size | **32** | **32** | 1 → can be 32 |
| ppo_mini_batch_size | **16** | 8 | 1 → can be 16 |
| ppo_micro_batch_size_per_gpu | 1 | 1 | 1 |
| rollout.n | 4 | 4 | 4 ✅ |
| max_prompt_length | 8192 | 8192 | 8192 ✅ |
| max_response_length | 122880 | 102400 | **32768 (cap)** |
| tensor_model_parallel_size | 2 | 4 | 1 |
| ulysses_sequence_parallel_size | 8 | 8 | 1 |
| total_epochs | 5 | 5 | 5 ✅ |
| max_steps / trajectory_timeout | 150 / 3600 | 150 / 5400 | 150 / 3600 ✅ |

NOTE on the 4B-vs-32B confusion: `train_batch_size=32` is IDENTICAL for both
models — it is the 4B value, not 32B-specific. The genuinely model-specific
differences are ppo_mini_batch (16 vs 8), max_response (122880 vs 102400),
tensor_model_parallel (2 vs 4), trajectory_timeout (3600 vs 5400) and GPU
count. For our 4B work the 4B column is the reference.

## What is reproducible on the single 143 GB H200 (4B)

GROUP A — already matched exactly (the METHOD): full-param (no LoRA),
param+optimizer offload, max_prompt=8192, rollout.n=4, max_steps=150,
trajectory_timeout=3600, total_epochs=5. Algorithm identical.

GROUP B — paper-exact and reproducible here, NOT memory-bound:
**train_batch_size=32, ppo_mini_batch_size=16.** With
ppo_micro_batch_size_per_gpu=1 + use_dynamic_bsz=False the actor backprops
ONE sequence at a time and only accumulates more micro-steps → VRAM peak
stays the deterministic ~128.6 GB regardless of batch. No extra total compute
(rollout work/epoch = num_tasks×n either way) — just slower per step, fewer
slower ckpt/val cycles. ONLY prerequisite: a ≥32-task train parquet (current
comprehensive parquet = 20 tasks). Expand via `build_comprehensive_swerebench.py`
(pulls nebius/SWE-rebench; raise the filter cap; its `docker_image` is the
public `swerebench/...` mirror).

GROUP C — IMPOSSIBLE on 1 GPU (irreducibly multi-GPU sequence sharding):
- **max_response_length=122880 — EXPLICITLY EXCLUDED. We do NOT want this.**
  The paper reaches 122K only via ulysses_sequence_parallel_size=8 (each
  long trajectory sharded across 8 GPUs). On 1 GPU the full-param backward
  of 122K tokens needs ~150 GB > 143 GB. **Hard cap stays max_response=32768**
  (peak ~128.6 GB, deterministic). Do not attempt 122880 single-GPU.
- tensor_model_parallel_size=2 / ulysses_sequence_parallel_size=8 require
  ≥2 / ≥8 GPUs by definition → forced to 1 here (this is what caps
  max_response).

DECISION: the runnable paper-faithful 4B config on this box is the GROUP A
method + GROUP B batch (train_batch_size=32, ppo_mini_batch_size=16) +
**max_response_length=32768 (NOT 122880)** + a ≥32-task SWE-rebench parquet.
The 32B recipe is fully out of scope (64 GPUs; 32B won't fit single-GPU
full-param PPO under any config).

---

# Paper-faithful hyperparameter update + dataset comparison (2026-05-19)

## Hyperparameters now matched to the paper 4B recipe (except multi-GPU sharding)

`train_comprehensive_swerebench_h200.sh` updated. All paper-exact 4B values
from `swe_rl_debug_hls_try_arsenal_long_step_and_timeout3600_4b.sh` adopted
EXCEPT the irreducibly multi-GPU sharding knobs.

| Paper 4B | Ours (1×H200) | status |
|---|---|---|
| `train_batch_size=32` | **32** (env `TRAIN_BATCH_SIZE`, default 32) | ✅ matched |
| `ppo_mini_batch_size=16` | **16** (env `PPO_MINI_BATCH_SIZE`, default 16) | ✅ matched |
| `ppo_micro_batch_size_per_gpu=1` | 1 | ✅ matched |
| `rollout.log_prob_micro_batch_size_per_gpu=4` | 4 | ✅ matched |
| `rollout.log_prob_use_dynamic_bsz=True` | True | ✅ matched |
| `data.filter_overlong_prompts_workers=32` | 32 | ✅ matched |
| `data.max_prompt_length=8192` | 8192 | ✅ matched |
| `rollout.n=4` | 4 | ✅ matched |
| `agent.max_steps=150 / timeout=3600` | 150 / 3600 | ✅ matched |
| `total_epochs=5` | 5 | ✅ matched |
| full-param, optimizer/param offload | same | ✅ matched |
| `data.max_response_length=122880` | **32768 (CAP — user decision)** | ❌ excluded |
| `tensor_model_parallel_size=2` | 1 | ❌ needs ≥2 GPUs |
| `ulysses_sequence_parallel_size=8` | 1 | ❌ needs ≥8 GPUs |
| `gpu_memory_utilization=0.70` | 0.30 | ⚠️ paper's 0.70 is per-GPU with 16-GPU FSDP-sharded optimizer (~3 GB/GPU); ours holds the full ~64 GB optimizer on 1 GPU, so vLLM gets less |

KEY: the batching changes are **memory-free** here (peak stays the
deterministic ~128.6 GB regardless of `train_batch_size`, because
`ppo_micro_batch_size_per_gpu=1` and `use_dynamic_bsz=False` keep the actor
backprop at one sequence at a time — bigger batch only adds gradient-
accumulation micro-steps before `optimizer.step()`).

## Dataset comparison vs the paper

**Paper RL training data** (from authors' 4B script
`...timeout3600_4b.sh`, `train_files=...`):
`/code/sunshuang/policy_rl/ss_debug/rllm/rllm/data/datasets/SWE_REBENCH_AND_GYM_OOD_RL_ADD_MAKE_TEST_SPEC/train_00000_verl.parquet`
— **SWE-rebench + SWE-Gym OOD combined**, with `make_test_spec` embedded.
This path is on the authors' internal infra and not publicly available.
Validation data: `SWE_BENCH_VERIFIED_TRY_DEBUG/test_00000_verl.parquet` —
SWE-bench Verified (held-out eval distribution).

**Distribution sizes from README table**:
| dataset | total | usable rows |
|---|---|---|
| SWE-Gym | Real | 2,438 |
| SWE-rebench | Real | 6,542 |

So the paper's training mixture is order-of-thousands of tasks across BOTH
sources.

**Our RL training data**
(`data_examples/rl_data/swerebench_comprehensive/{train,test}_00000_verl.parquet`):
built by `build_comprehensive_swerebench.py` from **`nebius/SWE-rebench`**
(the public mirror; `docker_image` IS already the public Docker Hub
`swerebench/...` image, no rebuild needed).

| dimension | paper | ours |
|---|---|---|
| Source distribution | SWE-rebench + SWE-Gym OOD | **SWE-rebench only** (no SWE-Gym component) |
| Train task count | thousands (combined) | **32** train tasks (filtered: public image exists + shorter problem statement) |
| Val task count | SWE-bench Verified (the eval distribution) | 4 val tasks from same SWE-rebench pool |
| Public images | private `harbor.weizhipin.com/arsenal-ai/swerebench/...` | public Docker Hub `swerebench/...` (verified pullable + nested containers run here) |
| `make_test_spec` embedded | yes | yes (we run the same `mts_swerebench`) |
| Verl schema | original `scripts/data/swe_dataset.py` schema | same |

**Honest diff:**
1. **No SWE-Gym OOD component.** Ours is SWE-rebench-only — a meaningful
   methodological difference vs the paper's mixed distribution. (SWE-Gym has
   public images too but the build_comprehensive script only pulls
   nebius/SWE-rebench; adding the SWE-Gym source is a follow-on enhancement.)
2. **Vastly smaller train set.** 32 tasks vs paper's combined thousands —
   sufficient for `train_batch_size=32` (1 step/epoch) but a smoke-scale
   subset, NOT a full paper-scale training run.
3. **No data leakage:** we do NOT train on SWE-bench (eval distribution);
   prior memory notes flag SWE-bench as eval-only.

**Important caveat on validation:** the paper validates on SWE-bench Verified
(separate dataset). Our val_files draw from the same SWE-rebench pool we
train on — adequate for a smoke-scale learning-signal sanity check but NOT a
fair held-out evaluation. A proper paper-aligned eval would point
`VAL_PARQUET` at a SWE-bench Verified parquet (separate from training).

## Parquet expansion + script default change

To enable `train_batch_size=32`, the train parquet was rebuilt with
`N_TRAIN=32 N_VAL=4 MAX_PROBE=200 build_comprehensive_swerebench.py`
(overwrites the prior 20-task parquet at the same path).

Default `MAX_RESPONSE` in the h200 script is now **32768** (matching the
documented cap; was 16384). Other env overrides unchanged.
