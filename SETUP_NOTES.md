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
