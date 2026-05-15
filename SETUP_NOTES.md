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
