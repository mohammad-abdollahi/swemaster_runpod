# SWE-World branch — RL training without docker

This branch (`setup/swe-world-rl`) merges
[RUCAIBox/SWE-World](https://github.com/RUCAIBox/SWE-World) into our repo so
that, on a machine with enough VRAM, the RL training pipeline can run
**without docker**. SWE-World replaces the docker sandbox with two
LLM-served *surrogate world models*:

- **SWT** — Simulated World Transition. A 32B Qwen2-arch model that predicts
  the `(stdout, stderr, exit_code)` of bash commands the agent runs.
- **SWR** — Simulated World Reward. A 32B Qwen2-arch model that predicts
  test-pass/fail and the binary task-success signal.

Both are served via vLLM (ports 8012 / 8013 by upstream convention) and the
training agent talks to them over an OpenAI-compatible HTTP API via litellm.

Models:
- SWT: https://huggingface.co/RUC-AIBOX/SWE-World-SWT-32B-wo-cot (65 GB bf16)
- SWR: https://huggingface.co/RUC-AIBOX/SWE-World-SWR-32B-w-cot (~65 GB bf16)

## Hardware reality on this RunPod pod

- We have **1 × H200 (143 GB VRAM)** and a 50 GB persistent-volume quota.
- A 4 B actor with vLLM + FSDP already uses ~95 GB of the H200.
- Two 32 B world models, even AWQ-int4, need ~18-20 GB each → ~40 GB.
- Total VRAM ~135 GB — tight but possible.
- **Disk quota is the binding constraint:** SWT + SWR raw bf16 = 130 GB, and
  our 50 GB quota cannot hold them. Confirmed by `OSError: [Errno 122] Disk
  quota exceeded` after pulling ~14 GB of SWT.

So the docker-free pipeline **cannot run end-to-end on this specific RunPod
pod** either. It needs:
1. ≥ 150 GB of disk (or stream the AWQ-quantized models direct from HF), and
2. ≥ 150 GB of VRAM, or a multi-GPU host where the actor and world models can
   sit on different GPUs.

This branch is a *restore point* for moving to such a host.

## What's already done on this branch

- `swe_world/` — the SWE-World code copied verbatim from upstream, including:
  - `swe_world/src/r2egym/agenthub/environment/sim_env.py` — `SimulatedEnv`
    that uses `LocalRuntime` instead of `DockerRuntime`.
  - `swe_world/src/r2egym/agenthub/agent/simulator.py` — `SimulatorAgent`
    that drives SWT/SWR via litellm.
  - `swe_world/src/r2egym/agenthub/runtime/local.py` — local file-system /
    git runtime that does NOT need docker.
  - `swe_world/scripts/launch_world_models.sh` — upstream's vLLM launcher
    for SWT and SWR.
  - `swe_world/src/simulation/models_yaml/example_world_models.yaml` — the
    config that points the SimulatorAgent at the SWT/SWR endpoints.
- `DeepSWE_RL/rllm/examples/swe/run_swe_rl_swe_world.sh` — upstream's RL
  training launcher, copied so you can adapt it.
- `DeepSWE_RL/rllm/rllm/environments/swe/swe_sim_swe_world.py` — the rllm
  `SWEEnv` wrapper that calls `SimulatedEnv` (the same class file ships in
  our `rllm/environments/swe/swe_sim.py` but the upstream version has
  imports we need to verify against the installed `r2egym`).
- `r2e-gym-fork` package is installed editable from `swe_world/`. It is a
  **superset** of the previous `R2E-Gym/` package — it has both
  `DockerRuntime` (for the docker pipeline on `setup/runpod-h200`) and
  `LocalRuntime` (for the docker-free pipeline here). `import r2egym`
  resolves to `swe_world/src/r2egym/__init__.py`.

Verified imports (with the venv in this repo's bootstrap recipe):
```python
from r2egym.agenthub.environment.sim_env import SimulatedEnv      # OK
from r2egym.agenthub.agent.simulator import SimulatorAgent        # OK
from r2egym.agenthub.runtime.local import LocalRuntime            # OK
from r2egym.agenthub.runtime.docker import DockerRuntime          # OK (kept)
```

## What's left to do on a docker-free, larger-disk host

The numbers below are the upstream defaults for an 8×8 H800 cluster. Scale
them down per host but the *steps* are the same.

### 1. Download SWT and SWR

```bash
source /workspace/venv/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=1
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='RUC-AIBOX/SWE-World-SWT-32B-wo-cot',
                  local_dir='/workspace/models/SWE-World-SWT-32B-wo-cot',
                  max_workers=8)
snapshot_download(repo_id='RUC-AIBOX/SWE-World-SWR-32B-w-cot',
                  local_dir='/workspace/models/SWE-World-SWR-32B-w-cot',
                  max_workers=8)
"
```
~130 GB total.

### 2. Quantize to AWQ-int4 (only if VRAM is tight)

If the host has multiple H100s/H200s, you can skip quantization and serve
bf16. On 1 × H200 you need AWQ.

```bash
pip install autoawq
python - <<'EOF'
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
for model_id, out_dir in [
    ("/workspace/models/SWE-World-SWT-32B-wo-cot",
     "/workspace/models/SWE-World-SWT-32B-AWQ"),
    ("/workspace/models/SWE-World-SWR-32B-w-cot",
     "/workspace/models/SWE-World-SWR-32B-AWQ"),
]:
    model = AutoAWQForCausalLM.from_pretrained(model_id, device_map="auto")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model.quantize(tok, quant_config={"w_bit": 4, "q_group_size": 128, "zero_point": True})
    model.save_quantized(out_dir)
    tok.save_pretrained(out_dir)
EOF
```
~1-2 hours per 32B model on a single H200.

### 3. Serve SWT and SWR via vLLM

Adapted from `swe_world/scripts/launch_world_models.sh`. For 1 × H200 + a 4B
actor you'd run:

```bash
# SWT on port 8012, ~20 GB VRAM
nohup vllm serve /workspace/models/SWE-World-SWT-32B-AWQ \
    --tensor-parallel-size=1 \
    --gpu-memory-utilization=0.15 \
    --served-model-name swt \
    --host 0.0.0.0 --port 8012 \
    --dtype auto --quantization awq --enable-prefix-caching \
    > /workspace/swt.log 2>&1 &

# SWR on port 8013, ~20 GB VRAM
nohup vllm serve /workspace/models/SWE-World-SWR-32B-AWQ \
    --tensor-parallel-size=1 \
    --gpu-memory-utilization=0.15 \
    --served-model-name swr \
    --host 0.0.0.0 --port 8013 \
    --dtype auto --quantization awq --enable-prefix-caching \
    > /workspace/swr.log 2>&1 &
```

Then point `simulator_yaml` at a config that talks to these endpoints:

```yaml
# swe_world/src/simulation/models_yaml/local_world_models.yaml
swt:
  - model_name: "openai/swt"
    base_url: "http://127.0.0.1:8012/v1"
    api_key: "EMPTY"
swr:
  - model_name: "openai/swr"
    base_url: "http://127.0.0.1:8013/v1"
    api_key: "EMPTY"
```

### 4. Run the RL training

Start from `DeepSWE_RL/rllm/examples/swe/run_swe_rl_swe_world.sh` and apply
the same 1×H200 shrinks that
`DeepSWE_RL/rllm/examples/swe/smoke_test_4b_single_h200.sh` already uses
(nnodes=1, n_gpus=1, TP=1, ulysses_sp=1, batch=1, max_response_length=4096,
enforce_eager=True). The key swap is the env block:

```bash
env.name=swe_sim \
+env.env_args.backend="simulated" \
+env.env_args.simulator_yaml="$(pwd)/swe_world/src/simulation/models_yaml/local_world_models.yaml" \
+env.env_args.sim_reward_max_workers=2 \
+env.env_args.scaffold="openhands" \
```

The actor + FSDP + vLLM bring-up reuses the patches already on
`setup/runpod-h200`:
- `RLLM_USE_PARENT_VENV=1` to skip Ray's broken pip clone
- `agentica-project/verl` fork in `DeepSWE_RL/rllm/verl/`
- `setuptools<81`, `transformers<5,>=4.45`, `huggingface_hub<1`,
  `tensorboard` installed

### 5. Observe rewards

`fit_agent` writes per-batch rollout info under
`trainer.rollout_data_dir`. The relevant fields are the final-step rewards
returned by `SimulatedEnv._calculate_simulated_reward_swebv` (binary 0/1
per task) plus the per-step trajectory rewards. Watch the running mean
across steps to know whether the policy is improving.

## What's *not* committed on this branch

- The two world-model weights (130 GB) — too big.
- The Python venv at `/workspace/venv` (~21 GB) — see the bootstrap recipe
  in `SETUP_NOTES.md`. The only extra dependency for SWE-World is
  `autoawq` if you need to quantize.
- `swe_world/.gitignore` is preserved upstream.

## Related branches

- `setup/runpod-h200` — the original docker-backed pipeline. Patches there
  are also needed here (the train_agent_ppo runtime-env fix, the verl fork,
  the package pins). They've already been carried into this branch.
- `master` — initial commit, no patches.
