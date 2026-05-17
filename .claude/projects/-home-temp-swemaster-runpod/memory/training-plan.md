---
name: training-plan
description: "What we're actually running for SWE-Master and the resume plan on the L40S VM"
metadata: 
  node_type: memory
  type: project
  originSessionId: 64bc627f-be17-4bd4-9bc9-9c7d614fe938
---

Goal: run the SWE-Master **RL** pipeline (DeepSWE_RL/rllm verl trainer) end-to-end as a downscaled single-GPU smoke test, getting the agent rollout (per-task Docker SWE env via r2egym) to actually execute — the step that was blocked on the H200.

State of prior work (see [SETUP_NOTES.md](../../../../swemaster_runpod/SETUP_NOTES.md)): on the H200 the trainer reached `fit_agent` → `run_agent_trajectory_async` and died at `docker.errors.TLSParameterError` then `unshare: operation not permitted`. Two rollout-boundary issues noted there: (1) `r2egym/agenthub/runtime/docker.py` hardcodes `DOCKER_TLS_VERIFY=1` / a bogus `DOCKER_CERT_PATH` → must make env-driven/unset; (2) container start needs working `unshare` (works on THIS VM).

Resume plan on this L40S VM:
1. Docker access for user `temp` (group/socket), verify nested `unshare` inside a container.
2. Rebuild env on `/home/temp` per SETUP_NOTES recipe (uv venv py3.11, torch cu128, **agentica fork of verl** not volcengine, rllm/r2egym editable, vllm 0.10.2, flash-attn 2.8.3, setuptools<81, transformers<5, tensorboard).
3. Download `RUC-AIBOX/SWE-Master-4B-SFT`.
4. L40S-tuned smoke script (46 GB: lower gpu_memory_utilization ~0.35, short seqs, local model + in-repo demo parquet `data_examples/rl_data/ood_data/.../train_00000_verl.parquet`).
5. Launch; fix the `r2egym` docker TLS hardcode at the rollout boundary; push past it.

Progress on this VM (2026-05-17): docker works for user `temp` (added to docker group + `chmod 666 /var/run/docker.sock`; nested containers start, unlike H200). Env rebuilt at `/home/temp/venv` (torch 2.8.0+cu128, vllm 0.10.2, flash-attn 2.8.3, transformers 4.57.6, agentica verl fork). Model at `/home/temp/models/SWE-Master-4B-SFT`. L40S script: `DeepSWE_RL/rllm/examples/swe/smoke_test_4b_single_l40s.sh` (gpu_mem_util 0.38, resp 3072, pre-starts Ray with tiny object store + RAY_memory_monitor_refresh_ms=0, RLLM_PROJECT_ROOT set). rllm needs lightweight deps installed manually (it's `-e --no-deps`): polars dm-tree datasets accelerate peft etc. r2egym docker TLS hardcode patched (env-driven, local socket when no remote ip).

KEY new constraint: 31 GB host RAM (H200 had ~2 TB). FSDP param/optimizer CPU offload of the 4B model OOMs the host at init (Ray killed WorkerDict at ~22 GB). Fix applied: **96 GB swapfile** at `/swapfile` (`sudo swapon`) so offload spills to the 1.2 TB disk — slow but lets the smoke run survive init/optimizer RAM peaks. Note: `actor_rollout_ref.rollout.swap_space` is NOT a valid verl config key (struct mode rejects it).

RESOLVED blockers (2026-05-17, all on branch setup/runpod-h200):
- Docker rollout (the H200 killer) WORKS here: containers start; `temp` added to docker group + `chmod 666 /var/run/docker.sock`.
- Demo RL parquet images are private (`harbor.weizhipin.com`) → UNPULLABLE. Public SWE-bench Verified images on Docker Hub work: `swebench/sweb.eval.x86_64.<id with __→_1776_>:latest`. Built `build_public_swe_parquet.py` → `data_examples/rl_data/public_swebench_verified/{train,test}_00000_verl.parquet` (instance astropy__astropy-12907, original swe_dataset.py schema).
- r2egym `docker.py`: TLS hardcode made env-driven + local-socket; **scoping bug** fixed (renamed local `make_test_spec`→`embedded_test_spec` at lines 203/227/249 so module import at L239 resolves).
- Host-RAM OOM (31 GB): full-param PPO of 4B needs ~40 GB CPU-offload RAM; pinned mem so swap can't help. FIX = **LoRA** (`actor_rollout_ref.model.lora_rank=32`, offload OFF) → optimizer is MB, RAM ~14 GB, no OOM. Also capped rollout concurrency to 1 via `RLLM_MAX_ROLLOUT_CONCURRENCY` env (patched rllm/engine/agent_execution_engine.py) — 31 GB can't run 2 concurrent SWE containers.
- `ppo_max_token_len_per_gpu` must be ≥ max_prompt+max_response (8192+2048=10240); set 12288.

Working run script: `DeepSWE_RL/rllm/examples/swe/train_public_swebench_l40s.sh` (original loop/GRPO pipeline, LoRA, n=2 sequential, real astropy SWE-bench container).

OUTCOME (2026-05-17, definitive): **GRPO PPO `step:1` completed end-to-end** — real container rollout → test-based reward → advantage → LoRA actor update → full metrics (saved `/home/temp/train_public_step1_SUCCESS.log`). GPU peak 40.7/46 GB, host RAM 19.8 GB (LoRA fix holds). Metrics: both n=2 rollouts reward 0.0 (TRUNCATION, model didn't solve astropy__astropy-12907 in 20 steps); `critic/rewards/mean=0`, `critic/advantages/mean=0`, `actor/pg_loss=0`, `actor/grad_norm=0`. VERDICT: pipeline 100% functional but **model is NOT learning** — single hard task → uniform-zero reward → zero GRPO advantage → zero gradient. Not a bug; needs reward variance (easier task / more steps / many diverse tasks). Secondary: verl LoRA+vLLM resync bug (`fsdp_vllm.py __collect_lora_params` in sharding_manager wake_up) blocks step 2+ — would need a fix for multi-step LoRA runs.

DATASET CORRECTION (user-flagged, important): SWE-Master trains on **SWE-rebench + SWE-Gym OOD**, NOT SWE-bench (SWE-bench Verified is the held-out `val_files` — training on it = eval contamination). The bundled `data_examples/rl_data/ood_data/equal_..._swe-gym_and_rebench_demo_for_RL/train_00000_verl.parquet` (3 swe-rebench tasks) IS the real training distribution. Its private `harbor.weizhipin.com/arsenal-ai/swerebench/sweb.eval.x86_64.X` images have a PUBLIC Docker Hub mirror: `swerebench/sweb.eval.x86_64.X:latest` (verified pullable for all 3: fair-workflows__nanopub-140, fairlearn__fairlearn-636, falconry__falcon-2254). `build_public_swerebench_train.py` rewrites only the image ref → `data_examples/rl_data/public_swerebench_train/{train,test}_00000_verl.parquet` (make_test_spec embedded → r2egym swerebench branch uses it, no fetch). Train script repointed; PROJECT swe_master_swerebench_ood. The earlier SWE-bench parquet was wrong (eval set) — do not use for training.

COMPREHENSIVE PIPELINE (2026-05-17): fixed a real verl bug blocking multi-step LoRA — `DeepSWE_RL/rllm/verl/verl/workers/sharding_manager/fsdp_vllm.py` lines 141 & 155 did `'cpu' in next(model.parameters()).device` (TypeError: torch.device not iterable) → changed to `.device.type == 'cpu'`. Without this every LoRA run dies at step 2 (vLLM resync). Data: `build_comprehensive_swerebench.py` pulls nebius/SWE-rebench (its `docker_image` IS already public `swerebench/...`), filters to image-verified + short-problem tasks, embeds make_test_spec → `data_examples/rl_data/swerebench_comprehensive/{train(20),test(4)}_00000_verl.parquet`. Run script `DeepSWE_RL/rllm/examples/swe/train_comprehensive_swerebench_l40s.sh`: rollout.n=4 (GRPO group), train_batch=1, total_epochs=2 (~40 PPO steps), test_freq=10, save_freq=10 (LoRA ckpts), LoRA r32, sequential rollout, all prior fixes. Log: /home/temp/train_comprehensive.log.

User wants autonomous progress ("continue your work"); don't over-ask — direction is fixed. See [[vm-resources]].
