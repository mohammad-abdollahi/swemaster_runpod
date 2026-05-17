---
name: vm-resources-h100
description: Current swemaster machine is now a 1xH100 80GB with 196GB RAM (replaced the L40S); RAM constraint gone
metadata:
  node_type: memory
  type: project
---

As of 2026-05-17 the swemaster work moved AGAIN, to a new machine. Repo still at `/home/temp/swemaster_runpod`, branch `setup/runpod-h200`. This supersedes [[vm-resources]] (which described the L40S).

Current machine: **1× NVIDIA H100 80GB HBM3**, CUDA 12.8 (driver 580.126.09 / CUDA 13.0), 16 vCPU, **196 GB RAM**, 1.2 TB free disk, passwordless sudo. Docker 29.4.2; `unshare`/nested containers work (PID1 `CapEff: 000001ffffffffff`) — the original H200 `unshare` blocker does NOT apply here.

Key change vs L40S: **196 GB RAM (was 31 GB) and 80 GB VRAM (was 46 GB).** The host-RAM OOM that forced LoRA + sequential rollouts + the 96 GB swapfile on the L40S is no longer a hard constraint here. The committed L40S-tuned `train_comprehensive_swerebench_l40s.sh` (LoRA r32, gpu_mem_util 0.38, `RLLM_MAX_ROLLOUT_CONCURRENCY=1`) still runs fine and is the proven recipe; scaling up (drop LoRA / full-FT, raise concurrency / gpu_mem_util / response len) is now feasible on this box if needed — see [[training-plan]] hardware-fit notes.

Env rebuilt fresh at `/home/temp/venv` via `build_env.sh` (nothing carried over from L40S/H200). Model re-downloaded to `/home/temp/models/SWE-Master-4B-SFT`. Datasets are committed in-repo.

Build gotchas hit on this machine (both fixed): see [[build-env-deps]].
