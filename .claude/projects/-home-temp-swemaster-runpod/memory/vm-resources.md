---
name: vm-resources
description: Hardware/resources of the current swemaster VM and how it differs from the prior H200 pod
metadata: 
  node_type: memory
  type: project
  originSessionId: 64bc627f-be17-4bd4-9bc9-9c7d614fe938
---

Current machine (the one Claude runs on now): repo at `/home/temp/swemaster_runpod`, **1× NVIDIA L40S 46 GB VRAM**, CUDA 13.0 (driver 580.126.09), 8 vCPU, 31 GB RAM, 1.2 TB free disk. Docker 29.4.2 daemon **active**; `sudo` is passwordless.

CRITICAL: this is NOT a fresh project. Substantial prior work was done on a **different RunPod 1×H200 (143 GB)** pod — fully captured in [SETUP_NOTES.md](../../../../swemaster_runpod/SETUP_NOTES.md) and git branch `setup/runpod-h200` (commits 54857dc, 8a58ed9; the 6 import patches + train_agent_ppo.py runtime_env gate are already committed). On the H200 the RL trainer ran end-to-end up to the first rollout, then hit a HARD blocker: RunPod's kernel blocked `unshare()` so Docker containers could not start.

**Why this VM exists:** to get past that Docker blocker. Verified here: `unshare --user --pid` works and PID 1 has full caps (`CapEff: 000001ffffffffff`). Nested containers ARE possible here. The H200 venv/model/`/workspace` do NOT exist here — env must be rebuilt on `/home/temp`.

Resource downgrade vs H200: 46 GB vs 143 GB VRAM. The committed `smoke_test_4b_single_h200.sh` used ~110 GB (vLLM ~94 GB + FSDP ~16 GB) and will OOM here — needs an L40S-tuned variant (lower gpu_memory_utilization, shorter seqs). 32B SFT/RL still infeasible; only 4B downscaled. See [[training-plan]].
