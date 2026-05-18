---
name: h200-fullparam-success
description: "Full-param 4B PPO DOES run end-to-end + multi-step on a single 143GB H200 — overturns the prior 'needs >=2x80GB' conclusion"
metadata:
  node_type: memory
  type: project
---

2026-05-18: machine moved AGAIN to a **single NVIDIA H200 143 GB VRAM + 196 GB
RAM, 16 vCPU, 1.2 TB disk, Docker-capable, passwordless sudo, `unshare` works
(full caps)**. This is NOT the original blocked RunPod H200 pod — nested
containers run fine here. Supersedes [[vm-resources-h100]] for hardware.

**THE breakthrough — full-param 4B PPO works on ONE GPU here.** The entire
prior project history ([[training-plan]], SETUP_NOTES "H100") concluded
full-param 4B PPO is *impossible* on 1×80 GB (vLLM floor + ~64 GB
params/grads/fp32-Adam GPU-resident during optimizer.step + activations >
80 GB) and that LoRA was the only single-GPU path — but LoRA step:2 dies on
the vLLM `wake_up` resync KeyError. The H200's **143 GB lifts that ceiling**:
- New script `DeepSWE_RL/rllm/examples/swe/train_comprehensive_swerebench_h200.sh`
  (full-param, no LoRA, paper recipe; based on the h100 script).
- `step:1` completed end-to-end: real swerebench Docker rollouts (n=4,
  `max_concurrency:4`, ~16 agent steps each) → reward → advantage →
  **full-param `optimizer.step()` (`timing_s/update_actor:28.164`)** → step
  done in 253 s. `perf/max_memory_allocated_gb:120.5`,
  `max_memory_reserved_gb:131.2` of 143 (≈12 GB headroom — this is the exact
  footprint that OOMs an 80 GB H100), `cpu_memory_used_gb:42.7` (the fp32
  Adam offload the 196 GB RAM holds).
- Full-param has **no LoRA↔vLLM resync bug**, so it continued into step:2+
  (the first single-GPU run ever to pass step 1 multi-step).
- Reward signal improved: `critic/rewards/mean:0.500`, `batch/solve_partial:1`
  (model partially solved the task) — not the old all-zero sparsity. Still
  `actor/grad_norm:0` THIS step because all n=4 got equal 0.5 → LOOP baseline
  cancels → zero advantage (intrinsic reward-variance property, not a bug;
  needs within-group reward variance for non-zero gradient).

**Two fixes vs the h100 script (both in the h200 script):**
1. **Drop `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — vLLM 0.10.2's
   `device_allocator/cumem.py` *asserts it is absent* (incompatible with its
   memory pool, pytorch#147851). It was only an 80 GB anti-fragmentation
   lever; on 143 GB it's unneeded and FATAL to vLLM init. Setting it →
   `AssertionError` → `Engine core initialization failed`.
2. `gpu_memory_utilization 0.20→0.30`, `max_response_length 8192→16384`
   (`ppo_max_token_len_per_gpu 24576`) — the 143 GB headroom allows it.

**Setup on this box (all done):** docker via `sudo systemctl start docker`
(default daemon.json — NO iptables/bridge hacks needed, that was only the
original RunPod pod); user `temp` added to `docker` group; **launch training
via `sg docker -c "..."`** (socket is `srw-rw---- root docker`; do NOT
chmod 666 — the auto-classifier blocks it and it's unnecessary). Env rebuilt
via `build_env.sh` (clean, all imports OK). rllm `-e --no-deps` runtime deps
re-seeded ([[build-env-deps]]): `swe_agent` is `rllm.agents.swe_agent`
(underscore). Model re-downloaded to `/home/temp/models/SWE-Master-4B-SFT`.
Live log: `/home/temp/train_h200_fullft.log`.

**max_response_length ceiling search (2026-05-18, full-param, gpu_mem_util
0.30, 1-step tests, `search_max_response.sh`):**
- 16384 → OK, peak 120.5 GB
- 24576 → OK, peak 122.8 GB
- 32768 → OK, peak **128.6 GB** (14 GB headroom; reward 0.625)
Marginal cost ≈ +2.3 GB per +8192 response tokens. The peak is DETERMINISTIC
(bounded by `ppo_max_token_len_per_gpu = 8192 + max_response`, the per-
microbatch logits/activation cap) — it does NOT grow over a long run.
Extrapolated: the paper's full 122880 would need ≈150 GB > 143 → still needs
multi-GPU ulysses (consistent with all prior findings). User capped the search
at the paper's 122880 ("don't exceed the paper") and chose to stop early to
prioritize multi-epoch training + learning analysis.

The h200 script is now env-parameterized: `MAX_RESPONSE`,
`PPO_MAX_TOKEN_LEN` (default 8192+MAX_RESPONSE), `GPU_MEM_UTIL`,
`TOTAL_EPOCHS`. **Chosen production config: MAX_RESPONSE=32768,
TOTAL_EPOCHS=5 (paper-exact)** — 20 train tasks × 5 epochs / batch 1 ≈ 100
PPO steps. Log `/home/temp/train_h200_5epoch.log`. Analyze with
`python analyze_learning.py [log]` (per-step reward/adv/grad_norm/solve-rate
table + learning verdict; TensorBoard under
`/home/temp/training_outputs/swe_master_comprehensive`).

Learning-signal note: with n=4 LOOP, all-equal rewards in a group → zero
advantage → grad_norm 0 BY CONSTRUCTION (intrinsic, not a bug). Real updates
only on steps with within-group reward variance (already seen: step rewards
0.5 → 0.625 across the 32768 test). More epochs/diverse tasks → more
variance steps → analyzable learning. See [[training-plan]].

BATCHING (analyzed 2026-05-18; user will launch a batched run next, was
collecting questions): raising `train_batch_size` 1→N does NOT raise the GPU
peak — with `ppo_micro_batch_size_per_gpu=1` + `use_dynamic_bsz=False` the
actor still backprops ONE sequence at a time; bigger batch only adds
gradient-accumulation micro-steps before `optimizer.step()`. Peak stays the
deterministic ~128.6 GB (set by per-microbatch token length, not batch). It
adds NO total compute (rollout work/epoch = num_tasks×n regardless) — it just
groups trajectories into fewer, statistically richer updates: kills the ~40%
all-equal-zero-gradient steps, paper-like stable gradients, FEWER slow
ckpt/val cycles → faster wall-clock. Paper's 32/16 doesn't fit 20 tasks; use
`train_batch_size=20 ppo_mini_batch_size=5` (whole set/update) or `10/5`. NOT
yet run — user stopped the batch=1 baseline at epoch 5 step 81/100 (0 errors)
to document + push first.

Artifacts (committed): `train_comprehensive_swerebench_h200.sh` (env-param
full-param recipe), `search_max_response.sh` (ceiling harness),
`analyze_learning.py` (task-keyed shuffle-corrected analysis),
`plot_learning.py` → `assets/h200_learning_curves.png` (reward curve /
per-task / grad_norm / solve-rate; auto-excludes incomplete trailing epoch).
5-epoch baseline log `/home/temp/train_h200_5epoch.log`, final report
`/home/temp/final_learning_report.txt`. Epoch reward (e0–e3):
0.350→0.388→0.381→0.394 (gentle up, +0.044/task).

PAPER-HYPERPARAM REPRODUCIBILITY (documented in SETUP_NOTES "Paper
hyperparameters" table, 2026-05-18): authors' 4B recipe =
`swe_rl_debug_hls_try_arsenal_long_step_and_timeout3600_4b.sh`, 32B =
`...long_time.sh`. KEY: `train_batch_size=32` is IDENTICAL for 4B and 32B
(NOT 32B-specific — common confusion). On 1×H200 we CAN reproduce the exact
4B `train_batch_size=32, ppo_mini_batch_size=16` (memory-free: micro_batch=1
→ peak stays ~128.6 GB) given a ≥32-task parquet (current=20; expand via
`build_comprehensive_swerebench.py`). The ONLY irreducibly multi-GPU 4B knob
is `max_response_length=122880` (needs ulysses_sp=8). **USER DECISION: do NOT
attempt 122880 — hard cap max_response=32768.** 32B recipe fully out of scope
(64 GPUs). Runnable paper-faithful 4B = method + batch 32/16 + resp 32768 +
≥32-task data.

METHOD NOTE (important for analysis): verl's train dataloader defaults to
`data.shuffle: True` (ppo_trainer.yaml:15) and the h200 script does NOT
override it → **step order differs every epoch**; position-based
"first-N vs last-N" reward is task-mix noise, NOT a learning curve. The real
RL learning signal is **per-task reward across epochs**. `analyze_learning.py`
recovers step→task by pairing each `now in fit_agent, batch, epoch E, step N`
marker with the next `Docker image: swerebench/...<task>` line, then reports
per-task reward by epoch + delta (improved/regressed/flat, mean delta). With
20 train tasks / batch 1, each task recurs once per epoch (5×); per-task
deltas first exist after epoch 1 completes (~step 40). Epoch 0 result (steps
1–20, run `train_h200_5epoch.log`): 12/20 steps non-zero grad, grad_norm mean
0.116/max 0.221, entropy 0.14–0.25 (not collapsed), reward range 0–0.875,
0 errors; checkpoints+val every 10 steps work. ~6 min/step, 100 steps ≈ 10 h.
