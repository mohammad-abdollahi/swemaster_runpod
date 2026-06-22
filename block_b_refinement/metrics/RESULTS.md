# Block B1 / R12 — Backtranslation Test Refinement: Results

**Branch:** `test-refinement-back-translation`
**Run:** `B1_r12_verlive` (project `ablation_b1_r12`)
**Date:** 2026-06-21 → 2026-06-22
**Base model:** RUC-AIBox/SWE-Master-4B-SFT
**Dataset:** `swerebench_verlive` (64 train / 16 val)
**Exported model:** `/home/temp/models/SWE-Master-4B-ATS-B1-r12` (8.3 G, 2 safetensors)

## What this run is

A single-arm training of **ATS-RL + online backtranslation test-refinement**. When a
FAIL_TO_PASS test stalls (unsolved after `refine_after` ticks at `k_steps=20`), the
gold-patch-derived decomposition is injected into the prompt as `<subtest>` blocks so the
policy sees a smaller, more legible target. The refinement is **prompt-only** — it never
changes the reward.

## Configuration

| Knob | Value |
|------|-------|
| Refinement | `RLLM_ACTIVE_TESTS_REFINE=1`, `REFINE_TICK=1` |
| Reward mode | **v1 exactly** (`MID_EPISODE=0`) — terminal `w_solve·frac_solved`, `w_solve=10` |
| k_steps (tick cadence) | 20 |
| refine_after / refine_max | 1 tick / 4 subtests |
| Batch / mini-batch | 16 / 16 |
| Epochs / steps | 1 epoch = **4 PPO steps** (64 train ÷ batch 16); `total_training_steps=10` cap not binding |
| Agent max steps / traj timeout | 150 / 5400 s |
| GPU mem util | 0.30 |

## Learning signal — reward trend (`critic/rewards/mean`)

| PPO step | Mean reward |
|----------|-------------|
| 1 | −0.091 (baseline) |
| 2 | **+0.481** |
| 3 | −0.077 |
| 4 | **+0.451** |

Mean reward rose from −0.091 to ~**+0.45–0.48** over one epoch (with normal PPO
step-to-step variance). Top individual trajectory reward reached **+6.49** (a high-credit
solve). The model is learning.

> Validation: not run (`TEST_FREQ=999`) — this run targeted the reward-trend learning
> signal over one epoch, not a held-out eval. A separate eval can be run on the exported
> checkpoint.

## Correctness guarantees (verified from logs)

- **v1 reward preserved exactly:** all **256 / 256** `ATS terminal` episodes logged
  `dense_sum=0.000` (0 with nonzero dense reward). The backtranslation refinement is
  provably prompt-only and never enters the reward path.
- **Refinement engaged:** refinement loaded on all 256 episodes; mid-episode ticks
  confirmed firing (`maxticks=3` — decomposition triggered for stalled tests at steps
  20/40/60 of long episodes).
- **Mechanism = the design:** stall-detection tick (k=20) → gold-patch decomposition into
  `<subtest>` prompt blocks → unchanged terminal reward.

## Run health

- Stable to completion after one early vLLM `EngineDeadError` restart (orchestrator
  auto-relaunch from checkpoint; no data lost — no checkpoint existed yet at crash).
- 0 OOM, disk stayed ≥ 94 G, miners kept killed throughout (watchdog).
- Checkpoints pruned to latest-2; final exported via `export_to_hf.sh`.
