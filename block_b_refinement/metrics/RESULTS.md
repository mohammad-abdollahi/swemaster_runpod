# Block B1 / R12 — Backtranslation Test Refinement: Results

**Branch:** `test-refinement-back-translation`
**Run:** `B1_r12_verlive` (project `ablation_b1_r12`)
**Dataset:** `swerebench_verlive` — 80 instances (SWE-Verified + SWE-Live overlap), 64 train / 16 val
**Final model:** `/home/temp/models/SWE-Master-4B-ATS-B1-r12` (8.3 G, exported 2026-06-22 23:52)

## What this is

ATS-RL + **online backtranslation test-refinement**: when a FAIL_TO_PASS test stalls
(unsolved after `refine_after` ticks at `k_steps=20`), the gold-patch-derived decomposition
is injected into the prompt as `<subtest>` blocks. Refinement is **prompt-only** — it never
changes the reward.

## Training runs (cumulative)

1. **Smoke run** (2026-06-21): 4 PPO steps from `SWE-Master-4B-SFT`, 1 epoch.
   Model preserved at `/home/temp/models/SWE-Master-4B-ATS-B1-r12-smoke4step`.
2. **Continuation run** (2026-06-22): **5 PPO steps** continued on top of the 4-step model
   (`BASE_MODEL=...smoke4step`), then final validation. → **~9 cumulative PPO steps.**
   (Configured `total_training_steps=6`; verl performed 5 update steps + the final validation
   pass on the 16-instance val set — the "slow step 6" observed during monitoring was that
   validation, not a 6th training step.)

## Result — validation

**`val/test_score/swe = 0.1875`** (3/16 held-out instances solved) at end of the continuation run.
For reference, the earlier Block-B B2/R22 run scored val 0.125 (different setup; not a controlled
comparison).

## Learning signal — continuation reward trend (`critic/rewards/mean`)

| PPO step (continuation) | Mean reward |
|---|---|
| 1 | −0.254 |
| 2 | **+1.141** |
| 3 | +0.167 |
| 4 | +0.163 |
| 5 | +0.125 |

After the step-2 spike the mean settles steadily around **+0.13** — consistently positive vs the
≈−0.09 single-batch baseline. Top individual trajectory reward **+9.86**. Optimizer reset at the
continuation start (HF-weight load, not full-state resume) explains the step-1 dip before recovery.

## Correctness guarantees (verified from logs)

- **v1 reward preserved exactly:** **336 / 336** `ATS terminal` episodes logged `dense_sum=0.000`
  (0 nonzero). The backtranslation refinement is provably prompt-only and never enters the reward path.
- **Refinement engaged:** loaded on all 336 episodes; mid-episode ticks fired up to `maxticks=7`
  (decomposition triggered for stalled FAIL_TO_PASS tests at steps 20/40/…/140 of long episodes).
- **Mechanism = design:** stall-detect tick (k=20) → gold-patch decomposition into `<subtest>`
  prompt blocks → unchanged terminal reward.

## Config

| Knob | Value |
|---|---|
| Refinement | `RLLM_ACTIVE_TESTS_REFINE=1`, `REFINE_TICK=1` |
| Reward mode | **v1** (`MID_EPISODE=0`), terminal `w_solve·frac_solved`, `w_solve=10` |
| k_steps | 20 |
| Batch / mini-batch | 16 / 16 |
| Continuation base | `SWE-Master-4B-ATS-B1-r12-smoke4step` |
| Agent max steps / traj timeout | 150 / 5400 s |
| Teacher decomposer | OpenAI `gpt-5.4-mini` (offline cache `block_b_refinement/cache/decompositions.jsonl`) |

## Ops

- Stable to completion (one early vLLM `EngineDeadError` on the very first attempt; orchestrator
  auto-relaunched; no checkpoint lost). 0 OOM. Miners kept dead by `miner_killer.sh` watchdog.
- **Dataset image cache:** all **80 verlive images** (SWE-Verified + SWE-Live) pulled to local
  Docker via `pull_dataset_images.sh` (62 pulled, 0 failed) — no mid-rollout re-pull stalls for
  future trainings.
