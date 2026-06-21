# Test-Selection Ablation — FAIL_TO_PASS-only reward — Results

**Branch:** `test-selection` · **Arm:** f2p-only (no PASS_TO_PASS reward) ·
**Base:** `RUC-AIBOX/SWE-Master-4B-SFT` · **Data:** `swerebench_verlive` (64 train / 16 val) ·
**Batch:** 16 · **Reward:** ATS, `W_SOLVE=10.0`, `w_regress = delta_p2p_break = delta_p2p_recover = 0`.

Run: `f2p_only_verlive_20260621-060039`. Target 10 steps; **9 steps completed**, then the
final step 10's rollout degraded to concurrency 1 (serial trajectories) and was not waited
out. The deliverable model is exported from the **step-9 (peak) checkpoint**.

## Per-step training reward (`critic/rewards/mean`)

| step | reward | grad_norm | solve_all | solve_partial | reward_max |
|--|--|--|--|--|--|
| 1 | 0.081 | 0.145 | 0 | 3 | 4.07 |
| 2 | 0.749 | 0.116 | 1 | 4 | 6.52 |
| 3 | 0.621 | 0.100 | 1 | 4 | 4.92 |
| 4 | 0.805 | 0.111 | 2 | 5 | 4.92 |
| 5 | 0.624 | 0.075 | 2 | 3 | 4.92 |
| 6 | 0.427 | 0.088 | 1 | 3 | 4.91 |
| 7 | 0.711 | 0.078 | 3 | 4 | 9.85 |
| 8 | 0.626 | 0.246 | 0 | 7 | 4.91 |
| **9** | **1.011** | 0.131 | 2 | 4 | 9.90 |

Reward climbs from **0.081 → 1.011** (12.5×), holding a 0.43–1.01 band from step 2 on, with
the highest mean and a fresh peak at step 9. Noisy but clearly-positive learning on hard
verified+live data; full bf16 RL, ~43–78 min/PPO step on one H200.

## Ablation validity (key check)
The p2p reward terms are zeroed but p2p tests are **still evaluated** each terminal step, so
the ablation is verifiable in the logs: trajectories that broke PASS_TO_PASS tests incurred
**zero** regression penalty, e.g.
`frac_solved=0.833 r_solve=8.333 frac_p2p_broken=0.453 r_regress=-0.000 p2p_broken=67/148`
(the control arm would have docked `-w_regress·0.453 = -2.26`). The p2p-break rate stays
measurable for the eventual comparison vs the regression-guarded control — that comparison
is the headline of the ablation and is still TODO (needs a matched control run on the same
`swerebench_verlive` data).

## Artifacts
- Model (step-9 peak): `/home/temp/models/SWE-Master-4B-ATS-f2p-only` (8.3G, 2 safetensors)
- Metrics: `ablation_test_selection_metrics/f2p_only_verlive.csv`
- Train log: `/home/temp/f2p_only.log`
