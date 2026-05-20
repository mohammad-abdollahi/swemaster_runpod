# SWE-World-4B-SFT scaled training run — live log

**Run started:** 2026-05-20 06:11 UTC
**Branch:** `reward/active-test-selection`
**Base policy:** `RUC-AIBOX/SWE-World-4B-SFT` (downloaded fresh to
`/home/temp/models/SWE-World-4B-SFT`; identical architecture and tokenizer
to `RUC-AIBOX/SWE-Master-4B-SFT`).
**Launcher:** `bash run_scaled_swe_world.sh > /home/temp/train_active_scaled_sweworld.log`
**Experiment:** `active_tests_swe_world/scaled_sweworld_minftp5_20260520-061041`
**Hardware:** 1 × NVIDIA H200 (143 GB), 196 GB host RAM, docker-capable.

## Configuration

| Knob | Value |
|---|---|
| Train parquet | `data_examples/rl_data/swerebench_min_ftp5/train_00000_verl.parquet` (16 tasks, mean `|F|=11.4`, range 5-37) |
| Val parquet | `swerebench_min_ftp5/test_00000_verl.parquet` (4 tasks) |
| `TRAIN_BATCH_SIZE` | 4 |
| `PPO_MINI_BATCH_SIZE` | 4 |
| `TOTAL_TRAINING_STEPS` | 8 |
| `AGENT_MAX_STEPS` | 50 |
| `MAX_RESPONSE` | 16384 |
| `RLLM_ACTIVE_TESTS_W_SOLVE` | 10.0 |
| `RLLM_ACTIVE_TESTS_LAM_TOKEN` | 0.1 |
| `LR` | 3e-6 |
| `n` (LOOP rollouts/task) | 4 |
| Trajectory timeout | 2400 s |
| Reward path | ATS v1 (terminal partial-credit + token penalty + active-tests prompt block) |

## Per-PPO-step metrics

(Filled from `grep "step:N - "` on the training log.)

| step | grad_norm | rew_mean | rew_max | rew_min | adv_max | adv_min | solve_partial | solve_all | solve_none | wall-clock |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **0.110** | +0.042 | **+1.609** | -0.074 | +1.668 | -0.564 | 1 | 0 | 3 | 593 s |
| 2 | **0.409** | **+1.009** | **+4.949** | -0.074 | **+5.011** | **-4.047** | **2** | 0 | 2 | 655 s |
| 3 | 0.002 | -0.065 | -0.043 | -0.074 | +0.025 | -0.011 | 0 | 0 | 4 | 673 s |
| 4 | TBD | | | | | | | | | |
| 5 | TBD | | | | | | | | | |
| 6 | TBD | | | | | | | | | |
| 7 | TBD | | | | | | | | | |
| 8 | TBD | | | | | | | | | |

## Observed solves (per trajectory)

From `ActiveTests terminal: frac_solved=X` lines:

| step | frac_solved values (16 trajectories) | non-zero count |
|---|---|---|
| 1 | 0.333, 15×0.000 | 1 |
| 2 | **1.000**, **0.857**, **0.857**, **0.714**, 12×0.000 | 4 |
| 3 | 16×0.000 (so far) | 0 |

The step-2 distribution is the headline result so far:
- A trajectory **fully solved** its 6-FTP task (frac_solved=1.000, total reward 9.897).
- Three trajectories achieved 5/6 or 6/7 solves (frac=0.857, 0.857, 0.714).
- 12 trajectories scored 0 (failed all FAIL_TO_PASS).

This produced `critic/rewards/max=4.949` (one full-solve trajectory whose
reward was halved by MAX_STEPS-truncation gating) and grad_norm=0.409 —
a 3.7× jump from step 1's 0.110.

## Step-3 collapse — honest reading

Step 3 saw **zero non-zero rewards across all 16 trajectories**. Advantages
collapsed to near-zero (`max=0.025, min=-0.011`), grad_norm dropped to
0.002 — essentially no PPO update. Two possible explanations:

1. **Task-sampling variance**: with only 16 train tasks and a 4-task batch,
   step 3 happened to draw the 4 hardest tasks (max FAIL_TO_PASS counts
   14, 35, 8, 6 — `35` is the long-tail task `mutalyzer/hgvs-parser-9`
   from `retired=0/35`). The agent could not solve any of them in 50
   steps, so the partial-credit signal degenerated.

2. **Token-penalty equalisation**: when every trajectory's token cost
   ends up in a similar bucket (most trajectories full-truncate at
   16384 response tokens — `response_length/clip_ratio=1.000`), the only
   source of within-group variance — the token term — collapses to a
   near-constant. ATS then offers *no* signal advantage over the binary
   baseline on that step.

Implication for the paper: the "always-on gradient" claim needs a
qualifier — "as long as either solve-rate or token-budget has within-group
variance". On steps where neither does (deep truncation + zero solves
across the board), even ATS contributes no gradient. The binary baseline
would have done exactly the same.

## Wallclock budget

Each PPO step takes ~10-11 minutes. 8 steps = ~85 minutes of
collect_trajectory + ~9 minutes of update_actor + setup. Expected
total: ~95-100 minutes from launch. ETA at write: ~07:50 UTC.

## Files / commands

- Live log: `/home/temp/train_active_scaled_sweworld.log`
- Tensorboard: `tensorboard --logdir /home/temp/swemaster_runpod/tensorboard_log/active_tests_swe_world`
- ATS analyzer: `python analyze_active_tests.py /home/temp/train_active_scaled_sweworld.log`
- Binary counterfactual: `python simulate_binary_baseline.py /home/temp/train_active_scaled_sweworld.log`
- Checkpoints land at `/home/temp/training_outputs/active_tests_swe_world/scaled_sweworld_minftp5_20260520-061041/`
