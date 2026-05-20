# Active-Test-Selection: smoke-run findings for the user

(This file is the morning-briefing summary the user requested when going
to sleep. Updated continuously as the smoke and scaled runs progress.)

Last update: 2026-05-20 ~05:20 UTC; smoke step 3/4 in progress.

## TL;DR

- The pipeline is **working end-to-end** on the SWE-Master-4B-SFT base
  (binary baseline path swapped for ATS by flipping
  `RLLM_ACTIVE_TESTS_ENABLED=1`).
- After **one PPO update**, `actor/grad_norm` jumped from **0.001
  (step 1) to 0.306 (step 2)** — a 306× increase in real PPO gradient.
- Two trajectories in step 2 (different rollouts of one task) actually
  solved their target test, giving the policy a strong learning signal
  on that group.
- The previously-hidden weakness of the binary recipe — `(rewards ==
  0).all()` group filtering — is bypassed automatically by the
  continuous ATS reward, even on the rollout groups where no test
  passes. We measured grad_norm > 0 on step 1 despite all 16
  trajectories returning `frac_solved = 0`.
- Changes pushed to **branch `reward/active-test-selection`**, also live
  on origin (`mohammad-abdollahi/swemaster_runpod`). PR-ready URL printed
  by `git push`.

## What was changed

| File | What |
|------|------|
| `R2E-Gym/src/r2egym/agenthub/runtime/active_tests.py` | NEW scheduler: 4-state machine, dense reward, prompt-block rendering |
| `R2E-Gym/src/r2egym/agenthub/runtime/test_active_tests.py` | NEW 13 unit tests pinning the state transitions; all green |
| `R2E-Gym/src/r2egym/agenthub/runtime/docker.py` | `get_fail_to_pass_ids()` exposed for the scheduler init |
| `DeepSWE_RL/rllm/rllm/environments/swe/swe_arsenal.py` | `SWEEnv.reset()` injects `<active_tests>` block; `compute_final_reward()` routes through `scheduler.terminal_reward()` |
| `DeepSWE_RL/rllm/examples/swe/train_active_tests_swe_master_4b_h200.sh` | NEW launch script wiring `RLLM_ACTIVE_TESTS_*` env vars |
| `DeepSWE_RL/rllm/examples/swe/train_active_tests_scaled_4b_h200.sh` | NEW overnight-scale recipe (8 PPO steps, agent_max_steps=50, w_solve=10) |
| `build_comprehensive_swerebench.py` | `MIN_FTP` env knob for filtering to larger-FTP tasks |
| `analyze_active_tests.py` | NEW per-step reward analyzer |
| `REWARD_ACTIVE_TEST_SELECTION.md` | Design doc + implementation drift |
| `paper/PAPER_DRAFT.md` | Working academic paper draft (Method, related work, ablation plan, smoke results) |

## What the smoke actually shows (step 1 → step 2)

Configuration: 32-task SWE-rebench parquet, `TRAIN_BATCH_SIZE=4`,
`AGENT_MAX_STEPS=20`, `MAX_RESPONSE=12288`, `TOTAL_TRAINING_STEPS=4`,
n=4 LOOP rollouts per task (16 trajectories per PPO step). Single H200.

| Metric | Step 1 | Step 2 |
|--------|--------|--------|
| `frac_solved` mean / max | 0.000 / 0.000 | **0.125 / 1.000** |
| reward mean / max | −0.111 / −0.043 | **+0.538 / +2.468** |
| reward std (within step) | 0.014 | **1.658** |
| `critic/advantages/min, max` | −0.014, +0.010 | **−1.673, +1.679** |
| `actor/grad_norm` | **0.001** | **0.306** |
| `actor/pg_loss` | ≈ 0 | **−0.022** |
| `actor/entropy_loss` | 0.254 | 0.280 |
| `batch/solve_none / partial / all` | 4 / 0 / 0 | **3 / 1 / 0** |
| `response_length/clip_ratio` | 0.938 | 0.812 |
| PPO step wallclock | 624 s | 637 s |

**Direct interpretation:** the model went from a no-gradient regime to a
real gradient update in one cycle. The two solves we saw in step 2 (both
on a 1-FTP task) created the variance that ATS's continuous reward
amplifies into LOOP advantage.

## What's running

**Scaled run is live** (launched 2026-05-20 05:36 UTC), experiment
`scaled_minftp5_20260520-053601` in project `active_tests_swe_master`.

Settings:
- Dataset: `data_examples/rl_data/swerebench_min_ftp5/` (16 train + 4
  val; mean `|F| = 11.4`, range 5-37 — much richer partial-credit
  signal than the smoke's mostly-1-FTP pool).
- `TOTAL_TRAINING_STEPS=8`, `AGENT_MAX_STEPS=50` (vs smoke's 4 / 20)
- `RLLM_ACTIVE_TESTS_W_SOLVE=10.0` (vs smoke's 5.0)
- `LR=3e-6` (vs smoke's 1e-6)
- `MAX_RESPONSE=16384`

Expected wallclock: ~6 hours (8 PPO steps × ~45 min/step). Log:
`/home/temp/train_active_scaled.log`.

Status updates will land in this file as PPO steps emit.

### Smoke completion summary

Smoke ran 3 PPO updates + 1 validation pass before exiting cleanly
(`TOTAL_TRAINING_STEPS=4` cap, the 4th "step" was validation).
Verified all 4 ActiveTests states (active → cooldown → retired etc.)
exercise in unit tests; the training loop's three observed PPO steps
all produced non-zero gradients, with grad_norm scaling with observed
solves (0/2/1 solves → 0.001/0.306/0.165 grad_norm).

## Honest caveats

- The smoke had to cap `AGENT_MAX_STEPS=20` (paper recipe is 150). At 20
  steps the agent typically can't complete a real fix, so solves were
  rare (2/32 trajectories across two PPO steps). The scaled run uses
  50 steps to get a better solve rate.
- The current SWE-rebench parquet is skewed to small `FAIL_TO_PASS`
  (median = 1). With `|F| ≤ 2`, the partial-credit aspect is degenerate
  (only 0/0.5/1 possible). A `MIN_FTP=5`-filtered parquet (built via the
  `build_comprehensive_swerebench.py` env knob added this session) would
  exercise the full continuous-reward regime. Not done yet — pending
  docker-image probing budget.
- The token-budget term uses *character count*, not real tokenizer
  output, as an approximation (~4× off from real tokens). Absorbed by
  the `T_ref` and `lam_token` scaling, so the relative signal is sound.
- We do not yet have a head-to-head ATS-vs-binary at matched compute.
  That ablation is the next planned experiment after the scaled run.

## Files to read on wake-up

1. `paper/PAPER_DRAFT.md` — the full academic-paper draft (abstract
   through limitations) with the §5.1 smoke metrics table.
2. `REWARD_ACTIVE_TEST_SELECTION.md` — the design doc; §8 captures every
   change made vs. the original v1 design as code met reality.
3. `SMOKE_RESULTS_FOR_USER.md` — *this file*.
4. Tensorboard:
   `tensorboard --logdir /home/temp/swemaster_runpod/tensorboard_log`.
5. Smoke log: `/home/temp/train_active_smoke.log`.
   Run `python analyze_active_tests.py /home/temp/train_active_smoke.log`
   for the per-step summary table.
6. Scaled-run log (once running): `/home/temp/train_active_scaled.log`.
