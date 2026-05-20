# Morning briefing — Active-Test-Selection RL

This file is the short version. Full detail in `SMOKE_RESULTS_FOR_USER.md`
and `paper/PAPER_DRAFT.md`. (Sleep update: 2026-05-20 ~05:38 UTC.)

## What works

The Active-Test-Selection (ATS) reward you sketched is built, wired into
the DeepSWE/SWE-Master pipeline, unit-tested (13 green), and demonstrably
producing real PPO gradient on the SWE-Master-4B-SFT base.

**Headline number** from a 3-PPO-step smoke (4 train tasks × 4 LOOP
rollouts = 16 trajectories per step, AGENT_MAX_STEPS=20):

| step | observed solves | `actor/grad_norm` |
|---|---|---|
| 1 | 0 | 0.001 (token-penalty signal only) |
| 2 | 2 | **0.306** |
| 3 | 1 | **0.165** |

For comparison, the binary baseline reward would have given grad_norm
= 0 on **all 4** unique-task groups in step 1 (all-fail → all-zero
reward → `(rewards == 0).all()` filter triggers), and 2 of 4 in step 2.
ATS gives signal on 100% of those.

## Where it lives

- Branch `reward/active-test-selection` (also pushed to origin).
- Reward scheme: `RUC-AIBOX/SWE-Master-4B-SFT` base + LOOP advantage, all
  scheduler knobs exposed as `RLLM_ACTIVE_TESTS_*` env vars. v1 default
  is prompt-injection + partial-credit terminal; the mid-episode pytest
  variant (v2) lives behind `RLLM_ACTIVE_TESTS_MID_EPISODE=1`.

## What's running now

`scaled_minftp5_20260520-053601` — overnight scaled run on the MIN_FTP=5
parquet (16 train + 4 val tasks, mean `|F|=11.4`). 8 PPO steps, max
50 agent turns per trajectory, `w_solve=10`, lr=3e-6. Expected wallclock
~6 hours from launch (so should be done around 11:30 UTC).

## When you wake up — first 3 things to read

1. `MORNING_BRIEF.md` — this file
2. `paper/PAPER_DRAFT.md` — full draft with §5.1 (smoke metrics) + §5.3
   (binary-baseline counterfactual: 100% ATS vs 50% binary gradient)
3. `SMOKE_RESULTS_FOR_USER.md` — implementation summary, what changed,
   honest caveats

To inspect the scaled run when you're back:

```bash
# live tail
tail -F /home/temp/train_active_scaled.log

# per-step metric summary
python /home/temp/swemaster_runpod/analyze_active_tests.py \
    /home/temp/train_active_scaled.log

# binary-baseline counterfactual
python /home/temp/swemaster_runpod/simulate_binary_baseline.py \
    /home/temp/train_active_scaled.log

# learning curves (figures saved to repo)
python /home/temp/swemaster_runpod/plot_learning.py \
    /home/temp/train_active_scaled.log \
    /home/temp/swemaster_runpod/paper/fig2_scaled_learning_curves.png

# tensorboard
tensorboard --logdir /home/temp/swemaster_runpod/tensorboard_log
```

## Decisions taken on your behalf

- Cleaned up the misplaced `SWE-World-4B-SFT` model (7.6 GB freed) per
  your correction; using local `SWE-Master-4B-SFT` snapshot.
- v1 is the default reward path (prompt injection + partial-credit
  terminal). v2 (mid-episode pytest ticks) is shipped behind an env-var
  for ablations — too expensive to be the default.
- Built a second parquet variant (`MIN_FTP=5`) to exercise the
  partial-credit signal more meaningfully; the original parquet's
  median `|F|=1` made the curriculum mostly degenerate.
- Token budget approximated by character count (vs real tokenizer);
  about 4× off in absolute units but absorbed by the scaling constant.
- Bumped `w_solve` 5 → 10 and `lr` 1e-6 → 3e-6 in the scaled run to
  amplify learning signal per PPO step.

## If something looks wrong

- If `actor/grad_norm` stays at 0.001 across all 8 scaled steps, the
  reward isn't getting strong-enough variance. Likely fix: bump
  `RLLM_ACTIVE_TESTS_W_SOLVE` further (e.g., 25) and re-launch.
- If the training fails / OOMs, check `/home/temp/train_active_scaled.log`
  for stack traces. The H200 fits this config with ~115/143 GB peak; we
  have headroom.
- If `n_total=1` dominates the schedulers, the dataset filter didn't take
  effect — the active-tests block degenerates to "show all". Confirm the
  scaled run used `swerebench_min_ftp5/`.
