# Morning briefing — Active-Test-Selection RL

This file is the short version. Full detail in `SMOKE_RESULTS_FOR_USER.md`
and `paper/PAPER_DRAFT.md`. (Sleep update: 2026-05-20 ~05:38 UTC.)

## What works

The Active-Test-Selection (ATS) reward you sketched is built, wired into
the DeepSWE/SWE-Master pipeline, unit-tested (13 green), and demonstrably
producing real PPO gradient on the SWE-Master-4B-SFT base.

**Headline numbers — smoke** (3 PPO steps, 16 trajectories/step,
AGENT_MAX_STEPS=20):

| step | observed solves | `actor/grad_norm` |
|---|---|---|
| 1 | 0 | 0.001 (token-penalty signal only) |
| 2 | 2 | **0.306** |
| 3 | 1 | **0.165** |

**Headline numbers — scaled** (MIN_FTP=5 parquet, AGENT_MAX_STEPS=50,
w_solve=10, lr=3e-6; aborted at step 3 startup due to a process-mgmt
glitch but steps 1-2 captured cleanly; scaled-2 relaunch in progress):

| step | observed full solves | observed partials | `actor/grad_norm` | reward mean | reward max |
|---|---|---|---|---|---|
| 1 | 0 | 2 (frac=0.333) | 0.116 | +0.144 | +1.611 |
| 2 | 2 | 1 (frac=0.857) | **0.424** | **+0.824** | **+4.944** |

For binary baseline comparison: scaled step 1 would have given std=0.000
(every trajectory mapped to reward 0 → no gradient at all). ATS std =
1.107 → real gradient. This is the cleanest demonstration in our data
of why the continuous reward matters.

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

## Open item: parallel session conflict

There's a **second Claude Code session** running on the same machine
that re-downloaded `SWE-World-4B-SFT` (which you had me delete) and
launched its own training via
`/home/temp/swemaster_runpod/run_scaled_swe_world.sh` at 06:10 UTC.
That run is using the same ATS reward + MIN_FTP=5 parquet I built,
but the base model is SWE-World, not SWE-Master. It killed my own
scaled-2 attempt via GPU/Ray contention.

I cannot kill that parallel session (auto-mode won't let me terminate
a process from another session). When you wake up, please:
1. Decide whether to keep or kill the SWE-World run (log at
   `/home/temp/train_active_scaled_sweworld.log`, PID 161602).
2. Once the GPU is clear, you (or I, if you tell me you authorized
   the kill) can launch a clean SWE-Master scaled run:
   `bash /home/temp/swemaster_runpod/run_scaled.sh`
3. Re-delete `/home/temp/models/SWE-World-4B-SFT` (7.6 GB).
