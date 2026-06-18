# ATS-RL Ablation — Results So Far (snapshot 2026-06-16 ~12:21 UTC)

Saved at user request before an afternoon break. Training left RUNNING (healthy,
checkpointed every step). Resume from here.

## 0. RESUME POINTER (read first — updated 2026-06-18 12:47 UTC)
- **CONTROL TRAINING COMPLETE (2026-06-18 12:39 UTC).** 20 PPO steps, clean `training done.`
  **Held-out val/test_score/swe = 0.125 (2/16)**; val pass@1 = 0.125.
  verl persisted `global_step_1..19`; it does NOT write a `global_step_20` ckpt (final-step
  save skipped by this fork) — `global_step_19` IS the final full-FT control model.
  Per-step reward (s1–19): 0.726/0.666/0.506/0.354/0.737/0.643/0.465/0.149/0.294/1.857/
  0.268/0.432/0.471/1.392/-0.127/0.713/0.527/0.39/0.799; grad_norm healthy 0.06–0.25 throughout.
- **Control model EXPORTED** → `/home/temp/models/SWE-Master-4B-ATS-control-live64`
  (2 safetensors shards + config + tokenizer; from global_step_19, CPU merge via export_to_hf.sh).
- **COMPLETION PIPELINE IN PROGRESS** (autonomous): frozen-SFT probe.
  - **PROBE BATCH FIX (14:08)**: first probe (12:47) ran with batch=16 (run_control_live64.sh
    line 19 *hardcoded* TRAIN_BATCH_SIZE=16) → 1 step only covered 16/64 instances. Fixed:
    edited line 19 to `${TRAIN_BATCH_SIZE:-16}` (overridable). Killed that probe; relaunched
    14:08 with **TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 TOTAL_TRAINING_STEPS=1** →
    1 batch = all 64 instances (micro_batch=1 bounds update memory). log /home/temp/probe_sft.log
    (old batch16 partial: probe_sft_batch16_partial.log). base=SWE-Master-4B-SFT confirmed.
  - next: probe_instances.py → probe_live64.csv (target ~64 rows) → build 3 arm parquets
    (S2/S1/S11) → train arms sequentially (MODEL_PATH=SWE-Master-4B-SFT explicit,
    TOTAL_TRAINING_STEPS=20) → export each to /home/temp/models/SWE-Master-4B-ATS-{S2-variance,S1-goldilocks,S11-tokeneff}.
  - ALL arm trainings use base /home/temp/models/SWE-Master-4B-SFT (RUC-AIBox/SWE-Master-4B-SFT),
    set MODEL_PATH explicitly (user-confirmed).
- EARLIER: crashed 06-17 12:23 (Ray SIGTERM during slow step-12), recovered+relaunched
  06-18 01:18 from global_step_11; steps 12–20 all clean. Lesson: don't leave this run
  fully unattended (slow-rollout/SIGTERM recurs).

## (archived) RESUME POINTER — 2026-06-18 01:18 UTC
- **CRASH + RECOVERY**: control was left running unattended after the 06-16 break;
  it ran abnormally SLOW (~24h step11→step12-update, i.e. hanging rollouts) then took
  a Ray-monitor **SIGTERM during the step-12 checkpoint write at 2026-06-17 12:23**,
  and stayed down ~13h (no babysitter). Disk was NOT the cause (255G free, cap=2 holds).
  Recovered 2026-06-18 01:18: GPU verified free + no zombie procs, old log saved to
  `control_live64.log.crash_0617`, **relaunched** (resume_mode=auto from `global_step_11`).
  LESSON: do not leave this run fully unattended for long; the slow-rollout/SIGTERM
  failure recurs. Babysit, or accept lost time.
- **Control progress: step 12/20 redoing** (latest ckpt `global_step_11`).
  ~75 min/step nominal (but watch for slow-rollout hangs).
- Trainer **alive**, **0 errors** all run, disk **255 GB free / 80%** (self-capping
  at 2 ckpts via `MAX_CKPT_KEEP=2`, so disk is safe unattended).
- reward steps 1–11: 0.726/0.666/0.506/0.354/0.737/0.643/0.465/0.149/0.294/1.857/0.268
  (noisy plateau; grad_norm healthy 0.13–0.25; learning confirmed at §2).
- **Left RUNNING unattended** — no babysitter. On afternoon return:
  (1) check `ls .../control_live64_20260616-052151/global_step_*`; if `global_step_20`
      exists and no trainer proc → control DONE, jump to §6 step 2 (export+probe+arms).
  (2) if still running → resume the 5-min watch loop (prune/df/trainer/containers/errors).
  (3) if crashed mid-run → relaunch `EXPERIMENT_NAME=control_live64_20260616-052151
      MAX_CKPT_KEEP=2 setsid nohup bash run_control_live64.sh > /home/temp/control_live64.log 2>&1 &`
      (verl auto-resumes from latest global_step_*).
- Autonomous arm pipeline (S2→S1→S11) is fully scripted in §6 + `block_a_selection/README.md`.

## 1. Dataset (DONE)
`data_examples/rl_data/swerebench_live64/` — **64 train / 16 val**, built by
`build_live_overlap_dataset.py`.
- Train = 39 validated v2-core instances + 25 SWE-bench-Live-overlap additions
  (1 each: loguru, textual, conan, networkx, tox, streamlink, falcon, starlette,
  dvc, kedro, pdm, ibis, fsspec, haystack, geopandas, certbot, briefcase, wemake,
  faker, mesa, beets, altair, sqllineage, structlog, mcp/python-sdk).
  Live additions: F2P∈[5,40], P2P∈[1,160].
- Val = 16 instance-level held-out live-overlap instances (fixes the v2 leakage
  where val was the first 8 train rows). No train/val leakage.
- Rows keep `make_test_spec` + gold `patch` + `test_patch` (for Block-B later).
- 41 new images pre-pulled (authenticated, ~80GB). Verified-overlap repos
  (astropy/sympy/xarray/...) excluded — no public swerebench images.
- Committed on branch `ablation/control-live64`.

## 2. CONTROL run (RUNNING)
- Experiment `control_live64_20260616-052151`,
  out `/home/temp/training_outputs/ablation_control/control_live64_20260616-052151/`,
  log `/home/temp/control_live64.log`, tensorboard `tensorboard_log/*26084*`.
- **Base: RUCAIBox/SWE-Master-4B-SFT** (`/home/temp/models/SWE-Master-4B-SFT`).
- **FULL fine-tuning** (full-param FSDP + param/optimizer CPU offload) — NOT LoRA.
  Verified: no lora/peft flags in the live process; checkpoints hold full
  model+optim shards. (`merge_lora_to_hf.py`/`lora32b_patch/` belong to a separate
  32B experiment.)
- Config: GRPO `adv_estimator=loop`, rollout.n=4, batch=16, max_steps=150,
  trajectory_timeout=5400, max_response=24576, lr=1e-6, save_freq=1,
  total_epochs=5 (=> ~20 PPO steps), ATS reward w_solve=10, token penalty capped.
  Launch script: `run_control_live64.sh`.
- **Per-step wall-clock ~77 min** (≈58 rollout + ~16 update).

### Learning results (per-step; batch=16 -> 4 steps/epoch, dataset cycles)
| step | reward_mean | grad_norm | solve all/part/none |
|--|--|--|--|
| 1 | 0.726 | 0.152 | 3/4/9 |
| 2 | 0.666 | 0.215 | 1/8/7 |
| 3 | 0.506 | 0.146 | 1/4/11 |
| 4 | 0.354 | 0.128 | 1/3/12 |
| 5 | **0.737** | 0.133 | 2/4/10 |

**KEY (epoch-over-epoch, controlled): step 5 reuses batch 1 (same 16 instances as
step 1).** reward 0.726 -> 0.737 (held/up), full-solve *trajectories* 3 -> 10,
grad stable. => the model is LEARNING; the epoch-1 per-step reward decline
(0.73->0.35) was batch-difficulty variance, not degradation. **0 errors all run.**
grad_norm healthy 0.13-0.22 throughout (no -494-style outliers; token cap works).

## 3. Block-A test-selection methods (CODE DONE, committed `ablation/block-a-selection`)
`block_a_selection/`:
- `probe_instances.py` — shared probe-pass extractor (`--from-log` tagged lines / `--synth`).
- `select_variance.py` (S2) — keep high `Var(reward)` groups (continuous DAPO).
- `select_goldilocks.py` (S1) — band `0<frac_solved<1`, weight to p*≈0.2.
- `select_token_eff.py` (S11) — rank by `Var(reward)/tokens`.
- `_common.py`, `README.md`. All validated on synthetic probe data.
- Also: tagged the `ATS terminal` log line with `instance_id=` (print-only) for
  reliable probe attribution under rollout concurrency.

## 4. Model-save plan (full fine-tuned models, per user)
`save_freq=1` already preserves every step's checkpoint per run. For the FINAL
model of each run, export to HF format via `export_to_hf.sh <exp_dir> <target>`:
| Run | Final model export target |
|--|--|
| Control | `/home/temp/models/SWE-Master-4B-ATS-control-live64` |
| A1 / S2 variance | `/home/temp/models/SWE-Master-4B-ATS-S2-variance` |
| A2 / S1 goldilocks | `/home/temp/models/SWE-Master-4B-ATS-S1-goldilocks` |
| A3 / S11 token-eff | `/home/temp/models/SWE-Master-4B-ATS-S11-tokeneff` |

## 5. Probe sourcing (decided)
`rollout_data_dir` stays empty (verl not dumping per-traj) and the control policy
drifts each step, so DON'T harvest the control log for the probe. Run a DEDICATED
frozen-SFT probe pass (MODEL_PATH=SWE-Master-4B-SFT over the live64 train parquet,
tagged ATS terminal lines) AFTER control frees the GPU, then
`probe_instances.py --from-log <log> --out block_a_selection/probe_live64.csv`.

## 6. Next steps (resume in the afternoon)
1. Let control finish (~step 20, total_epochs=5) — or stop early (any checkpoint is
   usable since save_freq=1). To resume a stopped run: relaunch `run_control_live64.sh`
   (verl auto-resumes from the latest `global_step_*`).
2. Export control final model (table §4).
3. Dedicated frozen-SFT probe pass -> `probe_live64.csv`.
4. Build 3 arm parquets (`select_*` -> `data_examples/rl_data/ats_A1_variance|ats_A2_goldilocks|ats_A3_tokeneff/`),
   create+commit branches `ablation/select-s2-variance|s1-goldilocks|s11-tokeneff`.
5. Train each arm sequentially (matched compute), export each final model (§4).
6. Compare arms vs control: held-out solve rate, zero-grad-step fraction,
   steps-to-target, tokens/solved (S11), grad stability.

## Ops reminders
- Single H200; never run two trainers at once (GPU/Ray contention crashes both).
- Launch long jobs `setsid nohup ... &`. Clean between runs: `ray stop --force`
  (mass `docker kill` is blocked by the safety classifier and is unnecessary).
- Monitor: `bash monitor_control.sh`; metrics via tensorboard EventAccumulator.
