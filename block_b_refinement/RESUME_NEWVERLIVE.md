# Resuming the B1/R12 NEW-DATA run (swerebench_newverlive)

This run was **paused by the user at global_step_5** (5 completed PPO updates of a planned
10-step run) on 2026-06-23 so the machine could be powered down. Step 6 was mid-rollout when
stopped and was **not** checkpointed — the last fully-saved state is `global_step_5`.

## What is saved

| Artifact | Path | Purpose |
|---|---|---|
| HF model (5 PPO steps) | `/home/temp/models/SWE-Master-4B-ATS-B1-r12-newverlive` (8.3G) | usable model / HF-weight continuation base |
| verl checkpoint + **optimizer state** | `/home/temp/training_outputs/ablation_b1_newdata/B1_newdata/global_step_5/` (63G) | **true resume** (optimizer/LR intact) |
| prior verl checkpoint | `…/global_step_4/` (63G) | safety / rollback |
| train/val data | `data_examples/rl_data/swerebench_newverlive/{train,test}_00000_verl.parquet` (64/16) | committed to git |
| decomposition cache | `block_b_refinement/cache/decompositions_newverlive.jsonl` (163 rec / 158 sub-tests, 64 inst) | committed to git |

> The 63G verl checkpoints live ONLY on disk (too big for git). If the machine's disk is wiped
> on power-down, true-resume is lost — but the **HF model is 8.3G and can be backed up / re-trained
> from** (HF-weight continuation, optimizer reset). Back up `…-newverlive` before wiping if unsure.

## Reward trend so far (5 PPO updates, NEW SWE-Verified+SWE-Live data)

`step1=+0.359 · step2=+1.342 · step3=+0.408 · step4=−0.216 · step5=+0.913` (avg ≈ +0.56, noisy
but clearly learning). v1 reward preserved: **364/364 ATS terminals logged dense_sum=0.000**
(refinement is prompt-only, as designed). 0 crashes.

## How to continue (two options)

### Option A — TRUE resume from global_step_5 (keeps optimizer/LR, no step-1 dip)
verl auto-resumes from the highest `global_step_*` in the output dir. Just relaunch the
orchestrator unchanged; it will pick up at step 6:
```bash
cd /home/temp/swemaster_runpod
setsid nohup bash b1_newdata_orchestrator.sh >/dev/null 2>&1 &
```
Requires `…/ablation_b1_newdata/B1_newdata/global_step_5/` to still exist on disk.
(If you want more than the original 10 steps, bump `TOTAL_TRAINING_STEPS`/`TOTAL_EPOCHS` in
`b1_newdata_orchestrator.sh` before relaunching.)

### Option B — HF-weight continuation from the exported model (optimizer reset → step-1 dip then recovery)
Use when the verl checkpoint is gone but the HF model survives. Point BASE_MODEL at the export
and use a FRESH output dir (so it doesn't collide with the old run):
```bash
BASE_MODEL=/home/temp/models/SWE-Master-4B-ATS-B1-r12-newverlive \
RLLM_ACTIVE_TESTS_REFINE=1 RLLM_ACTIVE_TESTS_REFINE_TICK=1 \
TRAIN_PARQUET=$PWD/data_examples/rl_data/swerebench_newverlive/train_00000_verl.parquet \
VAL_PARQUET=$PWD/data_examples/rl_data/swerebench_newverlive/test_00000_verl.parquet \
REFINE_CACHE=$PWD/block_b_refinement/cache/decompositions_newverlive.jsonl \
TOTAL_TRAINING_STEPS=10 TOTAL_EPOCHS=3 \
EXPERIMENT_NAME=B1_newdata_cont PROJECT_NAME=ablation_b1_newdata_cont \
setsid nohup bash run_refine_arm.sh > /home/temp/arm_B1_newdata_cont.log 2>&1 &
```

## On every restart, also
- Relaunch the miner watchdog: `setsid nohup bash miner_killer.sh >/dev/null 2>&1 &`
- All 80 verlive docker images are cached; new-data instances pull on first use. If images were
  wiped, re-pull as needed (swerebench/ namespace, lowercase, `__`→`_1776_`).
