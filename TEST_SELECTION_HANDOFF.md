# Test-Selection (Block-A) — Session Handoff

**Branch:** `ablation/test-selection`
**Last updated:** 2026-06-18 ~15:05 UTC
**Status:** PAUSED by user to rethink the design. Nothing running; GPU free. Probe was
killed mid-run (see §5). Read this top-to-bottom before resuming.

> Companion doc: `ABLATION_RESULTS_SO_FAR.md` (control-run details, dataset, learning
> evidence). This file is the focused test-selection handoff.

---

## 1. The idea (what we're testing)

Controlled ablation: does **instance selection** beat training on everything? Take the
frozen SFT model, measure how it behaves per training instance ("the probe"), then have
each of 3 selection methods pick a subset to RL-train on. Compare each arm vs the CONTROL
(RL on all 64).

Three selection methods (`block_a_selection/`):
| Arm | Method | Keeps instances where… | Knob |
|--|--|--|--|
| A1 / **S2** | variance | `Var(reward)` high (usable PPO gradient) | `select_variance.py --top-k 30` |
| A2 / **S1** | goldilocks | difficulty "just right", `0<frac_solved<1`, weight→p*≈0.2 | `select_goldilocks.py --mode filter --p-star 0.2` |
| A3 / **S11** | token-eff | `Var(reward)/tokens` high (signal per token) | `select_token_eff.py --top-k 30` |

**The selection scores do NOT exist in the dataset** — they must be measured by running
the frozen SFT model on each instance several times. That measurement = the probe.

---

## 2. Design parameters (user-confirmed 2026-06-18)

- **Base model for ALL arms:** `/home/temp/models/SWE-Master-4B-SFT` (= RUC-AIBox/SWE-Master-4B-SFT).
  Set `MODEL_PATH` EXPLICITLY every launch — do NOT inherit the control-trained model.
- **Select ~30 instances** per arm (`--top-k 30`).
- **Train 10 epochs** over the selected subset (NOT a fixed step count).
  30 instances × 10 epochs ≈ control's ~20 steps → compute stays roughly matched.
- **Full fine-tuning** (FSDP + CPU offload), NOT LoRA. (The `lora32b_*` / `merge_lora_to_hf.py`
  untracked files are a SEPARATE 32B experiment — ignore here.)

---

## 3. State of the pieces

| Piece | State |
|--|--|
| Dataset `data_examples/rl_data/swerebench_live64/` | DONE — 64 train / 16 val, leakage-fixed |
| CONTROL run | DONE — val/test_score/swe = **0.125 (2/16)**; model exported → `/home/temp/models/SWE-Master-4B-ATS-control-live64` |
| Selection code (S2/S1/S11 + probe parser) | DONE & committed (`2f867e7`) |
| Launcher env-overrides | DONE & committed (`8741035` batch, `e2291c5` epochs) |
| **Probe (frozen-SFT measurement)** | **NOT DONE** — too slow as configured, killed. See §5. |
| 3 arm parquets | NOT BUILT (need a valid probe CSV first) |
| 3 arm trainings + exports | NOT STARTED |

Target export names: `/home/temp/models/SWE-Master-4B-ATS-{S2-variance,S1-goldilocks,S11-tokeneff}`.

---

## 4. ⚠️ Open design decision (why it's paused)

User is reconsidering the probe approach. Two routes:

1. **Offline probe (current code):** one frozen-SFT pass scores all 64 → 3 cheap filters
   pick subsets → train arms on fixed data. Cleanest comparison (only the data differs).
   Cost = the probe pass (see §5 perf problem).
2. **Online / in-loop selection:** selection logic runs *inside* each arm's training loop
   (DAPO-style: skip all-solved/all-failed batches). No separate probe. Faster to start,
   but arms then differ in *mechanism* not just *data* → muddier ablation. Would need new code.

No decision made yet. If staying with route 1, fix the probe perf first (§5).

---

## 5. 🐌 The probe performance problem (MUST fix before re-running route 1)

Last probe (batch=64, all 64 instances, n=4 → 256 trajectories) ran ~50 min and completed
**0 of 64 instances** (0 `ATS terminal` lines). Killed it.

**Root cause:** `RLLM_MAX_ROLLOUT_CONCURRENCY` caps active trajectories at **8**
(launcher `run_control_live64.sh` line 23 pins it to 8). 256 trajectories ÷ 8 ≈ **32
sequential waves**, each wave a full agent rollout (~5–15 min) → multi-hour, first results
very late. (The 200+ Docker containers were trajectory envs across the 64 instances, NOT
200 instances — only 64 exist, ×4 rollouts.)

**Levers to make it tractable (pick before relaunch):**
- **Raise concurrency:** `RLLM_MAX_ROLLOUT_CONCURRENCY=32` (or higher) — more trajectories
  step in parallel. Watch GPU mem / vLLM throughput; H200 has headroom (probe used ~59 GB).
- **Fewer rollouts:** `rollout.n` is **HARDCODED =4** at line 152 of
  `DeepSWE_RL/rllm/examples/swe/train_active_tests_swe_master_4b_h200.sh` — it is NOT
  env-overridable. To use n=2 (halves trajectories), edit that line (and ideally make it
  `${ROLLOUT_N:-4}` so it's a knob). n=2 is the minimum for a variance estimate; S2 suffers
  most from low n.
- **Smaller candidate pool:** probe fewer than 64 (but then selection picks from fewer).

**Recommended fast config:** `RLLM_MAX_ROLLOUT_CONCURRENCY=32` + make `rollout.n` a knob and
set n=2 → 128 trajectories ÷ 32 ≈ 4 waves. Trade-off: noisier variance for S2.

---

## 6. Exact resume recipe (route 1 — offline probe)

Prereqs: GPU free (`nvidia-smi`), no trainer (`pgrep -f train_agent_ppo`), single H200 →
never two trainers at once. `python` is **`/home/temp/venv/bin/python3`**. `ray` lives in
`/home/temp/venv/bin/ray` (not on PATH); clean between runs with
`/home/temp/venv/bin/ray stop --force`. Launch long jobs with `setsid nohup ... &`.

### 6a. Run the probe (after applying a §5 speed fix)
```bash
cd /home/temp/swemaster_runpod
MODEL_PATH=/home/temp/models/SWE-Master-4B-SFT \
TRAIN_PARQUET=data_examples/rl_data/swerebench_live64/train_00000_verl.parquet \
VAL_PARQUET=data_examples/rl_data/swerebench_live64/test_00000_verl.parquet \
TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 TOTAL_TRAINING_STEPS=1 \
RLLM_MAX_ROLLOUT_CONCURRENCY=32 \
MAX_CKPT_KEEP=1 EXPERIMENT_NAME=probe_sft_$(date +%s) PROJECT_NAME=ablation_probe \
setsid nohup bash run_control_live64.sh > /home/temp/probe_sft.log 2>&1 &
```
Why batch=64 / 1 step: one step then covers ALL 64 instances in a single rollout (selectors
need every instance scored). The throwaway 1-step PPO update after the rollout can be ignored
— we only need the `ATS terminal: instance_id=...` lines the rollout emits.

Watch: `grep -c 'ATS terminal: instance_id=' /home/temp/probe_sft.log` → wait for ≥55 unique.

### 6b. Extract probe CSV
```bash
/home/temp/venv/bin/python3 block_a_selection/probe_instances.py \
  --from-log /home/temp/probe_sft.log --out block_a_selection/probe_live64.csv
wc -l block_a_selection/probe_live64.csv   # expect ~65 (64 instances + header)
```
NOTE: the `block_a_selection/probe_live64.csv` currently on disk (untracked) is the INVALID
16-instance output from a buggy batch=16 probe — DELETE/overwrite it; do not select from it.

### 6c. Build the 3 arm parquets (top-k 30)
```bash
SRC=data_examples/rl_data/swerebench_live64/train_00000_verl.parquet
P=block_a_selection/probe_live64.csv
mkdir -p data_examples/rl_data/{ats_A1_variance,ats_A2_goldilocks,ats_A3_tokeneff}
/home/temp/venv/bin/python3 block_a_selection/select_variance.py  --parquet $SRC --probe $P --out data_examples/rl_data/ats_A1_variance/train_00000_verl.parquet --top-k 30
/home/temp/venv/bin/python3 block_a_selection/select_token_eff.py --parquet $SRC --probe $P --out data_examples/rl_data/ats_A3_tokeneff/train_00000_verl.parquet --top-k 30
/home/temp/venv/bin/python3 block_a_selection/select_goldilocks.py --parquet $SRC --probe $P --out data_examples/rl_data/ats_A2_goldilocks/train_00000_verl.parquet --mode filter
# copy held-out val into each arm dir:
for d in ats_A1_variance ats_A2_goldilocks ats_A3_tokeneff; do
  cp data_examples/rl_data/swerebench_live64/test_00000_verl.parquet data_examples/rl_data/$d/
done
# record kept counts:
for d in ats_A1_variance ats_A2_goldilocks ats_A3_tokeneff; do
  /home/temp/venv/bin/python3 -c "import pandas as pd;print('$d', len(pd.read_parquet('data_examples/rl_data/$d/train_00000_verl.parquet')))"
done
```
**Goldilocks caveat:** `select_goldilocks.py` has NO `--top-k` — it keeps the whole
`0<frac<1` band. If that count >35, add a top-k (rank band by `|frac - 0.2|` ascending, keep 30)
and rerun, so all arms are ~30. Then commit the selectors + arm parquets.

### 6d. Train each arm (10 epochs, from SFT base), sequentially (single GPU)
Pick `TRAIN_BATCH_SIZE` to divide the kept count cleanly (kept≈30 → batch 15 → 2 steps/epoch
× 10 epochs = 20 steps; if kept differs use a near divisor 10/12/15). Do NOT set
`TOTAL_TRAINING_STEPS` — let it derive from epochs.
```bash
MODEL_PATH=/home/temp/models/SWE-Master-4B-SFT TOTAL_EPOCHS=10 \
TRAIN_BATCH_SIZE=15 PPO_MINI_BATCH_SIZE=15 \
EXPERIMENT_NAME=A1_variance_$(date +%s) PROJECT_NAME=ablation_block_a \
TRAIN_PARQUET=data_examples/rl_data/ats_A1_variance/train_00000_verl.parquet \
VAL_PARQUET=data_examples/rl_data/swerebench_live64/test_00000_verl.parquet \
MAX_CKPT_KEEP=2 setsid nohup bash run_control_live64.sh > /home/temp/arm_A1.log 2>&1 &
```
Verify the exec line in `arm_A1.log` shows `trainer.total_epochs=10`,
`model.path=/home/temp/models/SWE-Master-4B-SFT`, and the expected batch. Output dir:
`/home/temp/training_outputs/ablation_block_a/A1_variance_*/`. Babysit like control
(prune checkpoints to MAX_CKPT_KEEP, watch disk, GPU, errors).

Repeat for A2 (`ats_A2_goldilocks` → `EXPERIMENT_NAME=A2_goldilocks_*`) and A3
(`ats_A3_tokeneff` → `EXPERIMENT_NAME=A3_tokeneff_*`), each `MODEL_PATH=…SFT TOTAL_EPOCHS=10`.

### 6e. Export each final model
verl persists `global_step_1..N-1` but NOT the final-step ckpt (this fork skips it) — the
highest `global_step_*` present IS the final model.
```bash
CUDA_VISIBLE_DEVICES='' bash export_to_hf.sh <arm_out_dir>/ /home/temp/models/SWE-Master-4B-ATS-S2-variance
# A2 → …-ATS-S1-goldilocks ; A3 → …-ATS-S11-tokeneff
```

### 6f. Compare arms vs control
Held-out solve rate (val/test_score/swe), zero-gradient-step fraction, steps-to-target,
tokens/solved (S11), grad stability. Update `ABLATION_RESULTS_SO_FAR.md` + memory.

---

## 7. Key files & knobs

- **Launcher** `run_control_live64.sh` — env-overridable: `MODEL_PATH`, `TRAIN_BATCH_SIZE`,
  `PPO_MINI_BATCH_SIZE`, `TOTAL_EPOCHS`, `TOTAL_TRAINING_STEPS`, `RLLM_MAX_ROLLOUT_CONCURRENCY`,
  `EXPERIMENT_NAME`, `PROJECT_NAME`, `TRAIN_PARQUET`, `VAL_PARQUET`, `MAX_CKPT_KEEP`.
  (line 18 default MODEL_PATH=SFT, 19 batch, 20 epochs, 23 concurrency=8.)
- **Train script** `DeepSWE_RL/rllm/examples/swe/train_active_tests_swe_master_4b_h200.sh`
  — `rollout.n=4` HARDCODED at line 152 (make it a knob to change). GRPO `adv_estimator=loop`,
  save_freq=1, lr=1e-6, max_response=24576, ATS reward w_solve=10 (token penalty capped).
- **Selectors** `block_a_selection/{select_variance,select_goldilocks,select_token_eff,probe_instances}.py`
  + `_common.py` + `README.md`.
- **Probe ATS log line:** `ATS terminal: instance_id=...` (one per finished trajectory).
- Models: base SFT `/home/temp/models/SWE-Master-4B-SFT`; control
  `/home/temp/models/SWE-Master-4B-ATS-control-live64` (+ its final ckpt kept at
  `…/ablation_control/control_live64_20260616-052151/global_step_19`).

## 8. Ops reminders
- Single H200 (143 GB) — never two trainers at once (Ray/GPU contention crashes both).
- `setsid nohup … &` for long jobs. Clean between runs: `/home/temp/venv/bin/ray stop --force`.
- Mass `docker kill`/`rm -f` is blocked by the safety classifier and unnecessary — a GPU-free
  check (`nvidia-smi` ~11 MiB) is enough to confirm a clean slate. `docker container prune -f`
  (stopped only) + `docker image prune -f` (dangling only) are safe for disk.
- Babysit long runs — the control run took a Ray SIGTERM during a slow checkpoint write when
  left fully unattended; slow-rollout/SIGTERM recurs.
- Disk: keep SWE-bench instance images (re-pull is ~80 GB+ and hits docker pull limits).
