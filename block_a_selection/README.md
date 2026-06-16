# Block A — Test Selection ablation (S2 / S1 / S11)

Implements IMPLEMENTATION_PLAN.md §4. All three methods are **offline parquet
transforms** over one shared **probe pass** — the trainer and reward are
unchanged, so the only thing that differs between arms is the *training set*
(cleanest possible ablation). Every arm starts from the frozen CONTROL
(`ablation/control-live64`, base **RUCAIBox/SWE-Master-4B-SFT**, dataset
`swerebench_live64`, 64 train / 16 val).

## Pipeline

```
                         ┌─ select_variance.py  (S2)  → A1 parquet → train arm A1
probe_instances.py  ─────┼─ select_goldilocks.py (S1) → A2 parquet → train arm A2
   (probe_live64.csv)    └─ select_token_eff.py (S11) → A3 parquet → train arm A3
```

### 1. Probe pass (shared prerequisite)
Run the frozen base policy n=4 over the 64 train instances and record per-instance
outcomes. Two ways to get `probe_live64.csv`:

- **Harvest the control run** (free — it already rolls the policy over all 64):
  ```
  python block_a_selection/probe_instances.py \
      --from-log /home/temp/control_live64.log \
      --out block_a_selection/probe_live64.csv
  ```
  (Reliable now that the ATS terminal line carries `instance_id=`.)
- **Dedicated probe** (cleaner — pure step-0 policy, needs the GPU after control):
  run `run_control_live64.sh` with `TOTAL_TRAINING_STEPS=4 trainer.test_freq=...`
  in eval mode, or simply use the control run's first epoch.

### 2. Build each arm's parquet
```
SRC=data_examples/rl_data/swerebench_live64/train_00000_verl.parquet
P=block_a_selection/probe_live64.csv
python block_a_selection/select_variance.py   --parquet $SRC --probe $P --out data_examples/rl_data/ats_A1_variance/train_00000_verl.parquet
python block_a_selection/select_goldilocks.py --parquet $SRC --probe $P --out data_examples/rl_data/ats_A2_goldilocks/train_00000_verl.parquet
python block_a_selection/select_token_eff.py  --parquet $SRC --probe $P --out data_examples/rl_data/ats_A3_tokeneff/train_00000_verl.parquet
```
Val parquet is the SAME `swerebench_live64/test_00000_verl.parquet` for every arm
(held-out set never changes — that is the controlled comparison axis).

### 3. Train each arm (sequential — single H200, never two trainers at once)
Each arm reuses `run_control_live64.sh` with the arm parquet + a distinct
EXPERIMENT_NAME, e.g.:
```
EXPERIMENT_NAME=A1_variance_$(date +%s) PROJECT_NAME=ablation_block_a \
TRAIN_PARQUET=.../ats_A1_variance/train_00000_verl.parquet \
VAL_PARQUET=.../swerebench_live64/test_00000_verl.parquet \
... (same matched-compute knobs as control) ...
```
Matched compute: same TOTAL_EPOCHS, batch, rollout.n, max_steps as CONTROL.

## The three algorithms (distinct code, one file each)

| Arm | File | Score | Keeps | Headline |
|---|---|---|---|---|
| **A1 / S2** | `select_variance.py` | `Var(reward_per_rollout)` | `v>eps`, top-K | zero-grad-step fraction ↓ (continuous DAPO) |
| **A2 / S1** | `select_goldilocks.py` | `exp(-(f-p*)²/2σ²)`, p*≈0.2 | band `0<f<1` | steps-to-target ↓ (richest-signal difficulty) |
| **A3 / S11** | `select_token_eff.py` | `Var(reward)/tokens` | `score>eps`, top-K | tokens/solved ↓ |

S2 is a continuous generalization of DAPO dynamic sampling (binary = drop
all-solved/all-failed). S11 = S2 numerator ÷ rollout token cost.

## Metrics to compare across arms (vs CONTROL)
Per IMPLEMENTATION_PLAN §3: held-out solve rate (`val/test_score/swe`),
zero-gradient-step fraction, steps-to-target-reward, tokens/solved (S11),
grad_norm stability, GPU-hours. Tooling: `analyze_trend.py`,
`analyze_active_tests.py`, `dump_metrics_csv.py`. Append a row per arm to the
results table.

## Branches
- `ablation/control-live64` — frozen baseline (A0)
- `ablation/block-a-selection` — this shared code (probe + 3 selectors)
- `ablation/select-s2-variance`, `ablation/select-s1-goldilocks`,
  `ablation/select-s11-tokeneff` — per-arm branches (parquet + run cmd),
  branched off here once the real probe CSV exists.
