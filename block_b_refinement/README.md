# Block B — Test Refinement (ATS-RL ablation)

Offline test-refinement arms for ATS-RL. Each method rewrites/augments an instance's
`FAIL_TO_PASS` set *before* training; the ATS reward runs unchanged. See
[`../BLOCK_B_TEST_REFINEMENT_DESIGN.md`](../BLOCK_B_TEST_REFINEMENT_DESIGN.md) for the full
design.

| Arm | Method | LLM? | Status |
|--|--|--|--|
| B0 | control (original tests) | no | done (`/home/temp/models/SWE-Master-4B-ATS-control-live64`, val 0.125) |
| **B2 / R22** | mutation-specificity gate (drop non-discriminative tests) | **no** | implemented ✅ |
| B1 / R12 | backtranslation from gold patch (add tests) | yes | deferred — needs teacher LLM in `.env` |
| B3 / R3 | InfCode adversarial co-refinement (strengthen tests) | yes | deferred — needs teacher LLM in `.env` |

## Code
- `../R2E-Gym/src/r2egym/agenthub/runtime/test_refiner.py` — `OperatorMutator`,
  `DockerTestValidator`, `specificity_gate` (R22), `backtranslate`/`adversarial_strengthen`
  (R12/R3 stubs).
- `../refine_dataset.py` — driver (`--method r22`).
- `mutation_selfcheck.py` — offline mutator test (no Docker).
- `collect_metrics.py` — pull a run's metrics into `metrics/<arm>.csv`.
- `audit/` — per-run refinement audit trails (`<method>_<ts>.jsonl`).
- `metrics/` — per-arm training metrics + `RESULTS.md` comparison table.

## Run R22 end-to-end
```bash
cd /home/temp/swemaster_runpod
PY=/home/temp/venv/bin/python3

# 0) sanity: offline mutator
$PY block_b_refinement/mutation_selfcheck.py

# 1) smoke the executable gate on 2 instances (needs their Docker images)
$PY refine_dataset.py --method r22 --smoke --subset 2 \
   --in  data_examples/rl_data/swerebench_live64/train_00000_verl.parquet \
   --out /tmp/r22_smoke.parquet --mutants 4

# 2) full refine (parallel) -> gated parquet + audit
mkdir -p data_examples/rl_data/ats_B2_r22
$PY refine_dataset.py --method r22 --workers 12 --mutants 6 \
   --in  data_examples/rl_data/swerebench_live64/train_00000_verl.parquet \
   --out data_examples/rl_data/ats_B2_r22/train_00000_verl.parquet
cp data_examples/rl_data/swerebench_live64/test_00000_verl.parquet \
   data_examples/rl_data/ats_B2_r22/         # val unchanged (held-out generalization)

# 3) train 10 epochs from SFT base on the gated parquet
MODEL_PATH=/home/temp/models/SWE-Master-4B-SFT TOTAL_EPOCHS=10 \
EXPERIMENT_NAME=B2_r22_$(date +%s) PROJECT_NAME=ablation_block_b \
TRAIN_PARQUET=data_examples/rl_data/ats_B2_r22/train_00000_verl.parquet \
VAL_PARQUET=data_examples/rl_data/swerebench_live64/test_00000_verl.parquet \
MAX_CKPT_KEEP=2 setsid nohup bash run_control_live64.sh > /home/temp/arm_B2.log 2>&1 &

# 4) collect metrics + export
$PY block_b_refinement/collect_metrics.py --arm B2_r22 \
   --tb-dir /home/temp/training_outputs/ablation_block_b/B2_r22_<ts> --log /home/temp/arm_B2.log
CUDA_VISIBLE_DEVICES='' bash export_to_hf.sh \
   /home/temp/training_outputs/ablation_block_b/B2_r22_<ts>/ /home/temp/models/SWE-Master-4B-ATS-B2-r22
```

`--subset N` trains on the first N instances if the gated set is large (user-approved);
pick `TRAIN_BATCH_SIZE` to divide the kept count cleanly.

## Deferred (R12/R3): reminder
When ready, add a key to `.env` (`OPENAI_API_KEY=...`, optionally `OPENAI_API_BASE=...` for a
local/compatible endpoint), then implement the generation step in `test_refiner.backtranslate`
/ `adversarial_strengthen` and run `--method r12` / `--method r3`. The `DockerTestValidator`
+ audit plumbing are already shared.
