# Block B — Test Refinement Results

Comparison of test-refinement arms vs control. All arms: base `SWE-Master-4B-SFT`, dataset
`swerebench_live64` (64 train / 16 val), **10 epochs**, ATS reward unchanged. Only the
training-set `FAIL_TO_PASS` differs between arms.

Populate each row with `collect_metrics.py` after a run completes.

| Arm | Refiner | Tests kept (train) | Val solve-rate | Generalization gap | Tokens/solved | Peak reward | Notes |
|--|--|--|--|--|--|--|--|
| B0 control | — | original (all) | **0.125** (2/16) | — | — | — | `/home/temp/models/SWE-Master-4B-ATS-control-live64` |
| B2 / R22 | mutation gate | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _running/queued_ |
| B1 / R12 | backtranslation | _deferred_ | | | | | needs teacher LLM (.env) |
| B3 / R3 | adversarial | _deferred_ | | | | | needs teacher LLM (.env) |

**Headline (Block B):** generalization gap ↓ and solve-rate ↑ on instances that were
degenerate before refinement, vs control. R22 additionally reports **mutant kill-rate**
distribution (from the refine audit trail) as a test-quality score.

## Refinement audit (kill-rate) summary
Filled from `../audit/<method>_<ts>.jsonl` after each refine pass.

| Method | Rows | Changed | Tests orig→kept | % dropped | Fallbacks | Errors |
|--|--|--|--|--|--|--|
| R22 | _pending_ | | | | | |
