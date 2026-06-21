# Test-Selection Ablation — FAIL_TO_PASS-only reward

**Branch:** `test-selection` (forked from `reward/active-test-selection`, the ATS-RL paper branch)

## Hypothesis
The PASS_TO_PASS (p2p) term in the ATS reward is the regression guardrail. Without
it, ATS should optimize the FAIL_TO_PASS (f2p) subset more aggressively and **game
it — breaking p2p (real functionality) tests** even if raw f2p solve-rate looks
similar or higher. If so, the p2p-break rate rises in the f2p-only arm, which
justifies keeping the p2p term in the main method.

## The single variable
Same recipe as the verlive control / R22 arm — `SWE-Master-4B-SFT`, 10 steps,
batch 16, ATS enabled, `W_SOLVE=10.0`, token cap on — except the regression-guard
reward is zeroed:

| knob | default (control) | f2p-only |
|--|--|--|
| `RLLM_ACTIVE_TESTS_W_REGRESS` | `w_solve/2` (=5.0) | **0.0** |
| `RLLM_ACTIVE_TESTS_DELTA_P2P_BREAK` | 0.5 | **0.0** |
| `RLLM_ACTIVE_TESTS_DELTA_P2P_RECOVER` | 0.25 | **0.0** |

Nothing else changes. p2p tests are still **evaluated** every terminal step
(`get_f2p_and_p2p_status`), so the p2p-break rate stays measurable — it just no
longer shapes the reward. This isolates the regression-guard term, not the data.

## Dataset
`swerebench_verlive` (UNREFINED, 64 train / 16 val) — the same data the R22/B2
verlive arm trained on. Comparable to the (pending) **B0-control-on-verlive**
baseline and to the R22 mutation-gate arm.

## Code touchpoints
- `swe_arsenal.py:75-77` already reads the three p2p knobs from env at
  scheduler-build time (inside the rollout worker).
- `train_active_tests_swe_master_4b_h200.sh` now forwards them explicitly when
  set (guarded `+x` exports — behavior unchanged when unset).
- `run_f2p_only.sh` — the launcher; sets all three to 0.0 on `swerebench_verlive`.

## Metrics to compare (f2p-only vs control)
- f2p val solve-rate (headline)
- **p2p-break rate** at eval (the reward-hacking signal)
- train reward trend, grad-norm, tokens/solved

## Run
```bash
setsid nohup sg docker -c "bash run_f2p_only.sh" > /home/temp/f2p_only.log 2>&1 &
```
Output: `/home/temp/training_outputs/ablation_test_selection/<EXP>/`, checkpoints
every step (`save_freq=1`). Export the final/step-10 checkpoint to HF after.
