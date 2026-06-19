# Block B — Test Refinement Results

Comparison of test-refinement arms vs control. All arms: base `SWE-Master-4B-SFT`,
**10 steps** (batch 16 → 10 epochs over the 64-instance set), ATS reward unchanged.
Only the training-set `FAIL_TO_PASS` differs between arms.

⚠️ **Dataset note:** the **B2 / R22** run below was trained on `swerebench_verlive`
(64 train / 16 val = SWE-bench-Verified ∩ SWE-rebench repos + multi-test SWE-bench-Live
repos), **not** the original `swerebench_live64`. The `B0 control` row is the older
live64 control, so it is **NOT a clean comparator** for B2. A **B0-control-on-verlive**
(original, unrefined tests, same `swerebench_verlive` data) is still **needed** before any
B2-vs-control claim can be made. See "Pending" below.

| Arm | Refiner | Tests kept (train) | Val solve-rate | Peak train reward | Notes |
|--|--|--|--|--|--|
| B0 control (live64) | — | original (all) | **0.125** (2/16) | — | `…/SWE-Master-4B-ATS-control-live64` — **different dataset** (live64) |
| **B2 / R22 (verlive)** | mutation gate | **254/378** (32.8% dropped) | **0.125** (2/16) | **+0.521** (step 9) | final `…/SWE-Master-4B-ATS-B2-r22` (8.3G); step-5 snapshot `…-step5` |
| B1 / R12 | backtranslation | _deferred_ | | | needs teacher LLM (.env) |
| B3 / R3 | adversarial | _deferred_ | | | needs teacher LLM (.env) |

**B2 / R22 training reward trend** (`critic/rewards/mean`, batch16 → 4 batches A–D × ~2–3 epochs):

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|--|--|--|--|--|--|--|--|--|--|--|
| reward | −0.194 | +0.405 | +0.028 | +0.289 | −0.077 | +0.218 | +0.495 | −0.018 | **+0.521** | _(val only)_ |

Same-batch epoch-over-epoch: **A** s1→s5→s9 **−0.194 → −0.077 → +0.521** (clean 3-epoch
climb, strongest signal); **B** s2→s6 +0.405→+0.218; **C** s3→s7 +0.028→**+0.495**;
**D** s4→s8 +0.289→−0.018. Epoch-1 mean +0.132 → epoch-2 mean +0.155. Net: **noisy but
modestly-positive learning** on hard verified+live data; batch A is the clearest evidence
the model improves on instances it revisits. Final post-training val = **0.125** (2/16).

**Headline (Block B, provisional):** R22 mutation-specificity gate drops 32.8% of train
`FAIL_TO_PASS` tests (least-specific ones, by mutant kill-rate) while keeping the ATS
reward mechanism unchanged. Training is stable and learns (esp. batch A). A clean
solve-rate vs-control claim awaits the **B0-control-on-verlive** baseline.

## Pending for clean comparison
- **B0-control-on-verlive**: rerun the control arm (original/unrefined `FAIL_TO_PASS`)
  on `swerebench_verlive` for 10 steps, SFT base — the only valid comparator for B2.
- **B1 / R12** (backtranslation) and **B3 / R3** (adversarial) — both need a teacher LLM
  API key in `.env` (deferred per user).

## Refinement audit (kill-rate) summary
Filled from `../audit/<method>_<ts>.jsonl` after each refine pass.

| Method | Rows | Changed | Tests orig→kept | % dropped | Fallbacks | Errors |
|--|--|--|--|--|--|--|
| R22 (verlive) | 64 | 34 | 378→254 | 32.8% | 20 | 0 |

Audit trail: `../audit/r22_1781829363.jsonl`. "Fallback" = gate would have dropped all
tests, so kept the single highest-kill-rate test to avoid an empty `FAIL_TO_PASS`.
