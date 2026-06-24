# Test-to-OpenAI Refinement — Design

Branch: `test-refinement-test-to-openai`. Base model: `RUC-AIBox/SWE-Master-4B-SFT`
(from scratch). Dataset: **`data_examples/rl_data/swerebench_verified`** (22 train / 4 val) —
instances sourced from SWE-rebench whose **repos are all SWE-bench Verified** (the 7-repo
SWE-bench-Verified ∩ SWE-rebench overlap from `verified_swerebench_overlap.tsv`: astropy,
matplotlib, scikit-learn, seaborn, sphinx, sympy, xarray; only astropy/matplotlib/sympy/sphinx/
xarray actually have instances in SWE-rebench → 26 total, 22/4 split). Built by
`build_verified_swerebench_dataset.py`. Pipeline: the **original ATS-RL pipeline** (no
in-training refinement during phase 1). Override with `DATASET=swerebench_comprehensive` to use
the original ATS-RL set instead.

## Idea (user's spec)

1. Train the model with the **original test cases** — both `FAIL_TO_PASS` (F2P) and
   `PASS_TO_PASS` (P2P) — for **70 PPO steps** (plain ATS-RL, refinement OFF).
2. After 70 steps, for each F2P test that is **still unsolved** but whose rollout was a
   **meaningful / insightful trajectory**, send `(test, trajectory)` to the **OpenAI API**
   (`gpt-5.4`) and ask it to **refine that test — change it or break it into better tests**.
3. Validate the refined tests, splice them into the dataset, and **continue training (+10 steps)**
   on the refined tests.

This differs from the prior B1/R12 (backtranslation) and B3/R3 (adversarial) refinement arms:
those inject cached decompositions into the **prompt** (reward never changes). Here the refinement
**rewrites the actual training tests** between two training phases, driven by the model's **own
failed trajectory** after real training — not an offline cache built before training.

## Phases

```
        ┌─────────────────────────────────────────────────────────────┐
Phase 1 │ ATS-RL, REFINE=0, swerebench_comprehensive, SFT base,        │
70 steps│ F2P+P2P original tests. SAVE_FREQ=1. Last checkpoint = θ70.   │
        └───────────────────────────┬─────────────────────────────────┘
                                     │
        ┌────────────────────────────▼────────────────────────────────┐
Collect │ Run θ70 over the 32 TRAIN instances (validation-style rollout │
        │ with RLLM_DUMP_TRAJ_DIR set). Per episode dump:               │
        │   {instance_id, f2p_pass_map, p2p_pass_map, frac_solved,      │
        │    transcript[(step,action,obs)], n_steps, patch_diff}        │
        └───────────────────────────┬─────────────────────────────────┘
                                     │
        ┌────────────────────────────▼────────────────────────────────┐
Refine  │ test_to_openai_refiner.py                                     │
        │ 1. HEURISTIC pre-filter (per unsolved F2P test):              │
        │      insightful ⇔ trajectory made ≥1 code edit AND ran tests  │
        │      AND used ≥MIN_STEPS steps AND did not crash-out early.    │
        │ 2. OpenAI gpt-5.4 JUDGE+REFINE: given problem, gold patch,    │
        │      original test src, and the failed trajectory →           │
        │      {insightful: bool, action: rewrite|split|skip,           │
        │       tests:[{name,code}], rationale}. Skip if not insightful.│
        │ 3. Docker TWO-RUN GATE (reuse test_refiner.validate_*):       │
        │      keep refined tests that FAIL pre-patch AND PASS gold.     │
        │ → cache JSONL: refinements_test_to_openai.jsonl               │
        └───────────────────────────┬─────────────────────────────────┘
                                     │
        ┌────────────────────────────▼────────────────────────────────┐
Build   │ build_refined_dataset.py                                      │
        │ For each instance with ≥1 gated refined test:                 │
        │   • add refined test file to test_patch (post-image)          │
        │   • FAIL_TO_PASS: replace the refined-away node id(s) with     │
        │     the new node id(s); untouched F2P + ALL P2P unchanged      │
        │   • rebuild make_test_spec via swebench_fork_swerebench        │
        │ → data_examples/rl_data/swerebench_comprehensive_refined/     │
        │   (instances with no refinement are copied verbatim)          │
        └───────────────────────────┬─────────────────────────────────┘
                                     │
        ┌────────────────────────────▼────────────────────────────────┐
Phase 2 │ Continue from θ70 (BASE_MODEL=phase-1 export), REFINE=0,      │
+10 step│ on swerebench_comprehensive_refined. +10 steps → θ80.        │
        │ HF-weight continuation (optimizer reset → step-1 dip then     │
        │ recovery, as in prior continuations). Export final model.     │
        └──────────────────────────────────────────────────────────────┘
```

## Selection criteria (decision: heuristic pre-filter → OpenAI judge)

A `(instance, F2P test)` is a refinement candidate iff, in θ70's rollout for that instance:
- the test is **unsolved** (`f2p_pass_map[test] is False`), AND
- the trajectory is **insightful** (heuristic): `n_code_edits ≥ 1` (file_editor/str_replace/insert
  actions) AND `ran_tests ≥ 1` (a pytest/run_tests action) AND `n_steps ≥ MIN_STEPS` (default 5)
  AND not an immediate error/empty trajectory.

OpenAI then makes the final call (`insightful: true/false`). Non-insightful → skipped (kept as-is).
This avoids wasting OpenAI calls + Docker validation on trivial/degenerate failures and respects
the user's "meaningful and insightful trajectory" gate.

## Refinement output (decision: REPLACE)

OpenAI returns `action ∈ {rewrite, split, skip}`:
- `rewrite` → one improved test replaces the original F2P node id.
- `split` → several smaller tests replace the original F2P node id (its credit is now spread over
  the split tests; the original is removed from FAIL_TO_PASS).
- `skip` → leave the original test untouched.

Refined tests must still validate the **gold fix** (Docker gate). P2P tests are never modified.

## What is reused vs. new

Reused (no behavior change):
- `train_active_tests_swe_master_4b_h200.sh`, `run_refine_arm.sh` (with REFINE=0 = plain ATS-RL).
- `test_refiner.py`: `load_env_value`, OpenAI client pattern, `extract_test_source`,
  `validate_decomposition` / `run_adhoc_tests` (two-run Docker gate), DockerRuntime usage.
- `build_decomposition_cache.py` orchestration pattern (per-instance, 1 container, gold-apply once).
- `swebench_fork_swerebench.harness.test_spec.test_spec.make_test_spec` for rebuilding make_test_spec.

New:
- `swe_arsenal.py`: a trajectory+passmap **dump hook** gated by `RLLM_DUMP_TRAJ_DIR`
  (unset ⇒ ZERO change ⇒ original pipeline byte-identical; v1 reward untouched, dense_sum=0).
- `test_to_openai_refiner.py`: heuristic filter + gpt-5.4 judge/rewrite/split + Docker gate → cache.
- `build_refined_dataset.py`: splice refined tests into a new parquet dataset.
- `test_to_openai_orchestrator.sh`: phase1 → collect → refine → build → phase2, with
  crash-relaunch, miner-kill watchdog, checkpoint prune, and final export.

## Invariants / safety

- **Phase 1 = the ORIGINAL ATS-RL pipeline, unchanged.** REFINE=0, MID_EPISODE=0 ⇒ the v1 reward
  is bit-exact (dense_sum=0). The dump hook is read-only and no-op when `RLLM_DUMP_TRAJ_DIR` unset.
- Refined tests are admitted to phase-2 **only if** they pass the executable gate (fail pre-patch,
  pass gold) — no unverifiable tests enter training.
- P2P regression-guard tests are never altered.
- Models preserved; new model exported to `/home/temp/models/SWE-Master-4B-TestToOpenAI`.

## Config knobs (env)

| Var | Default | Meaning |
|---|---|---|
| `RLLM_DUMP_TRAJ_DIR` | unset | when set, dump per-episode trajectory+passmap JSONL there |
| `RLLM_DUMP_TRAJ_OBS_CAP` | 2000 | per-observation char cap in the dump |
| phase-1 `TOTAL_TRAINING_STEPS` | 70 | step budget for phase 1 |
| phase-2 `TOTAL_TRAINING_STEPS` | 10 | step budget for phase 2 (continue) |
| refiner `--model` | gpt-5.4 | OpenAI model |
| refiner `--min-steps` | 5 | heuristic min trajectory length |
| refiner `--max-refined` | 4 | max refined tests per original (split) |

## Open follow-ups (not blocking phase-1)

- Exact node-id mapping for `split` (new test file path + `::name`) is finalized in
  `build_refined_dataset.py`; validated by the gate before it enters training.
- Collection pass uses the val-rollout path with `VAL_PARQUET=train`; if that proves awkward in
  verl, fall back to a standalone AgentExecutionEngine collection script (same dump format).
