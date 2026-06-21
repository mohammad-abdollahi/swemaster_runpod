# Block B / B1 — R12: Backtranslation of minimal tests from the gold patch

**Branch:** `test-refinement-back-translation` (off `test-selection`, the main ATS-RL
training branch). **Status:** design / plan — no training code touched yet.
**Companion:** B2/R22 (mutation-specificity gate) already done on the `test-refinement`
branch — see its `block_b_refinement/` + `RESULTS.md` there. R12 reuses the same shared
infrastructure (ported, not branched-from) and the same dataset for comparability.

---

## 0. What "test refinement" is in ATS-RL (and why it matters)

ATS-RL's reward is **defined entirely by the instance's `FAIL_TO_PASS` test set.** In
`active_tests.py::terminal_reward()`:

```
frac_solved = |{F2P tests that PASS after the agent's patch}| / |F2P|
r_solve     = w_solve · frac_solved          # w_solve = 10
r_final     = r_solve + r_regress(P2P) + r_token
```

So **the `FAIL_TO_PASS` tests are the reward function.** The policy is optimized to make
those specific tests pass. Two consequences:

1. **Today there is no refinement** — we take each instance's `FAIL_TO_PASS` exactly as it
   ships in the SWE-rebench dataset and train against it unchanged. "Test refinement"
   (Block B of the ablation) is the category of **offline transforms that improve the
   quality of that test set *before* training**, while leaving the ATS reward machinery
   itself untouched.
2. **Reward quality is bounded by test quality.** Failure modes a raw dataset can have:
   - *Degenerate / thin*: only one F2P test, or a weak one — coarse, high-variance signal.
   - *Non-discriminative*: a test that also passes on **wrong** patches → the policy can
     "solve" it without actually fixing the bug (reward hacking surface).
   - *Under-specified*: the issue is broad but the test pins only a narrow slice of the
     fixed behavior → sparse gradient.

The three Block-B arms attack this from different angles, all as **offline parquet
transforms** (no trainer change → cleanest possible ablation):

| Arm | Idea | Direction | Needs |
|--|--|--|--|
| **B2 / R22** (done) | mutation-specificity **gate** | **removes** non-discriminative tests | harness only |
| **B1 / R12** (this doc) | **backtranslation** from gold patch | **adds** discriminative tests | teacher LLM |
| B3 / R3 | adversarial co-refinement | **strengthens** tests vs wrong patches | teacher LLM |

R22 subtracts bad tests; **R12 adds good ones.** They compose (generate → gate = arm B4).

---

## 1. R12 idea in one paragraph

The **gold patch** (`extra_info.patch`) is the diff that actually fixes the bug — it
*exactly encodes the changed behavior*. A test derived from the gold diff therefore carries
strictly more signal than a test guessed from the issue text alone. R12: for each instance,
ask a **teacher LLM** to write a minimal pytest that exercises the changed behavior (given
issue + gold diff + changed symbols), then keep it **only if execution proves it
discriminates** — it must **PASS on the gold-patched repo** and **FAIL on the pre-patch
repo** (the textbook FAIL_TO_PASS invariant). No LLM-judge: execution is ground truth.
Survivors are appended to the instance's `FAIL_TO_PASS`. Paper anchor: SWE-RL "patches as
supervision" (arXiv:2502.18449); code reference `facebookresearch/swe-rl`.

---

## 2. The mechanism that makes R12 harder than R22 (verified)

R22 only **filtered** existing `FAIL_TO_PASS` ids — it never had to inject new test code.
R12 **generates new test code**, so it must reach the training harness. The harness builds
each instance's run from `extra_info.make_test_spec`, whose `eval_script_list` literally
injects the gold tests inline (verified on `swerebench_verlive`):

```
11: git checkout <base_commit> <gold_test_file>           # reset test file to base
12: git apply -v - <<'EOF' … gold test_patch … EOF        # inject gold tests
14: pytest … <gold_test_file>                              # run them
```

`FAIL_TO_PASS` ids reference tests **inside that applied file.** So to add a backtranslated
test, the offline transform must edit **`make_test_spec`** (the authoritative source the
runtime reads — NOT just `extra_info.FAIL_TO_PASS`), specifically:

- **(a)** inject the generated test — cleanest as its **own new file**
  `r2e_tests/test_r12_<instance>.py` written via an added heredoc step in
  `eval_script_list` (avoids munging the gold `git apply` diff);
- **(b)** add that file to the **pytest target** line in `eval_script_list`;
- **(c)** append `r2e_tests/test_r12_<instance>.py::<func>` to **`FAIL_TO_PASS`** (in both
  `make_test_spec.FAIL_TO_PASS` and `extra_info.FAIL_TO_PASS` for consistency).

> ⚠️ **Verify-before-coding (carry-over from B2):** the runtime reads
> `TestSpec(**json.loads(ds['make_test_spec']))` and results are keyed to
> `test_spec.FAIL_TO_PASS`. Confirm the eval-script edit + new id round-trips through
> `docker.py::get_f2p_and_p2p_status` on **one** instance before batch-refining all 64.

This injection design is the main new engineering vs R22 and the highest-risk part.

---

## 3. Pipeline (per instance)

```
entry (issue, gold patch, gold test_patch, base_commit, docker_image)
  │
  1. PROMPT teacher LLM: issue + gold diff + changed files/symbols
  │     → "write ONE minimal pytest test exercising the changed behavior;
  │        self-contained; standard pytest; no network; assert the new behavior."
  │     (parse fenced code block → candidate test source + chosen func name)
  │
  2. EXECUTABLE GATE (mandatory, in the instance Docker image):
  │     a. checkout base_commit  → write candidate to r2e_tests/test_r12_<id>.py
  │        → pytest that file      → must FAIL  (bug present, test is discriminative)
  │     b. apply gold patch        → (same test file) → pytest  → must PASS (test is valid)
  │     KEEP iff (fails_prepatch AND passes_gold). Else discard (optionally 1 retry
  │     with the failure captured back to the LLM — bounded to ≤2 calls/instance).
  │
  3. WRITE-BACK into make_test_spec: add inject step + pytest target + F2P id (§2 a/b/c).
  │     Provenance row in audit/ (instance, n_candidates, kept, gate verdicts, llm_model).
```

Outcome: each instance's F2P grows from *k* gold tests to *k + (#survivors)*; the reward
becomes denser/more discriminative without touching `active_tests.py`.

---

## 4. Code plan

**Reuse (port from `test-refinement` branch — already validated for B2):**
- `R2E-Gym/src/r2egym/agenthub/runtime/test_refiner.py` — keep `DockerTestValidator`
  (`_apply_verified()` checkout/apply-with-`--3way`-fallback, `status_for_patch`). The
  `OperatorMutator` (R22-only) is not needed here.
- `refine_dataset.py` (repo-root CLI: `--method --in --out --workers --subset --smoke`,
  `ParallelProcessExecutor`). Add a `--method r12` branch.
- `block_b_refinement/` scaffold (README, audit/, collect_metrics.py, RESULTS.md table).

**New for R12:**
1. `test_refiner.py::TeacherClient` — provider-agnostic LLM wrapper; reads creds from
   `.env`. One `generate_test(issue, gold_diff, changed_symbols) -> (code, func)` method
   (+ optional 1 refine-retry given a failure trace). **Default teacher = Claude** (latest
   Opus/Sonnet) via `anthropic` SDK; pluggable.
2. `test_refiner.py::DockerTestValidator.validate_new_test(entry, test_code, test_path)
   -> {passes_gold, fails_prepatch, gold_err, prepatch_err}` — the §3.2 two-run gate
   (writes the file into the container, runs pytest, reads exit/markers). This is the new
   capability vs B2 (B2 only ran *existing* ids; R12 runs *injected* code).
3. `test_refiner.py::backtranslate(entry, teacher, k_max=2)
   -> {kept:[(path,func,code)], audit:{...}}` — orchestrates prompt → gate → (retry).
4. `refine_dataset.py` r12 worker — for each surviving test, edit `make_test_spec`
   (§2 a/b/c), re-serialize `make_test_spec` + `extra_info`, write refined parquet
   + `audit/r12_<ts>.jsonl`.

**Untouched (the invariant):** `active_tests.py`, `swe_arsenal.py`, the launcher reward
flags — identical to control, exactly as in B2. R12 changes *only which tests are in
`FAIL_TO_PASS`* (here, adds to it).

---

## 5. Dataset & training (matched to B2 for comparability)

- **Input parquet:** `data_examples/rl_data/swerebench_verlive/` (64 train / 16 val) — the
  **same** set B2 used, so B1/B2/B0 are mutually comparable. (Port it from `test-refinement`
  or rebuild via `build_verified_live_dataset.py`.) → output
  `swerebench_verlive_r12/`.
- **Training:** SFT base `SWE-Master-4B-SFT`, **10 steps**, `TRAIN_BATCH_SIZE=16`,
  `rollout.n=4`, ATS flags identical to control (`RLLM_ACTIVE_TESTS_ENABLED=1`,
  `W_SOLVE=10`). Reuse the B2 launcher/orchestrator pattern (`run_control_live64.sh` with
  `TRAIN_PARQUET`/`VAL_PARQUET` overrides; detached orchestrator with checkpoint pruning,
  step-5 snapshot, save_freq=1, MAX_CKPT_KEEP=2).
- **Models out:** `/home/temp/models/SWE-Master-4B-ATS-B1-r12` (+ `-step5` snapshot).
- **The shared comparator is still B0-control-on-verlive** (original, unrefined tests, same
  data) — needed for *both* B1 and B2 vs-control claims; see RESULTS.md "Pending".

---

## 6. Metrics (per IMPLEMENTATION_PLAN §3, Refinement headline)

- Val solve-rate `val/test_score/swe` (vs B0-verlive, B2).
- **Generalization gap** = (solve on shown/active F2P) − (solve on hidden tests) per
  instance — the Block-B headline; R12 should *narrow* it on instances that were thin.
- **Backtranslation yield**: candidates generated, gate pass-rate, tests added/instance,
  F2P size before→after (audit trail).
- Reward trend `critic/rewards/mean` per step; grad-norm stability; tokens/solved.

---

## 7. Risks & mitigations

| Risk | Mitigation |
|--|--|
| Generated test passes pre-patch (non-discriminative) | hard gate drops it; only `fails_prepatch ∧ passes_gold` survive |
| Test fails pre-patch for the **wrong** reason (import/collection error, not the bug) | require `passes_gold=True` (proves runnable) *and* prefer assertion-level failure markers in `prepatch_err`; log both reasons in audit |
| Injected test id never runs in harness (spec edit wrong) | §2 single-instance round-trip check through `get_f2p_and_p2p_status` **before** batch |
| Teacher cost / flakiness | ≤2 LLM calls/instance; 64 instances; cache raw responses; `--subset`/`--smoke` first |
| Non-determinism / network in generated test | prompt forbids network/sleep/random; gate re-runs twice (stability) before keeping |
| Secret leakage | `.env` git-ignored (done); never log the key |

---

## 8. Open decisions (need confirmation before coding)

1. **Teacher LLM** — provider + model (you opened `.env` — presumably to add the key).
   Recommend **Claude** (latest Opus/Sonnet via `anthropic`); the client is pluggable
   (OpenAI/etc.) but I'll wire one provider first.
2. **Dataset** — confirm **`swerebench_verlive`** (recommended, matches B2). Alternative:
   the control's `swerebench_min_ftp5_v2` or `live64`, but those break B1-vs-B2 comparability.
3. **Scope of "add"** — *augment* (append survivors to existing F2P, recommended) vs
   *replace-if-degenerate* (only inject when F2P is a single/weak test). Augment is simpler
   and the cleaner ablation.

---

## 9. Definition of done (per IMPLEMENTATION_PLAN §7)

Env-gated & default-off · `backtranslate`/`validate_new_test` unit-tested offline ·
single-instance harness round-trip confirmed · `--smoke` refine on ~3 instances · full
64-instance refine with audit trail · 10-step matched-compute run · models exported ·
`RESULTS.md` B1 row + reward trend + yield filled · commit/push to
`test-refinement-back-translation`.
