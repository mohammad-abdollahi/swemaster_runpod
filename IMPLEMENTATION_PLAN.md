# ATS-RL — Controlled Ablation Study: Implementation Plan

**Audience:** a fresh Claude Code session implementing and running the experiments.
**Read first, in order:** [`PIPELINE_HANDOFF.md`](PIPELINE_HANDOFF.md) (operational),
[`REWARD_ACTIVE_TEST_SELECTION.md`](REWARD_ACTIVE_TEST_SELECTION.md) (design),
[`METHODS_RANKING_README.md`](METHODS_RANKING_README.md) (what & why), then this file.

---

## 1. Goal & experimental design

We want to **measure the isolated effect** of test-selection, test-refinement, and
token-reduction methods on the ATS-RL pipeline. The design is **one factor at a
time (OFAT)**: freeze a single control configuration, then in each block vary only
the methods in *one* category, holding the other two categories fixed at the
control. This attributes any metric change to the method under test.

```
                 ┌── Block A: TEST SELECTION    (vary S2,S1,S11 | refinement=off, token=control)
ATS-RL control ──┼── Block B: TEST REFINEMENT   (vary R12,R22,R2 | selection=control, token=control)
                 └── Block C: TOKEN REDUCTION   (vary T1,T8,T24,T12/T17 | selection=control, refinement=off)
```

Every method is **env-var / config gated and OFF by default**, so the control is
recoverable exactly by setting all new flags off. Run each arm at **matched
compute** (same `TOTAL_EPOCHS`, `train_batch_size`, `rollout.n`, `max_steps`, same
parquet unless the method *is* the parquet change) against the control.

> **Implementation note on overlap.** DAPO Dynamic Sampling appears as both a
> selection idea (S2) and a token idea (T24). To keep the ablation clean we split
> them mechanistically: **S2 = offline instance curation by reward-variance**
> (changes *which instances* are in the parquet); **T24 = online group dropping at
> rollout time** (changes *which rollouts produce gradient*). They are tested in
> different blocks and never both on within one arm.

---

## 2. The frozen control configuration

This is ATS-RL exactly as it runs today (PIPELINE_HANDOFF §4 canonical launch),
**plus the two Phase-0 reward-correctness fixes folded in** (T1 + the truncation
masking) so the control itself is not buggy. Run the two fixes once, confirm the
smoke is clean, freeze, and treat this as `CONTROL` for all three blocks.

| Axis | Control value |
|---|---|
| Base model | `SWE-Master-4B-SFT` |
| Algorithm | GRPO, `algorithm.adv_estimator=loop`, `rollout.n=4` |
| Reward | ATS terminal: `w_solve=10`, `w_regress=w_solve/2`, bounded token penalty (T1 applied) |
| Truncation handling | mask (T8 applied) — see §5 Block C control note |
| Selection | none beyond existing `MIN_FTP` filter |
| Refinement | none |
| Parquet | `swerebench_min_ftp5_v2` (39 train / 8 val) |
| Steps | `TOTAL_EPOCHS=5`, `TRAIN_BATCH_SIZE=16`, `AGENT_MAX_STEPS=150`, `TRAJECTORY_TIMEOUT=5400` |

> If you prefer the control to be the *literal* current pipeline (token penalty +
> reward-halving, both un-fixed), run T1 and T8 as their own Block-C arms instead.
> Recommended: fold both into control (they are correctness fixes, not research
> variables) and still keep `=half` / unbounded as reachable ablation arms.

### Code surfaces (unchanged from prior plan)

| Concern | File | Symbol / lines |
|---|---|---|
| Reward state machine + terminal reward | `R2E-Gym/src/r2egym/agenthub/runtime/active_tests.py` | `ActiveTestScheduler`, `SchedulerConfig`, `terminal_reward()` L367-427, token penalty L411-414 |
| Env wiring | `DeepSWE_RL/rllm/rllm/environments/swe/swe_arsenal.py` | `reset()` L195, `compute_final_reward()` L301, `step()` L342, `_scheduler_config_from_env()` L50 |
| Trajectory reward / truncation | `DeepSWE_RL/rllm/rllm/engine/agent_execution_engine.py` | overlong_filter block L511-534 |
| Pytest / test universes in Docker | `R2E-Gym/src/r2egym/agenthub/runtime/docker.py` | `get_fail_to_pass_ids` L1738, `get_pass_to_pass_ids` L1693, `get_f2p_and_p2p_status` L1708, `run_tests` L1368 |
| Launcher + knobs | `DeepSWE_RL/rllm/examples/swe/train_active_tests_swe_master_4b_h200.sh` | — |
| Dataset build | `build_comprehensive_swerebench.py` | — |
| Scheduler unit tests | `R2E-Gym/src/r2egym/agenthub/runtime/test_active_tests.py` | 23 tests |

---

## 3. Metrics (log identically for every arm)

Report all of these per arm; the headline depends on the block.

| Metric | How measured | Headline for |
|---|---|---|
| **Solve rate** (held-out) | `val/test_score/swe` on the val parquet | all |
| **Zero-gradient-step fraction** | fraction of PPO steps where every group has `std(reward)=0` (parse `!!!!!!! ATS terminal` + group ids) | Selection |
| **Steps-to-target-reward** | PPO steps to reach a fixed reward-mean threshold | Selection, Token |
| **Tokens per solved instance** | Σ(prompt+completion tokens) ÷ #solved (derive from engine token counters; SWE-Effi EuTB-style) | Token |
| **Generalization gap** | (solve rate on shown/ACTIVE tests) − (solve rate on hidden tests) per instance | Refinement |
| **Valid-test yield / mutant kill-rate** | refiner audit trail | Refinement |
| **grad_norm stability** | TensorBoard | all (sanity) |
| **Wall-clock / GPU-hours per run** | run timestamps | all (cost) |

Tooling already exists (PIPELINE_HANDOFF §6): `analyze_trend.py`,
`analyze_active_tests.py`, `simulate_binary_baseline.py`, `dump_metrics_csv.py`,
`plot_metrics_csv.py`. Extend these to emit the new columns (zero-grad fraction,
tokens/solved, generalization gap). Save each arm to a distinct
`EXPERIMENT_NAME=<block>_<arm>_<ts>`.

**Per-arm checklist:** unit tests green → 3-5 step smoke (stable grad-norm, no
reward outliers) → full matched-compute run → append a result row to the block's
results table in `paper/PAPER_DRAFT.md §5`.

---

## 4. BLOCK A — TEST SELECTION

**Hold fixed:** refinement OFF, token = control. **Vary:** the selection method.

| Arm | Method | What changes | New flag(s) |
|---|---|---|---|
| A0 | Control | nothing | — |
| A1 | **S2** reward-variance instance curation | which instances are in the parquet | offline `select_variance.py` |
| A2 | **S1** Goldilocks difficulty banding | which instances are in the parquet | offline `select_goldilocks.py` |
| A3 | **S11** token-efficiency-weighted selection | which instances are in the parquet | offline `select_token_eff.py` |
| A4 (opt) | S2+S1 combined | both filters chained | — |

All three are **offline parquet transforms** (no trainer change → cleanest possible
ablation; the only thing that differs between arms is the training set). Each
consumes a one-time **probe pass**: run the control policy `n=4` times over the
candidate pool and record per-instance outcomes.

### Shared prerequisite — the probe pass
Write `probe_instances.py`: load the candidate parquet, run the SFT policy `n` times
per instance through the existing rollout path (reuse `SWEEnv` + the eval harness),
and dump a CSV with columns: `instance_id, frac_solved_mean, frac_solved_per_rollout
(list), reward_per_rollout (list), prompt_tokens_mean, completion_tokens_mean`.
Every Block-A arm is a different ranking/filter over this one CSV.

### A1 — S2: Reward-variance instance curation
- **Algorithm.** GRPO advantage for a group is `(r_i − mean(r)) / std(r)`; if
  `std(r)=0` the whole group contributes **zero gradient**. So prefer instances
  whose `n` rollouts have **high reward variance**. With our continuous
  partial-credit reward, score each instance `v(i) = Var(reward_per_rollout)` (or
  `Var(frac_solved_per_rollout)`); keep the top-K, or sample instances with
  probability ∝ `v(i)^α` (α≈1). Binary special case (DAPO): drop instances where
  all rollouts are all-correct or all-wrong.
- **Implement.** `select_variance.py`: read probe CSV → compute `v(i)` → write a
  filtered parquet keeping `v(i)>0` and top-K by variance. Optional: emit a
  `priority` column for the persistent-priority extension below.
- **Novel extension (the paper claim).** Persist priorities across epochs:
  `instance_priority[id] = EMA(reward_std)`; re-rank between epochs. This is
  *prioritized instance sampling*, inspired by PER (NOT literal PER — no TD-error /
  replay buffer in on-policy GRPO). For the sum-tree sampling primitive see PER ref.
- **Paper.** DAPO arXiv:2503.14476 (NeurIPS 2025); GRPO arXiv:2402.03300; PER
  arXiv:1511.05952.
- **GitHub.** Dynamic-sampling logic to mirror: `volcengine/verl` →
  `recipe/dapo/dapo_ray_trainer.py` and `verl/trainer/ppo/core_algos.py`
  (group std → advantage). PER sum-tree reference: `openai/baselines` →
  `baselines/deepq/replay_buffer.py`. (We implement the *offline* curation
  ourselves; these are for the variance/advantage math and the priority sampler.)

### A2 — S1: Goldilocks difficulty banding
- **Algorithm.** Train where the signal is richest: keep instances with
  `0 < frac_solved_mean < 1`, weight toward a target difficulty `p* ≈ 0.2`
  (SSR's cited optimum), and drop `frac_solved_mean ∈ {0,1}`. Weight function e.g.
  `w(i) = exp(−(frac_solved_mean(i) − p*)² / 2σ²)`.
- **Implement.** `select_goldilocks.py`: read probe CSV → apply band + weight →
  write parquet (or duplicate rows ∝ weight for a simple sampler). This *replaces*
  the coarse `MIN_FTP` heuristic with a measured difficulty estimate. The novelty
  vs SSR: we band on the **partial-credit pass-fraction** (finer) over **real
  SWE-rebench** instances, not self-generated bugs.
- **Paper.** Open-Reasoner-Zero arXiv:2503.24290 (annealing on a mined "hard" set —
  our primary code anchor); Bengio Curriculum Learning, ICML 2009. The p*≈0.2
  band target is a *conceptual* borrow from SSR arXiv:2512.18552 (no code there).
- **GitHub.** ✓ **`Open-Reasoner-Zero/Open-Reasoner-Zero`** — public PPO + curriculum
  code: difficulty-annealing stage uses `data/orz_math_13k_collection_hard.json`
  (the mined hard set; our analogue is the high-`frac_solved` tail), training
  scripts in `playground/`, core PPO in `orz/ppo/`. The banding/weighting itself is
  ~30 lines we write over the probe CSV; ORZ shows the annealing-stage wiring.

### A3 — S11: Token-efficiency-weighted selection
- **Algorithm.** Prefer instances that yield learning signal cheaply:
  `score(i) = signal(i) / cost(i)`, where `signal(i) = Var(reward)` (from A1) and
  `cost(i) = prompt_tokens_mean + completion_tokens_mean`. Keep top-K by `score`.
  This biases training toward informative-but-cheap instances → fewer tokens/solve.
- **Implement.** `select_token_eff.py`: read probe CSV → compute `score(i)` → write
  parquet. Reuse the EuTB framing (effectiveness under token budget) for reporting.
- **Paper.** Novel framing; SWE-Effi arXiv:2509.09853 (EuTB = AUC of resolve-rate
  vs tokens/issue).
- **GitHub.** ⚠️ Not a code method — SWE-Effi is a benchmark/metric:
  project site `centre-for-software-excellence.github.io/SWE-Effi/` (leaderboard +
  metric code). We implement the selection scorer ourselves.

**Block A headline:** zero-gradient-step fraction ↓ and steps-to-target ↓ vs A0,
at equal or better solve rate. (S11 additionally: tokens/solved ↓.)

---

## 5. BLOCK B — TEST REFINEMENT  (the core contribution)

**Hold fixed:** selection = control, token = control. **Vary:** the refinement
method. All three are **offline parquet transforms** that rewrite/augment each
instance's `FAIL_TO_PASS` *before* training; ATS then runs unchanged.

| Arm | Method | What changes | Output |
|---|---|---|---|
| B0 | Control | original tests | control parquet |
| B1 | **R12** backtranslation from gold patch | adds discriminative F2P tests | refined parquet |
| B2 | **R22** mutation-specificity gate | filters non-discriminative tests | gated parquet |
| B3 | **R3 (InfCode)** adversarial test–patch co-refinement | strengthens tests against wrong patches | refined parquet |
| B4 (opt) | R12→R22 pipeline | generate then gate | best parquet |

> **Replacement note.** B3 was originally R2 (Otter/e-Otter++ execution-feedback),
> but Otter's *system* code is not public (only its benchmark). We replace it with
> **R3 (InfCode)**, which has a public, runnable implementation and occupies the
> same "iteratively refine tests against execution feedback" role — with the
> stronger adversarial framing that exactly matches our discrimination goal.

### Shared infrastructure
- New module `R2E-Gym/src/r2egym/agenthub/runtime/test_refiner.py`.
- New driver `refine_dataset.py` (repo root): reads a parquet, refines each
  instance, writes a new parquet with an **audit trail** (`refined`, `refiner`,
  original vs new test ids, kill-rate). Never silently mutate — always keep provenance.
- **Confirm columns first:** the gold patch and test patch ride on the dataset row
  (`SWEEnv.entry`) under SWE-bench keys. Run
  `python -c "import pandas; print(pandas.read_parquet('<train>.parquet').columns.tolist())"`
  to confirm `patch` / `test_patch` / test-spec names before coding.
- **Executable validation requires the Docker image** per instance — reuse the same
  `DockerRuntime` the env uses; candidate tests are run there.

### B1 — R12: Backtranslation of minimal tests from the gold patch
- **Algorithm.** The gold diff exactly encodes the changed behavior → derive tests
  from it (strictly more signal than issue-only generation). Per instance:
  1. Teacher LLM prompt = issue + **gold diff** + changed files/symbols → "write a
     minimal pytest exercising the changed behavior."
  2. **Executable gate (mandatory):** candidate must **PASS on the gold-patched
     repo** AND **FAIL on the pre-patch repo**. Discard otherwise. (No LLM-judge
     needed — execution is ground truth.)
  3. Add survivors to `FAIL_TO_PASS` (or replace a degenerate single test).
- **Implement.** `test_refiner.backtranslate(instance) -> list[Test]`; validate via
  two `run_tests` invocations (pre/post patch) in the instance image.
- **Difficulty.** Easy-Medium (1 LLM call + 2 harness runs we already have).
- **Paper.** Extends SWE-RL "patches as supervision" arXiv:2502.18449 (NeurIPS 2025).
- **GitHub.** `facebookresearch/swe-rl` — reward + oracle-patch handling patterns
  (the most reusable reference; test *generation* from the diff is our addition).

### B2 — R22: Mutation-validated specificity gate
- **Algorithm.** Inverse mutation testing using our **rollout failure logs as a
  free mutant pool**. Keep a test only if it **passes gold** AND **fails on ≥M of k
  logged wrong-patch rollouts** (default `M = ceil(0.5k)`). Tests that pass wrong
  patches are non-discriminative → drop. Record kill-rate as a quality score.
- **Implement.** `test_refiner.specificity_gate(test, gold_patch, wrong_patches)`;
  parse wrong patches from training logs / trajectory dumps. Fallback when rollout
  logs are thin: synthetic AST-operator mutants (note in audit trail). Composes
  after B1 (generate → gate = arm B4).
- **Difficulty.** Easy-Medium — **harness-only, needs no external code** (run a test
  against k known-wrong patches, keep if it kills ≥M). Not blocked by the lack of
  SSR code.
- **Paper.** Jia & Harman, IEEE TSE 37(5) 2011 (mutation testing); inverse-mutation
  framing also in SSR arXiv:2512.18552 (conceptual; no code).
- **GitHub.** ✓ For the discrimination loop, reference **`Tokfinity/InfCode`**
  (`src/` Patch-Selector + Test-Patch-Generator) — but we reuse rollout-failure
  patches as the mutant pool instead of running its Code-Patch-Generator. The gate
  itself is ~20 lines over our existing `run_tests` harness.

### B3 — R3 (InfCode): Adversarial test–patch co-refinement  *(replaces R2)*
- **Algorithm.** A Test-Patch-Generator strengthens the test to expose bad
  behavior; a Code-Patch-Generator produces candidate (often wrong) patches; a
  Selector keeps the most reliable. Run this as an **offline test-strengthening
  loop**: for each instance, iteratively harden the `FAIL_TO_PASS` test until it
  **passes the gold patch** and **fails the candidate wrong patches**. Crucially,
  **we supply the wrong patches from our rollout logs for free** — so we run only
  InfCode's Test-Patch-Generator + Selector, not its Code-Patch-Generator. This is
  the execution-feedback refinement role (original R2) with a published, code-backed
  adversarial loop and a discrimination objective identical to ours.
- **Implement.** `test_refiner.adversarial_strengthen(instance, wrong_patches, K=3)`,
  mirroring InfCode's `src/` Test-Patch-Generator + Patch-Selector against the
  instance Docker image. Reuse rollout-failure patches (same pool as R22). Bounded
  to `K ≤ 3` iterations to cap teacher cost. Output: refined parquet + audit trail.
- **Difficulty.** Medium (adapt a public loop; we skip the patch-generator half).
- **Paper.** InfCode arXiv:2511.16004 (adversarial test/patch refinement;
  79.4% SWE-bench Verified).
- **GitHub.** ✓ **`Tokfinity/InfCode`** — runnable code. Read `src/` (Test-Patch-
  Generator + Code-Patch-Generator + Patch-Selector + tools: File Editor, ripgrep
  searcher, Bash executor, LLM API manager); entry points `_run.py` (single),
  `_batch_run.py` (batch), `_build_image.py` (Docker). We reuse the Test-Patch-
  Generator + Selector and feed our rollout failures as the adversarial patches.
- **Optional complement (public library, zero reimplementation):** for instances
  where example-based tests underspecify, add **property-based tests** via the
  `hypothesis` pip library (`HypothesisWorks/hypothesis`) — generate invariants
  instead of fixed assertions. This is consolidated-table idea **R8** (MacIver &
  Hatfield-Dodds, JOSS 2019), usable as-is without writing a generator.

**Targeting glue (R19, recommended, not a separate arm):** rank instances by
*signal informativeness* (per-instance reward-variance from Block A's probe CSV)
and spend refinement teacher-budget on the **lowest-signal instances first**.
Pure analysis over data we already have — this is the "active refinement" thread.

**Block B headline:** generalization gap small + solve-rate ↑ on instances that
were degenerate before refinement (the A/B = refined parquet vs control parquet).

---

## 6. BLOCK C — TOKEN REDUCTION

**Hold fixed:** selection = control, refinement OFF. **Vary:** the token method.
These are **trainer/reward changes** (not parquet changes).

| Arm | Method | Where it acts | New flag |
|---|---|---|---|
| C0 | Control | — | — |
| C1 | **T1** bounded overlong reward shaping | terminal reward | `RLLM_ACTIVE_TESTS_LEN_*` |
| C2 | **T8** compact filtering (mask truncated) | trajectory loss | `RLLM_COMPACT_FILTER` |
| C3 | **T24** dynamic sampling (online group drop) | rollout→loss batch | `RLLM_DROP_FLAT_GROUPS` / verl `filter_groups` |
| C4 | **T12/T17 (SWE-Master)** context manager + LSP | rollout scaffold (tokens/turn) | `RLLM_CONTEXT_MANAGER` / `RLLM_LSP` |

> **Replacement note.** C4 was originally T23 (R4P/Mini-SE fast verifier), but R4P
> has **no public code or model**. We replace it with **T12/T17 (SWE-Master Context
> Manager + LSP)** — the token-reduction setup from our *own base model's* paper,
> with public code that lives in the **same repo family as our pipeline**
> (`RUCAIBox/SWE-Master` contains `DeepSWE_RL/rllm/` and `R2E-Gym/`). Different
> mechanism (cuts tokens-per-trajectory rather than pytest cost) but the same
> headline metric and a clean, code-backed integration.

> Control note: if T1 and T8 are folded into the frozen control (§2 recommended),
> then C1/C2 become *ablation-off* arms (turn them back to the old buggy behavior to
> measure their contribution), and C3/C4 are the genuinely new arms. Keep both
> framings reproducible via the flags.

### C1 — T1: Bounded overlong reward shaping
- **Algorithm.** Replace the linear `tokens/token_ref` penalty (which produced the
  −989 outlier, PIPELINE_HANDOFF §7.3) with a DAPO soft band bounded to `[−1,0]`:
  ```
  over = tokens_used − (len_max − len_cache)
  band = 0 if over ≤ 0 else min(over / len_cache, 1)      # ∈ [0,1]
  r_token = −lam_token · band                              # ∈ [−lam_token, 0]
  ```
  Outliers become structurally impossible.
- **Implement.** Edit `active_tests.py` `terminal_reward()` L411-414; add
  `len_cache_tokens=4096`, `len_max_tokens=24576` (=`MAX_RESPONSE`) to
  `SchedulerConfig`; expose `RLLM_ACTIVE_TESTS_LEN_CACHE/LEN_MAX` in
  `_scheduler_config_from_env()`. **Also use real token counts**, not
  `tokens_used_approx` (char proxy ≈ 4× real): plumb the engine's
  `prompt_tokens`/`env_tokens_len` into `compute_final_reward`, or divide proxy by
  4 and document.
- **Paper.** DAPO arXiv:2503.14476 (Overlong Reward Shaping = `overlong_buffer`).
- **GitHub.** `volcengine/verl` → `recipe/dapo/` (the `overlong_buffer` penalty
  config + linear penalty up to `penalty_factor`); `BytedTsinghua-SIA/DAPO` (paper/configs).
- **Validate.** Unit test: `tokens_used=1e9` ⇒ `r_token ≥ −lam_token`; normal
  trajectory ⇒ `r_token ≈ 0`.

### C2 — T8: Compact filtering (mask, don't halve, truncated trajectories)
- **Algorithm.** Loss-mask trajectories that hit max-context / timeout / max-steps
  instead of reinforcing a halved reward (DeepSWE's recipe). Only deliberate
  submissions get gradient.
- **Implement.** `agent_execution_engine.py` L522-524 currently sets
  `reward_half=True` for `TRUNCATION/MAX_STEPS/TRAJ_TIMEOUT`. Add `RLLM_COMPACT_FILTER`:
  `mask` → set `masked_out=True` (the commented line at L524 is exactly this);
  `half` → current behavior (ablation). **First check** whether
  `algorithm.mask_truncated_samples=True` (already a verl knob in the launcher)
  achieves this at the trainer level — prefer the config flip if so.
- **Caveat.** Masking removes signal on never-finishable instances → keep
  `max_steps=150` (PIPELINE_HANDOFF §7.2). Measure solve-rate + grad-norm both ways.
- **Paper.** DeepSWE, Together AI 2025 (blog; no arXiv).
- **GitHub.** `rllm-org/rllm` (active; `agentica-project/rllm` is archived →
  redirects). Entry: `examples/swe/run_deepswe.py`; GRPO++ / compact-filtering
  loss-masking + length-norm in the `rllm/` trainer package.

### C3 — T24: Online dynamic sampling (drop flat groups)
- **Algorithm.** After a prompt's `n=4` rollouts are scored, if `std(reward)=0`
  (all-correct or all-wrong) the group gives zero gradient → drop it and oversample
  to refill the batch. DAPO reports faster convergence per the paper.
- **Implement.** **First check** if the installed verl exposes DAPO `filter_groups`:
  ```
  python -c "import verl, os; print(os.path.dirname(verl.__file__))"
  grep -rn "filter_groups\|num_gen_batches" <that path>/trainer/ppo/ray_trainer.py
  ```
  If present → enable in launcher (`algorithm.filter_groups.enable=True`,
  `max_num_gen_batches=...`). If absent → implement at the rllm rollout layer:
  compute per-prompt `std(reward)`; drop `std==0` groups from the loss batch behind
  `RLLM_DROP_FLAT_GROUPS=1`, logging `dropped_flat_groups` per step (drop-only
  first; refill/oversample is the fuller version).
- **Distinction from S2.** S2 curates the *dataset offline*; T24 drops *rollouts
  online*. Never run both in the same arm.
- **Paper.** DAPO arXiv:2503.14476.
- **GitHub.** `volcengine/verl` → `recipe/dapo/dapo_ray_trainer.py` (the canonical
  dynamic-sampling loop to mirror).

### C4 — T12/T17 (SWE-Master): Context Manager + LSP  *(replaces T23)*
- **Algorithm.** Two orthogonal token cuts from our base model's own paper:
  - **T12 Context Manager:** compress old history into NL summaries + keep a
    high-fidelity recent-turns sliding window (reported −42% input / −50% peak
    context on a large model). Fewer tokens carried per turn → fewer tokens/solve.
  - **T17 LSP integration:** replace text/grep navigation with deterministic
    language-server tools (reported −23.7% input / −16.3% output / −17.5% turns).
- **Implement.** Both ship as code in the **same repo family as ours**:
  - Context Manager: port `R2E-Gym/rollout_foundation_model/memory/` (e.g.
    `run_deepseek_v3_memory.sh`, `sft_process_memory_data_01.py`) into the rollout
    path; wire into `SWEEnv` / the r2egym scaffold behind `RLLM_CONTEXT_MANAGER=1`.
  - LSP: port `R2E-Gym/rollout_foundation_model/lsp/` tools; expose them to the
    agent behind `RLLM_LSP=1`. Add the LSP server to the instance Docker images.
  - Scaffold/inference change only — **does not touch the GRPO loop**, so it's a
    clean ablation arm. Measure tokens/solved and turns vs control.
- **Difficulty.** Medium (Context Manager) / Medium-Hard (LSP needs a language
  server per image).
- **Paper.** SWE-Master arXiv:2602.03411 (Context Manager + LSP token reductions).
- **GitHub.** ✓ **`RUCAIBox/SWE-Master`** (MIT) — contains `R2E-Gym/rollout_foundation_model/memory/`
  (Context Manager) and `R2E-Gym/rollout_foundation_model/lsp/` (LSP), plus
  `DeepSWE_RL/rllm/` and `R2E-Gym/` — i.e. the upstream of our own tree, so the
  code is directly mergeable.
- **Deferred (was T23):** the R4P fast-verifier idea (attack pytest cost to unlock
  v2 dense reward) is parked until a public R4P implementation/model exists. If you
  still want a fast-verifier arm, it must be trained/obtained first — out of scope
  for this study.

**Block C headline:** tokens/solved ↓ and grad stability ↑ at equal solve rate;
C4 (Context Manager + LSP) additionally cuts tokens/turn and episode length.

---

## 7. Run plan & bookkeeping

1. **Land control:** apply T1 + T8 fixes, run smoke, freeze as `CONTROL`. Record its
   metrics — every arm is reported as a delta vs CONTROL.
2. **Block A:** probe pass → A1/A2/A3 (and A4) → fill the Selection results table.
3. **Block B:** build refined parquets → B1/B2/B3 (and B4) → fill the Refinement table.
4. **Block C:** C1/C2/C3 (and C4 if pursued) → fill the Token table.
5. Each arm: distinct `EXPERIMENT_NAME=<block>_<arm>_<ts>`, outputs under
   `/home/temp/training_outputs/active_tests/`. Clean containers + `ray stop` between
   runs (PIPELINE_HANDOFF §7.6). Confirm latest checkpoint after each run (§7.1).
6. After all blocks: a final **combined arm** (best selection + best refinement +
   best token method) to test additivity.

### Definition of done (per arm)
Env-var/config gated & default-off · unit tests green · clean 3-5 step smoke ·
matched-compute full run · parse-stable `!!!!!!!` log line · result row appended to
`paper/PAPER_DRAFT.md §5`.

---

## 8. Verified references & repositories

All selected arms now have a public-code anchor (methods whose original grounding
lacked code were replaced or re-anchored — see the replacement notes in §5–§6).

| Arm | Method | Paper | Official GitHub — ✓ usable code |
|---|---|---|---|
| A1 | DAPO variance / dynamic sampling (S2) | arXiv:2503.14476, NeurIPS 2025 | ✓ `BytedTsinghua-SIA/DAPO`; impl `volcengine/verl` `recipe/dapo/dapo_ray_trainer.py`, `trainer/ppo/core_algos.py` |
| A2 | Goldilocks banding (S1) | arXiv:2503.24290 (code); p* from arXiv:2512.18552 (concept) | ✓ `Open-Reasoner-Zero/Open-Reasoner-Zero` (`orz/ppo/`, `playground/`, hard-set anneal) |
| A3 | Token-efficiency selection (S11) | arXiv:2509.09853 (metric) | site `centre-for-software-excellence.github.io/SWE-Effi/`; scorer is ours |
| — | GRPO baseline | arXiv:2402.03300 | ✓ `volcengine/verl` `trainer/ppo/core_algos.py`, `ray_trainer.py` |
| — | PER priority sampler (S2 ext) | arXiv:1511.05952 | ✓ `openai/baselines` `deepq/replay_buffer.py` |
| B1 | Backtranslation from gold patch (R12) | arXiv:2502.18449, NeurIPS 2025 | ✓ `facebookresearch/swe-rl` (reward/oracle patterns) |
| B2 | Mutation specificity gate (R22) | IEEE TSE 37(5) 2011 | ✓ `Tokfinity/InfCode` (selector ref); gate is harness-only |
| B3 | **Adversarial co-refinement (R3 — replaces R2)** | arXiv:2511.16004 | ✓ `Tokfinity/InfCode` (`src/`, `_run.py`, `_batch_run.py`) |
| B3+ | Property-based tests (R8, optional) | JOSS 2019 | ✓ `HypothesisWorks/hypothesis` (pip library, use as-is) |
| C1 | Overlong reward shaping (T1) | arXiv:2503.14476 | ✓ `volcengine/verl` `recipe/dapo/` (`overlong_buffer`); also ✓ `NVlabs/DLER` (`verl-dler/`) for the corrected-penalty variant |
| C2 | Compact filtering (T8) | Together AI blog 2025 | ✓ `rllm-org/rllm` (`examples/swe/run_deepswe.py`) |
| C3 | Dynamic sampling online (T24) | arXiv:2503.14476 | ✓ `volcengine/verl` `recipe/dapo/dapo_ray_trainer.py` |
| C4 | **Context Manager + LSP (T12/T17 — replaces T23)** | arXiv:2602.03411 | ✓ `RUCAIBox/SWE-Master` (`R2E-Gym/rollout_foundation_model/memory/` + `/lsp/`) — upstream of our tree |

**Replaced (no public code):** R2 Otter (benchmark-only `IBM/TDD-Bench-Verified`) →
**R3 InfCode**; T23 R4P/Mini-SE (no code, arXiv:2510.22775) → **T12/T17 SWE-Master**.
**Re-anchored off code-less SSR (arXiv:2512.18552):** S1 → Open-Reasoner-Zero code;
R22 → harness-only + InfCode selector reference.

**Citation corrections vs the source doc:** e-Otter++ is `2502.05368`/`2508.06365`
(NOT `2605.04320`, the Java variant); 80/20-rule is `2506.01939` (NOT `2601.01718`,
and it has **no usable code** — website only); DAPO is `2503.14476`.
