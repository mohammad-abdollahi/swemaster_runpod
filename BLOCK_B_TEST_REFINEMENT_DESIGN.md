# Block B — Test Refinement: Design & Implementation

**Branch:** `test-refinement`
**Created:** 2026-06-18
**Scope:** Ablation of *test refinement* for ATS-RL. Three methods, each an **offline
parquet transform** that rewrites/augments each instance's `FAIL_TO_PASS` set *before*
training; the ATS reward machinery runs unchanged (per IMPLEMENTATION_PLAN.md §5).

> Implementation order (user-directed 2026-06-18): **R22 first, end-to-end, no API key.**
> R12 and R3 need a teacher LLM — deferred until an API key is added to a `.env`
> (see §6 "Deferred"). Remind the user before starting those.

---

## 1. The three arms

| Arm | Method | Needs LLM? | What it does to `FAIL_TO_PASS` |
|--|--|--|--|
| **B0** | Control | no | unchanged (original tests) |
| **B2 / R22** | mutation-specificity gate | **NO** ✅ | **drops** non-discriminative tests (keep only those that kill mutants) |
| **B1 / R12** | backtranslation from gold patch | yes (teacher) | **adds** minimal tests derived from the gold diff |
| **B3 / R3** | InfCode adversarial co-refinement | yes (teacher) | **strengthens** tests against wrong patches |
| B4 (opt) | R12→R22 pipeline | yes | generate then gate |

Headline metric (Block B): **generalization gap** (solve-rate on shown/ACTIVE tests −
solve-rate on hidden tests) ↓, and solve-rate ↑ on instances that were degenerate before
refinement. Plus **valid-test yield / mutant kill-rate** from the refiner audit trail.

---

## 2. Confirmed code surfaces (verified 2026-06-18)

- **Dataset row.** Parquet (verl format) columns: `data_source, prompt, ability,
  reward_model, extra_info`. The SWE-bench fields live inside **`extra_info`** (a JSON
  string): keys include `patch` (gold), `test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`,
  `base_commit`, `docker_image`, `image_name`, `make_test_spec`, `instance_id`, …
  → Refiners parse `extra_info`, rewrite `FAIL_TO_PASS`, re-serialize.
- **Per-instance Docker validation.** `DockerRuntime(ds=<extra_info dict>, command=
  ["/bin/bash","-l"], backend="docker")` — the `ds` is exactly the `extra_info` dict the
  training env uses (`swe_arsenal.py`: `self.entry = extra_info; EnvArgs(ds=self.entry)`).
  Methods: `.checkout(commit)`, `.apply_patch(patch_str)`, `.run_tests()`,
  `.get_f2p_and_p2p_status(timeout)` → `({f2p_id: passed_bool}, {p2p_id: passed_bool})`.
  (`R2E-Gym/src/r2egym/agenthub/runtime/docker.py`.)
- **Teacher LLM (for R12/R3 only).** Codebase calls LLMs via `litellm.completion(...)` or
  `openai.Client(base_url=..., api_key=...)`. **No API key set** → R12/R3 deferred.
- **Wrong-patch pool (for R22/R3).** Rollout dumps
  `…/<run>/chat_completions/*.jsonl` carry `traj_score` + the final patch, but **no
  `instance_id`** → unreliable to map back. **R22 default uses synthetic operator mutants
  (offline, self-contained)**; rollout-log patches are an optional add-on later.

---

## 3. R22 — Mutation-validated specificity gate (THIS PHASE)

**Goal.** Keep a `FAIL_TO_PASS` test only if it actually *discriminates* the fix: it must
**pass the gold-patched repo** and **fail on ≥M of k wrong-patch "mutants"**
(`M = ceil(0.5·k)`). Non-discriminative tests (pass even wrong code) are dropped. The
fraction of mutants a test kills = its **kill-rate** (recorded as a quality score).

**Mutant pool (no LLM, no rollout logs needed):**
1. **Mutant #0 = pre-patch state** (the original bug). A real F2P test MUST fail here by
   definition — this alone catches mislabeled/degenerate tests cheaply.
2. **Synthetic operator mutants of the gold diff.** Mutate single tokens in the gold
   patch's *added* (`+`) code lines — comparison flips (`==`↔`!=`, `<`↔`>=`…), arithmetic
   swaps (`+`↔`-`, `*`↔`/`), boolean (`and`↔`or`, `True`↔`False`, ±`not`), numeric-literal
   tweaks (`0`↔`1`, n→n+1). Each single-token change yields a still-applyable patch
   (line counts unchanged) representing "gold with one bug" — a good test should kill it.

**Per-instance procedure** (`specificity_gate`):
```
v = DockerTestValidator(extra_info)              # builds DockerRuntime
gold_status = v.status_for_patch(gold_patch)     # checkout base → apply gold → run F2P
base_status = v.status_for_patch(None)           # pre-patch (mutant #0)
mutants = OperatorMutator(gold_patch).generate(k)
mutant_status = [v.status_for_patch(m) for m in mutants]
for t in FAIL_TO_PASS:
    if not gold_status[t]: drop(t, reason="fails on gold")          # mislabeled
    kills = (base_status[t] is False) + Σ(mutant_status[i][t] is False)
    kill_rate = kills / (k + 1)
    keep(t) iff kill_rate >= 0.5
new_F2P = survivors  (fallback: if empty, keep the single highest-kill-rate test + flag)
```
**Cost control.** One test-suite run per mutant (not per test); parallelize across
instances (multiprocessing, concurrency≈8–16 containers like the rollout). `k` configurable
(default 6 → 7 suite-runs/instance). Subset of instances allowed (user OK'd).

**Audit trail (never silently mutate):** per instance record `instance_id, refiner="R22",
orig_f2p, kept_f2p, dropped_f2p, kill_rate_per_test, k, M, fallback_used`. Written to
`block_b_refinement/audit/<arm>_<ts>.jsonl`.

---

## 4. Files

| File | Role |
|--|--|
| `R2E-Gym/src/r2egym/agenthub/runtime/test_refiner.py` | core: `OperatorMutator`, `DockerTestValidator`, `specificity_gate` (R22), `backtranslate` (R12 — deferred stub), `adversarial_strengthen` (R3 — deferred stub), audit dataclasses |
| `refine_dataset.py` (repo root) | CLI driver: read parquet → refine each instance (`--method r22`) → write refined parquet + audit trail; `--subset N`, `--mutants k`, `--workers W` |
| `block_b_refinement/mutation_selfcheck.py` | offline unit test of `OperatorMutator` (no Docker) |
| `block_b_refinement/collect_metrics.py` | pull a run's metrics (solve rate, grad-norm, tokens, zero-grad frac) from TB/log → `metrics/<arm>.csv` row |
| `block_b_refinement/README.md` | how to run each arm + reproduce |
| `block_b_refinement/metrics/RESULTS.md` | comparison table: control vs B1/B2/B3 |
| `block_b_refinement/audit/` | refinement audit trails |

---

## 5. Run plan (R22 end-to-end)

1. **Refine:** `python refine_dataset.py --method r22 --in <live64 train> --out
   data_examples/rl_data/ats_B2_r22/train_00000_verl.parquet --mutants 6 --workers 12`
   (copy the live64 val parquet into the arm dir unchanged — we refine TRAIN only; val
   measures held-out generalization). Smoke first on `--subset 2` to confirm the
   base→FAIL / gold→PASS invariant holds in-container.
2. **Train 10 epochs** from SFT base on the gated parquet (reuse `run_control_live64.sh`
   with `TOTAL_EPOCHS=10`, `MODEL_PATH=/home/temp/models/SWE-Master-4B-SFT`,
   `EXPERIMENT_NAME=B2_r22_<ts>`, `PROJECT_NAME=ablation_block_b`). Subset of instances if
   the gated set is large; pick `TRAIN_BATCH_SIZE` dividing the kept count.
3. **Metrics:** after the run, `collect_metrics.py` → `metrics/B2_r22.csv`; append a row to
   `metrics/RESULTS.md` next to control (`/home/temp/models/SWE-Master-4B-ATS-control-live64`,
   val 0.125). Export the model → `/home/temp/models/SWE-Master-4B-ATS-B2-r22`.
4. Compare vs control on: held-out solve rate, generalization gap, kept/dropped counts,
   kill-rate distribution.

---

## 6. Deferred (need API key → `.env`)

- **B1 / R12 backtranslation:** teacher prompt = issue + gold diff → minimal pytest;
  executable gate = pass gold ∧ fail pre-patch; add survivors to F2P.
- **B3 / R3 adversarial:** teacher Test-Patch-Generator strengthens F2P against wrong
  patches (rollout failures or synthetic mutants), K≤3 iters; keep if pass-gold ∧
  fail-wrong.
- Both will reuse `DockerTestValidator` + the audit-trail plumbing built now; only the
  generation step is new. **Reminder to user:** add `OPENAI_API_KEY` (or provider key +
  `OPENAI_API_BASE`) to `.env`, then `--method r12 / r3`.
