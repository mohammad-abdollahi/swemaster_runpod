# B1 / R12 — Online backtranslation: decompose *stalled* F2P tests mid-episode

**Branch:** `test-refinement-back-translation` (off `test-selection`, the main ATS-RL
training branch). **Status:** design / plan — no training or runtime code changed yet.

> **Correction vs the IMPLEMENTATION_PLAN.** The plan framed R12 as an *offline* parquet
> transform that adds gold-patch tests to every instance. That is **not** this idea. The
> real idea (user, this session) is an **online, mid-episode refinement inside the
> ActiveTestScheduler**: when a `FAIL_TO_PASS` test stays *unsolved* for too long, decompose
> it (via backtranslation) into smaller, more meaningful sub-tests the policy can actually
> make progress on. This doc supersedes the offline framing.

---

## 0. The ATS-RL structure this plugs into (recap)

ATS reward is defined by an **`ActiveTestScheduler`** (`active_tests.py`) per instance — a
state machine over `FAIL_TO_PASS` (+ `PASS_TO_PASS` as a regression guard). Each F2P test
moves through four buckets:

- **POOL** → not shown yet.
- **ACTIVE** → shown in the injected `<active_tests>` prompt block (`prompt_block()`); size
  ∈ [`m_init=2`, `m_max=6`]; **each ACTIVE test has a stall counter** `self.active[t]`.
- **COOLDOWN** → recently passed; masked for `cooldown_steps=10`, then re-checked → RETIRED
  (still green) or ACTIVE (regressed).
- **RETIRED** → counts toward terminal reward.

**Tick loop (`swe_arsenal.py:371-400` → `active_tests.py::tick`).** With
`RLLM_ACTIVE_TESTS_MID_EPISODE=1`, every `k_steps=5` agent steps the runtime runs the tests
(`get_f2p_and_p2p_status`), builds a `pass_map`, and calls `scheduler.tick(pass_map)`. Per
ACTIVE test: **passed →** COOLDOWN (`+α`, `solves++`); **failed →** `self.active[t] += 1`,
and **when the stall counter reaches `k_steps` the code "stall-expands"** — promotes 2 more
POOL tests into ACTIVE (`−β`) and resets the counter (`active_tests.py:274-281`).

**Rewards.** Dense (per tick): `α·solves − β·expansions − δ·regressions − γ +
η/(|active|+1) + P2P`. Terminal: `w_solve·frac_solved` over `FAIL_TO_PASS` + P2P + token.

**The opening this idea exploits:** the scheduler *already* knows a test is stuck
(`self.active[t] >= k_steps`). Today the only response is to show *more* tests; it never
helps the agent understand the one it's failing.

---

## 1. The idea (corrected)

**Problem.** Some `FAIL_TO_PASS` tests are too hard/opaque for the current policy — a single
broad test that asserts several behaviors at once. The agent fails it for many steps; it
contributes **no learning signal** (binary-stuck), and stall-expand just adds noise.

**Refinement.** When a F2P test `t` stays unsolved past a stall threshold, **backtranslate
`t` into a small set of smaller, minimal, more-meaningful sub-tests** `{t₁…t_j}` — each
checking *one* facet of the behavior `t` covers — and surface *those* in the prompt instead
of `t`. The agent now gets a **decomposed curriculum**: it can pass `t₁`, then `t₂`, earning
dense reward and a clearer target, rather than staring at one all-or-nothing test.

**Why "backtranslation" works here.** The **gold patch** (`extra_info.patch`) encodes the
exact fixed behavior, and the **original test source** shows what `t` checks. Given (issue +
gold diff + `t`'s source), the teacher LLM can split `t` into minimal assertions that are
*derived from ground truth*, not guessed — strictly more reliable than issue-only splitting.

**Online, not offline.** Trigger = the scheduler's per-test stall counter, mid-episode. Only
genuinely-stuck tests are ever refined (selective, cheap-by-default).

---

## 2. Where it hooks in

`active_tests.py::tick()` Step 2, the **stall branch**
(`self.active[t] >= cfg.k_steps` → currently stall-expand). Add a refinement hook *there*:

```
if self.active[t] >= cfg.refine_after:        # new knob; default = k_steps (the stall point)
    subtests = REFINE(t)                       # decompose t → [t1..tj]  (cached; see §4)
    if subtests:
        replace t in ACTIVE with t1..tj        # t masked; sub-tests shown in prompt_block()
        record (t -> [t1..tj]) for terminal roll-up
    else:
        <fall back to existing stall-expand>
```

The refinement is **gated** by a new `RLLM_ACTIVE_TESTS_REFINE=1` (default **off** → the
scheduler is byte-for-byte the current control; exact reproducibility preserved).

---

## 3. Terminal-reward stance (key decision): **scaffold-only**

Sub-tests drive the **mid-episode dense reward + the prompt** (the curriculum), but the
**terminal `frac_solved` stays computed over the ORIGINAL `FAIL_TO_PASS`.** Rationale:

- Keeps the terminal reward (the cross-arm comparison metric) **identical to control/B2** —
  refinement changes *how the agent learns within an episode*, not the grading criterion.
- Avoids changing the reward denominator `N = |fail_to_pass|` (which `replace`-in-F2P would).
- **Roll-up rule:** when *all* sub-tests `{t₁…t_j}` pass, mark the parent `t` solved → move
  `t` to COOLDOWN (its normal terminal re-check then confirms the real fix). Passing all
  minimal sub-tests ⟹ the behavior is fixed ⟹ `t` should pass.

So the dense signal gets richer/decomposed; the terminal correctness bar is unchanged.

---

## 4. Components to build

1. **Scheduler trigger + bookkeeping** (`active_tests.py`):
   - `SchedulerConfig`: `refine_enabled=False`, `refine_after:int=k_steps`,
     `refine_max_subtests:int=4`, `refine_max_per_episode:int=3`.
   - In `tick()`: detect stall → call refinement hook → swap `t`→`{t₁…t_j}` in `active`;
     track `parent_of[t_i]=t` + `subtests_of[t]`; roll-up `t` when all children pass.
   - New `SchedulerEvent` fields: `refinements`, `subtests_added`, `subtests_solved`.
   - The scheduler holds a **callback** `refine_fn(test_id) -> [(node_id, code)]` injected by
     the env (keeps `active_tests.py` free of LLM/Docker deps; unit-testable with a stub).

2. **`TestDecomposer`** (`test_refiner.py`, teacher = OpenAI **`gpt-5.4-mini`**, key from
   `.env` var **`API_KEY_OPENAI`**, `openai` SDK v2.37): `decompose(issue, gold_diff,
   original_test_src) -> [(name, code)]` — "split this test into ≤K minimal pytest tests,
   each asserting one facet; self-contained; no network/sleep/random." ≤1 refine-retry.

3. **Mid-episode injection + gate** (`docker.py` runtime + `test_refiner.py`):
   - `runtime.run_adhoc_tests(test_code, node_ids, timeout) -> {node_id: bool}` — write the
     sub-test file into `/testbed`, run pytest, return pass/fail. (New capability vs B2/R22,
     which only ran *existing* spec ids; mirrors how `eval_script_list` injects gold tests.)
   - **First-time executable gate** for each sub-test (cached): must **fail on pre-patch**
     repo AND **pass on gold-patched** repo (the F2P invariant) — discard non-discriminating
     splits. Runs once per (instance, test) — see cache.

4. **Decomposition cache** (persistent, keyed `instance_id::test_id`): the same hard tests
   stall across rollouts/epochs/`rollout.n`. Compute the LLM split + gate **once**, store to
   disk (`block_b_refinement/cache/decompositions.jsonl`), reuse everywhere. This is what
   makes mid-episode LLM refinement affordable.

5. **Env wiring** (`swe_arsenal.py`): build the `refine_fn` callback (decomposer + cache +
   `run_adhoc_tests`), pass it into the scheduler when `RLLM_ACTIVE_TESTS_REFINE=1`; extend
   the mid-episode tick to also fetch sub-test pass status; log refinement telemetry on the
   `!!!!!!!` line.

**Untouched:** `terminal_reward()` math, `w_solve`, the parquet, the launcher reward flags
— identical to control. Refinement only changes the *mid-episode curriculum*.

---

## 5. Cost control (mid-episode LLM is the main risk)

- **Persistent decomposition cache** (the big win — first stall pays, all reuse).
- `refine_max_per_episode` cap; refine each `t` **≤1×/episode**.
- Gate sub-tests once, cache verdicts; never re-validate a cached split.
- Optional: precompute decompositions for the *known-historically-stuck* tests offline
  (from B2's rollout logs) to warm the cache before training.

---

## 6. Decisions (locked)

| Decision | Value |
|--|--|
| Teacher LLM | OpenAI **`gpt-5.4-mini`**, key `.env::API_KEY_OPENAI`, `openai` SDK v2.37 |
| Dataset | `swerebench_verlive` (64/16) — matches B2 → `…_r12` not needed (refinement is online; parquet unchanged) |
| Terminal reward | **scaffold-only** (original F2P unchanged; sub-tests = mid-episode curriculum) |
| Gating | `RLLM_ACTIVE_TESTS_REFINE=1` (default off); needs `RLLM_ACTIVE_TESTS_MID_EPISODE=1` |
| Train | **10 steps**, batch 16, rollout.n=4, SFT base `SWE-Master-4B-SFT` |
| Trigger | stall counter ≥ `refine_after` (default = `k_steps`); ≤`refine_max_subtests=4` |

---

## 7. ⚠️ Baseline note (matched ablation)

Refinement **requires the mid-episode pytest path** (`MID_EPISODE=1`), which is heavier than
the v1 prompt+terminal path that the B2/R22 and control runs used. So the clean comparator
for this arm is **ATS with mid-episode ON + refine OFF** (`MID_EPISODE=1`,
`REFINE=0`), *not* the v1 control. Plan both:
- **B0-mid** = ATS mid-episode, refine off (matched baseline for this arm).
- **B1-refine** = ATS mid-episode, refine on.
(The earlier `B0-control-on-verlive` flagged in RESULTS.md is the v1 baseline; still useful,
but the refine arm needs the mid-episode-matched B0-mid.)

---

## 8. Metrics

- Val solve-rate `val/test_score/swe` (B1-refine vs B0-mid).
- **Stuck-test recovery**: fraction of stalled tests that get solved *after* refinement
  (subtests_solved → parent roll-up) vs the stall-expand baseline.
- Dense-reward trajectory shape; `critic/rewards/mean` per step; grad-norm.
- Refinement telemetry: refinements/episode, sub-tests added/solved, cache hit-rate,
  teacher calls + cost.
- Generalization gap (Block-B headline).

---

## 9. Build sequence (each phase verifiable before the next)

1. **Scheduler trigger + roll-up + stub `refine_fn`** in `active_tests.py`; unit tests
   (stall → refine → swap → roll-up → terminal unchanged). No LLM/Docker yet.
2. **`TestDecomposer`** (gpt-5.4-mini) + **decomposition cache**; offline-test on a couple
   of known-stuck tests from B2 logs (eyeball the splits).
3. **`run_adhoc_tests`** in the runtime + the **executable gate**; round-trip-verify on ONE
   instance that an injected sub-test runs and scores in the live container (highest-risk).
4. **Env wiring** (`swe_arsenal.py` builds `refine_fn`); smoke a 3-5 step run with
   `MID_EPISODE=1 REFINE=1` — confirm refinements fire, telemetry logs, no crash.
5. **B0-mid** (refine off) + **B1-refine** (refine on), 10 steps each, verlive, SFT base,
   B2's launcher/orchestrator pattern → export `SWE-Master-4B-ATS-B1-r12` (+ step-5).
6. RESULTS.md B1 row (vs B0-mid, B2) + reward/recovery telemetry → commit/push.

## 10. Definition of done
Gated & default-off (control exactly recoverable) · scheduler unit tests green · single-
instance ad-hoc-injection round-trip confirmed · cache working · 3-5 step smoke with refine
on · B0-mid + B1-refine 10-step runs · models exported · RESULTS.md row + telemetry · commit
/push to `test-refinement-back-translation`.
