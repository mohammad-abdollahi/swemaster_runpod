# B1 / R12 — Online backtranslation: decompose *stalled* F2P tests into prompt guidance

**Branch:** `test-refinement-back-translation` (off `test-selection`, the main ATS-RL
training branch). **Status:** design / plan — no training or runtime code changed yet.

> **Correction vs the IMPLEMENTATION_PLAN.** The plan framed R12 as an *offline* parquet
> transform that adds gold-patch tests to every instance. That is **not** this idea. The
> real idea (user, this session) is an **online, mid-episode refinement inside the
> ActiveTestScheduler**: when a `FAIL_TO_PASS` test stays *unsolved* for too long, decompose
> it (via backtranslation) into smaller, more meaningful sub-tests so the policy can
> understand and make progress on it. This doc supersedes the offline framing.

> **HARD CONSTRAINT (user): the reward is unchanged.** Refinement must NOT alter the ATS-RL
> reward in any way — neither the per-tick dense reward (`tick()`) nor the terminal reward
> (`terminal_reward()`), and both stay computed over the **original** `FAIL_TO_PASS`. The
> decomposition is **prompt-only**: it changes *what the agent reads* about a stuck test,
> never how anything is scored. Sub-tests are never executed for reward and never enter the
> scheduler's reward bookkeeping.

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
and **when the stall counter reaches `k_steps` the code "stall-expands"** — promotes 2 POOL
tests into ACTIVE (`−β`) and resets the counter (`active_tests.py:274-281`).

**Rewards (unchanged by this work).** Dense (per tick): `α·solves − β·expansions −
δ·regressions − γ + η/(|active|+1) + P2P`. Terminal: `w_solve·frac_solved` over
`FAIL_TO_PASS` + P2P + token.

**The opening this idea exploits:** the scheduler *already* knows a test is stuck
(`self.active[t]`). Today the only response is to show *more* tests; it never helps the agent
*understand* the one it keeps failing.

---

## 1. The idea (corrected) — reward-free, prompt-only

**Problem.** Some `FAIL_TO_PASS` tests are too hard/opaque — one broad test asserting several
behaviors at once. The agent fails it for many steps; it gives little learning signal and
stall-expand just piles on more tests.

**Refinement.** When a F2P test `t` stays unsolved past a stall threshold, **backtranslate
`t` into a small set of smaller, minimal, more-meaningful sub-tests** `{t₁…t_j}` — each
checking *one* facet of what `t` covers — and **inject them as guidance into `t`'s entry in
the prompt block.** The agent now reads a decomposed, clearer spec of what `t` requires
instead of one opaque all-or-nothing test.

**The reward never moves.** `t` stays the ACTIVE test; it is still evaluated and rewarded
exactly as ATS-RL does today (dense + terminal, over the original `FAIL_TO_PASS`). The
sub-tests are **shown, never scored** — the only effect is richer prompt content for `t`.
(The decomposition text costs prompt tokens, which the *existing* `r_token` already accounts
for — that is ATS-RL's normal information↔token trade-off, not a reward change.)

**Why "backtranslation" makes the decomposition trustworthy.** The **gold patch**
(`extra_info.patch`) encodes the exact fixed behavior and the **original test source** shows
what `t` checks; given (issue + gold diff + `t`'s source), the teacher LLM splits `t` into
minimal sub-tests derived from ground truth, not guessed.

---

## 2. Where it hooks in (prompt-only; reward branches untouched)

`active_tests.py::tick()` already increments the stall counter and runs stall-expand. We
**leave all of that exactly as-is** (so the reward and state machine are identical) and add a
*side annotation* that only `prompt_block()` reads:

```
# inside tick(), at the existing stall detection — ADD, do not replace:
if cfg.refine_enabled and self.active.get(t, 0) >= cfg.refine_after:
    if t not in self.refined and self.refine_fn is not None:
        self.refined[t] = self.refine_fn(t)        # cached decomposition (sub-test specs)
    # NOTE: stall-expand and every reward branch below run UNCHANGED.

# prompt_block(): for an ACTIVE test t, if t in self.refined, render its sub-tests as the
# guidance body for t (using the existing show_bodies/test_bodies path).
```

`self.refined` is **prompt-only state** — nothing in `tick()`'s reward computation
(`r_alpha…r_eta`, expansions, solves, regressions, P2P) or in `terminal_reward()` reads it.
Gated by **`RLLM_ACTIVE_TESTS_REFINE=1`** (default **off** → scheduler is byte-for-byte the
current control; exact reproducibility preserved).

---

## 3. Reward stance: **zero change** (the constraint)

| ATS-RL today | With refinement on |
|--|--|
| dense per-tick reward over original F2P | **identical** |
| terminal `w_solve·frac_solved` over original F2P | **identical** |
| stall-expand `−β`, regressions, P2P, token terms | **identical** |
| what the agent *reads* for a stuck test | **enriched** with the decomposition |

Verification obligation: a unit test asserting that for any rollout trace, the sequence of
`SchedulerEvent.reward` values and the `terminal_reward` are **bit-identical** with
`refine_enabled` on vs off (given the same `pass_map` stream). Refinement may only diverge in
`prompt_block()` output.

---

## 4. Components to build

1. **Scheduler annotation + plumbing** (`active_tests.py`) — *no reward changes*:
   - `SchedulerConfig`: `refine_enabled=False`, `refine_after:int=k_steps`,
     `refine_max_subtests:int=4`.
   - `self.refined: Dict[str, list]` + an injected callback `self.refine_fn(test_id) ->
     [(name, code)]` (set by the env; `None` in unit tests). Populate `self.refined[t]` at
     the stall point; render it in `prompt_block()`.
   - Telemetry-only `SchedulerEvent.refinements` counter (does **not** enter `reward`).

2. **`TestDecomposer`** (`test_refiner.py`, teacher = OpenAI **`gpt-5.4-mini`**, key from
   `.env::API_KEY_OPENAI`, `openai` SDK v2.37): `decompose(issue, gold_diff,
   original_test_src) -> [(name, code)]` — "split this test into ≤K minimal pytest tests,
   each asserting one facet; self-contained; no network/sleep/random." ≤1 refine-retry.

3. **Offline validation gate (cached, NOT on the reward path)** — to ensure the guidance we
   show is *correct*, validate each sub-test **once** in the instance Docker image: must
   **fail on pre-patch** AND **pass on gold-patched** (the F2P invariant). Discard splits
   that don't discriminate. Reuse B2's `DockerTestValidator` + a `run_adhoc_tests(code,
   node_ids)` helper (writes the file into `/testbed`, runs pytest). This runs in the
   **cache-build step**, never inside the reward loop — so the make-or-break mid-episode
   injection risk from the earlier draft is gone.

4. **Decomposition cache** (persistent, keyed `instance_id::test_id`): the same hard tests
   stall across rollouts/epochs/`rollout.n`; compute the split + gate **once**, store to
   `block_b_refinement/cache/decompositions.jsonl`, reuse everywhere. `refine_fn` is just a
   cache lookup at runtime (compute-on-miss optional; prefer warm cache).

5. **Env wiring** (`swe_arsenal.py`): build `refine_fn` = cache lookup (→ decomposer on miss)
   and pass it to the scheduler when `RLLM_ACTIVE_TESTS_REFINE=1`; log refinement telemetry
   on the `!!!!!!!` line. No change to the mid-episode reward call.

**Untouched (the constraint):** `tick()` reward math, `terminal_reward()`, `w_solve`, the
parquet, the launcher reward flags. Refinement only enriches `prompt_block()` output.

---

## 5. Cost control (the teacher call is the only cost; off the reward path)

- **Persistent decomposition cache** — first stall of a given test pays; everything reuses.
- **Warm the cache offline** from B2's rollout logs (the historically-stuck tests) so no
  teacher call ever happens inside a rollout.
- Gate + cache each split once; never re-validate.
- `refine_after` and `refine_max_subtests` keep the set small and selective.

---

## 6. Decisions (locked)

| Decision | Value |
|--|--|
| **Reward** | **UNCHANGED — exactly ATS-RL** (dense + terminal, over original F2P) |
| Refinement effect | **prompt-only** (decomposition shown as guidance for the stuck test) |
| Teacher LLM | OpenAI **`gpt-5.4-mini`**, key `.env::API_KEY_OPENAI`, `openai` SDK v2.37 |
| Dataset | `swerebench_verlive` (64/16) — matches B2; **parquet unchanged** (online refinement) |
| Gating | `RLLM_ACTIVE_TESTS_REFINE=1` (default off); needs `RLLM_ACTIVE_TESTS_MID_EPISODE=1` |
| Train | **10 steps**, batch 16, rollout.n=4, SFT base `SWE-Master-4B-SFT` |
| Trigger | stall counter ≥ `refine_after` (default = `k_steps`); ≤ `refine_max_subtests=4` |

---

## 7. ⚠️ Baseline note (matched ablation)

Stall *detection* needs the **mid-episode pytest path** (`MID_EPISODE=1`) — that's how the
scheduler learns a test is unsolved (it's also what gives ATS its dense reward). The B2/R22
and control runs used the v1 prompt+terminal path, whose reward differs from v2. So the clean
comparator for this arm is **ATS mid-episode ON + refine OFF** — same reward, only the prompt
differs:
- **B0-mid** = `MID_EPISODE=1`, `REFINE=0` (matched baseline).
- **B1-refine** = `MID_EPISODE=1`, `REFINE=1`.
By construction (§3) these two have **identical reward**; the only difference is the enriched
prompt for stalled tests — so any solve-rate delta is attributable purely to the guidance.

---

## 8. Metrics

- Val solve-rate `val/test_score/swe` (B1-refine vs B0-mid).
- **Stuck-test recovery**: fraction of stalled tests that get solved *after* their
  decomposition is shown (vs B0-mid where it isn't).
- `critic/rewards/mean` per step; grad-norm (sanity — reward dists should match B0-mid).
- Refinement telemetry: refinements/episode, cache hit-rate, teacher calls + cost.
- Generalization gap (Block-B headline).

---

## 9. Build sequence (each phase verifiable before the next)

1. **Scheduler annotation** in `active_tests.py` (`refined` state + `prompt_block` rendering
   + stub `refine_fn`). Unit tests: (a) stall → decomposition shows in prompt; (b) **reward
   stream + terminal are bit-identical refine-on vs refine-off** (§3). No LLM/Docker.
2. **`TestDecomposer`** (gpt-5.4-mini) + **decomposition cache**; eyeball splits on a couple
   of known-stuck tests from B2 logs.
3. **Offline validation gate** (`run_adhoc_tests` + fail-pre/pass-gold) in the cache-build
   step; confirm sub-tests discriminate on one instance. (Off the reward path → low risk.)
4. **Env wiring** (`swe_arsenal.py` builds `refine_fn`=cache lookup); smoke 3-5 steps with
   `MID_EPISODE=1 REFINE=1` — confirm refinements fire, prompt shows decompositions,
   telemetry logs, reward dist matches a refine-off smoke.
5. **B0-mid** (refine off) + **B1-refine** (refine on), 10 steps each, verlive, SFT base,
   B2's launcher/orchestrator pattern → export `SWE-Master-4B-ATS-B1-r12` (+ step-5).
6. RESULTS.md B1 row (vs B0-mid, B2) + recovery telemetry → commit/push.

## 10. Definition of done
Gated & default-off (control exactly recoverable) · **reward-invariance unit test green
(refine on==off)** · decomposition cache + offline gate working · 3-5 step smoke with refine
on · B0-mid + B1-refine 10-step runs · models exported · RESULTS.md row + telemetry · commit
/push to `test-refinement-back-translation`.
