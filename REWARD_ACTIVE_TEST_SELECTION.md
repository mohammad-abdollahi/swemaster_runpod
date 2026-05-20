# Active-Test-Selection Reward — Design Doc

Branch: `reward/active-test-selection`
Status: design / pre-implementation
Author note: this document is the deliverable for the current iteration. No code under
`DeepSWE_RL/`, `R2E-Gym/`, or training configs has been modified yet — design first.

---

## 1. Motivation

The current SWE-Master RL setup gives the policy a **sparse binary terminal reward**
(see [docker.py:1523](R2E-Gym/src/r2egym/agenthub/runtime/docker.py#L1523) and
[swe.py:126](DeepSWE_RL/rllm/rllm/environments/swe/swe.py#L126)). Reward only fires
when the agent emits `finish` / `submit`, and only as `int(all FAIL_TO_PASS resolved)`.
Per-step reward returned from `env.step` is always `0`
([env.py:298](R2E-Gym/src/r2egym/agenthub/environment/env.py#L298),
[env.py:338](R2E-Gym/src/r2egym/agenthub/environment/env.py#L338)).

Two consequences:

1. **Credit assignment is hard.** The agent gets one bit at the end of a 50-turn episode.
2. **The agent never sees the tests.** All test names/bodies are hidden, so the policy
   has no anchor to "what does success look like" beyond the natural-language task
   description.

The new direction proposed by the user: **expose a small, dynamically-managed subset
of tests to the agent each step**, so that

- prompt length stays bounded (token savings),
- reward becomes dense and tied to observable progress (better credit assignment),
- the curriculum keeps difficulty roughly matched to current policy capability.

The user's first sketch:

> reward = `1 / (n_active_tests + 1)`; if a test is solved within `k=5` steps it is
> deactivated for `10` steps; if not solved within `k=5` steps it is "reduced to two
> new tests"; continue for some horizon; if the task is not finished, return the
> reward.

This document evaluates that sketch and proposes a refined version.

---

## 2. Does the sketch make sense?

**Direction: yes.** The core insight — a curriculum of currently-tracked tests, with
"solve → cooldown" and "stall → expand" — is sound. It is essentially a
multi-armed-bandit-style scheduler over the task's own assertion set, and it gives
us a natural dense signal.

**Concerns with the literal formulation:**

| Issue | Why it bites |
| --- | --- |
| `1/(n_active+1)` is not action-conditioned. | Reward only depends on the *size* of the set, which changes on a *schedule*, not directly because of what the agent did this turn. The agent learns "smaller set good", not "fix this test good". |
| `1/(n+1)` is non-monotonic in difficulty. | Hard tasks legitimately have more tests; the policy gets punished for working on hard instances even when it's doing the right thing. |
| "Reduce to two new tests" is undefined. | The dataset gives a fixed `FAIL_TO_PASS` list per instance. There is no `split_test()` operator. We need a concrete pool semantics (see §3). |
| Cooldown without re-check is hackable. | If a test passes once and is then masked for 10 steps, the agent can regress it (e.g., by reverting a file) and never pay. Need re-verification on cooldown expiry. |
| Terminal "return the reward" is ambiguous. | If the task doesn't finish, what is the credit? Partial credit on the full `FAIL_TO_PASS` is the only stable choice. |
| Cost of running tests every step. | The current `_calculate_reward_*` runs the full pytest suite. We need a subset-runner; otherwise dense reward is prohibitively expensive in Docker. |

The sketch's good ideas:
- **Cooldown for solved tests** — prevents the agent from harvesting the same easy
  test forever.
- **Expansion on stall** — gives the agent more anchors when it's stuck.
- **Bounded active set** — directly delivers the token-saving win.

---

## 3. Refined design

### 3.1 Per-task test state machine

For each training instance, the agent's `FAIL_TO_PASS` list (call it `F`, size `N`)
is partitioned into four buckets that evolve over the episode:

- `ACTIVE` — tests currently shown in the prompt and used in the dense reward.
  Initial size `M_init = 2`.
- `POOL` — tests not yet shown. `F \ ACTIVE` at episode start.
- `COOLDOWN` — recently solved tests, masked from the prompt; each has a
  remaining-step counter (default 10).
- `RETIRED` — tests that passed re-verification after cooldown. These are *kept
  green* but not shown.

Invariants:
- `ACTIVE ∪ POOL ∪ COOLDOWN ∪ RETIRED = F` and the four sets are disjoint.
- The prompt only ever contains names/bodies from `ACTIVE`. `|ACTIVE|` is bounded
  by `M_max = 6` (hard cap) for token control.

### 3.2 The k-step accounting window

A counter `stall[t]` per active test `t` increments each step `t` stays in `ACTIVE`
without a green pytest result.

At every step:
1. Run pytest on the subset `ACTIVE ∪ COOLDOWN` only (see §3.4 for runtime). Get
   pass/fail per test id.
2. For each `t ∈ ACTIVE`:
   - If green → move to `COOLDOWN` with timer `D = 10`; emit per-test solve event.
   - Else → `stall[t] += 1`.
   - If `stall[t] == k` (default `k = 5`) → **stall-expand**: keep `t` in `ACTIVE`,
     and additionally promote up to `2` tests from `POOL` to `ACTIVE` (subject to
     `M_max`). Reset `stall[t] = 0` so the next expansion is another `k` steps away.
3. For each `t ∈ COOLDOWN`: decrement timer. When `0`:
   - Re-run `t` once. If still green → move to `RETIRED`. If regressed → move
     back to `ACTIVE` with `stall[t] = 0` and emit a regression event (penalty).
4. If `ACTIVE` becomes empty and `POOL` is non-empty → promote `M_init` from
   `POOL` (the episode shouldn't end early just because the easy ones are done).

This addresses the four-test interpretation ambiguity cleanly: "split into two new
tests" becomes "promote two from the hidden pool" — no synthetic test generation
needed.

### 3.3 Reward shape

Let, for step `i`:
- `s_i` = number of `ACTIVE` tests that turned green this step (solve events).
- `x_i` = number of stall-expansions this step (each promotes ≤ 2 tests).
- `g_i` = number of regression events (cooldown re-check failed).
- `n_i` = `|ACTIVE|` *after* the step's bookkeeping.

Dense per-step reward:

```
r_dense_i = α · s_i  −  β · x_i  −  δ · g_i  −  γ  +  η · 1/(n_i + 1)
```

with suggested initial weights:

- `α = 1.0`     — direct credit for tests solved this turn (the strongest signal)
- `β = 0.2`     — mild cost of getting stuck (stall-expand happened)
- `δ = 0.5`     — penalty for regression (you broke something you'd fixed)
- `γ = 0.01`    — per-step cost to encourage finishing
- `η = 0.05`    — *retains the user's `1/(n+1)` term* as a small shaping bonus,
  rewarding compact active sets without overpowering the action-conditioned
  signal.

This keeps the spirit of the original `1/(n+1)` idea but demotes it from "the
reward" to a shaping term, with the bulk of the signal coming from observable
agent behavior (`s_i`, `g_i`, `x_i`).

Terminal reward (at `done`, `max_steps`, or `truncation`):

```
r_final = w_solve · (|RETIRED| + |COOLDOWN_passing|) / N
         − λ_token · tokens_used_in_prompt_history / T_ref
```

with `w_solve = 5.0`, `λ_token = 0.1`, `T_ref = 32_000` (so the token penalty is
~`-0.1` per 32k tokens consumed).

Total trajectory reward = `Σ r_dense_i + r_final`.

**Why this is better than `1/(n+1)` alone:**
- Dense credit is tied to what the agent *did* this step (`s_i`, `g_i`), not just
  what the curriculum did to the set size.
- Regression penalty closes the cooldown-hacking loophole.
- Token penalty makes the token-saving incentive explicit rather than implicit.

### 3.4 Subset test execution

The current Docker runner executes `/run_tests.sh` and parses the *whole*
suite ([docker.py:1511-1526](R2E-Gym/src/r2egym/agenthub/runtime/docker.py#L1511-L1526)).
That parses fine, but running the full suite every step is wasteful.

Two-phase approach:
1. **Phase A (cheap):** if pytest is the runner, we can run with
   `pytest <node_ids...> --tb=no -q` for just the `ACTIVE ∪ COOLDOWN` ids. Add a
   new method `_run_tests_subset(test_ids: list[str], timeout: int)` on
   `DockerRuntime` that writes a `/run_tests_subset.sh` wrapper.
2. **Phase B (terminal):** at episode end, run the full `/run_tests.sh` exactly
   as today so `r_final` is computed against the same suite the eval pipeline uses
   (apples-to-apples with current paper numbers).

For non-pytest runners (some legacy `r2e` images), fall back to running the full
script but post-filtering `eval_status_map` to the subset — same Docker cost, but
zero code-path divergence.

### 3.5 What the agent sees in the prompt

Today, the system + task prompts describe the bug informally with no test bodies.
Under this scheme, after the task description we inject:

```
<active_tests count="{|ACTIVE|}">
  <test id="{t}">
    {test source or signature, truncated to ~600 chars}
    Last result: {fail|pass(in cooldown)}
  </test>
  ...
</active_tests>
```

Token budget: with `M_max = 6` and ~600 chars per test, this block is bounded at
~3.6k chars (~1k tokens) — easily fits and is *much* smaller than the typical
prompt today, which loads the bug description plus full file contents on demand.
Net effect on token usage depends on how the agent currently uses file-read tools;
we expect a wash or modest savings on prompt, with the larger savings coming from
**shorter episodes** (the agent has clearer targets and finishes faster).

This is the lever for the user's clarification ("help reduce token usage and
similar learning/performance"): the dense, test-anchored reward should let the
agent learn to act decisively earlier in the trajectory.

### 3.6 Anti-overfitting guard

There is a real risk that the agent learns to pass only `ACTIVE` tests and not
the hidden ones in `POOL`. Mitigations baked into the design:

- `r_final` is computed against the full `F` (via `/run_tests.sh`), not just what
  was shown. This is the dominant terminal signal.
- Stall-expand promotes *previously-hidden* tests; an agent over-fitting to the
  visible set will get unfamiliar tests added on stall and pay the `β` and `γ`
  costs.
- The cooldown re-check (§3.2 step 3) catches regressions explicitly.

---

## 4. Why this could yield a paper

Contribution claim: **a curriculum-based dense reward that exposes the
target-test set to the policy via a bandit-style scheduler, achieving comparable
solve-rate with significantly fewer training tokens.**

What to measure (paper experiments):
1. **Solve rate** on SWE-bench-Verified / SWE-rebench-public vs. the binary-reward
   baseline at matched training compute.
2. **Tokens per solved instance** (prompt + completion). The headline number.
3. **Episode length** distribution — does the dense signal really shorten
   episodes?
4. **Generalization gap** = (active-tests solve rate) − (held-out tests solve
   rate) on the same instance. Should be small if the anti-overfitting guards
   work.
5. **Ablations:** drop `η · 1/(n+1)`; drop cooldown re-check; drop stall-expand;
   set `M_max = N` (no curriculum). Each should hurt one metric and isolate the
   contribution.

Honest risks:
- Subset pytest invocation may be slow on instances with heavy test fixtures
  (collection still dominates). Need to measure on 5-10 representative instances
  before committing.
- The agent may learn to game `β` (avoid stall-expand) by emitting `finish` early
  with low solve rate. The `w_solve` term on the terminal should outweigh this;
  measure on the first run and re-tune.
- Some `swerebench` instances have only 1-2 `FAIL_TO_PASS` tests — for those the
  curriculum collapses to "show all". That's fine, just notable.

---

## 5. Concrete code surfaces to change (when we implement)

Recording these now so the next session has a punch list; **no edits yet**.

1. **State container & scheduler**
   - New module: `R2E-Gym/src/r2egym/agenthub/runtime/active_tests.py`
   - Class `ActiveTestScheduler(F, k=5, D=10, M_init=2, M_max=6)` with `step(pass_map)
     → SchedulerEvent(s_i, x_i, g_i, n_i, prompt_block)`.

2. **Per-step pytest subset run**
   - New method on `DockerRuntime` in
     [docker.py](R2E-Gym/src/r2egym/agenthub/runtime/docker.py):
     `run_tests_subset(test_ids, timeout) -> dict[test_id -> pass|fail]`.
   - Mirror `_calculate_reward_*` parsers but feed the filtered output.

3. **Env-level wiring**
   - [env.py:298](R2E-Gym/src/r2egym/agenthub/environment/env.py#L298) currently
     returns `0` from `calculate_reward`. Replace with a call into the scheduler:
     `reward, prompt_block = self.scheduler.step(self.runtime.run_tests_subset(...))`.
   - Inject `prompt_block` into the observation so the agent sees it next turn.

4. **Rollout reward path**
   - [swe.py:146](DeepSWE_RL/rllm/rllm/environments/swe/swe.py#L146) — propagate
     the per-step reward (currently always 0). Also call
     `compute_final_reward()` to add `r_final` on `done`.

5. **Config**
   - Add a config block under `reward_model` in the PPO yaml selecting
     `active_test_selection` and exposing `k, D, M_init, M_max, α, β, γ, δ, η,
     w_solve, λ_token`.
   - Register a custom reward function via `reward_model.custom_reward_function`
     pointing at a new file
     `DeepSWE_RL/rllm/rllm/rewards/active_test_reward.py` that just aggregates the
     scheduler-emitted step rewards (the scheduler does the work in the env; the
     reward fn is a passthrough that matches verl's interface).

6. **Telemetry**
   - Log to tensorboard: `n_active_mean`, `solve_rate_active`, `solve_rate_full`,
     `regression_count`, `stall_expand_count`, `tokens_per_episode`.

Estimated work: ~2-3 days for the scheduler + docker subset runner + integration,
plus another ~2 days for a first run on the existing H200 4B setup and an
ablation pass.

---

## 6. Open questions for the user

(Listed but answered with my best guesses per auto-mode; flag if any of these is
wrong and we'll revise.)

1. **Should `PASS_TO_PASS` be part of the active set?** Default: no. Those are
   regression-guard tests; they belong as a single "regression block" that
   contributes to `r_final` but not to dense reward.  *Assumed: no.*
2. **k=5 fixed or annealed?** Default: fixed in v1. We can anneal `k` down as
   training progresses (more impatient curriculum) if the first run looks good.
   *Assumed: fixed.*
3. **`M_init = 2`?** This is the most token-aggressive setting. If solve rate
   tanks, try `M_init = N` (all tests visible) as a sanity check.  *Assumed: 2.*

---

## 7. Status / next step

- Branch `reward/active-test-selection` created off `setup/runpod-h200`.
- This doc lives at `REWARD_ACTIVE_TEST_SELECTION.md` in the repo root.
- No source under the training/runtime packages has been edited.
- Next concrete action when we resume implementation: write
  `R2E-Gym/src/r2egym/agenthub/runtime/active_tests.py` with the scheduler and a
  unit test exercising the four state-machine transitions, before touching
  Docker or PPO wiring.
