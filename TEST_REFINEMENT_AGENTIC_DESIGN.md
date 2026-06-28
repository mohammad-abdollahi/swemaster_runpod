# Agentic test refinement — design (branch `test-refinement-agentic`)

Status: DESIGN (no runtime code yet). Base: branched off `test-refinement-test-to-openai`
(reuses ~94% of its plumbing; only the LLM-call layer changes).

## 1. Goal / what changes vs the OpenAI branch

Same trigger and same scheduler integration as the OpenAI online-refine pipeline, but the
refiner becomes the **Claude Code agent via `claude -p`** (headless), driven on the user's **Max
subscription** (no API charge), instead of a single OpenAI chat call. Deltas:

| Aspect | OpenAI branch (current) | Agentic branch (this) |
|---|---|---|
| Refiner | 1 `chat.completions` call | **`claude -p`** (Claude Code agent) — explores repo, runs the test, self-verifies |
| Gold patch in prompt | YES | **NO** (removed) |
| Trajectory input | current env only, 7K-char summary, **inert** (DUMP flag off) | **multiple diverse** trajectories, always-on capture |
| Trajectory selection | n/a | regex-normalized similarity; include if **>30% mismatch** |
| F2P guarantee | "currently fails" only (weak; admits NameErrors) | **fail-on-buggy AND pass-on-fixed**, both in clean containers |
| Provider | OpenAI gpt-5.4-mini | Claude via **`claude -p`** on the Max subscription (no API charge) |
| Input truncation | issue/patch/test/trajectory all char-capped | **none — full untruncated** issue + test + trajectories |

Refinement is applied **only to FAIL_TO_PASS tests** (unchanged — the scheduler only refines
stalled F2P tests).

## 2. Trigger (unchanged from online-refine)

ATS scheduler `tick()`: a FAIL_TO_PASS test that stays unsolved for `refine_after` (=2) stall
ticks → `refine_fn(test_id)`. With `refine_replace=True`, surviving refined nodes REPLACE the
stalled test in `fail_to_pass`+`active` and drive `frac_solved` at terminal. (Note: the original
"after 70 steps" framing is superseded by the integrated stall trigger; refine fires per-test on
stall, not at a fixed global step.)

## 3. Trajectory preprocessor (NEW — diversity selection)

**Source pool.** This branch makes transcript capture **always-on** (fixes the `RLLM_DUMP_TRAJ_DIR`
gap) and writes each rollout's compact transcript to a per-instance store
`…/agentic_traj/<instance_id>.jsonl` (one JSON line per completed trajectory: `{step, fn, action, obs}` list).
Envs append as trajectories finish, so by refine time the store holds the instance's trajectories
across the rollout group (n=4) and prior steps. The stalled trajectory is always included.

**Diversity filter (regex-based).**
1. Normalize each trajectory to a canonical action-sequence string: regex-strip volatile tokens
   (hex addrs `0x[0-9a-f]+`, line numbers, tmp paths `/tmp/\S+`, container ids, timestamps,
   pytest durations) and collapse whitespace; keep the ordered `(fn, action-head)` tokens.
2. Similarity = `difflib.SequenceMatcher(None, A, B).ratio()` on the normalized strings.
3. Greedy select: seed with the stalled trajectory; add another trajectory iff its **max similarity
   to all already-selected < 0.70** (i.e. **>30% mismatch**). Cap at `K=RLLM_AGENTIC_MAX_TRAJ` (default 3)
   to bound prompt size.
4. Selected trajectories are passed to `claude -p` **in full — no step/char/token truncation**
   (the agent also reads the repo directly, and Claude's context is large). The diversity cap `K`
   bounds the *number* of trajectories, not their length.

Result: Claude sees a few *materially different* failure modes, not 4 near-duplicates.

## 4. Prompt (gold patch removed)

System (gold-fix references removed to match "no gold patch"):
> You are a senior Python test engineer helping an RL agent STUCK on a failing pytest test during
> training. You get the issue, the original hard test, and the agent's trajectories so far. REFINE
> the test so it is more learnable: REWRITE it as one cleaner test OR SPLIT it into a few smaller
> minimal tests — each asserting ONE facet, descriptive name, standard pytest only (no
> network/sleep/randomness/external services). CRITICAL: the refined test(s) must FAIL on the
> buggy/unfixed code and PASS only once the bug is correctly fixed — never write a test that passes
> trivially. You may use the provided tools to read the repo and RUN your candidate test (it must
> currently fail). Return the final refined test(s).

User body: `## Issue`, `## Original (hard) FAIL_TO_PASS test {test_id}`, `## Agent trajectories (N diverse, it has NOT made this test pass)`. **No gold patch.**

**No input-length conditions.** Unlike the OpenAI branch (which capped issue ≤3500, gold patch ≤7000,
test ≤3500, trajectory ≤7000 chars, and per-step action/obs to 300/200), this branch passes the issue,
the original test, and the selected trajectories **verbatim and uncapped**. (The user accepts the
context/quota implications; the diversity filter + `claude -p`'s own repo access keep it manageable.)

## 5. Agentic refiner — `claude -p` (only backend)

Module: `R2E-Gym/.../runtime/agentic_refine.py`. The single refiner shells out to Claude Code headless
mode — the real Claude Code agent (bash/read/edit/grep), so it
explores the buggy repo, drafts a candidate test, runs the failing test, iterates, and self-verifies
it fails. No custom tools to build. Authenticated by `claude login` with the user's **Max**
subscription, so refinement does not bill the API.

```python
subprocess.run(
    ["claude", "-p", prompt,
     "--output-format", "json",
     "--model", model,                       # RLLM_AGENTIC_REFINE_MODEL, default claude-sonnet-4-6
     "--max-turns", str(max_turns),          # RLLM_AGENTIC_MAX_TURNS, default 12
     "--permission-mode", "acceptEdits",     # headless: no interactive approval prompts
     "--allowedTools", "Bash,Read,Edit,Write,Grep"],
    cwd=buggy_repo_dir,                       # clean checkout at base_commit → true repo access
    env=_env_without_anthropic_api_key(),    # ← force subscription auth (see gotcha)
    timeout=refine_timeout,                  # RLLM_AGENTIC_REFINE_TIMEOUT
)
```

- **Working dir = clean checkout at `base_commit`** (buggy code) so "fails on buggy" is genuine and the
  agent can run the test itself.
- **Auth gotcha:** Claude Code prefers `ANTHROPIC_API_KEY` over the subscription login — the subprocess
  env MUST strip `ANTHROPIC_API_KEY` (and `ANTHROPIC_AUTH_TOKEN` unless a profile is intended) or it
  silently bills the API instead of the Max subscription.
- **Model:** `--model claude-sonnet-4-6` default (lighter/quota-friendlier); `claude-opus-4-8` selectable.
- **Output:** `--output-format json`; the agent writes the refined test to a known path in the checkout
  (or returns it in the result), which the F2P post-processor then validates.
- **Prereq:** the `claude` CLI isn't installed here — one-time `npm i -g @anthropic-ai/claude-code` +
  `claude login` (Max), done by the orchestrator on startup. If `claude` is missing or not logged in,
  the refiner no-ops (keeps the original test) rather than crashing the rollout.

The refiner returns refined test source; everything downstream (diversity preprocessor, F2P
post-processor, cache, `_replace_test`) consumes it directly.

## 6. Post-processor — F2P guarantee (gold patch used internally only)

For each candidate test function, validate in **clean containers** (agent never saw the patch):
1. **Buggy** (`base_commit`): run → must **FAIL** with a genuine assertion/logic failure.
   Reject if it errors on **collection/import/NameError** (fixes the OpenAI-branch bug where such
   errors were mis-admitted as "fails on buggy").
2. **Fixed** (`base_commit` + gold `patch`): run → must **PASS**.
3. Keep only functions that are **fail-then-pass (true F2P)**. Drop the rest.

Surviving F2P nodes → `_replace_test` swaps them into the scheduler. If none survive, keep the
original test (no-op), cache the negative result.

Container strategy: one buggy + one fixed clean container per refined instance, created from
`docker_image`, pooled/reused across a refine call, torn down after; cached result means each test
is validated once per run. (Fallback knob: reuse the agent's container for the fail-check only — cheaper,
less rigorous — `RLLM_AGENTIC_REUSE_CONTAINER=1`.)

## 7. Caching & wiring

- Cache: per `instance_id::test_id` JSONL (reuse `OnlineRefineCache` shape) — one agentic refine + F2P
  validation per test per run.
- `swe_arsenal._make_refine_fn`: add an `agentic` branch gated by `RLLM_ACTIVE_TESTS_AGENTIC_REFINE=1`
  (mutually exclusive with the OpenAI online path); returns full node ids for `_replace_test`.
- Reuse `_replace_test`, terminal F2P merge, `compute_final_reward` unchanged.
- New env: `RLLM_ACTIVE_TESTS_AGENTIC_REFINE`, `RLLM_AGENTIC_REFINE_MODEL` (claude-sonnet-4-6),
  `RLLM_AGENTIC_MAX_TRAJ` (3), `RLLM_AGENTIC_TRAJ_DIVERSITY` (0.30), `RLLM_AGENTIC_MAX_TURNS` (12),
  `RLLM_AGENTIC_MAX_CONCURRENCY` (1–2), `RLLM_AGENTIC_REFINE_TIMEOUT`. No API key needed (subscription
  auth). Launcher/orchestrator cloned to `run_agentic_arm.sh` / `agentic_orchestrator.sh` (the
  orchestrator installs the `claude` CLI + verifies `claude login` on startup).

## 8. Quota / rate-limit handling (Max subscription)

`claude -p` on the Max subscription shares one quota with interactive Claude Code (rolling ~5h windows
+ weekly caps), and each refine is a full multi-turn agentic session. Measures to keep a run within
quota and avoid lock-out:

- **Global concurrency cap** on refine subprocesses (`RLLM_AGENTIC_MAX_CONCURRENCY`, default 1–2) via a
  cross-rollout semaphore — several rollouts stall at once, so unbounded fan-out would spike usage.
- **Aggressive caching** (per `instance_id::test_id`) → each test refined once per run (~few hundred).
- **Backoff + skip on rate-limit:** detect Claude Code's usage-limit exit, exponential backoff, and
  after N retries **skip refinement** for that test (keep the original) rather than stall the rollout.
- **`--max-turns` cap + per-refine wall-clock `timeout`** bound each session's spend.
- **Sonnet default** (`--model`) over opus for lower per-session cost.
- **Pause/resume:** the cache persists, so a run throttled mid-way resumes reusing prior refines; on a
  hard usage-limit the refiner no-ops (original test kept) so training continues uninterrupted.

## 9. Open sub-decisions (resolved with rationale; correct on review)
- Refiner = `claude -p` only (Max subscription, no API charge); no SDK/API backend.
- Inputs passed verbatim/uncapped (no length conditions).
- Trajectory pool = per-instance disk store accumulated across rollout group + steps (vs current-env-only).
- F2P validation in clean containers (rigorous) vs reuse agent container (cheap) → default clean, knob to reuse.
- `claude -p` `--model` default sonnet-4-6 (vs opus) for quota/speed.
- Diversity metric = normalized `SequenceMatcher` ratio, threshold 0.70 (>30% mismatch).

## 10. Build order
1. `agentic_refine.py`: `claude -p` refiner (subprocess, JSON parse, auth-strip, no-op-if-missing) +
   prompt builder (verbatim inputs) + trajectory diversity preprocessor + clean-container F2P post-processor + cache.
2. Always-on transcript capture + per-instance trajectory store in `swe_arsenal.py`.
3. `_make_refine_fn` agentic branch + env wiring + concurrency semaphore.
4. Unit test (diversity filter + F2P post-processor) + 1 live `claude -p` refine on one instance.
5. `run_agentic_arm.sh` / `agentic_orchestrator.sh` (CLI install + login check); smoke run.
