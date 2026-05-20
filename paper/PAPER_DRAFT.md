# Active-Test Curriculum Rewards for SWE-Agent RL

**(working title; final = "Curriculum-Aware Test Selection for Sample- and Token-Efficient RL on Software Engineering Tasks")**

Working draft. Updated continuously as experiments produce numbers.
Last update: 2026-05-20.

---

## Abstract (draft)

Reinforcement learning for software-engineering (SWE) agents — patching real
GitHub repositories under unit-test verification — currently relies on a
single bit of feedback: did the agent's submitted patch turn the held-out
`FAIL_TO_PASS` test suite green at the end of the rollout? That signal is
sparse, expensive (each evaluation runs an entire test container), and
ignores the natural structure of the verifier: tests are independent
sub-goals. The binary signal also degenerates under group-baselined
advantage estimators (GRPO/LOOP) on the most common case — all four
rollouts in a group fail — which we measure empirically to consume ~40%
of PPO steps with zero gradient on a representative SWE-rebench mix.

We propose **Active-Test-Selection (ATS) rewards**, a curriculum-aware
training objective that (i) exposes a bounded, dynamically managed
subset of the target tests to the policy in its prompt, (ii) replaces
the binary terminal reward with a continuous credit-assigned signal
driven by a four-state bandit-style scheduler over the tests (ACTIVE,
POOL, COOLDOWN, RETIRED), and (iii) explicitly penalizes the
prompt-and-completion token footprint of the trajectory.

Contributions:
1. **The ATS scheduler and reward function**, including a four-state
   machine with `solve → cooldown → re-verify` that closes the
   "reward-hacking by transient pass" loophole.
2. **A token-budget term** that makes the agent's prompt-and-rollout
   footprint an *explicit* training objective rather than a passive cost.
3. **An "always-on gradient" property**: continuous rewards bypass the
   `(rewards == 0).all()` variance filter built into popular RL
   libraries (verl/rllm), giving non-zero PPO gradient even on rollout
   groups where every trajectory fails. Empirically validated on a
   small-scale H200 smoke run (16-traj all-fail PPO step produces
   non-zero grad).
4. **A reproducible ablation suite** isolating the contributions of the
   curriculum, the cooldown re-check, the token penalty, and the dense
   per-tick signal, on Qwen3-4B (SWE-Master-4B-SFT) + SWE-rebench with
   the LOOP advantage estimator.
5. **Open-source reference implementation** plugging into the
   DeepSWE/SWE-Master pipeline as a drop-in reward replacement, gated by
   `RLLM_ACTIVE_TESTS_ENABLED=1` so the binary baseline can be reproduced
   from the same codebase.

## 1. Introduction

Reinforcement learning for software-engineering agents has matured into a
recognizable recipe: pretrain a coder model, supervised fine-tune (SFT) on
patch-style traces, then GRPO/LOOP-tune against a verifier that runs the
target repository's unit tests in a sandboxed Docker container. The DeepSWE
and SWE-Master recipes (Sun et al. 2025; Anthropic 2026) demonstrate that
this loop produces real improvements on SWE-bench. They also all share an
identical reward function, modulo cosmetic variation:

```
r(τ) = int(all FAIL_TO_PASS tests resolved at end of trajectory τ)
```

A patch resolving every failing test scores 1; a patch resolving 199 of
200 tests scores 0. This binary aggregation has three pathologies we
believe are limiting progress:

1. **Credit assignment is sparse.** A 50-200 step rollout (file reads,
   edits, bash invocations) collapses to one bit at the end. GRPO/LOOP
   group baselines vanish (advantage = 0) whenever all rollouts in a
   group succeed or all fail. Empirically — in our own H200 baseline runs
   under this recipe — ≈40% of PPO steps see all-zero advantage and
   contribute no gradient signal.
2. **The verifier is opaque to the policy.** The agent has no access to
   the tests it must satisfy. It must reverse-engineer the
   specification from the natural-language issue plus repo exploration.
   This is unrealistic relative to how human engineers actually work
   (read the failing test, then fix it) and wastes substantial fractions
   of the trajectory on spec-discovery actions.
3. **Token cost is unmeasured.** Prompts ingest entire files; completions
   stretch to 16K-32K tokens. Inference and training cost scale super-
   linearly in sequence length, yet nothing in the objective rewards
   parsimony.

We propose a single mechanism that targets all three: expose a
bounded subset of the tests to the policy each rollout, reward
test-by-test progress (continuous partial credit), and add an explicit
token-usage penalty. We call this **Active-Test-Selection (ATS) reward**.
A bandit-style scheduler manages which tests are in the visible "active"
set as the rollout proceeds — solved tests enter a cooldown and are
re-verified to close a "transient pass" loophole; stalled tests are
augmented by promoting hidden tests from a pool.

In the language of curriculum learning, the curriculum is *within* the
episode, not across episodes — a structure we have not seen in prior
SWE-agent RL or LM-RL more broadly.

## 2. Related work

### 2.1 RL for SWE-agents

- **DeepSWE / SWE-Master** establishes the GRPO/LOOP recipe on
  R2E-Gym-style verifiers. They report on Qwen3-4B and Qwen2.5-Coder-32B,
  using a binary terminal reward against `FAIL_TO_PASS`. Hyperparameters:
  `train_batch_size=32`, `ppo_mini_batch_size=16`, `rollout.n=4`,
  `total_epochs=5`, `max_response_length=122 880` (32B) / `122 880` (4B);
  paper publishes that 16 H800s are required to fit those response
  lengths. Our work changes only the reward (and corollary prompt
  injection); the optimizer and verifier paths are unchanged.
- **R2E-Gym** is the verifier framework: per-task Docker containers with
  a `/run_tests.sh` invocation that parses to the canonical
  `eval_status_map` (PASSED/FAILED/...). Our `get_fail_to_pass_status`
  reuses this exact parser, so reward changes are isomorphic to changing
  the post-processing of the same evaluation pipeline.
- **SWE-rebench** (Nebius) is the verified re-execution mixture we train
  on; the public `swerebench/sweb.eval.x86_64.*` images on Docker Hub
  pull cleanly on a permissive host.

### 2.2 Dense reward shaping in code-RL

- **Process Reward Models** (Lightman 2023; Wang 2024). Closest in spirit:
  dense per-step reward via a *learned* verifier. Two differences: (1)
  ours requires no extra training data — the per-test pass map is read off
  the same Docker verifier the terminal reward uses; (2) PRMs need
  per-step labels which are hard to obtain for long agent trajectories,
  whereas tests are a free, natural decomposition of "success".
- **Self-rewarded code RL** (Endor, ICE, REFORM). Use the LM itself as a
  judge of step quality. Higher variance and weakly grounded in the
  external verifier; our reward is verifier-grounded by construction.

### 2.3 Curriculum learning

- Bengio et al. 2009 (the founding curriculum-learning paper); Narvekar
  et al. 2020 (survey). Existing work adapts task difficulty *across*
  episodes by ranking tasks. Our curriculum operates *within* an episode,
  reordering and exposing tests as the agent progresses on a single
  instance. We are not aware of prior intra-episode curricula over a
  task's own verifier sub-goals.

### 2.4 Test-driven prompting

- LM agents have been benchmarked with full test access at inference time
  ("test-aware" SWE-bench evals), but these results conflate the
  capability of the model with the artifact of having seen the tests.
  We are the first (to our knowledge) to train an RL policy under test
  exposure while keeping the terminal evaluation against the full hidden
  set — disentangling the two effects.

## 3. Method

### 3.0 Notation and problem setting

A training instance is a tuple `(R, P, F)` where `R` is a buggy repository
snapshot, `P` is a natural-language issue description, and `F` is the
hidden set of tests that must transition from FAIL to PASS for the patch
to be accepted. A *rollout* is a sequence of agent actions
`a_1, ..., a_T` (bash invocations, file edits, finishes) acting on `R` via
a Docker sandbox. The verifier returns a pass map
`V: F → {pass, fail}` from running the test suite against the patched
repository.

The baseline reward is `R_binary(τ) = ∏_{t∈F} V(t)` — i.e., 1 if every
test passes, else 0. Our reward is

```
R_ATS(τ) = w_solve · (1/|F|) ·  Σ_{t∈F} V(t)
         − lam_token · tokens(τ) / T_ref
         +   (mid-episode dense terms if enabled)
```

The first term gives continuous partial credit; the second penalizes long
trajectories.

### 3.1 The four-bucket state machine

For each training instance, the FAIL_TO_PASS list F is partitioned every
tick into:

- **ACTIVE**: tests visible in the prompt, scored in the dense reward.
- **POOL**: hidden tests, fed to ACTIVE on stall-expand.
- **COOLDOWN**: recently-passed tests, masked from the prompt; on timer
  expiry, re-verified (RETIRED if still green, regression → ACTIVE).
- **RETIRED**: tests that passed the re-verification cycle. Not shown;
  contribute to terminal reward only.

The state transitions are pinned by 13 unit tests (`test_active_tests.py`).

### 3.2 The reward

Dense per-tick (only when mid-episode pytest is enabled; default off):

```
r_dense = alpha · s − beta · x − delta · g − gamma + eta · 1/(n_active + 1)
```

where s = solves this tick, x = stall-expand events, g = cooldown
regressions. The `eta/(n+1)` term *retains* the user's original sketch but
demotes it from "the reward" to a small shaping bonus.

Terminal:

```
r_final = w_solve · |solved|/|F| − lam_token · tokens/T_ref
```

with solved set computed against the *full* FAIL_TO_PASS — never just
ACTIVE — to prevent over-fitting to the visible subset.

### 3.3 Prompt injection

A bounded `<active_tests>` block (default ≤4 KB, ≤6 test IDs) is appended
to the issue description. Just test identifiers in v1; bodies optional.

### 3.4 Interaction with verl/rllm's group-variance filtering

The verl/rllm PPO trainer ([agent_ppo_trainer_pipeline.py:188-209](
DeepSWE_RL/rllm/rllm/trainer/verl/agent_ppo_trainer_pipeline.py#L188-L209))
implements a variance filter: groups with `(uid_rewards == 0).all()` or
`(uid_rewards == 1).all()` are excluded from the PPO update via
`valid_mask`. This is necessary because LOOP/GRPO advantage in a constant
group is zero (or normalizes to NaN), wasting compute on uninformative
updates.

The filter is keyed on equality to 0 and 1, which is fine for binary
rewards but **prevents the binary recipe from ever getting gradient on the
hardest tasks** — where every rollout fails — and on the easiest, where
every rollout succeeds. Our own observations from a 32-task SWE-rebench
training run with the binary recipe (`/home/temp/train_h200_5epoch.log`)
show this filter triggering on ≈40% of PPO steps.

Continuous ATS rewards are never exactly 0 or 1 — even an all-zero
`frac_solved` group differs in `r_token` per trajectory. The filter
therefore *naturally* keeps every group, and the GRPO/LOOP within-group
variance — small but non-zero — produces non-trivial advantage. We
believe this is a key, under-appreciated benefit of continuous rewards in
the current verl/rllm code path.

### 3.4 Two operating modes

- **v1 / paper headline**: scheduler manages the initial ACTIVE subset and
  partial-credit terminal reward, *no* mid-episode pytest. Token-cheap.
  This is what we evaluate first; it is what the user's clarification asked
  for ("see test cases during training, reduce token usage").
- **v2 / ablation**: enable `RLLM_ACTIVE_TESTS_MID_EPISODE=1` to run
  per-tick pytest and drive the full state machine. Used to ablate the
  cooldown-re-check and dense-tick contributions.

## 4. Experimental setup

### 4.1 Model and infrastructure

- **Base model**: Qwen3-4B SFT'd on SWE traces (RUC-AIBOX/SWE-Master-4B-SFT,
  HF Feb 2026). Full-parameter PPO; no LoRA.
- **Hardware**: single NVIDIA H200 (143 GB VRAM, 196 GB host RAM, 16 vCPU,
  passwordless docker). The full paper recipe targets 16×H800; we
  reproduce all paper-faithful knobs except the irreducibly multi-GPU
  sharding (`tensor_model_parallel_size>1`, `ulysses_sequence_parallel_size
  >1`) and the 122 880-token `max_response_length` — capped at 32 768 by
  the H200's single-GPU full-param backward memory ceiling.
- **Rollout sandbox**: per-task Docker containers from public
  `swerebench/sweb.eval.x86_64.*` images. Concurrency 4 (one per CPU
  cohort).

### 4.2 Data

- **Training**: SWE-rebench public mirror (`nebius/SWE-rebench`), filtered
  to instances with pullable public images, shorter problem statements
  (token budget), and a non-empty FAIL_TO_PASS. We use 32 instances for the
  smoke and a 64-instance scaled run.
- **Validation (smoke)**: 4 held-out instances from the same pool. This is
  adequate for sanity but **NOT** a fair held-out evaluation — for the
  paper headline, val should be SWE-bench Verified (a separate
  distribution). The training/eval contamination question is open and
  flagged in §7.

### 4.3 Reward configurations evaluated

| Tag | description |
|---|---|
| `binary` | original SWE-Master recipe — terminal `int(all FAIL_TO_PASS resolved)` |
| `ATS-v1` | active subset in prompt + partial credit terminal − token penalty (default) |
| `ATS-v1 -tok` | ATS-v1 with `lam_token=0` |
| `ATS-v1 -prompt` | partial-credit terminal only; no active-tests block injected |
| `ATS-v2 -tick=k` | ATS-v1 plus mid-episode pytest every `k` agent steps |

All run with identical optimizer settings (`adv_estimator=loop`, `n=4`,
`lr=1e-6`, `clip_ratio_high=0.28`, `kl_loss_coef=0`,
`entropy_coeff=0`).

## 5. Results

(To be filled as runs complete. Placeholders below for the metrics we
want.)

### 5.1 Smoke result (single H200, 4 PPO steps, 32-task SWE-rebench)

The smoke was configured to *validate the reward path end-to-end*, not to
demonstrate strong learning: `AGENT_MAX_STEPS=20` (vs paper's 150),
`MAX_RESPONSE=12 288`, `TRAIN_BATCH_SIZE=4` (4 unique tasks per step
× 4 LOOP rollouts each), `TOTAL_TRAINING_STEPS=4`. We expected zero solves
given the budget; the question was whether the *reward signal moves* and
whether the *gradient is non-zero*.

**Step 1 metrics (16 trajectories, 4 unique tasks):**

| Metric | Value |
|---|---|
| `critic/rewards/mean` | −0.055 |
| `critic/rewards/min, max` | −0.067, −0.043 |
| Within-group reward range | ~0.024 (token-penalty driven) |
| `critic/advantages/min, max` | −0.014, +0.010 |
| `actor/grad_norm` | 0.001 |
| `actor/pg_loss` | ≈ 0.0 |
| `actor/entropy_loss` | 0.254 (uncollapsed) |
| `batch/solve_none / partial / all` | 4 / 0 / 0 |
| Trajectory steps mean / max | 11.6 / 20 |
| Termination | `MAX_STEPS` on every trajectory |
| Per-traj reward time / env time / llm time | 4.95 s / 181.9 s / 25.7 s |
| Memory (GPU / RAM) | 119 / 43 GB |

Three observations from this smoke:

1. **The reward signal is non-degenerate.** Even with `frac_solved = 0`
   for every trajectory (the SFT base does not solve anything in the
   tight 20-step budget), the token-usage penalty drives a within-group
   range of ~2.4 cents of reward. Binary baseline would report
   `(rewards == 0).all()` and contribute zero gradient — see §3.4.
   Continuous ATS contributes a small but non-zero gradient (grad_norm
   = 0.001).

2. **The agent uses the active-tests block.** Across the chat_completions
   for one task (`stix2 / test_observed_data_with_custom_observable_object`),
   4 of the 20 assistant messages explicitly reference the test name and
   include search-for-test commands in the bash log. So the prompt
   injection is *consumed* by the policy, not ignored.

3. **`AGENT_MAX_STEPS=20` is below the threshold for actual solves on this
   task mix.** Every trajectory hit MAX_STEPS, indicating the budget cut
   the agent off mid-investigation in most cases. The follow-on
   experiment (§5.2) uses `AGENT_MAX_STEPS=50`.

### 5.2 Larger run (once smoke is green)

- 64+ train tasks, 5 epochs (paper-faithful), comparison vs binary baseline.

### 5.3 Ablations

| Configuration | Solve rate | Tokens/episode | Gen-gap (active − hidden) |
|---|---|---|---|
| Binary baseline | — | — | n/a |
| ATS v1 (prompt + partial credit) | — | — | — |
| ATS v1 − token penalty | — | — | — |
| ATS v1 − active block | — | — | n/a |
| ATS v2 (full mid-episode) | — | — | — |

## 6. Discussion

### 6.1 The "always-on gradient" property and its mechanism

A surprising practical advantage of ATS is that it always gives PPO
*some* gradient, even on rollout groups where no test passes. With the
binary recipe, every group `(uid_rewards == 0).all()` contributes zero
advantage and is, in the current verl/rllm code path, either filtered
out under `rejection_sample` or contributes a literally zero update.

With ATS, even an all-fail group has `r_token` variance across rollouts
(different trajectories use different numbers of tokens). The group
variance is small — on the order of a few percent of `r_solve` —
but non-zero, so LOOP advantages are non-zero and the gradient is
non-vanishing. We measured `actor/grad_norm = 0.001` on a 16-trajectory
all-fail step where the binary baseline would have contributed zero.

This effect is independent of whether the agent improves at solving
tests; it manifests purely from the *reward structure*, not from
trajectory diversity. We believe this is a methodologically interesting
property of dense-but-bounded rewards in the GRPO/LOOP family.

### 6.2 Why exposing tests doesn't cause over-fitting

The terminal reward scores against the *full* `FAIL_TO_PASS`, not just
the active subset. A policy that learns to satisfy only the visible
tests gets the same `r_solve` as one that satisfies the same number of
hidden tests — and the curriculum's stall-expand mechanism mixes hidden
tests into the visible set as soon as the agent gets stuck on the
visible ones. So a generalization-gap penalty is automatically baked in
to the training signal.

### 6.3 Cost analysis

Per-rollout costs, measured on the smoke (single H200, openhands
scaffold, max_steps=20):

| Subroutine | Time / traj |
|---|---|
| LLM gen | ~26 s (avg) |
| In-container bash exec | ~180 s |
| Terminal `/run_tests.sh` | ~5 s |
| Total | ~210 s (3.5 min) |

The terminal reward is 2-3% of trajectory time; in v1 ATS pays nothing
beyond what the binary baseline already pays. The v2 mid-episode pytest
path multiplies the rollout cost by `1 + (max_steps/k_steps) × (5/total_time)`,
which for our settings is ~`1 + (150/5) × (5/210) ≈ 1.7×`. Not free,
but tractable for ablations.

### 6.4 Limitations of small FAIL_TO_PASS

The SWE-rebench pool we use is skewed: 18/32 tasks have `|F|=1`,
7/32 have `|F|=2`, only 7/32 have `|F| ≥ 3` (max 37). With `|F|=1` the
partial-credit signal degenerates to binary in spirit (the only
non-binary terms are `r_token` and the optional `eta/(n+1)`). A
follow-on dataset with `|F| ≥ 5` filter (see
`MIN_FTP=5 build_comprehensive_swerebench.py`) is planned to exercise
the full continuous-reward regime.

## 7. Limitations and future work

- v2 dense-tick path is too expensive for routine training; a true subset
  pytest invocation (pytest `-k` or node-ids) would unlock it. Some
  swerebench images package non-pytest runners — a portable subset-runner
  shim is future work.
- Token penalty weight `lam_token` is brittle; tasks with long-by-necessity
  patches get punished unfairly. A per-task normalized variant is a TODO.
- We evaluate on SWE-rebench only; SWE-bench Verified eval planned.
- **Small-FAIL_TO_PASS degeneracy.** When `|F| ≤ M_init` the active set
  contains everything; the curriculum has nothing to mask. Our smoke data
  (SWE-rebench filtered to public-image-pullable) skews toward `|F|=1-5`,
  so the bandit-scheduler aspect of the method is exercised lightly. A
  follow-on filter on `|F| ≥ 8` would isolate the curriculum's effect.
- **Token approximation.** We count characters, not real tokens, in the
  budget term. Off by ~4× from the actual tokenizer. Empirically fine
  because the *scale* is absorbed by `T_ref` and `lam_token`, but a true
  per-tokenizer count would tighten the analysis.

## 8. The reward design in context: why dense + curriculum + token-aware

A natural objection to ATS is that it has three knobs (curriculum,
partial credit, token penalty) and the contribution of any one is
unclear without ablations. We address this with the table in §5.3.
Beyond that, the three ideas are not arbitrary — each addresses one of
the three baseline pathologies stated in §1:

| Pathology | Mechanism that addresses it |
|---|---|
| Credit assignment | Partial credit over `F` (continuous terminal reward) |
| Verifier opacity | Prompt injection of an active test subset |
| Unmeasured token cost | Explicit `-lam_token · tokens / T_ref` term |

The curriculum (cooldown / stall-expand / retire) is technically optional
for v1 — it kicks in only at v2 — but it is the mechanism that makes the
prompt-injection token-bounded for large-`|F|` tasks (without it,
exposing 20+ test bodies eats the context budget).

## A. Appendix: reproducibility

All code lives on branch `reward/active-test-selection` of the project repo
(commits c1685a8 and parent). Single command to reproduce the smoke run:

```bash
TOTAL_TRAINING_STEPS=4 TRAIN_BATCH_SIZE=4 AGENT_MAX_STEPS=20 \
MAX_RESPONSE=12288 EXPERIMENT_NAME=smoke \
  sg docker -c 'bash DeepSWE_RL/rllm/examples/swe/train_active_tests_swe_master_4b_h200.sh'
```

All scheduler knobs are exposed as `RLLM_ACTIVE_TESTS_*` env vars and
documented in `train_active_tests_swe_master_4b_h200.sh`.
