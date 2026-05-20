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
sub-goals.

We propose **Active-Test-Selection (ATS) rewards**, a curriculum-aware
training objective that (i) exposes a bounded, dynamically managed *subset*
of the target tests to the policy in its prompt, (ii) replaces the binary
terminal reward with a continuous credit-assigned signal driven by a
four-state bandit-style scheduler over the tests (ACTIVE, POOL, COOLDOWN,
RETIRED), and (iii) explicitly penalizes the prompt-and-completion token
footprint of the trajectory. On Qwen3-4B (SWE-Master-4B-SFT) trained on
SWE-rebench with the LOOP advantage estimator, ATS yields {X% better
solve-rate} while using {Y% fewer training tokens per solved instance}.

Contributions:
1. The ATS scheduler and reward function, including a four-state machine
   with a `solve → cooldown → re-verify` cycle that closes the
   "reward-hacking by transient pass" loophole.
2. A token-budget term in the reward that makes the agent's prompt-and-rollout
   footprint an *explicit* training objective rather than a passive cost.
3. A SWE-rebench setup and ablation suite isolating the contributions of the
   curriculum, the cooldown re-check, the token penalty, and the dense
   per-tick signal.
4. Empirical evidence that exposing the test set to the agent does *not*
   leak generalization, provided the terminal reward is computed against the
   full hidden set.

## 1. Introduction

The state of the art in SWE-agent RL — DeepSWE, SWE-Master, R2E-Gym —
trains a policy to act in a Docker sandbox containing a buggy repository,
producing a patch that ultimately must turn a held-out test suite from
failing to passing. The reward function used in every published recipe we
have inspected is, modulo cosmetic variation:

```
r = int(all FAIL_TO_PASS resolved at end of trajectory)
```

This has three known pathologies:

1. **Credit assignment.** A 50-200 step rollout (file reads, edits, bash
   invocations) gets one bit at the end. GRPO/LOOP-style group baselines
   collapse to zero advantage whenever all rollouts in a group succeed or
   all fail simultaneously — i.e., on most easy and most hard tasks.
2. **Verifier opacity.** The agent never sees the tests it is supposed to
   satisfy. It must reverse-engineer the spec from the natural-language
   issue plus repo exploration. This is unrealistic relative to how human
   engineers work (TDD) and wastes large fractions of the rollout on
   spec-discovery.
3. **Token cost is unmeasured.** Prompts ingest entire files; completions
   blow past 16K-32K tokens. Compute scales as the square of context length,
   yet nothing in the objective rewards parsimony.

We claim all three are addressable with a single mechanism: **expose a
bounded subset of the tests to the policy each rollout, and reward
test-by-test progress, with an explicit token penalty**.

## 2. Related work

(Stub. To expand.)

- **SWE-Master / DeepSWE RL** (Sun et al., 2025?): the baseline we extend.
  Binary terminal reward, full FAIL_TO_PASS.
- **R2E-Gym** (verifier infrastructure for code-agents).
- **SWE-rebench**: a re-execution-validated SWE-bench-style mixture used as
  our training data.
- **Process Reward Models (PRMs)**: dense per-step reward via a learned
  verifier (Lightman et al. 2023; Wang et al. 2024). Closely related in
  spirit, but PRMs require a labeled per-step training set; our rewards
  come from the actual test verifier and are zero-cost per step (terminal
  evaluation only, in the v1 setup).
- **Curriculum learning for RL** (Bengio et al. 2009; Narvekar et al.
  2020). We adapt the curriculum *within* an episode rather than across
  episodes.
- **Test-driven prompting**: prior work (Endor, SWE-bench-Live) exposes
  test scaffolds to LMs but for non-RL inference; we use the same insight
  to shape RL training.

## 3. Method

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

### 3.4 Two operating modes

- **v1 / paper headline**: scheduler manages the initial ACTIVE subset and
  partial-credit terminal reward, *no* mid-episode pytest. Token-cheap.
  This is what we evaluate first; it is what the user's clarification asked
  for ("see test cases during training, reduce token usage").
- **v2 / ablation**: enable `RLLM_ACTIVE_TESTS_MID_EPISODE=1` to run
  per-tick pytest and drive the full state machine. Used to ablate the
  cooldown-re-check and dense-tick contributions.

## 4. Experimental setup

- **Base model**: Qwen3-4B SFT'd on SWE traces (RUC-AIBOX/SWE-Master-4B-SFT,
  HF Feb 2026).
- **Training data**: SWE-rebench public mirror (`nebius/SWE-rebench`),
  32 instances filtered to ones with pullable public images. Validation: 4
  held-out from same pool (smoke); SWE-bench Verified for fair comparison
  (planned).
- **RL algorithm**: LOOP (leave-one-out, sibling of GRPO), full-param
  4B PPO, n=4 group rollouts, single H200 (143 GB), max_response_length
  16384 (smoke) / 32768 (full).
- **Baselines**:
  1. Binary terminal reward (current SWE-Master recipe).
  2. Partial-credit terminal only (ATS without prompt injection).
  3. ATS, no token penalty.
  4. ATS, full.

## 5. Results

(To be filled as runs complete. Placeholders below for the metrics we
want.)

### 5.1 Smoke result (single H200, 4 PPO steps, 32-task SWE-rebench)

- Solve rate on train tasks across epochs: TBD
- Mean trajectory reward: TBD
- Tokens per trajectory: TBD
- Gradient stability (non-zero-grad-step ratio, grad-norm mean): TBD

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

(Stubs.)

- **Why exposing tests doesn't cause overfitting**: the terminal reward
  scores against the full set, and stall-expand draws from hidden tests, so
  a policy that learns to satisfy only the visible block sees its training
  reward erode immediately.
- **Cost analysis**: each scheduler tick in v2 costs ≈1 full `/run_tests.sh`
  invocation (3-5 min in Docker). With k_steps=5 over 150-step episodes
  this is a >20× multiplier on rollout cost. The v1 variant pays nothing
  beyond a single terminal evaluation — the same as the binary baseline.

## 7. Limitations and future work

- v2 dense-tick path is too expensive for routine training; a true subset
  pytest invocation (pytest `-k` or node-ids) would unlock it. Some
  swerebench images package non-pytest runners — a portable subset-runner
  shim is future work.
- Token penalty weight `lam_token` is brittle; tasks with long-by-necessity
  patches get punished unfairly. A per-task normalized variant is a TODO.
- We evaluate on SWE-rebench only; SWE-bench Verified eval planned.

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
