# ATS-RL Methods Ranking — Test Selection, Test Refinement, Token Reduction

Ranked shortlist of ideas from [`consolidated_methods_table.md`](consolidated_methods_table.md),
evaluated against the **ATS-RL** pipeline (Active-Test-Selection RL for SWE agents:
partial-credit reward over `FAIL_TO_PASS`, bounded test subset shown in the prompt,
`PASS_TO_PASS` regression penalty, capped token penalty; PPO via rllm/verl on a single
H200, base = SWE-Master-4B-SFT, SWE-rebench data).

All arXiv IDs / venues below were verified by reading the source papers; corrections to the
original doc are listed at the bottom.

---

## Selection rubric

Each candidate scored 1–5 (5 = best) on five axes. **Fit is double-weighted** — an idea only
matters if it amplifies one of ATS-RL's two headline claims or fixes a known pain point.

**Headline claims:** (1) denser gradient signal (~100% of PPO steps vs binary's ~50%),
(2) fewer tokens per solved instance.

**Known pain points:** (i) pytest in Docker costs minutes → v2 dense reward is disabled;
(ii) the char-based token penalty once produced a −989 outlier; (iii) reward is halved on
truncated/timeout/max-step trajectories; (iv) degenerate instances (`|F|=1`, all-pass/all-fail
groups) give no signal.

- **Ease** — 5 = verl/rllm config flag or <20 lines; 1 = train/obtain a new model.
- **Novelty/Promise** — newness *in the SWE-RL test context* × strength of evidence.
- **Fit** — how directly it improves ATS-RL specifically.
- **Evidence** — venue + quantitative backing.

---

## Table 1 — Test Selection (top 3 of S1–S11)

| Rank | Idea | Ease | Novelty | Fit×2 | Evidence | What it buys ATS-RL | Venue (verified) |
|---|---|---|---|---|---|---|---|
| **1** | **S2 — Reward-variance / advantage-based selection** (continuous generalization of DAPO dynamic sampling: keep instances with non-zero within-group partial-credit variance) | 5 | 4 | 5 | 5 | **Directly amplifies headline #1.** DAPO discards only all-0/all-G groups (binary); our partial-credit reward gives a *continuous* variance signal to rank on — DAPO becomes the binary special case of our rule. Already native in verl. | DAPO, **arXiv:2503.14476, NeurIPS 2025**; GRPO, arXiv:2402.03300 |
| **2** | **S1 — Goldilocks solve-rate banding** (weight toward p\*≈0.2 band; penalize solve-rate 0/1) | 3 | 4 | 4 | 5 | Most direct SWE-RL difficulty precedent. We can band on the *partial-credit pass-fraction* (a finer difficulty estimator than SSR's binary solve-rate) over real SWE-rebench tasks — a clean novelty contrast. Replaces our coarse MIN_FTP filter. | Self-play SWE-RL (SSR), **arXiv:2512.18552**, Meta 2025 (+10.4 SWE-bench Verified) |
| **3** | **S11 — Token-efficiency-weighted selection** (score = learning-signal ÷ rollout-token-cost) | 3 | 5 | 4 | 3 | **Amplifies headline #2.** Unpublished in SWE-RL; SWE-Effi gives the EuTB framing to ground it. Naturally prioritizes cheap-but-informative instances → fewer tokens/solve. | Novel; SWE-Effi, **arXiv:2509.09853** (EuTB = AUC of resolve-rate vs tokens) |

**Honorable mention:** S7 annealed curriculum (Open-Reasoner-Zero, arXiv:2503.24290) — natural
extension of S1 but more schedule/tuning machinery.

**Note:** S4 disagreement-selection ≈ S2 mathematically (the G rollouts *are* the committee) —
useful for framing, not a distinct mechanism. Don't cite an arXiv ID for Settles (UW TR1648) or
Bengio (ICML 2009) — neither is on arXiv.

---

## Table 2 — Test Refinement (top 3 of R1–R22)

| Rank | Idea | Ease | Novelty | Fit×2 | Evidence | What it buys ATS-RL | Venue (verified) |
|---|---|---|---|---|---|---|---|
| **1** | **R12 — Backtranslation of minimal tests from the gold patch** (condition an LLM on the gold diff; validate pass-on-gold / fail-on-pre-patch) | 4 | 5 | 5 | 3 | **Exploits an asset every other generator deliberately avoids — we already have the oracle patch.** Strictly more signal than issue-only generation; manufactures discriminative tests for degenerate `|F|=1` / no-signal instances. One LLM call + existing harness check. | Novel; extends SWE-RL "patches as supervision", **arXiv:2502.18449, NeurIPS 2025** |
| **2** | **R22 — Mutation-validated specificity gate** (keep a refined test only if it passes gold AND fails on ≥k logged wrong patches) | 4 | 5 | 5 | 4 | **Inverse mutation testing using our rollout failures as a free mutant pool.** Guarantees refined tests are *discriminative*, raising reward-signal quality — the core ATS bet. No new model; pure harness reuse. Pairs perfectly with R12 (generate → gate). | SSR **arXiv:2512.18552**; Jia & Harman, IEEE TSE 2011 |
| **3** | **R2 — Execution-feedback iterative refinement** (generate → run → feed traces → regenerate) | 3 | 3 | 4 | 5 | Most *proven* refiner; the feedback loop maps onto our test harness directly. Best for instances where backtranslation alone underspecifies. | Otter **arXiv:2502.05368, ICML 2025**; e-Otter++ **arXiv:2508.06365** (63% F2P, SOTA on TDD-Bench) — ⚠ doc's `2605.04320` is the *Java* variant |

**Honorable mentions:**
- **R19 active refinement from failure patterns** (Easy, novel) is the **targeting layer** — mine
  rollout logs for low-signal instances and spend the R12/R22 budget there; wire it as
  orchestration glue rather than a 4th standalone idea.
- **R3 InfCode** (arXiv:2511.16004) is the most conceptually aligned published work (adversarial
  test-strengthening = our discrimination objective) but Medium-Hard multi-agent; our rollout logs
  already supply its "bad-patch" population for free.
- ⚠ **R21 Shi et al. (arXiv:2511.10865) judges *patches*, not tests** — repurposable but a *soft*
  gate, strictly weaker than R22's executable gate.

---

## Table 3 — Token Usage Reduction (top 4 of T1–T47)

| Rank | Idea | Ease | Novelty | Fit×2 | Evidence | What it buys ATS-RL | Venue (verified) |
|---|---|---|---|---|---|---|---|
| **1** | **T1 — DAPO Overlong Reward Shaping** (soft linear penalty in an `L_cache` band, bounded to [−1, 0]) | 5 | 2 | 5 | 4 | **Fixes pain point (ii) outright.** Bounded by construction → the −989 char-penalty outlier becomes *impossible*. ~10 lines, drop-in replacement for our crude token penalty. | DAPO, **arXiv:2503.14476, NeurIPS 2025** |
| **2** | **T8 — DeepSWE Compact Filtering** (mask the loss for truncated / timeout / max-step trajectories instead of rewarding them) | 4 | 3 | 5 | 4 | **Fixes pain point (iii).** The field's consensus is *mask, don't halve* — our reward-halving still trains on accidental/partial endings. Prevents reward collapse. | DeepSWE, Together AI blog 2025 (**no arXiv**); GRPO++ |
| **3** | **T24 — DAPO Dynamic Sampling** (drop all-0/all-G groups, oversample to refill) | 5 | 2 | 4 | 4 | Compute/step lever (DAPO converges in ~50% steps as a *system*). Native in verl. **Shares its mechanism with S2** — implement once, count it for both headlines. | DAPO, **arXiv:2503.14476, NeurIPS 2025** |
| **4** | **T23 — R4P / Mini-SE learned patch verifier** (replace test execution as reward; ~50s → <1s, 50×) | 1 | 5 | 4 | 4 | **Highest payoff for our #1 bottleneck (pytest = minutes).** Could *unlock the disabled v2 dense reward*. High-risk: 72.2% verifier accuracy = reward noise, and it's validated only as *final* reward, not dense — pilot as a fast pre-filter before pytest. | **arXiv:2510.22775** (72.2% on SWE-bench Verified) |

**Honorable mentions:**
- **T12 SWE-Master Context Manager / T17 LSP** (arXiv:2602.03411) — our *own base model's*
  recommended scaffold (−23.7% input, −16.3% output, −17.5% turns via LSP); orthogonal to the
  GRPO loop, so it's "free" expected token savings to layer on.
- **T2 DLER (arXiv:2510.15110) / T4 DRPO (arXiv:2510.04474)** explain *why* a naive length penalty
  corrupts the GRPO advantage (length-penalizing a long-correct rollout can flip its advantage
  negative) — the theoretical backing for doing T1 carefully.

---

## Citation corrections to fold back into the source doc

- **[au] Yuan3.0 80/20 rule → wrong source.** `2601.01718` is "Yuan3.0 Flash" (method = RAPO). The
  real 80/20 paper is **"Beyond the 80/20 Rule…", arXiv:2506.01939** (Qwen, Jun 2025).
- **[n] e-Otter++ →** `2605.04320` is the *Java* follow-up; cite **2502.05368 (Otter, ICML 2025) +
  2508.06365 (e-Otter++)** for the core method.
- **[z] Shi et al. (2511.10865)** is patch evaluation, not test acceptance — reframe the R21
  description.
- Confirmed correct: DAPO is **2503.14476 (NeurIPS 2025)**; SSR 2512.18552, SWE-Tester 2601.13713,
  SWE-RL 2502.18449, SWE-Effi 2509.09853, DLER/DRPO/R4P/SWE-Master all verified real. DeepSWE has
  no arXiv (Together AI blog only).

---

## The through-line for the paper

Every #1 pick exploits something ATS-RL **already has that the published baselines lack**: a
*continuous partial-credit signal* (S2 generalizes DAPO's binary filter), the *gold patch*
(R12 backtranslation, R22 mutant gate), and *rollout logs as a free mutant/failure pool*
(R22, R19). That's a single defensible novelty thread, not three unrelated bolt-ons.

**Cheapest high-value implementation sequence:**
1. **T1 + T8** — fix the two reward bugs (~1 day).
2. **S2 / T24** — one variance-selection mechanism (verl flag), serves both headlines.
3. **R12 + R22** — the refinement contribution (generate → gate).
4. *(optional)* **T23** — to unlock the disabled v2 dense reward.
