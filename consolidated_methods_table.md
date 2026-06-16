# Consolidated Methods Table: Test Selection, Test Refinement, and Token Efficiency in RL for SWE

Three tables — one per stage of the pipeline — covering every idea and method discussed across the previous documents. Each row cites the original paper or the closest published reference where one exists. Where an idea is novel or unpublished in this exact form, the citation column lists the nearest neighbor work that grounds the technique.

## Table 1 — Test Selection Strategies

| # | Method | Description | Original / Reference Paper |
|---|---|---|---|
| S1 | **Goldilocks solve-rate banding (static)** | Train only on instances with solve rate in [0.2, 0.8] to maximize within-group advantage in GRPO | Self-play SWE-RL [a]; "ideal-difficulty" framing |
| S2 | **Reward-variance-based selection** | Select on within-group reward *variance* (proxy for the GRPO advantage itself) rather than on solve rate | GRPO advantage formulation, Shao et al. [b] |
| S3 | **Prioritized experience replay for SWE-RL** | Sample instances proportional to recent loss^α, importance-weight by (1/p)^β | Schaul et al., Prioritized Experience Replay [c] |
| S4 | **Disagreement / self-consistency selection** | Cluster k high-temperature rollouts by patch semantics; select where patches semantically disagree | Settles, Active Learning Survey [d] |
| S5 | **Diversity-aware (instance-level)** | Cluster instances by bug type / file / patch size; enforce per-batch balance | Standard balanced-sampling literature; SWE-Bench-CL [e] |
| S6 | **Within-instance mutation-score-based test ranking** | Within an instance with multiple tests, rank tests by mutation kill rate | Jia & Harman, Mutation Testing Survey [f] |
| S7 | **Annealed difficulty curriculum** | Slide the Goldilocks band upward as policy capability grows | Bengio et al., Curriculum Learning [g]; Open-Reasoner-Zero [h] |
| S8 | **Influence-function-based selection** | Pick instances that most improve held-out validation performance | Koh & Liang, Influence Functions [i] |
| S9 | **Failure-mode-targeted selection** | Diagnose policy's weak bug-types; bias batch toward them | Bug-type taxonomy (no direct paper); related to SWE-bench analysis [j] |
| S10 | **Bandit over selection strategies (meta-selection)** | Multi-armed bandit chooses which selection strategy to use per step | UCB / Thompson sampling — Auer et al. [k]; no published SWE-RL instance |
| S11 | **Token-efficiency-weighted selection** | Score by expected (learning signal) / (rollout-token cost) | Novel framing aligned with SWE-Effi [l] |

## Table 2 — Test Refinement Approaches

| # | Method | Description | Original / Reference Paper |
|---|---|---|---|
| R1 | **Direct LLM rewriting** | Single-shot teacher rewrite of test from issue + buggy code + oracle patch | SWE-Tester [m] (distilled variant) |
| R2 | **Execution-feedback iterative refinement** | Generate → run → feed traces back → regenerate, for K iterations | Otter / e-Otter++ [n] |
| R3 | **Adversarial test–patch co-refinement** | Test generator and patch generator iterate against each other; selector picks reliable fix | InfCode [o] |
| R4 | **Coverage-guided refinement** | Refined test must execute every line modified by the oracle patch | Mutation-testing & coverage literature [f]; no direct SWE paper |
| R5 | **Test decomposition (curriculum within instance)** | Split a hard test into a sequence of easier sub-tests; reward = fraction passed | Novel; nearest neighbor is process-reward work, PRIME [p] |
| R6 | **Issue-description refinement** | Enrich the issue text rather than the test (without leaking the fix) | Under-explored; nearest is no-spoiler prompt engineering |
| R7 | **Specification-first test generation** | Extract structured spec from issue, derive tests from spec | Formal-methods test generation literature |
| R8 | **Property-based test generation** | Generate invariants instead of example assertions | MacIver & Hatfield-Dodds, Hypothesis [q] |
| R9 | **Cross-family generate-and-intersect** | Generate tests from multiple teacher families; keep intersection | Self-consistency [r]; no direct SWE paper |
| R10 | **Distilled refiner** | Use frontier teacher offline to build a dataset; fine-tune a small open refiner | SWE-Tester [m] |
| R11 | **Hybrid LLM + search-based testing** | LLM proposes scaffold; search-based testing finds boundary inputs | Kitsios et al. [s] |
| R12 | **Backtranslation from oracle patch** | Reverse-engineer minimal tests directly from the gold diff | Novel; nearest is "patches as supervision" in SWE-RL [t] |
| R13 | **Delta debugging / test minimization** | Iteratively remove statements from an over-asserting test while keeping pre/post gates | Zeller & Hildebrandt, Delta Debugging [u] |
| R14 | **PR-discussion / code-review mining** | Extract reviewer-stated expectations from GitHub PR threads to ground refinement | SWE-bench data construction [v]; no direct refinement paper |
| R15 | **Regression test mining from post-fix git history** | Harvest tests added in commits after the fix in the same files | Standard repo-mining technique; no direct SWE-RL paper |
| R16 | **Metamorphic testing** | Assert relations between outputs under input transformations | Chen, Cheung & Yiu, Metamorphic Testing [w] |
| R17 | **Behavioral-diff testing** | Execute buggy + fixed code, assert only on observed differences | Differential testing — McKeeman [x] |
| R18 | **Fuzzing-guided refinement** | Coverage-guided fuzzer biased toward patched lines to find discriminating inputs | Google Atheris [y]; AFL — Zalewski |
| R19 | **Active refinement from policy failure patterns** | Cluster the policy's failed rollouts; refine tests to discriminate near-misses from the correct fix | Novel; nearest is active learning [d] |
| R20 | **Bug-type classification with templates** | Classify bug type, apply type-specific test template | APR taxonomy literature; no direct paper |
| R21 | **LLM-as-judge with structured rubric** | Score candidate refined tests on a multi-dimensional rubric before acceptance | Shi et al., LLM-as-Judge for Patch Eval [z] |
| R22 | **Mutation-validated specificity gate** | Refined test must fail on ≥M of N plausible wrong-patch mutants | Self-play SWE-RL inverse mutation testing [a] |

## Table 3 — Token Efficiency Methods in RL for SWE

| # | Category | Method | Description | Original Paper |
|---|---|---|---|---|
| T1 | Length penalty | **DAPO Overlong Reward Shaping** | Soft penalty in [L_max − L_cache, L_max], hard penalty beyond | Yu et al., DAPO [aa] |
| T2 | Length penalty | **DLER** | Corrected length penalty that avoids GRPO-advantage corruption | DLER [ab] |
| T3 | Length penalty | **GRPO-LEAD** | Length-dependent *accuracy* reward, conciseness rewarded within group | GRPO-LEAD [ac] |
| T4 | Length penalty | **DRPO** | Decouples length signal from GRPO advantage entirely | DRPO [ad] |
| T5 | Length penalty | **AutoThink** | Length decreases reward when correct, increases when incorrect | AutoThink — Tu et al. [ae] |
| T6 | Length penalty | **Short-RL / ASRR** | Conditional penalty — only when correct, or off when batch accuracy drops | Yuan et al. / Zhang et al. [af] |
| T7 | Length penalty | **Optimizing Length Compression** | Group-relative length reward: 1 − \|o_i\|/max\|o_j\| | Optimizing Length Compression [ag] |
| T8 | Loss masking | **DeepSWE Compact Filtering** | Mask loss for trajectories that hit max context / timeout / max steps | DeepSWE [ah] |
| T9 | Loss masking | **SWE-RM tool-response masking** | Mask tool outputs from gradient; agent only credited for its own tokens | SWE-RM [ai] |
| T10 | Loss masking | **Overlong filtering (DAPO variant)** | Mask truncated samples rather than negatively reinforce them | DAPO [aa]; GRPO++ [aj] |
| T11 | Context compression | **SWE-AGILE Compression Rate Reward** | Trajectory-level compression reward inside RLVR | SWE-AGILE [ak] |
| T12 | Context compression | **SWE-Master Context Manager** | -42% input / -50% peak per trajectory; comparable resolve rate | SWE-Master [al] |
| T13 | Context compression | **CoT compression with mastery gating** | Cross-domain CoT compression; -67% length / -52% rounds on SWE-Verified | CoT Compression [am] |
| T14 | Context compression | **WebAgent-R1 dynamic context compression** | M-GRPO multi-turn RL with dynamic compression | WebAgent-R1 [an] |
| T15 | Inference pruning | **SWE-Pruner** | Self-adaptive context pruning; -23–38% tokens, +1.2–1.4 pts | SWE-Pruner [ao] |
| T16 | Inference pruning | **AgentDiet** | Remove waste (cache files, dead enumerations) from trajectories | AgentDiet [ap] |
| T17 | Scaffold / tools | **SWE-Master LSP integration** | Distill semantic LSP-based navigation; -23.7% in / -16.3% out / -17.5% turns | SWE-Master [al] |
| T18 | Scaffold / tools | **SkyRL-Agent AST search** | AST-based code-navigation tool; 2× cost reduction at matched performance | SkyRL-Agent [aq] |
| T19 | Difficulty filtering | **SWE-RM static difficulty filter** | Drop too-easy and too-hard instances offline | SWE-RM [ai] |
| T20 | Difficulty filtering | **Self-play SWE-RL adaptive difficulty** | Bug-injector generates instances of ideal difficulty for current policy | Self-play SWE-RL [a] |
| T21 | Evaluation framing | **SWE-Effi (EuTB metric)** | Effectiveness under Token Budget; names "expensive failures" + "token snowball" | SWE-Effi [l] |
| T22 | Test-time scaling | **DeepSWE hybrid TTS** | Execution-based + execution-free verifiers reach 59% from 42% base | DeepSWE [ah] |
| T23 | Test-time scaling | **R4P / Mini-SE patch verifier** | Learned verifier replaces test execution, 50× faster | R4P / Mini-SE [ar] |
| T24 | Dynamic sampling | **DAPO Dynamic Sampling** | Discard groups with all-0 or all-G accuracy; 50% fewer training steps | DAPO [aa] |
| T25 | Dynamic sampling | **DSPO** | `0 < Σ R < G` retention condition for agentic search | DSPO [as] |
| T26 | Dynamic sampling | **Skywork-OR1 history resampling** | Drop samples uniformly solved in last epoch | Skywork-OR1 [at] |
| T27 | Dynamic sampling | **Yuan3.0 ADS** | Resample from highest-pass-rate prompts; 52.91% training-efficiency gain | Yuan3.0 [au] |
| T28 | Dynamic sampling | **QiMeng-CodeV-R1 adaptive batch size** | Dynamically estimated valid-sample ratio; adapts generation batch size | QiMeng-CodeV-R1 [av] |
| T29 | Dynamic sampling | **RubricRL rollout-level filtering** | Filter individual rollouts inside a prompt, not whole prompts | RubricRL [aw] |
| T30 | Async / Infra | **ROLL Flash** | Environment-level async + redundant rollouts; 1.81× SWE / 2.72× ALFWorld | ROLL Flash [ax] |
| T31 | Async / Infra | **SkyRL-Agent async dispatcher** | 1.55× speedup over naive async batching; SA-SWE-32B reaches 39.4% | SkyRL-Agent [aq] |
| T32 | Async / Infra | **Klear-AgentForge disaggregated arch** | Separate rollout vs. train nodes to avoid GPU idle | Klear-AgentForge [ay] |
| T33 | Async / Infra | **ROLLART** | Disaggregated infra with async reward computation + Reward-as-a-Service | ROLLART [az] |
| T34 | Async / Infra | **Forge async controller** | Resolves Strict-FIFO straggler + Greedy distribution-shift trade-off | Forge / MiniMax [ba] |
| T35 | Prefix caching | **Tree Training** | Tree-structured training preserves shared-prefix reuse in backward pass | Tree Training [bb] |
| T36 | Prefix caching | **REDSearcher two-tier load balancing** | Affinity within rollout + round-robin across engines for prefix-cache hits | REDSearcher [bc] |
| T37 | Prefix caching | **Sequence packing** | Concatenate short trajectories into fixed-length training examples | Krell et al. [bd] |
| T38 | Step-level rewards | **SWE-TRACE rubric PRM** | LLM multi-task cascading + rubric process rewards + heuristic-guided TTS | SWE-TRACE [be] |
| T39 | Step-level rewards | **RTMC (Rollout-Tree MC)** | Tree-based per-step advantage estimation, critic-free | RTMC [bf] |
| T40 | Step-level rewards | **PRIME** | Implicit process reward derived online from outcome labels | PRIME — Cui et al. [p] |
| T41 | Step-level rewards | **VinePPO** | Token-level branching for credit assignment | VinePPO [bg] |
| T42 | Step-level rewards | **Tree-GRPO** | Turn-level branching in multi-turn settings | Tree-GRPO [bh] |
| T43 | Token-level loss | **DAPO Token-Level Policy Gradient Loss** | Per-token weighting instead of sample-level dilution in long-CoT | DAPO [aa] |
| T44 | Token-level loss | **Yuan3.0 80/20 rule** | Update gradients only on top-20% highest-entropy tokens | Yuan3.0 [au] |
| T45 | Curriculum / history | **Open-Reasoner-Zero two-step curriculum** | Two-stage difficulty progression | Open-Reasoner-Zero [h] |
| T46 | Curriculum / history | **SRPO math-then-code** | First math for reasoning, then code on the foundation | SRPO [bi] |
| T47 | Curriculum / history | **Light-R1 two-sided clipping** | Broader importance-sampling clipping for stable off-policy reuse | Light-R1 [bj] |
| T48 | **Ours** | **Adaptive test refinement + selection** | Proactively refine tests in low-signal instances + difficulty-aware selection | This work |

## Reference Key

[a] Meta (2025). *Toward Training Superintelligent Software Agents through Self-Play SWE-RL (SSR).* arXiv:2512.18552.
[b] Shao, Z. et al. (2024). *DeepSeekMath.* (GRPO.) arXiv:2402.03300.
[c] Schaul, T. et al. (2016). *Prioritized Experience Replay.* ICLR.
[d] Settles, B. (2009). *Active Learning Literature Survey.* U Wisconsin TR 1648.
[e] (2025). *SWE-Bench-CL: Continual Learning for Coding Agents.* arXiv:2507.00014.
[f] Jia, Y. & Harman, M. (2011). *Analysis and Survey of the Development of Mutation Testing.* IEEE TSE 37(5).
[g] Bengio, Y. et al. (2009). *Curriculum Learning.* ICML.
[h] Hu et al. (2025). *Open-Reasoner-Zero.*
[i] Koh, P. W. & Liang, P. (2017). *Understanding Black-box Predictions via Influence Functions.* ICML.
[j] Jimenez, C. E. et al. (2024). *SWE-bench.* ICLR.
[k] Auer, P. et al. (2002). *Finite-time Analysis of the Multiarmed Bandit Problem.* Machine Learning.
[l] Vasilevski, K. et al. (2025). *SWE-Effi.* arXiv:2509.09853.
[m] Soni, A. B. et al. (2026). *SWE-Tester.* arXiv:2601.13713.
[n] Ahmed, T. et al. (2025). *e-Otter++.* arXiv:2605.04320.
[o] Li, K. et al. (2025). *InfCode.* arXiv:2511.16004.
[p] Cui, Y. et al. (2025). *PRIME: Process Reinforcement through IMplicit rEwards.*
[q] MacIver, D. R. & Hatfield-Dodds, Z. (2019). *Hypothesis.* JOSS.
[r] Wang, X. et al. (2023). *Self-Consistency Improves Chain of Thought Reasoning.* ICLR.
[s] Kitsios, K. et al. (2025). *Automated Generation of Issue-Reproducing Tests by Combining LLMs and Search-Based Testing.* arXiv:2509.01616.
[t] Wei, Y. et al. (2025). *SWE-RL.* arXiv:2502.18449.
[u] Zeller, A. & Hildebrandt, R. (2002). *Simplifying and Isolating Failure-Inducing Input.* IEEE TSE 28(2).
[v] Jimenez, C. E. et al. (2024). *SWE-bench.* ICLR.
[w] Chen, T. Y., Cheung, S. C. & Yiu, S. M. (1998). *Metamorphic Testing.* TR HKUST-CS98-01.
[x] McKeeman, W. M. (1998). *Differential Testing for Software.* Digital Technical Journal.
[y] Google (2020). *Atheris: Coverage-Guided Native Python Fuzzer.* github.com/google/atheris.
[z] Shi, S. et al. (2025). *Human-in-the-Loop Framework for Reliable Patch Evaluation Using LLM-as-a-Judge.* arXiv:2511.10865.
[aa] Yu, Y. et al. (2025). *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* dapo-sia.github.io.
[ab] (2025). *DLER: Doing Length pEnalty Right.* arXiv:2510.15110.
[ac] (2025). *GRPO-LEAD.* arXiv:2504.09696.
[ad] (2025). *DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization.* arXiv:2510.04474.
[ae] Tu et al. (2025). *AutoThink.* (Referenced in survey [bk].)
[af] Yuan et al. (2025). *Short-RL* / Zhang et al. (2025). *ASRR.* (Both in survey [bk].)
[ag] (2025). *Optimizing Length Compression in Large Reasoning Models.* arXiv:2506.14755.
[ah] Together AI (2025). *DeepSWE.* together.ai/blog/deepswe.
[ai] (2025). *SWE-RM: Execution-free Feedback for Software Engineering Agents.* arXiv:2512.21919.
[aj] Wolfe, C. R. (2026). *GRPO++.* cameronrwolfe.substack.com.
[ak] (2026). *SWE-AGILE.* arXiv:2604.11716.
[al] (2026). *SWE-Master: Unleashing the Potential of Software Engineering Agents via Post-Training.* arXiv:2602.03411.
[am] (2026). *RL for Chain of Thought Compression with One-Domain-to-All Generalization.* arXiv:2601.06052.
[an] (2025). *WebAgent-R1.* arXiv:2505.16421.
[ao] (2026). *SWE-Pruner.* arXiv:2601.16746.
[ap] (2025). *AgentDiet.* arXiv:2509.23586.
[aq] (2025). *SkyRL-Agent.* arXiv:2511.16108.
[ar] (2025). *Scalable Supervising Software Agents with Patch Reasoner (R4P / Mini-SE).* arXiv:2510.22775.
[as] (2025). *DSPO: Stable and Efficient Policy Optimization for Agentic Search and Reasoning.* arXiv:2510.09255.
[at] He et al. (2025). *Skywork-OR1.*
[au] (2026). *Yuan3.0 Flash.* arXiv:2601.01718.
[av] (2025). *QiMeng-CodeV-R1.* arXiv:2505.24183.
[aw] (2025). *RubricRL.* arXiv:2511.20651.
[ax] (2025). *ROLL Flash.* arXiv:2510.11345.
[ay] (2025). *Klear-AgentForge.* arXiv:2511.05951.
[az] (2025). *ROLLART.* arXiv:2512.22560.
[ba] MiniMax (2026). *Forge.* HuggingFace blog.
[bb] (2025). *Tree Training: Accelerating Agentic LLMs Training via Shared Prefix Reuse.* arXiv:2511.00413.
[bc] (2026). *REDSearcher.* arXiv:2602.14234.
[bd] Krell, M. M. et al. (2021). *Sequence Packing.*
[be] (2026). *SWE-TRACE.* arXiv:2604.14820.
[bf] (2026). *RTMC: Step-Level Credit Assignment via Rollout Trees.* arXiv:2604.11037.
[bg] Kazemnejad, A. et al. (2024). *VinePPO.*
[bh] Ji et al. (2025). *Tree-GRPO.*
[bi] Zhang et al. (2025). *SRPO.*
[bj] Wen et al. (2025). *Light-R1.*
[bk] (2025). *Towards Concise and Adaptive Thinking in Large Reasoning Models: A Survey.* arXiv:2507.09662.

---

**Note on verification.** Some arxiv IDs were reconstructed from search snippets; verify each against the actual arxiv listing before submitting in any work. Where a paper has multiple authors or partial metadata, the canonical full citation should be checked on the official venue or arxiv page.
