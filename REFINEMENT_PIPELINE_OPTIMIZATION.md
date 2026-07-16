# Refinement Pipeline Optimization: Reduce Claude Code Usage

## Current State Analysis

### Architecture
Two parallel refinement pipelines integrate into ATS-RL training:

1. **Agentic Refinement** (`agentic_refine.py`):
   - Triggers: `RLLM_ACTIVE_TESTS_AGENTIC_REFINE=1` + stalled test
   - Calls: `claude -p` (Claude Code agent) headless on Max subscription
   - Per-call cost: Full project prompt + trajectory iteration overhead
   - Resource: Spawns 2-3 Docker containers + Claude session (2-3 min per call)

2. **Online Refinement** (`online_refine.py`):
   - Triggers: `RLLM_ACTIVE_TESTS_REFINE=1` + stalled test
   - Calls: OpenAI `gpt-5.4` API
   - Per-call cost: Much smaller (API call, no Docker)
   - Resource: Negligible overhead

### Usage Drain Points

**High-volume callers:**
- `build_prompt()` (agentic_refine.py:155): **Zero content trimming**
  - Sends full problem statement (can be 1000s of tokens)
  - Sends full gold patch (can be 1000s of tokens)
  - Sends FULL original test source (200-500 tokens)
  - Sends **ALL trajectories verbatim** (100-500 tokens each, up to 3 trajectories = 1500 tokens)
  - **Total per refinement call: 3000-5000 tokens of prompt overhead**

- `run_claude_p()` (agentic_refine.py:181): **Multiple concurrent sessions**
  - Bounded semaphore `_MAX_CONCURRENCY=2` but could spawn many over a training run
  - If 100 tests stall during training → 100+ Claude Code calls

- `_make_buggy_checkout()` (agentic_refine.py:298): **Docker I/O overhead**
  - `docker cp /testbed` can be slow (copies entire repo, 1GB+)
  - Blocks agent for 30-60 seconds per call

- `validate_f2p()` (agentic_refine.py:260): **2 containers per refinement**
  - Spins up buggy + fixed container
  - Tests each refined test (slow)
  - Only keeps tests that truly F2P (many refined tests fail validation)

---

## Usage Drain: Quantified Example

**Scenario**: 40-epoch training with high stall rate
- 10 unique FAIL_TO_PASS tests across instances
- Each stalls at least once → 10+ refinements
- Each refinement = 1 Claude -p call + 2 containers + 2-3 validations
- **Overhead per refinement: 5,000 tokens + 2-3 min wall-clock**

**Conservative estimate**: 50k tokens + 200 min (3+ hours) wall-clock for refinement phase

---

## Optimization Opportunities

### TIER 1: Content Trimming (Immediate, High ROI)
**Goal**: Reduce prompt tokens per call by 60-80%

#### 1.1 Trajectory Summarization
**Current**:
```python
def build_prompt(entry, test_id, test_src, trajs):
    # trajs = [trajectory_1, trajectory_2, trajectory_3]  # 3 full trajectories
    for i, t in enumerate(trajs, 1):
        blocks.append(f"\n## Agent trajectory #{i}\n" + _render_trajectory(t))
```

**Problem**: Full trajectory (100-500 tokens) × 3 = 300-1500 tokens wasted

**Solution**: Summarize instead of verbatim
```python
def _summarize_trajectory(steps: List[dict], max_chars: int = 500) -> str:
    """Compress trajectory: last 5 steps + action count."""
    if not steps:
        return "(no steps)"
    summary = []
    summary.append(f"Total steps: {len(steps)}")
    # Count action types
    actions = {}
    for s in steps:
        act = s.get("action", "").split()[0]  # first word
        actions[act] = actions.get(act, 0) + 1
    summary.append("Actions: " + ", ".join(f"{k}×{v}" for k, v in actions.items()))
    # Last 3 steps only
    summary.append("\nLast 3 steps:")
    for s in steps[-3:]:
        fn = s.get("fn", "?").split("::")[-1]
        act = (s.get("action", "") or "").split("\n")[0][:80]
        summary.append(f"  [{fn}] {act}")
    return "\n".join(summary)

# In build_prompt:
for i, t in enumerate(trajs, 1):
    blocks.append(f"\n## Agent trajectory #{i} summary\n" + _summarize_trajectory(t))
```

**Savings**: 200-800 tokens per call (300-1500 → 200-300)

#### 1.2 Problem Statement Summarization
**Current**: Full `problem_statement` (often 500-1000 words)

**Solution**: Extract and compress
```python
def _extract_issue_core(issue: str, max_chars: int = 800) -> str:
    """Keep first N words + final assertion, trim middle."""
    lines = issue.split("\n")
    if len(issue) <= max_chars:
        return issue
    # Keep first 3 lines + last 3 lines
    core = lines[:3] + ["..."] + lines[-3:]
    text = "\n".join(core)
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text

# In build_prompt:
blocks.append("\n## Issue\n" + _extract_issue_core(issue))
```

**Savings**: 200-500 tokens per call

#### 1.3 Gold Patch Summarization
**Current**: Full `patch` (can be 1000-3000 tokens for large diffs)

**Solution**: Show diff stats + key changes only
```python
def _summarize_patch(patch: str, max_lines: int = 15) -> str:
    """Show patch summary: affected files + sample changes."""
    lines = patch.split("\n")
    files = set()
    changes = []
    for line in lines:
        if line.startswith("---"):
            files.add(line.split()[-1])
        elif line.startswith("+") and not line.startswith("+++"):
            changes.append(line[:100])
    summary = [f"Affected files: {', '.join(sorted(files))}"]
    summary.append(f"Total lines in patch: {len(lines)}")
    summary.append("\nKey changes (sample):")
    summary.extend(changes[:max_lines])
    return "\n".join(summary)

# In build_prompt:
blocks.append("\n## Gold patch summary\n" + _summarize_patch(gold))
```

**Savings**: 500-2000 tokens per call

#### 1.4 Test Source Trimming
**Current**: Full test source (200-500 tokens)

**Solution**: Keep only the failing function, drop helpers
```python
def _extract_test_function_only(test_src: str, test_id: str) -> str:
    """Extract just the test function, skip helpers."""
    import re
    # Extract function name from test_id (e.g. "test_foo::test_bar" -> "test_bar")
    fn_name = test_id.split("::")[-1]
    pattern = rf"^def {re.escape(fn_name)}\([^)]*\):.*?(?=\ndef |\Z)"
    m = re.search(pattern, test_src, re.DOTALL | re.MULTILINE)
    if m:
        return m.group(0)[:1000]  # cap at 1000 chars
    return test_src[:1000]

# In build_prompt:
test_src = _extract_test_function_only(test_src, test_id)
```

**Savings**: 100-300 tokens per call

**TIER 1 Total**: 1000-3600 tokens per call → **60-80% reduction**

---

### TIER 2: Caching & Deduplication (Medium ROI, Moderate effort)

#### 2.1 Trajectory Deduplication
**Current**: Sends up to 3 trajectories per refinement, even if identical

**Solution**: Deduplicate before sending
```python
def _deduplicate_trajectories(trajs: List[List[dict]], similarity_threshold: float = 0.8) -> List[List[dict]]:
    """Keep only diverse trajectories (>20% different)."""
    if not trajs:
        return []
    from difflib import SequenceMatcher
    selected = [trajs[0]]
    for traj in trajs[1:]:
        norm_new = _normalize_traj(traj)
        if all(SequenceMatcher(None, norm_new, _normalize_traj(sel)).ratio() < similarity_threshold
               for sel in selected):
            selected.append(traj)
    return selected

# In refine_test_agentic:
trajs = _deduplicate_trajectories(trajectories)
```

**Savings**: 200-500 tokens per call (fewer trajectories sent)

#### 2.2 Cached Prompt Templates
**Current**: Rebuilds full prompt each time

**Solution**: Cache template, only fill in data
```python
# Compute once per refinement:
CACHED_SYSTEM = "[cached system prompt from _SYSTEM]"
# Build prompt:
blocks = [CACHED_SYSTEM, ...]  # reuse constant, don't rebuild string
```

**Savings**: Minimal token savings, but faster execution

#### 2.3 Cross-Refinement Cache
**Current**: Cache by `(instance_id, test_id)` → each unique test refined once

**Solution**: Also cache by **problem_statement hash** → reuse refinement across instances with same issue
```python
def _refinement_key(entry: dict, test_id: str) -> str:
    """Generate cache key: hash(problem_statement + test_id)."""
    import hashlib
    issue = entry.get("problem_statement", "")
    h = hashlib.md5(issue.encode()).hexdigest()[:8]
    return f"{h}_{test_id.split('::')[-1]}"

# Check cache by this key before calling claude -p
```

**Savings**: If same problem appears in 3 instances, 2 refinement calls skipped = 5000 tokens × 2 = 10k tokens per epoch

---

### TIER 3: Lazy Evaluation (Lower ROI, Complex)

#### 3.1 Disable Validation Containers
**Current**: Validates every refined test in 2 containers (buggy + fixed)

**Solution**: Skip validation for obvious failures
```python
def _quick_syntax_check(code: str) -> bool:
    """Quick check: can the code parse? (skips Docker)."""
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False

def validate_f2p(entry, pairs, full_code, ...):
    # Pre-filter with syntax check
    pairs = [(n, c) for n, c in pairs if _quick_syntax_check(c)]
    if not pairs:
        return []
    # Only then spin up containers
    ...
```

**Savings**: 60-90 seconds per call if syntactically invalid test is rejected upfront

#### 3.2 Disable Agentic Refinement by Default
**Current**: Agentic refinement enabled on some configs (Max subscription cost)

**Solution**: Keep enabled ONLY when:
- `RLLM_ACTIVE_TESTS_AGENTIC_REFINE=0` (default)
- Only enable explicitly for small test sets (<10 tests)

**Savings**: All agentic calls skipped unless explicitly enabled

---

### TIER 4: Alternative Approaches (Architectural)

#### 4.1 Use Lightweight Refiner (OpenAI)
**Current**: Agentic refiner calls Claude Code
**Alternative**: Use `online_refine.py` (OpenAI API only) by default
- Cost: ~$0.10 per refinement (API) vs. 5000 tokens Claude Code
- Speed: Instant response vs. 2-3 min
- Quality: Similar output (both ask same question)

**Recommendation**: Make OpenAI the default, agentic optional

#### 4.2 Disable Refinement During Training, Enable Post-Hoc
**Current**: Refinement on stall **during training** (blocks agent)
**Alternative**: Collect stalled tests during training, refine **offline after epoch**
- Benefit: No training latency, batch refinement opportunities
- Cost: Slightly lower per-test quality (no real-time feedback)
- Savings: Reduce concurrency burden from 2 to 0 during training

#### 4.3 Refinement Only on First N Stalls
**Current**: Every stall triggers refinement
**Alternative**: Refine only first 2 stalls per test
```python
stall_count = cache.get_stall_count(iid, test_id)
if stall_count <= 2:
    nodes = refine_test_agentic(...)
```

**Savings**: 70-80% fewer refinement calls for tests that stall repeatedly

---

## Recommended Implementation Plan

### Phase 1: Quick Wins (1-2 hours, 60% savings)
1. **TIER 1.1-1.4**: Content trimming in `build_prompt()`
   - Add `_summarize_trajectory()`, `_extract_issue_core()`, `_summarize_patch()`, `_extract_test_function_only()`
   - Edit `build_prompt()` to use summaries instead of verbatim
   - Result: 1000-3600 tokens per call → ~1200 tokens (67% reduction)

2. **TIER 2.1**: Trajectory deduplication
   - Add `_deduplicate_trajectories()` before selecting diverse set
   - Result: Further 200-500 tokens saved

### Phase 2: Configuration Tuning (15 min, 10-30% savings)
1. **TIER 3.2**: Disable agentic refine by default
   - Set `RLLM_ACTIVE_TESTS_AGENTIC_REFINE=0` in default scripts
   - Users can enable explicitly with `--agentic-refine` flag
   - Result: 0 Claude Code usage for standard training

2. **TIER 4.1**: Use OpenAI by default (if agentic disabled)
   - Make `RLLM_ACTIVE_TESTS_REFINE=1` default (uses online_refine.py, cheap API)
   - Result: Refinement still works, minimal cost

3. **TIER 3.1**: Quick syntax check before Docker validation
   - Add `_quick_syntax_check()` to skip obvious failures
   - Result: 30-60 sec faster per invalid refined test

### Phase 3: Architectural (1-2 hours, 40-80% savings)
1. **TIER 3.3**: Limit refinement stalls
   - Track stall count per test
   - Refine only first 2-3 stalls, then give up
   - Result: Fewer calls for low-learning tests

2. **TIER 4.2** (optional): Offline refinement
   - Collect stalled tests in training, refine post-epoch
   - More complex, but removes training blocking
   - Result: Better throughput, batch efficiencies

---

## Configuration Changes (Immediate)

Update `/home/mohammad/fun/swemaster_runpod/DeepSWE_RL/rllm/examples/swe/train_active_tests_swe_master_4b_8xh100.sh`:

```bash
# Before:
export RLLM_ACTIVE_TESTS_AGENTIC_REFINE=${RLLM_ACTIVE_TESTS_AGENTIC_REFINE:-0}  # default off (good)

# After (use API instead):
export RLLM_ACTIVE_TESTS_REFINE=${RLLM_ACTIVE_TESTS_REFINE:-1}  # enable cheap API refinement
export RLLM_ONLINE_REFINE_MODEL=${RLLM_ONLINE_REFINE_MODEL:-gpt-5.4-mini}  # use cheapest model
export RLLM_AGENTIC_MAX_CONCURRENCY=0  # disable agentic entirely unless overridden
```

---

## Expected Impact

| Approach | Tokens Saved | Implementation | Wall-Clock Impact |
|----------|--------------|-----------------|-------------------|
| **TIER 1** (content trim) | 1000-3600 / call | 2 hours | -60-80% per call |
| **TIER 2.1** (dedupe) | 200-500 / call | 30 min | -10-20% per call |
| **TIER 3.2** (agentic off) | **ALL** agentic calls | 5 min config | -100% agentic usage |
| **TIER 4.1** (OpenAI default) | 4000 tokens / call | 0 min | N/A (uses API) |
| **TIER 3.3** (limit stalls) | 60-80% of refinement calls | 1 hour | -60-80% total refinement |

**Conservative estimate after Phase 1+2**: 80-90% reduction in Claude Code usage for refinement pipeline

---

## Rollback Plan
- All changes are additive (new summary functions don't break existing code)
- Config changes are env-var based (easily revert with `RLLM_ACTIVE_TESTS_AGENTIC_REFINE=1`)
- No persistent state changes

