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

#### 1.1 Trajectory File Reference (Not Inline)
**Current**:
```python
def build_prompt(entry, test_id, test_src, trajs):
    # trajs = [trajectory_1, trajectory_2, trajectory_3]  # 3 full trajectories
    for i, t in enumerate(trajs, 1):
        blocks.append(f"\n## Agent trajectory #{i}\n" + _render_trajectory(t))
```

**Problem**: Full trajectory (100-500 tokens) × 3 = 300-1500 tokens inlined in prompt

**Solution**: Write trajectories to disk files, pass file paths to agent instead
```python
def build_prompt(entry, test_id, test_src, trajs, work_dir: str):
    """Build prompt, write trajectories to disk for agent to read."""
    blocks = [_SYSTEM, ...]
    
    # Write trajectories to files, reference by path
    traj_refs = []
    for i, t in enumerate(trajs, 1):
        traj_file = os.path.join(work_dir, f"trajectory_{i}.jsonl")
        with open(traj_file, "w") as f:
            for step in t:
                f.write(json.dumps(step) + "\n")
        traj_refs.append(f"  trajectory_{i}.jsonl (in current directory)")
    
    if traj_refs:
        blocks.append("\n## Agent trajectories (read from files)\n" + 
                     "The following trajectory files are available for you to read:\n" +
                     "\n".join(traj_refs))
    
    return "\n".join(blocks)
```

**Outcome**: Agent can `cat trajectory_1.jsonl` to read trajectories on demand, but prompt doesn't embed them
**Savings**: 300-1500 tokens per call (trajectories now 0 tokens in prompt)

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

#### 1.3 Remove Gold Patch Entirely
**Current**: Full `patch` included in prompt (can be 1000-3000 tokens for large diffs)

**Rationale**: Agent will explore repository files directly and understand the fix by reading the code. The patch doesn't help the agent refine the test; the agent needs to understand what makes the code correct by inspecting it.

**Solution**: Don't send patch at all
```python
def build_prompt(entry, test_id, test_src, trajs, work_dir: str):
    blocks = [_SYSTEM]  # Remove patch from here
    blocks.append("\n## Issue\n" + _extract_issue_core(issue))
    blocks.append(f"\n## Original (hard) FAIL_TO_PASS test `{test_id}`\n" + test_src)
    # Trajectories reference (not inline)
    blocks.append(f"\n## Agent trajectories (read from files)\n" + traj_refs)
    # NO PATCH
    return "\n".join(blocks)
```

**Savings**: 1000-3000 tokens per call (gold patch removed entirely)

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

**TIER 1 Total**: 
- Trajectories removed from prompt (file refs only): 300-1500 tokens saved
- Gold patch removed entirely: 1000-3000 tokens saved
- **Total: 1300-4500 tokens per call = 60-75% reduction**

---

### TIER 2: Configuration Changes
**SKIPPED per user feedback** — keeping agentic refinement as-is with just TIER 1 content trimming

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

---

## Recommended Implementation Plan

### Phase 1: TIER 1 Content Trimming (30 min, 60-75% token reduction)
1. **Trajectory File References** (instead of inline)
   - Write trajectories to `trajectory_1.jsonl`, `trajectory_2.jsonl`, etc. in work directory
   - Pass file references in prompt (not the trajectory data)
   - Agent can `cat trajectory_1.jsonl` to read on demand
   - Result: 300-1500 tokens saved per call

2. **Remove Gold Patch Entirely**
   - Don't send patch in prompt
   - Agent will explore repo files directly and understand the fix
   - Result: 1000-3000 tokens saved per call

**Total TIER 1**: 1300-4500 tokens saved per call (60-75% reduction)

### Phase 2: TIER 3 Validation Optimization (30 min, wall-clock improvement)
1. **Quick Syntax Check Before Docker**
   - Add `_quick_syntax_check()` to validate Python syntax before spinning Docker containers
   - Skip Docker validation entirely if all refined tests fail syntax check
   - Result: 60-90 seconds faster per call if agent produces syntactically invalid test(s)

---

## Configuration Changes (None)
No configuration changes required. All improvements are code-level in `agentic_refine.py`.

---

## Expected Impact

| Approach | Tokens Saved | Implementation | Impact |
|----------|--------------|-----------------|--------|
| **TIER 1.1** (trajectory file refs) | 300-1500 / call | 20 min | Remove trajectories from prompt |
| **TIER 1.2** (remove patch) | 1000-3000 / call | 10 min | Agent explores repo directly |
| **TIER 3.1** (syntax check) | N/A tokens | 20 min | -60-90 sec per call wall-clock |

**Combined TIER 1**: 1300-4500 tokens saved per call = **60-75% reduction in prompt size**

**Combined TIER 1+3**: 60-75% token reduction + faster validation (skip Docker for broken tests)

---

## Rollback Plan
- All changes are additive (new summary functions don't break existing code)
- Config changes are env-var based (easily revert with `RLLM_ACTIVE_TESTS_AGENTIC_REFINE=1`)
- No persistent state changes

