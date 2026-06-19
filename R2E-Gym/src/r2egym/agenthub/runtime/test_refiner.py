"""
Block-B test refinement (offline) for ATS-RL.

Three refinement methods, each an OFFLINE transform over a dataset row's
``FAIL_TO_PASS`` set. The ATS reward machinery is unchanged — we only change which
tests an instance carries before training (IMPLEMENTATION_PLAN.md §5).

  * R22  specificity_gate(...)        -- NO LLM. Keep only tests that kill mutants.
  * R12  backtranslate(...)           -- needs a teacher LLM (deferred until .env key).
  * R3   adversarial_strengthen(...)  -- needs a teacher LLM (deferred until .env key).

Validation is executable ground truth via the same DockerRuntime the env uses:
``DockerRuntime(ds=<extra_info dict>)`` exposes ``checkout`` / ``apply_patch`` /
``get_f2p_and_p2p_status``. A candidate/derived test is judged by running the instance's
test suite in its own Docker image — no LLM-judge.

This module is import-light at top level so the offline mutator can be unit-tested
without Docker (the DockerRuntime import is lazy, inside DockerTestValidator).
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("test_refiner")


# --------------------------------------------------------------------------------------
# Audit trail (never silently mutate — always record provenance)
# --------------------------------------------------------------------------------------
@dataclass
class RefineResult:
    instance_id: str
    refiner: str                       # "R22" | "R12" | "R3" | "control"
    orig_f2p: List[str] = field(default_factory=list)
    kept_f2p: List[str] = field(default_factory=list)
    dropped_f2p: List[str] = field(default_factory=list)
    added_f2p: List[str] = field(default_factory=list)
    kill_rate_per_test: Dict[str, float] = field(default_factory=dict)
    n_mutants: int = 0
    m_threshold: int = 0
    fallback_used: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def changed(self) -> bool:
        return bool(self.dropped_f2p or self.added_f2p)


# --------------------------------------------------------------------------------------
# Offline operator mutator (no Docker) -- the R22 synthetic mutant pool
# --------------------------------------------------------------------------------------
class OperatorMutator:
    """Generate "gold-with-one-bug" patches by mutating single tokens in the *added*
    (``+``) code lines of the gold diff. Each mutation changes characters *within* one
    line (line counts unchanged) so the resulting unified diff still applies cleanly.

    A discriminative test should FAIL on these mutants (it kills them).
    """

    # (compiled_regex, replacement) — each rule mutates ONE token occurrence.
    # Order matters only for determinism; we collect all candidate single-edits and the
    # caller samples k of them.
    _RULES: List[Tuple[str, str]] = [
        # comparison operators (word-free; match exact operator tokens)
        (r"==", "!="),
        (r"!=", "=="),
        (r"<=", ">"),
        (r">=", "<"),
        (r"(?<![<>=!])<(?![=])", ">="),
        (r"(?<![<>=!])>(?![=])", "<="),
        # arithmetic
        (r"(?<![\+\-=])\+(?![\+=])", "-"),
        (r"(?<![\+\-=])\-(?![\-=>])", "+"),
        (r"(?<![\*])\*(?![\*=])", "/"),
        # boolean / keywords (word-bounded)
        (r"\bTrue\b", "False"),
        (r"\bFalse\b", "True"),
        (r"\band\b", "or"),
        (r"\bor\b", "and"),
        (r"\bis not\b", "is"),
        (r"\bnot in\b", "in"),
        # numeric literal bump (standalone integer)
        (r"(?<![\w.])0(?![\w.])", "1"),
        (r"(?<![\w.])1(?![\w.])", "2"),
    ]

    def __init__(self, gold_patch: str, seed: int = 0):
        self.gold_patch = gold_patch
        self.rng = random.Random(seed)
        self._compiled = [(re.compile(p), r) for p, r in self._RULES]

    def _added_code_lines(self) -> List[int]:
        """Indices (into split lines) of `+` code lines (not the `+++` file header)."""
        out = []
        for i, ln in enumerate(self.gold_patch.splitlines()):
            if ln.startswith("+") and not ln.startswith("+++"):
                out.append(i)
        return out

    def _candidate_edits(self) -> List[Tuple[int, int, int, str]]:
        """All single-token edits: (line_idx, start, end, replacement)."""
        lines = self.gold_patch.splitlines()
        edits: List[Tuple[int, int, int, str]] = []
        for idx in self._added_code_lines():
            body = lines[idx][1:]  # drop leading '+'
            # skip comment-only / docstring-ish lines
            stripped = body.strip()
            if stripped.startswith("#") or not stripped:
                continue
            for rx, repl in self._compiled:
                for m in rx.finditer(body):
                    # +1 to account for the leading '+' we stripped
                    edits.append((idx, m.start() + 1, m.end() + 1, repl))
        return edits

    def generate(self, k: int) -> List[str]:
        """Return up to ``k`` distinct mutant patch strings."""
        edits = self._candidate_edits()
        self.rng.shuffle(edits)
        lines = self.gold_patch.splitlines(keepends=False)
        seen = set()
        mutants: List[str] = []
        for idx, start, end, repl in edits:
            orig_line = lines[idx]
            mutated_line = orig_line[:start] + repl + orig_line[end:]
            if mutated_line == orig_line:
                continue
            new_lines = list(lines)
            new_lines[idx] = mutated_line
            patch = "\n".join(new_lines)
            if self.gold_patch.endswith("\n"):
                patch += "\n"
            if patch in seen:
                continue
            seen.add(patch)
            mutants.append(patch)
            if len(mutants) >= k:
                break
        return mutants


# --------------------------------------------------------------------------------------
# Executable validation in the instance's Docker image
# --------------------------------------------------------------------------------------
class DockerTestValidator:
    """Thin wrapper over DockerRuntime to evaluate F2P test outcomes under arbitrary
    code patches. Construct with the instance ``extra_info`` dict (== the env's ``ds``).

    Lifecycle: one container per instance; call ``status_for_patch`` repeatedly (gold,
    None=pre-patch, each mutant), then ``close()``.
    """

    def __init__(self, entry: dict, timeout: int = 300, verbose: bool = False):
        self.entry = entry
        self.timeout = timeout
        self.verbose = verbose
        self.base_commit = entry.get("base_commit")
        self.gold_patch = entry.get("patch", "")
        self.f2p_ids = list(entry.get("FAIL_TO_PASS", []) or [])
        self.runtime = None  # lazy

    def _ensure_runtime(self):
        if self.runtime is not None:
            return
        from r2egym.agenthub.runtime.docker import DockerRuntime  # lazy
        lg = logging.getLogger("test_refiner.docker")
        if not self.verbose:
            lg.setLevel(logging.ERROR)
        self.runtime = DockerRuntime(
            ds=self.entry,
            command=["/bin/bash", "-l"],
            backend="docker",
            logger=lg,
        )

    def _reset_to_base(self):
        """Discard all working changes and untracked files; back to clean base code.
        (eval_script handles the test files on each run.)"""
        self.runtime.run("git checkout -f -- . 2>/dev/null || true")
        if self.base_commit:
            self.runtime.run(f"git checkout -f {self.base_commit} 2>/dev/null || true")
        # remove untracked files (test files dropped by eval_script, stray artifacts),
        # but NOT ignored build dirs/venv (no -x).
        self.runtime.run("git clean -fdq 2>/dev/null || true")

    def _apply_verified(self, code_patch: str) -> bool:
        """Apply ``code_patch`` and confirm it took effect via ``git diff`` (robust to
        cosmetic whitespace warnings that make git apply return non-zero). Falls back to
        a 3-way apply. Returns True iff the working tree changed."""
        self.runtime.apply_patch(code_patch)
        diff, _ = self.runtime.run("git diff --stat 2>/dev/null || true")
        if diff and diff.strip():
            return True
        # fallback: 3-way merge apply (handles fuzzy context)
        import uuid as _uuid
        p = f"/tmp/refiner_{self.entry.get('instance_id','x')}_{_uuid.uuid4().hex[:8]}.patch"
        self.runtime.run(f"cat > {p} <<'R2E_EOF_PATCH'\n{code_patch}\nR2E_EOF_PATCH")
        self.runtime.run(f"git apply --3way --whitespace=fix {p} 2>/dev/null || "
                         f"git apply --whitespace=fix {p} 2>/dev/null || true")
        diff, _ = self.runtime.run("git diff --stat 2>/dev/null || true")
        return bool(diff and diff.strip())

    def status_for_patch(self, code_patch: Optional[str]) -> Dict[str, bool]:
        """Reset to base, optionally apply ``code_patch``, run the F2P suite once.
        Returns ``{test_id: passed_bool}`` for the instance's FAIL_TO_PASS tests.
        On a genuine apply failure or error returns ``{}`` (caller treats empty as
        "unknown" and fail-safes)."""
        self._ensure_runtime()
        try:
            self._reset_to_base()
            if code_patch:
                if not self._apply_verified(code_patch):
                    logger.warning("patch did not apply for %s (no tree change)",
                                   self.entry.get("instance_id"))
                    return {}
            f2p_status, _p2p = self.runtime.get_f2p_and_p2p_status(timeout=self.timeout)
            return {k: bool(v) for k, v in (f2p_status or {}).items()}
        except Exception as e:  # noqa: BLE001
            logger.warning("status_for_patch error for %s: %s",
                           self.entry.get("instance_id"), e)
            return {}

    def close(self):
        if self.runtime is not None:
            try:
                self.runtime.stop_container()
            except Exception:  # noqa: BLE001
                try:
                    self.runtime.close()
                except Exception:  # noqa: BLE001
                    pass
            self.runtime = None


# --------------------------------------------------------------------------------------
# R22 — mutation-validated specificity gate (NO LLM)
# --------------------------------------------------------------------------------------
def specificity_gate(
    entry: dict,
    k: int = 6,
    m_frac: float = 0.5,
    seed: int = 0,
    timeout: int = 300,
    verbose: bool = False,
) -> RefineResult:
    """Keep a FAIL_TO_PASS test iff it PASSES gold AND fails on >= M of (k+1) mutants
    (mutant #0 = pre-patch state; the rest are operator mutants of the gold diff).

    Returns a RefineResult with the new (kept) F2P set + kill-rate audit. On Docker/setup
    error, returns the original tests unchanged with ``error`` set (never emits empty F2P).
    """
    iid = entry.get("instance_id", "unknown")
    orig = list(entry.get("FAIL_TO_PASS", []) or [])
    res = RefineResult(instance_id=iid, refiner="R22", orig_f2p=orig, n_mutants=k)

    if not orig:
        res.error = "no FAIL_TO_PASS"
        res.kept_f2p = orig
        return res
    if not entry.get("patch"):
        res.error = "no gold patch"
        res.kept_f2p = orig
        return res

    mutants = OperatorMutator(entry["patch"], seed=seed).generate(k)
    total_mutants = len(mutants) + 1  # +1 for pre-patch
    res.n_mutants = total_mutants
    res.m_threshold = 1   # principled gate: require >=1 operator-mutant kill (when any exist)

    v = DockerTestValidator(entry, timeout=timeout, verbose=verbose)
    try:
        gold_status = v.status_for_patch(entry["patch"])
        if not gold_status:
            res.error = "gold suite produced no status (build/apply failure)"
            res.kept_f2p = orig            # fail safe: keep originals
            return res
        base_status = v.status_for_patch(None)
        mutant_statuses = [v.status_for_patch(m) for m in mutants]
    finally:
        v.close()

    # Principled gate (gentler than a pure mutant-kill threshold): the ground-truth
    # discrimination signal is failing the PRE-PATCH state (the test catches the real
    # bug). Keep any test that passes gold AND fails pre-patch; operator-mutant kills are
    # a quality SCORE. Drop a test only if it: (a) fails on gold (mislabeled), (b) passes
    # pre-patch (doesn't catch the bug -> non-discriminative), or (c) catches the bug but
    # kills ZERO operator mutants when mutants exist (completely non-specific/loose).
    n_op = len(mutants)  # operator mutants (excludes pre-patch)
    kept, dropped = [], []
    for t in orig:
        passes_gold = gold_status.get(t, False)
        fails_prepatch = base_status.get(t, True) is False     # caught the real bug
        op_kills = sum(1 for st in mutant_statuses if st and st.get(t, True) is False)
        total_kills = (1 if fails_prepatch else 0) + op_kills
        res.kill_rate_per_test[t] = round(total_kills / total_mutants, 4)
        if not passes_gold:
            dropped.append(t); continue            # mislabeled (fails on gold-patched repo)
        if not fails_prepatch:
            dropped.append(t); continue            # passes pre-patch -> doesn't test the fix
        if n_op > 0 and op_kills == 0:
            dropped.append(t); continue            # bug-catcher but fully non-specific
        kept.append(t)

    if not kept:
        # never ship an empty F2P set: keep the single best discriminator (or originals)
        if res.kill_rate_per_test:
            best = max(res.kill_rate_per_test, key=res.kill_rate_per_test.get)
            kept = [best]
            dropped = [t for t in orig if t != best]
        else:
            kept = orig
            dropped = []
        res.fallback_used = True

    res.kept_f2p = kept
    res.dropped_f2p = dropped
    return res


# --------------------------------------------------------------------------------------
# R12 / R3 — need a teacher LLM. Deferred until an API key is added to .env.
# --------------------------------------------------------------------------------------
def backtranslate(entry: dict, llm: Optional[Callable] = None, **kw) -> RefineResult:
    """R12: derive minimal pytest tests from the gold diff via a teacher LLM, gate by
    execution (pass gold ∧ fail pre-patch), add survivors to FAIL_TO_PASS.

    Deferred: requires a teacher LLM. See BLOCK_B_TEST_REFINEMENT_DESIGN.md §6.
    """
    if llm is None:
        raise NotImplementedError(
            "R12 backtranslation needs a teacher LLM. Add an API key to .env and pass an "
            "llm client (see BLOCK_B_TEST_REFINEMENT_DESIGN.md §6)."
        )
    raise NotImplementedError("R12 generation loop not yet wired (teacher LLM pending).")


def adversarial_strengthen(entry: dict, wrong_patches: Optional[List[str]] = None,
                           llm: Optional[Callable] = None, K: int = 3, **kw) -> RefineResult:
    """R3 (InfCode): iteratively harden FAIL_TO_PASS against wrong patches via a teacher
    Test-Patch-Generator until it passes gold ∧ fails the wrong patches (K<=3 iters).

    Deferred: requires a teacher LLM. See BLOCK_B_TEST_REFINEMENT_DESIGN.md §6.
    """
    if llm is None:
        raise NotImplementedError(
            "R3 adversarial co-refinement needs a teacher LLM. Add an API key to .env and "
            "pass an llm client (see BLOCK_B_TEST_REFINEMENT_DESIGN.md §6)."
        )
    raise NotImplementedError("R3 generation loop not yet wired (teacher LLM pending).")
