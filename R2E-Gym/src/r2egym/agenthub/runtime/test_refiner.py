"""Test refinement for ATS-RL — B1/R12 online backtranslation.

This module *generates* and *caches* decompositions of hard `FAIL_TO_PASS` tests.
It is **off the reward path**: the `ActiveTestScheduler` only renders a cached
decomposition into the prompt as guidance for a stalled test (see
`active_tests.py` `refine_*`). Nothing here ever scores a test.

Pieces
------
- `TestDecomposer`  : OpenAI `gpt-5.4-mini` (key from `.env::API_KEY_OPENAI`).
    `decompose(issue, gold_diff, original_test_src) -> Decomposition`.
- `DecompositionCache`: persistent JSONL keyed by `instance_id::test_id`.
- `extract_test_source`: pull a single test's source out of the gold `test_patch`.

Design doc: `R12_BACKTRANSLATION_DESIGN.md`.
"""

from __future__ import annotations

import ast
import json
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# .env helper (no python-dotenv dependency)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[5]  # .../swemaster_runpod


def load_env_value(key: str, env_path: Optional[str] = None) -> Optional[str]:
    """Read a single `KEY=value` from `.env` (process env wins if already set)."""
    if os.environ.get(key):
        return os.environ[key]
    path = Path(env_path) if env_path else (_REPO_ROOT / ".env")
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Decomposition:
    """A backtranslated decomposition of one hard test."""

    instance_id: str
    test_id: str
    full_code: str                       # complete runnable test file (imports + subtests)
    subtests: List[Tuple[str, str]]      # [(func_name, func_source)] for prompt rendering
    model: str = ""
    n_subtests: int = 0
    error: str = ""                      # non-empty if generation/parse failed

    def to_record(self) -> dict:
        d = asdict(self)
        d["subtests"] = [list(s) for s in self.subtests]  # tuples -> lists for JSON
        return d

    @staticmethod
    def from_record(d: dict) -> "Decomposition":
        d = dict(d)
        d["subtests"] = [tuple(s) for s in d.get("subtests", [])]
        return Decomposition(**d)


# ---------------------------------------------------------------------------
# Helpers: extract a test's source, split a file into test functions
# ---------------------------------------------------------------------------
def _extract_func_block(lines: List[str], func: str) -> str:
    """Slice `def func(...)`'s block out of `lines` by indentation (handles methods).

    Returns the dedented block, or "" if not found.
    """
    import textwrap

    pat = re.compile(r"^(\s*)(?:async\s+)?def\s+" + re.escape(func) + r"\b")
    for i, ln in enumerate(lines):
        m = pat.match(ln)
        if not m:
            continue
        indent = len(m.group(1))
        # include a preceding decorator line if present
        start = i
        while start - 1 >= 0 and lines[start - 1].lstrip().startswith("@"):
            start -= 1
        block = list(lines[start:i + 1])
        for ln2 in lines[i + 1:]:
            if ln2.strip() == "":
                block.append(ln2)
                continue
            cur_indent = len(ln2) - len(ln2.lstrip())
            if cur_indent <= indent:
                break
            block.append(ln2)
        return textwrap.dedent("\n".join(block)).strip()
    return ""


def extract_test_source(test_patch: str, test_id: str) -> str:
    """Best-effort: pull the source of `test_id` out of the gold `test_patch`.

    `test_id` looks like `path/test_file.py::Class::test_method` or
    `path::test_method`. We reconstruct each touched file's *post-image* (context
    + added lines, dropping removed lines) and slice the target function block by
    indentation — robust to methods inside classes. Returns "" on failure (the
    decomposer still works from issue+diff, just with less context).
    """
    if not test_patch or "::" not in test_id:
        return ""
    # Strip a pytest parametrization suffix: `test_x[param]` -> `test_x`.
    func = test_id.split("::")[-1].split("[")[0]
    # Post-image per file: keep context (' ') and added ('+') lines; drop '-'.
    files: Dict[str, List[str]] = {}
    cur = None
    for line in test_patch.splitlines():
        if line.startswith("+++ b/"):
            cur = line[len("+++ b/"):].strip()
            files[cur] = []
        elif line.startswith(("+++", "---", "diff ", "index ", "@@")):
            continue
        elif cur is not None:
            if line.startswith("+"):
                files[cur].append(line[1:])
            elif line.startswith("-"):
                continue
            else:  # context line (leading space, or empty)
                files[cur].append(line[1:] if line.startswith(" ") else line)
    for _fname, post in files.items():
        block = _extract_func_block(post, func)
        if block:
            return block
    return ""


def split_test_functions(code: str, max_subtests: int) -> List[Tuple[str, str]]:
    """Parse a python file and return [(name, source)] for top-level + method
    test functions whose name starts with `test`."""
    out: List[Tuple[str, str]] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            seg = ast.get_source_segment(code, node)
            if seg:
                out.append((node.name, seg))
            if len(out) >= max_subtests:
                break
    return out


def _strip_code_fence(text: str) -> str:
    """Extract the first fenced ```python block, or return the text as-is."""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


# ---------------------------------------------------------------------------
# Decomposer (OpenAI gpt-5.4-mini)
# ---------------------------------------------------------------------------
_SYSTEM = (
    "You are a senior Python test engineer. You decompose one large, hard pytest "
    "test into a few SMALLER, minimal, self-contained pytest tests. Each smaller "
    "test must assert exactly ONE facet of the behavior the original test checks, "
    "have a descriptive name, and use only standard pytest (no network, no sleep, "
    "no randomness, no external services). Together the smaller tests should cover "
    "the original test's behavior. Output ONLY a single python code block."
)

_USER_TMPL = """A reinforcement-learning agent is stuck trying to make this test pass.
Break it into at most {k} smaller, more understandable pytest tests.

## Issue
{issue}

## Gold patch (the fix that makes the original test pass — ground truth of the changed behavior)
{gold_diff}

## Original (hard) test{which}
{test_src}

Return ONE ```python code block containing the imports plus the {k}-or-fewer smaller
`test_*` functions. No prose."""


@dataclass
class TestDecomposer:
    model: str = "gpt-5.4-mini"
    api_key_var: str = "API_KEY_OPENAI"
    base_url: Optional[str] = None
    max_subtests: int = 4
    max_issue_chars: int = 4000
    max_diff_chars: int = 8000
    max_completion_tokens: int = 2048
    _client: object = field(default=None, repr=False)

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            key = load_env_value(self.api_key_var)
            if not key:
                raise RuntimeError(f"{self.api_key_var} not found in env/.env")
            kwargs = {"api_key": key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _complete(self, system: str, user: str) -> str:
        client = self._ensure_client()
        # Use chat.completions (stable surface). New models reject custom
        # temperature and want `max_completion_tokens`; fall back if needed.
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        try:
            resp = client.chat.completions.create(
                model=self.model, messages=msgs,
                max_completion_tokens=self.max_completion_tokens,
            )
        except TypeError:
            resp = client.chat.completions.create(
                model=self.model, messages=msgs,
                max_tokens=self.max_completion_tokens,
            )
        return resp.choices[0].message.content or ""

    def decompose(
        self,
        instance_id: str,
        test_id: str,
        issue: str,
        gold_diff: str,
        original_test_src: str = "",
    ) -> Decomposition:
        which = f" (`{test_id}`)" if test_id else ""
        user = _USER_TMPL.format(
            k=self.max_subtests,
            issue=(issue or "")[: self.max_issue_chars],
            gold_diff=(gold_diff or "")[: self.max_diff_chars],
            which=which,
            test_src=(original_test_src or "(source unavailable — derive from the patch)"),
        )
        try:
            raw = self._complete(_SYSTEM, user)
        except Exception as e:  # network / auth / model errors
            return Decomposition(instance_id, test_id, "", [], self.model, 0, error=repr(e))
        code = _strip_code_fence(raw).strip()
        subs = split_test_functions(code, self.max_subtests)
        if not subs:
            return Decomposition(instance_id, test_id, code, [], self.model, 0,
                                 error="no test_* functions parsed")
        return Decomposition(instance_id, test_id, code, subs, self.model, len(subs))


# ---------------------------------------------------------------------------
# Persistent cache
# ---------------------------------------------------------------------------
class DecompositionCache:
    """Append-only JSONL cache keyed by `instance_id::test_id`.

    The same hard tests stall across rollouts/epochs, so a decomposition is
    computed (LLM + offline gate) once and reused everywhere. Thread-safe for
    the in-process dict; file append is mutex-guarded.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._mem: Dict[str, Decomposition] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    d = Decomposition.from_record(rec)
                    self._mem[f"{d.instance_id}::{d.test_id}"] = d
                except Exception:
                    continue

    @staticmethod
    def key(instance_id: str, test_id: str) -> str:
        return f"{instance_id}::{test_id}"

    def get(self, instance_id: str, test_id: str) -> Optional[Decomposition]:
        return self._mem.get(self.key(instance_id, test_id))

    def put(self, d: Decomposition) -> None:
        with self._lock:
            self._mem[self.key(d.instance_id, d.test_id)] = d
            with self.path.open("a") as fh:
                fh.write(json.dumps(d.to_record()) + "\n")

    def __len__(self) -> int:
        return len(self._mem)


# ---------------------------------------------------------------------------
# Offline executable gate (Docker) — validates a decomposition's correctness.
# NOT on the reward path: this runs once during cache-build so the guidance we
# later show in the prompt is verified to discriminate (fails pre / passes gold).
# ---------------------------------------------------------------------------
import base64 as _base64
import os as _os
import tempfile as _tempfile

_PYTEST = (
    "python -m pytest --no-header -rA --tb=no -p no:cacheprovider "
    "-W ignore::DeprecationWarning"
)


def _container_write_file(rt, container_path: str, content: str) -> None:
    """Write `content` to `container_path` inside the running container."""
    tmp = _tempfile.mktemp(suffix=".py")
    with open(tmp, "w") as fh:
        fh.write(content)
    try:
        d = _os.path.dirname(container_path)
        if d:
            rt.run(f"mkdir -p {d}")
        rt.copy_to_container(tmp, container_path)
    finally:
        try:
            _os.remove(tmp)
        except OSError:
            pass


def run_adhoc_tests(rt, test_file: str, func_names: List[str], timeout: int = 300
                    ) -> Tuple[Dict[str, bool], str]:
    """Run `test_file` (a path under /testbed) in the container's conda env and
    return ({func_name: passed}, raw_output). Unseen funcs default to False.
    """
    runner = (
        "source /opt/miniconda3/bin/activate 2>/dev/null\n"
        "conda activate testbed 2>/dev/null\n"
        "cd /testbed\n"
        f"{_PYTEST} {test_file} 2>&1\n"
    )
    _container_write_file(rt, "/tmp/r12_runner.sh", runner)
    out, _ = rt.run("bash /tmp/r12_runner.sh", timeout=timeout)
    out = re.sub(r"\x1b\[[0-9;]*m|\r", "", out or "")
    verdict: Dict[str, bool] = {}
    for m in re.finditer(r"^(PASSED|FAILED|ERROR)\s+(\S+)", out, re.MULTILINE):
        status, node = m.group(1), m.group(2)
        fn = node.split("::")[-1].split("[")[0]
        # first definitive PASSED wins; otherwise keep failing
        verdict[fn] = verdict.get(fn, False) or (status == "PASSED")
    for fn in func_names:
        verdict.setdefault(fn, False)
    return verdict, out


def validate_decomposition(entry: dict, dec: Decomposition, timeout: int = 300
                           ) -> Tuple[List[Tuple[str, str]], dict]:
    """Two-run executable gate in the instance image: keep sub-tests that FAIL on
    the pre-patch repo AND PASS on the gold-patched repo. Returns (survivors,
    verdicts). Requires the instance Docker image to be available locally.
    """
    import logging

    from r2egym.agenthub.runtime.docker import DockerRuntime

    instid = dec.instance_id
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", instid)
    test_file = f"r2e_tests/test_r12_{safe}.py"
    abs_file = f"/testbed/{test_file}"
    names = [n for n, _ in dec.subtests]

    # NB: pass an explicit logger — DockerRuntime.__init__ references self.backend
    # before assigning it in the logger==None branch.
    rt = DockerRuntime(ds=dict(entry), command=["/bin/bash", "-l"], backend="docker",
                       logger=logging.getLogger("r12_gate"))
    try:
        # pre-patch (container starts at base_commit)
        _container_write_file(rt, abs_file, dec.full_code)
        pre, _ = run_adhoc_tests(rt, test_file, names, timeout)
        # gold-patched (apply the code fix; sub-tests should now pass)
        rt.apply_patch(entry.get("patch", "") or "")
        _container_write_file(rt, abs_file, dec.full_code)  # ensure still present
        gold, _ = run_adhoc_tests(rt, test_file, names, timeout)
    finally:
        try:
            rt.close()
        except Exception:
            pass

    survivors = [(n, c) for (n, c) in dec.subtests
                 if pre.get(n) is False and gold.get(n) is True]
    verdicts = {n: {"pre": pre.get(n), "gold": gold.get(n)} for n in names}
    return survivors, verdicts


def build_decomposition(entry: dict, test_id: str, decomposer: "TestDecomposer",
                        cache: "DecompositionCache", gate: bool = True,
                        timeout: int = 300) -> Decomposition:
    """Cache-aware: return a cached decomposition or compute (LLM [+ gate]) and
    cache it. Only gate-surviving sub-tests are stored when `gate=True`.
    """
    instid = entry["instance_id"]
    cached = cache.get(instid, test_id)
    if cached is not None:
        return cached
    src = extract_test_source(entry.get("test_patch", "") or "", test_id)
    dec = decomposer.decompose(
        instid, test_id, entry.get("problem_statement", "") or "",
        entry.get("patch", "") or "", src,
    )
    if dec.subtests and gate:
        survivors, verdicts = validate_decomposition(entry, dec, timeout)
        dec.subtests = survivors
        dec.n_subtests = len(survivors)
        if not survivors:
            dec.error = (dec.error + "; " if dec.error else "") + "no sub-tests survived gate"
    cache.put(dec)
    return dec
