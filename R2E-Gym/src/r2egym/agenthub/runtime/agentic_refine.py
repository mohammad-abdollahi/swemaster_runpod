"""Agentic test refinement via `claude -p` (branch test-refinement-agentic).

INTEGRATED into the ATS-RL loop, like the OpenAI online-refine pipeline, but the refiner is the
**Claude Code agent run headless (`claude -p`)** on the user's Max subscription (no API charge),
NOT a single chat call.  On stall, the scheduler's refine hook calls `refine_test_agentic`, which:

  1. selects DIVERSE prior trajectories (regex-normalized `SequenceMatcher`; keep one only if it is
     >30% different from those already chosen; cap K),
  2. builds a VERBATIM prompt (issue + original FAIL_TO_PASS test + the diverse trajectories;
     **no gold patch**, **no length caps**),
  3. runs `claude -p` in a clean `base_commit` checkout so the agent can read the code and run the
     failing test itself, and write the refined test(s) to `refined_test.py`,
  4. validates TRUE F2P in clean containers — fails on the buggy code AND passes on the gold-patched
     code (the gold patch is applied by US, internally; the agent never sees it).  Requiring
     pass-on-fixed automatically rejects dead tests (a NameError/import-error test fails on both),
  5. caches per `instance_id::test_id`; returns full node ids `[(path::name, code)]` for `_replace_test`.

Reuses test_refiner: extract_test_source, split_test_functions, _strip_code_fence,
run_adhoc_tests, _container_write_file.  Reuses online_refine.OnlineRefineCache for the JSONL cache.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from typing import Dict, List, Optional, Tuple

from r2egym.agenthub.runtime.test_refiner import (
    extract_test_source,
    split_test_functions,
    _strip_code_fence,
    run_adhoc_tests,
    _container_write_file,
)
from r2egym.agenthub.runtime.online_refine import OnlineRefineCache, OnlineRefinement

# --------------------------------------------------------------------------- #
# Config (env-overridable)
# --------------------------------------------------------------------------- #
def _default_claude_bin() -> str:
    cand = os.environ.get("RLLM_AGENTIC_CLAUDE_BIN")
    if cand and os.path.exists(cand):
        return cand
    # the native binary that ships with the VS Code extension (present on this host)
    import glob
    hits = sorted(glob.glob(
        "/home/temp/.vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude"))
    if hits:
        return hits[-1]
    return shutil.which("claude") or "claude"

_CLAUDE_BIN = _default_claude_bin()
_MODEL = os.environ.get("RLLM_AGENTIC_REFINE_MODEL", "claude-sonnet-4-6")
_MAX_TURNS = int(os.environ.get("RLLM_AGENTIC_MAX_TURNS", "12"))
_MAX_TRAJ = int(os.environ.get("RLLM_AGENTIC_MAX_TRAJ", "3"))
_MISMATCH = float(os.environ.get("RLLM_AGENTIC_TRAJ_DIVERSITY", "0.30"))  # >30% different => include
_TIMEOUT = int(os.environ.get("RLLM_AGENTIC_REFINE_TIMEOUT", "900"))
_MAX_REFINED = int(os.environ.get("RLLM_AGENTIC_REFINE_MAX", "4"))

_SYSTEM = (
    "You are a senior Python test engineer helping a reinforcement-learning agent that is STUCK on a "
    "failing pytest test during training. You get the issue, the gold fix (ground truth of the "
    "intended behavior), the original hard test, and the agent's trajectories so far. REFINE the test "
    "so it is more learnable: either REWRITE it as one cleaner test or SPLIT it into a few smaller "
    "minimal tests - each asserting ONE facet, descriptive name, standard pytest only "
    "(no network/sleep/randomness/external services). CRITICAL: the refined test(s) must FAIL on the "
    "buggy/unfixed code and PASS only once the correct fix (the gold patch) is applied - never write a "
    "test that passes trivially. You may read the repository and RUN your candidate with `./run_test.sh` "
    "(it runs the test in the buggy checkout; it MUST currently fail). "
    "When done, WRITE the final refined test(s) to a file named `refined_test.py` in the current "
    "directory - imports at top, one or more `def test_...():` functions, nothing else."
)


# --------------------------------------------------------------------------- #
# 1. Trajectory diversity preprocessor (regex-normalized similarity)
# --------------------------------------------------------------------------- #
_VOLATILE = [
    (re.compile(r"0x[0-9a-fA-F]+"), "0xADDR"),
    (re.compile(r"/tmp/\S+"), "/tmp/X"),
    (re.compile(r"/testbed/\S+?:\d+"), "FILE:N"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\S*"), "TS"),
    (re.compile(r"\bline \d+"), "line N"),
    (re.compile(r"in \d+\.\d+s"), "in Ns"),
    (re.compile(r"\b[0-9a-f]{12,}\b"), "HASH"),
    (re.compile(r"\s+"), " "),
]


def _normalize_traj(steps: List[dict]) -> str:
    parts = []
    for s in steps or []:
        fn = (s.get("fn") or "?")
        act = (s.get("action") or "")
        parts.append(f"{fn}|{act}")
    text = "\n".join(parts)
    for rx, repl in _VOLATILE:
        text = rx.sub(repl, text)
    return text.strip()


def _similar(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def select_diverse_trajectories(stalled: List[dict], pool: List[List[dict]],
                                k: int = _MAX_TRAJ, mismatch: float = _MISMATCH) -> List[List[dict]]:
    """Greedy: seed with the stalled trajectory, then add any trajectory whose max similarity to the
    already-selected set is < (1 - mismatch) (i.e. >mismatch different). Cap at k."""
    sim_thr = 1.0 - mismatch
    selected: List[List[dict]] = []
    norms: List[str] = []
    if stalled:
        selected.append(stalled)
        norms.append(_normalize_traj(stalled))
    for traj in pool:
        if len(selected) >= k:
            break
        if traj is stalled:
            continue
        n = _normalize_traj(traj)
        if not selected:
            selected.append(traj); norms.append(n); continue
        if max(_similar(n, m) for m in norms) < sim_thr:
            selected.append(traj); norms.append(n)
    return selected[:k]


# --------------------------------------------------------------------------- #
# 2. Prompt builder (verbatim, uncapped)
# --------------------------------------------------------------------------- #
def _render_trajectory(steps: List[dict]) -> str:
    lines = []
    for s in steps or []:
        fn = s.get("fn") or "?"
        act = (s.get("action") or "").strip()
        obs = (s.get("obs") or "").strip()
        lines.append(f"[step {s.get('step')}] ({fn}) {act} -> {obs}")
    return "\n".join(lines)


def build_prompt(entry: dict, test_id: str, test_src: str, trajs: List[List[dict]]) -> str:
    issue = entry.get("problem_statement", "") or ""
    gold = entry.get("patch", "") or ""
    blocks = [
        _SYSTEM,
        "\n## Issue\n" + issue,
        "\n## Gold patch (the fix that makes the intended behavior correct - ground truth)\n" + gold,
        f"\n## Original (hard) FAIL_TO_PASS test `{test_id}`\n" + (test_src or "(source unavailable)"),
    ]
    for i, t in enumerate(trajs, 1):
        blocks.append(f"\n## Agent trajectory #{i} (it has NOT made this test pass)\n" + _render_trajectory(t))
    return "\n".join(blocks)


# --------------------------------------------------------------------------- #
# 3. claude -p runner (headless, Max subscription)
# --------------------------------------------------------------------------- #
def _clean_env() -> dict:
    """Strip CLAUDE*/CLAUDECODE/ANTHROPIC* so the subprocess runs as a fresh headless agent on the
    subscription login (a stray ANTHROPIC_API_KEY would silently switch it to API billing)."""
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("CLAUDE") or k == "CLAUDECODE" or k.startswith("ANTHROPIC"))}
    env.setdefault("HOME", os.path.expanduser("~"))
    return env


def run_claude_p(prompt: str, cwd: str, model: str = _MODEL, max_turns: int = _MAX_TURNS,
                 timeout: int = _TIMEOUT) -> str:
    """Run `claude -p` in `cwd`; return its final result text ("" on any failure / usage-limit /
    missing CLI). Never raises into the rollout."""
    if not (_CLAUDE_BIN and (os.path.exists(_CLAUDE_BIN) or shutil.which(_CLAUDE_BIN))):
        print(f"!!!!!!! agentic_refine: claude binary not found ({_CLAUDE_BIN}); skipping")
        return ""
    cmd = [_CLAUDE_BIN, "-p", prompt,
           "--output-format", "json",
           "--model", model,
           "--max-turns", str(max_turns),
           "--permission-mode", "acceptEdits",
           "--allowedTools", "Bash,Read,Edit,Write,Grep,Glob"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           cwd=cwd, env=_clean_env())
    except subprocess.TimeoutExpired:
        print(f"!!!!!!! agentic_refine: claude -p timed out after {timeout}s")
        return ""
    except Exception as e:
        print(f"!!!!!!! agentic_refine: claude -p failed: {e!r}")
        return ""
    out = r.stdout or ""
    try:
        obj = json.loads(out)
        if obj.get("is_error"):
            print(f"!!!!!!! agentic_refine: claude -p is_error ({obj.get('subtype')})")
            return ""
        return obj.get("result", "") or ""
    except Exception:
        return out  # non-JSON fallback


# --------------------------------------------------------------------------- #
# 4. Clean-container F2P validation (gold patch applied internally)
# --------------------------------------------------------------------------- #
class _DockerRT:
    """Minimal rt shim so we can reuse run_adhoc_tests / _container_write_file against a fresh
    container created from the instance image."""
    def __init__(self, container):
        self._c = container

    def run(self, cmd: str, timeout: int = 600):
        res = self._c.exec_run(cmd=["bash", "-lc", cmd], demux=False)
        out = res.output.decode("utf-8", "replace") if res.output else ""
        return out, res.exit_code

    def copy_to_container(self, local_path: str, container_path: str):
        import io, tarfile
        data = open(local_path, "rb").read()
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            ti = tarfile.TarInfo(name=os.path.basename(container_path))
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
        buf.seek(0)
        self._c.put_archive(os.path.dirname(container_path) or "/", buf.getvalue())


def _start_container(image: str):
    import docker
    cli = docker.from_env()
    return cli.containers.run(image, command="sleep infinity", detach=True, auto_remove=False,
                              network_mode="none")


def _apply_gold_patch(rt: "_DockerRT", patch: str) -> bool:
    if not patch:
        return False
    _container_write_file(rt, "/tmp/gold.patch", patch)
    for cmd in ("cd /testbed && git apply -p1 /tmp/gold.patch",
                "cd /testbed && git apply --3way /tmp/gold.patch",
                "cd /testbed && patch -p1 < /tmp/gold.patch"):
        out, rc = rt.run(cmd, timeout=120)
        if rc == 0:
            return True
    return False


def validate_f2p(entry: dict, pairs: List[Tuple[str, str]], full_code: str, test_file: str,
                 timeout: int = 300) -> List[Tuple[str, str]]:
    """Keep only functions that FAIL on the buggy image AND PASS on the gold-patched image.
    Requiring pass-on-fixed automatically rejects dead (import/NameError) tests."""
    image = entry.get("docker_image") or entry.get("image_name")
    patch = entry.get("patch", "") or ""
    if not image:
        print("!!!!!!! agentic_refine: no docker_image; cannot validate F2P")
        return []
    names = [n for n, _ in pairs]
    buggy = fixed = None
    try:
        buggy = _start_container(image)
        rt_b = _DockerRT(buggy)
        _container_write_file(rt_b, f"/testbed/{test_file}", full_code)
        v_buggy, _ = run_adhoc_tests(rt_b, test_file, names, timeout)

        fixed = _start_container(image)
        rt_f = _DockerRT(fixed)
        applied = _apply_gold_patch(rt_f, patch)
        if not applied:
            print("!!!!!!! agentic_refine: gold patch did not apply; cannot confirm pass-on-fixed")
            return []
        _container_write_file(rt_f, f"/testbed/{test_file}", full_code)
        v_fixed, _ = run_adhoc_tests(rt_f, test_file, names, timeout)
        print(f"!!!!!!! agentic_refine F2P: patch_applied={applied} buggy={v_buggy} fixed={v_fixed}")
    finally:
        for c in (buggy, fixed):
            try:
                if c: c.remove(force=True)
            except Exception:
                pass
    return [(n, c) for (n, c) in pairs if v_buggy.get(n) is False and v_fixed.get(n) is True]


# --------------------------------------------------------------------------- #
# 5. Buggy host checkout for the agent to explore + a run_test.sh bridge
# --------------------------------------------------------------------------- #
def _make_buggy_checkout(image: str) -> Optional[Tuple[str, "object"]]:
    """Start a buggy container, docker-cp /testbed to a host temp dir, drop a run_test.sh that runs a
    candidate inside the live container. Returns (host_dir, container) or None."""
    try:
        c = _start_container(image)
    except Exception as e:
        print(f"!!!!!!! agentic_refine: could not start buggy container: {e!r}")
        return None
    host_dir = tempfile.mkdtemp(prefix="agentic_refine_")
    try:
        # docker cp <cid>:/testbed -> host_dir/testbed
        subprocess.run(["bash", "-lc",
                        f"sg docker -c 'docker cp {c.id}:/testbed {host_dir}/testbed'"],
                       check=True, capture_output=True, timeout=600)
    except Exception as e:
        print(f"!!!!!!! agentic_refine: docker cp /testbed failed: {e!r}")
        try: c.remove(force=True)
        except Exception: pass
        shutil.rmtree(host_dir, ignore_errors=True)
        return None
    work = os.path.join(host_dir, "testbed")
    bridge = (
        "#!/bin/bash\n"
        "# Run the candidate refined test inside the buggy checkout. Usage: ./run_test.sh [file]\n"
        "F=\"${1:-refined_test.py}\"\n"
        f"sg docker -c \"docker cp $F {c.id}:/testbed/r2e_cand.py\"\n"
        f"sg docker -c \"docker exec {c.id} bash -lc 'source /opt/miniconda3/bin/activate 2>/dev/null; "
        "conda activate testbed 2>/dev/null; cd /testbed && python -m pytest -rA r2e_cand.py 2>&1'\"\n"
    )
    bp = os.path.join(work, "run_test.sh")
    with open(bp, "w") as fh:
        fh.write(bridge)
    os.chmod(bp, 0o755)
    return work, c


# --------------------------------------------------------------------------- #
# 6. Top-level entry
# --------------------------------------------------------------------------- #
def refine_test_agentic(entry: dict, test_id: str, trajectories: List[List[dict]],
                        cache: OnlineRefineCache, model: str = _MODEL,
                        timeout: int = _TIMEOUT) -> List[Tuple[str, str]]:
    """Agentically refine `test_id`. Returns [(node_id, code)] of F2P-validated refined tests, or []."""
    iid = entry.get("instance_id", "")
    cached = cache.get(iid, test_id)
    if cached is not None:
        return list(cached.refined)

    o = OnlineRefinement(instance_id=iid, test_id=test_id, model=f"claude-cli:{model}")
    test_src = extract_test_source(entry.get("test_patch", "") or "", test_id) \
        or "(source unavailable - derive from the repo)"
    trajs = select_diverse_trajectories(
        trajectories[0] if trajectories else [], trajectories or [])
    prompt = build_prompt(entry, test_id, test_src, trajs)

    image = entry.get("docker_image") or entry.get("image_name")
    chk = _make_buggy_checkout(image) if image else None
    if chk is None:
        o.error = "no buggy checkout"; cache.put(o); return []
    work, container = chk
    try:
        run_claude_p(prompt, cwd=work, model=model, timeout=timeout)
        refined_path = os.path.join(work, "refined_test.py")
        if not os.path.exists(refined_path):
            o.error = "agent wrote no refined_test.py"; cache.put(o); return []
        full_code = open(refined_path).read()
        if os.environ.get("RLLM_AGENTIC_DEBUG"):
            shutil.copy(refined_path, "/tmp/agentic_last_refined.py")
            print(f"!!!!!!! agentic_refine: refined_test.py ({len(full_code)} chars) -> /tmp/agentic_last_refined.py")
    finally:
        try: container.remove(force=True)
        except Exception: pass
        shutil.rmtree(os.path.dirname(work), ignore_errors=True)

    pairs = split_test_functions(_strip_code_fence(full_code), _MAX_REFINED)
    if not pairs:
        o.error = "no test functions parsed"; cache.put(o); return []
    # Write claude's file VERBATIM (keeps module-level imports + helper defs the tests rely on);
    # split_test_functions only enumerates the test-node NAMES. _run_refined_nodes runs the file
    # by path, so every node carries the full file as its code.
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{iid}_{test_id.split('::')[-1]}")[:80]
    test_file = f"r2e_tests/test_agentic_{safe}.py"

    survivors = validate_f2p(entry, pairs, full_code, test_file, timeout=300)
    o.full_code = full_code
    o.test_file = test_file
    o.refined = [(f"{test_file}::{n}", full_code) for (n, _c) in survivors]
    if not o.refined and not o.error:
        o.error = "no refined func is true-F2P (fail-on-buggy AND pass-on-fixed)"
    cache.put(o)
    return list(o.refined)
