"""Paper-faithful, Docker-free SWE-bench test reward.

The original SWE-Master reward (R2E-Gym `_calculate_reward_swebench`) is:

    apply model patch -> run /run_tests.sh in the per-instance docker image
    -> R = 1 iff every FAIL_TO_PASS test now passes AND every PASS_TO_PASS
            test still passes, else R = 0

We don't have the internal docker images, so we reproduce the SWE-bench
evaluation logic locally:

  1. The sandbox worktree already contains the model's patch (it edited files
     during the rollout).
  2. Reset ONLY the test files to ``base_commit`` (so the model can't cheat
     by editing tests) then apply the gold ``test_patch``.
  3. Run the FAIL_TO_PASS + PASS_TO_PASS tests with the spec's ``test_cmd``
     inside the per-instance conda env (built by 04_setup_test_envs.sh).
  4. Parse pytest ``-rA`` output; R = 1 iff resolution is FULL.

This is binary {0, 1}, exactly like the paper.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any

TEST_ENVS_ROOT = os.environ.get(
    "SWE_TEST_ENVS_ROOT",
    f"/scratch/{os.environ.get('USER','user')}/swemaster/test_envs",
)


def _as_list(v: Any) -> list[str]:
    """Parquet stores list-ish columns as numpy arrays / stringified lists."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return [str(x) for x in v.tolist()]
    except Exception:
        pass
    s = str(v).strip()
    if s.startswith("[") and s.endswith("]"):
        # e.g. "['a::b' 'c::d']" (numpy repr) or "['a::b', 'c::d']"
        inner = s[1:-1]
        parts = re.findall(r"'([^']+)'|\"([^\"]+)\"", inner)
        flat = [a or b for a, b in parts]
        if flat:
            return flat
    return [s] if s else []


def _test_files(test_ids: list[str]) -> list[str]:
    """``tests/test_x.py::test_fn`` -> ``tests/test_x.py``."""
    return sorted({t.split("::", 1)[0] for t in test_ids if "::" in t or t})


# pytest -rA prints a summary block:  "PASSED tests/x.py::test_y"
_PYTEST_LINE = re.compile(
    r"^(?P<status>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(?P<test>\S+)",
    re.MULTILINE,
)


def parse_pytest_log(output: str) -> dict[str, str]:
    """Map ``test_id -> STATUS`` from pytest ``-rA`` output."""
    status: dict[str, str] = {}
    for m in _PYTEST_LINE.finditer(output):
        status[m.group("test")] = m.group("status")
    return status


def env_python_for(instance_id: str) -> str | None:
    """Path to the per-instance conda env's python, or None if not built."""
    p = os.path.join(TEST_ENVS_ROOT, instance_id, "bin", "python")
    return p if os.path.isfile(p) else None


def compute_test_reward(
    sandbox,
    row: dict[str, Any],
    timeout: int = 600,
) -> tuple[float, str]:
    """Return ``(reward, detail)``. reward is 1.0 or 0.0.

    ``sandbox`` must already have the model's patch in the worktree (the
    rollout edited files in place). We DON'T reset it here.
    """
    instance_id = row["instance_id"]
    spec = row.get("make_test_spec")
    if isinstance(spec, str):
        spec = json.loads(spec)
    install_cfg = spec.get("install_config", {}) if spec else {}
    test_cmd = install_cfg.get(
        "test_cmd",
        "pytest --no-header -rA --tb=short --color=no -p no:cacheprovider",
    )

    f2p = _as_list(row.get("FAIL_TO_PASS"))
    p2p = _as_list(row.get("PASS_TO_PASS"))
    if not f2p:
        return 0.0, "no FAIL_TO_PASS tests in row"

    env_py = env_python_for(instance_id)
    if env_py is None:
        return 0.0, (f"test env not built for {instance_id}; run "
                     f"TRL_RL/scripts/04_setup_test_envs.sh")

    cwd = sandbox.cwd
    base = sandbox.base_commit
    test_patch = row.get("test_patch") or ""

    # 1. Reset ONLY the test files to base, keep model's source edits.
    tfiles = _test_files(f2p + p2p)
    if tfiles:
        subprocess.run(["git", "checkout", base, "--", *tfiles],
                        cwd=cwd, capture_output=True, text=True, timeout=60)

    # 2. Apply the gold test patch (adds/updates the discriminating tests).
    if test_patch.strip():
        ap = subprocess.run(["git", "apply", "--whitespace=nowarn"],
                             input=test_patch, cwd=cwd,
                             capture_output=True, text=True, timeout=60)
        if ap.returncode != 0:
            # try 3way then reject — same forgiveness as the sandbox
            for extra in (["--3way"], ["--reject"]):
                ap = subprocess.run(["git", "apply", "--whitespace=nowarn", *extra],
                                     input=test_patch, cwd=cwd,
                                     capture_output=True, text=True, timeout=60)
                if ap.returncode == 0:
                    break
            if ap.returncode != 0:
                return 0.0, f"failed to apply gold test_patch: {ap.stderr[:300]}"

    # 3. Run the targeted tests with the spec's command.
    cmd = f"{env_py} -m {test_cmd} " + " ".join(f2p + p2p)
    try:
        proc = subprocess.run(
            cmd, shell=True, cwd=cwd,
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 0.0, f"test run timed out after {timeout}s"
    log = proc.stdout + "\n" + proc.stderr

    # 4. Parse + resolution check (SWE-bench FULL criterion).
    status = parse_pytest_log(log)
    f2p_ok = all(status.get(t) == "PASSED" for t in f2p)
    p2p_ok = all(status.get(t) == "PASSED" for t in p2p) if p2p else True
    resolved = f2p_ok and p2p_ok
    detail = (f"f2p_ok={f2p_ok} p2p_ok={p2p_ok} "
              f"statuses={{ {', '.join(f'{k}:{v}' for k,v in list(status.items())[:8])} }}")
    return (1.0 if resolved else 0.0), detail
