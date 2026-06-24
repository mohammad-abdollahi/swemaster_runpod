"""Online test refinement for ATS-RL (branch test-refinement-test-to-openai).

INTEGRATED into the ATS-RL loop (not a separate phase): when a FAIL_TO_PASS test
stalls during a rollout, the scheduler's refine hook calls `refine_test_online`,
which sends the test + the agent's trajectory-so-far to OpenAI (gpt-5.4) and asks it
to rewrite or split the test into better test(s). The refined test(s) then REPLACE the
original in the scheduler's scored set and drive the reward.

Reward-hacking safeguard (runs in the agent's EXISTING container — no extra container):
a refined test is admitted only if it (a) is collectible/runnable AND (b) CURRENTLY
FAILS in the agent's container (the agent could not pass the original; a refined target
that already passes the agent's incomplete code is trivial → rejected). Pass-on-the-fix
is steered by the prompt. Results are cached per (instance_id::test_id) in-process + JSONL.

Reuses test_refiner.py: extract_test_source, run_adhoc_tests, _container_write_file,
split_test_functions, _strip_code_fence, load_env_value.
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from r2egym.agenthub.runtime.test_refiner import (
    load_env_value,
    extract_test_source,
    run_adhoc_tests,
    _container_write_file,
    split_test_functions,
    _strip_code_fence,
)

_MODEL = os.environ.get("RLLM_ONLINE_REFINE_MODEL", "gpt-5.4")
_MAX_REFINED = int(os.environ.get("RLLM_ONLINE_REFINE_MAX", "4"))

_SYSTEM = (
    "You are a senior Python test engineer helping a reinforcement-learning agent that "
    "is STUCK on a failing pytest test during training. You get the issue, the gold fix "
    "(ground truth of the intended behavior), the original hard test, and the agent's "
    "trajectory so far. REFINE the test so it is more learnable: either REWRITE it as one "
    "cleaner test or SPLIT it into a few smaller minimal tests — each asserting ONE facet, "
    "descriptive name, standard pytest only (no network/sleep/randomness/external services). "
    "CRITICAL: the refined test(s) must FAIL on the buggy/unfixed code and PASS only once "
    "the correct fix (the gold patch) is applied — never write a test that passes trivially."
)

_USER_TMPL = """## Issue
{issue}

## Gold patch (the fix that makes the intended behavior correct — ground truth)
{gold_diff}

## Original (hard) FAIL_TO_PASS test `{test_id}`
{test_src}

## The agent's trajectory so far (it has NOT made this test pass)
{trajectory}

Return ONLY a JSON object (no prose, no code fence):
{{
  "action": "rewrite"|"split",
  "tests": [{{"name":"test_name","code":"def test_name():\\n    ..."}}],  // 1 for rewrite, up to {k} for split
  "imports": "import ...\\n..."   // imports the tests need (may be "")
}}"""


def _summarize_transcript(transcript: List[dict], max_chars: int = 7000) -> str:
    lines = []
    for s in (transcript or []):
        fn = s.get("fn") or "?"
        act = (s.get("action") or "").strip().replace("\n", " ")[:300]
        obs = (s.get("obs") or "").strip().replace("\n", " ")[:200]
        lines.append(f"[step {s.get('step')}] ({fn}) {act} -> {obs}")
    text = "\n".join(lines)
    return ("...(earlier elided)...\n" + text[-max_chars:]) if len(text) > max_chars else text


def _parse_json_obj(raw: str) -> Optional[dict]:
    raw = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    i = raw.find("{")
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(raw)):
        if raw[j] == "{":
            depth += 1
        elif raw[j] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[i:j + 1])
                except Exception:
                    return None
    return None


# --------------------------------------------------------------------------- #
# Cache (JSONL keyed by instance_id::test_id) — one OpenAI call + gate per pair
# across the whole run, reused on every later stall/epoch.
# --------------------------------------------------------------------------- #
@dataclass
class OnlineRefinement:
    instance_id: str
    test_id: str
    refined: List[Tuple[str, str]] = field(default_factory=list)  # [(node_id, code)]
    test_file: str = ""        # path under /testbed the refined funcs live in
    full_code: str = ""
    model: str = _MODEL
    error: str = ""

    def to_record(self) -> dict:
        d = asdict(self)
        d["refined"] = [list(t) for t in self.refined]
        return d


class OnlineRefineCache:
    def __init__(self, path: Optional[str]):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._mem: Dict[str, OnlineRefinement] = {}
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    o = OnlineRefinement(
                        instance_id=r["instance_id"], test_id=r["test_id"],
                        refined=[tuple(x) for x in r.get("refined", [])],
                        test_file=r.get("test_file", ""), full_code=r.get("full_code", ""),
                        model=r.get("model", _MODEL), error=r.get("error", ""))
                    self._mem[f"{o.instance_id}::{o.test_id}"] = o
                except Exception:
                    continue

    def get(self, iid, tid):
        return self._mem.get(f"{iid}::{tid}")

    def put(self, o: OnlineRefinement):
        with self._lock:
            self._mem[f"{o.instance_id}::{o.test_id}"] = o
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as fh:
                    fh.write(json.dumps(o.to_record()) + "\n")

    def __len__(self) -> int:
        return len(self._mem)


_client = None
_client_lock = threading.Lock()


def _get_client():
    global _client
    with _client_lock:
        if _client is None:
            from openai import OpenAI
            key = load_env_value("API_KEY_OPENAI")
            if not key:
                raise RuntimeError("API_KEY_OPENAI not found in env/.env")
            _client = OpenAI(api_key=key)
    return _client


def _complete(system: str, user: str, model: str, max_tokens: int = 3072) -> str:
    client = _get_client()
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        resp = client.chat.completions.create(model=model, messages=msgs,
                                              max_completion_tokens=max_tokens)
    except TypeError:
        resp = client.chat.completions.create(model=model, messages=msgs,
                                              max_tokens=max_tokens)
    return resp.choices[0].message.content or ""


def refine_test_online(entry: dict, test_id: str, transcript: List[dict], agent_rt,
                       cache: OnlineRefineCache, model: str = _MODEL,
                       timeout: int = 180) -> List[Tuple[str, str]]:
    """Refine `test_id` via OpenAI using the trajectory-so-far, write the validated
    refined test file into the AGENT's container, and return [(node_id, code)] for the
    scheduler to swap in. Empty list ⇒ no usable refinement (caller keeps the original).

    Safeguard: a refined function is kept only if it CURRENTLY FAILS in the agent's
    container (not trivially passing) — blocks reward hacking without a 2nd container.
    """
    iid = entry.get("instance_id", "")
    cached = cache.get(iid, test_id)
    if cached is not None:
        if cached.refined and agent_rt is not None:
            try:  # ensure the file exists in THIS container too
                _container_write_file(agent_rt, f"/testbed/{cached.test_file}", cached.full_code)
            except Exception:
                pass
        return list(cached.refined)

    o = OnlineRefinement(instance_id=iid, test_id=test_id, model=model)
    test_src = extract_test_source(entry.get("test_patch", "") or "", test_id) \
        or "(source unavailable — derive from the gold patch)"
    user = _USER_TMPL.format(
        issue=(entry.get("problem_statement", "") or "")[:3500],
        gold_diff=(entry.get("patch", "") or "")[:7000],
        test_id=test_id, test_src=test_src[:3500],
        trajectory=_summarize_transcript(transcript), k=_MAX_REFINED)
    try:
        raw = _complete(_SYSTEM, user, model)
    except Exception as e:
        o.error = f"openai: {e!r}"; cache.put(o); return []
    obj = _parse_json_obj(raw)
    if not obj:
        o.error = "unparseable"; cache.put(o); return []
    pairs: List[Tuple[str, str]] = []
    for t in (obj.get("tests") or [])[:_MAX_REFINED]:
        if t.get("name") and t.get("code"):
            pairs.append((t["name"], t["code"]))
    if not pairs:
        pairs = split_test_functions(_strip_code_fence(
            "\n".join(t.get("code", "") for t in (obj.get("tests") or []))), _MAX_REFINED)
    if not pairs:
        o.error = "no tests parsed"; cache.put(o); return []
    imports = obj.get("imports") or ""
    full_code = (imports + "\n\n" if imports else "") + "\n\n".join(c for _, c in pairs)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{iid}_{test_id.split('::')[-1]}")[:80]
    test_file = f"r2e_tests/test_online_{safe}.py"

    # write into the agent's container + safeguard run (must currently FAIL)
    survivors: List[Tuple[str, str]] = []
    if agent_rt is not None:
        try:
            _container_write_file(agent_rt, f"/testbed/{test_file}", full_code)
            names = [n for n, _ in pairs]
            verdict, _out = run_adhoc_tests(agent_rt, test_file, names, timeout)
            # keep funcs that are collectible AND currently failing (verdict False)
            survivors = [(n, c) for (n, c) in pairs if verdict.get(n) is False]
        except Exception as e:
            o.error = f"safeguard: {e!r}"
    else:
        survivors = pairs  # no container (unit context) — trust prompt
    o.full_code = full_code
    o.test_file = test_file
    o.refined = [(f"{test_file}::{n}", c) for (n, c) in survivors]
    if not o.refined and not o.error:
        o.error = "no refined func currently fails (rejected as trivial)"
    cache.put(o)
    return list(o.refined)
