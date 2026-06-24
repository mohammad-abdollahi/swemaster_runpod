#!/usr/bin/env python3
"""Test-to-OpenAI refiner — branch `test-refinement-test-to-openai`.

Reads phase-1 trajectory dumps (written by swe_arsenal's RLLM_DUMP_TRAJ_DIR hook),
finds FAIL_TO_PASS tests that are STILL UNSOLVED after 70 steps but whose rollout was
a MEANINGFUL / INSIGHTFUL trajectory, sends `(problem, gold patch, original test,
failed trajectory)` to the OpenAI API (gpt-5.4) and asks it to **rewrite or split**
the test into better test(s). Refined tests are validated with the same two-run Docker
gate used by the decomposition cache (FAIL pre-patch, PASS gold) and written to a JSONL
refinement cache that `build_refined_dataset.py` splices into the phase-2 dataset.

Design: TEST_TO_OPENAI_DESIGN.md. Reuses test_refiner.py (OpenAI client, .env loader,
extract_test_source, Decomposition + validate_decomposition Docker gate).

Usage:
  python test_to_openai_refiner.py \
      --dump-dir   /home/temp/training_outputs/test_to_openai/traj_dump \
      --train-parquet data_examples/rl_data/swerebench_comprehensive/train_00000_verl.parquet \
      --out        block_b_refinement/cache/refinements_test_to_openai.jsonl \
      --model gpt-5.4 --min-steps 5 --max-refined 4 --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import glob
import dataclasses
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make the r2egym test_refiner helpers importable (OpenAI client, .env, gate).
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "R2E-Gym" / "src"))
from r2egym.agenthub.runtime.test_refiner import (  # noqa: E402
    load_env_value,
    extract_test_source,
    Decomposition,
    validate_decomposition,
    split_test_functions,
    _strip_code_fence,
)

# --------------------------------------------------------------------------- #
# Editor / test action heuristics (openhands + r2egym scaffolds)
# --------------------------------------------------------------------------- #
_EDIT_FNS = {"str_replace_editor", "file_editor", "editor", "edit", "oh_editor"}
_EDIT_MARKERS = ("str_replace", "insert", "create", "apply_patch", "edit_file")
_TEST_MARKERS = ("pytest", "run_tests", "run_test", "/run_tests.sh", "py.test", "unittest")


def _as_list(v) -> list:
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return []
    return []


def _entry_of(row) -> dict:
    ei = row["extra_info"]
    return json.loads(ei) if isinstance(ei, str) else dict(ei)


def load_latest_episodes(dump_dir: str) -> Dict[str, dict]:
    """One record per instance = the LAST-written episode (closest to the final
    θ70 model). Each dump file is <instance>.jsonl with one JSON object per line."""
    out: Dict[str, dict] = {}
    for fp in sorted(glob.glob(os.path.join(dump_dir, "*.jsonl"))):
        last = None
        for line in Path(fp).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except Exception:
                continue
        if last and last.get("instance_id"):
            out[last["instance_id"]] = last
    return out


def trajectory_is_insightful(episode: dict, min_steps: int) -> Tuple[bool, dict]:
    """Heuristic: a meaningful trajectory made ≥1 code edit AND ran tests AND used
    ≥min_steps steps. Returns (ok, stats)."""
    tr = episode.get("transcript", []) or []
    n_edits = n_tests = 0
    for s in tr:
        fn = (s.get("fn") or "").lower()
        act = (s.get("action") or "").lower()
        if fn in _EDIT_FNS or any(m in act for m in _EDIT_MARKERS):
            n_edits += 1
        if any(m in act for m in _TEST_MARKERS) or any(m in fn for m in ("pytest", "test")):
            n_tests += 1
    stats = {"n_steps": episode.get("n_steps", len(tr)), "n_edits": n_edits, "n_tests": n_tests}
    ok = (stats["n_steps"] >= min_steps) and (n_edits >= 1) and (n_tests >= 1)
    return ok, stats


def summarize_trajectory(episode: dict, max_chars: int = 9000) -> str:
    """Compact, OpenAI-friendly rendering of the failed trajectory: the actions the
    agent took and (truncated) what it observed — emphasising the tail (final edits
    and test output)."""
    tr = episode.get("transcript", []) or []
    lines = []
    for s in tr:
        fn = s.get("fn") or "?"
        act = (s.get("action") or "").strip().replace("\n", " ")[:400]
        obs = (s.get("obs") or "").strip().replace("\n", " ")[:300]
        lines.append(f"[step {s.get('step')}] ({fn}) action: {act}\n    -> obs: {obs}")
    text = "\n".join(lines)
    if len(text) > max_chars:  # keep the tail (most informative)
        text = "...(earlier steps elided)...\n" + text[-max_chars:]
    return text


# --------------------------------------------------------------------------- #
# OpenAI judge + refiner (gpt-5.4)
# --------------------------------------------------------------------------- #
_SYSTEM = (
    "You are a senior Python test engineer helping an RL agent that is STUCK on a "
    "failing pytest test. You are given the issue, the gold fix (ground truth of the "
    "intended behavior), the original hard test, and the agent's full failed trajectory. "
    "Decide whether the trajectory is genuinely INSIGHTFUL (the agent understood the "
    "problem and made real progress, but the test is too coarse / brittle / entangled to "
    "learn from), and if so REFINE the test: either REWRITE it as one cleaner, more "
    "learnable test, or SPLIT it into a few smaller minimal tests — each asserting ONE "
    "facet, descriptive name, standard pytest only (no network/sleep/randomness/external "
    "services). The refined tests MUST still pass ONLY when the gold fix is applied. "
    "If the trajectory is trivial/degenerate (no real progress) answer skip."
)

_USER_TMPL = """## Issue
{issue}

## Gold patch (the fix that makes the intended behavior correct — ground truth)
{gold_diff}

## Original (hard) FAIL_TO_PASS test `{test_id}`
{test_src}

## The RL agent's failed trajectory after 70 training steps (it did NOT make this test pass)
{trajectory}

Return ONLY a JSON object (no prose, no code fence) with this schema:
{{
  "insightful": true|false,           // was the trajectory meaningful/insightful?
  "action": "rewrite"|"split"|"skip", // skip iff not insightful
  "rationale": "one sentence",
  "tests": [                          // [] when action=skip; 1 for rewrite; up to {k} for split
    {{"name": "test_descriptive_name", "code": "def test_descriptive_name():\\n    ..."}}
  ],
  "imports": "import ...\\n..."       // any imports the test functions need (may be "")
}}"""


def _complete(client, model: str, system: str, user: str, max_tokens: int = 4096) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        resp = client.chat.completions.create(
            model=model, messages=msgs, max_completion_tokens=max_tokens,
        )
    except TypeError:
        resp = client.chat.completions.create(
            model=model, messages=msgs, max_tokens=max_tokens,
        )
    return resp.choices[0].message.content or ""


def _parse_json_obj(raw: str) -> Optional[dict]:
    raw = raw.strip()
    # strip a code fence if present
    m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    # fallback: first balanced {...}
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


def _make_client(api_key_var: str = "API_KEY_OPENAI", base_url: Optional[str] = None):
    from openai import OpenAI
    key = load_env_value(api_key_var)
    if not key:
        raise RuntimeError(f"{api_key_var} not found in env/.env")
    kwargs = {"api_key": key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


@dataclasses.dataclass
class RefineResult:
    instance_id: str
    test_id: str
    insightful: bool = False
    action: str = "skip"
    rationale: str = ""
    refined: List[Tuple[str, str]] = dataclasses.field(default_factory=list)  # gated survivors
    full_code: str = ""
    model: str = ""
    stats: dict = dataclasses.field(default_factory=dict)
    verdicts: dict = dataclasses.field(default_factory=dict)
    error: str = ""

    def to_record(self) -> dict:
        d = dataclasses.asdict(self)
        d["refined"] = [list(t) for t in self.refined]
        return d


def refine_one(client, model: str, entry: dict, test_id: str, episode: dict,
               max_refined: int, gate: bool, timeout: int) -> RefineResult:
    iid = entry["instance_id"]
    res = RefineResult(instance_id=iid, test_id=test_id, model=model,
                       stats=episode.get("_stats", {}))
    test_src = extract_test_source(entry.get("test_patch", "") or "", test_id) \
        or "(source unavailable — derive from the gold patch)"
    user = _USER_TMPL.format(
        issue=(entry.get("problem_statement", "") or "")[:4000],
        gold_diff=(entry.get("patch", "") or "")[:8000],
        test_id=test_id,
        test_src=test_src[:4000],
        trajectory=summarize_trajectory(episode),
        k=max_refined,
    )
    try:
        raw = _complete(client, model, _SYSTEM, user)
    except Exception as e:
        res.error = f"openai: {e!r}"
        return res
    obj = _parse_json_obj(raw)
    if not obj:
        res.error = "unparseable openai response"
        return res
    res.insightful = bool(obj.get("insightful"))
    res.action = obj.get("action", "skip")
    res.rationale = (obj.get("rationale") or "")[:300]
    if not res.insightful or res.action == "skip":
        return res
    tests = obj.get("tests") or []
    imports = obj.get("imports") or ""
    # assemble [(name, code)] and a full runnable file
    pairs: List[Tuple[str, str]] = []
    for t in tests[:max_refined]:
        name, code = t.get("name"), t.get("code")
        if name and code:
            pairs.append((name, code))
    if not pairs:
        # tolerate models that returned a code block instead of structured tests
        pairs = split_test_functions(_strip_code_fence("\n".join(
            t.get("code", "") for t in tests)), max_refined)
    if not pairs:
        res.error = "no refined test functions parsed"
        return res
    full_code = (imports + "\n\n" if imports else "") + "\n\n".join(c for _, c in pairs)
    res.full_code = full_code
    # Docker two-run gate: keep refined tests that FAIL pre-patch AND PASS gold.
    if gate:
        dec = Decomposition(instance_id=iid, test_id=test_id, full_code=full_code,
                            subtests=pairs, model=model, n_subtests=len(pairs))
        try:
            survivors, verdicts = validate_decomposition(entry, dec, timeout)
        except Exception as e:
            res.error = f"gate: {e!r}"
            return res
        res.refined = survivors
        res.verdicts = verdicts
        if not survivors:
            res.error = "no refined tests survived gate"
    else:
        res.refined = pairs
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--train-parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--min-steps", type=int, default=5)
    ap.add_argument("--max-refined", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-gate", action="store_true", help="skip the Docker validation gate")
    ap.add_argument("--limit", type=int, default=0, help="cap number of (instance,test) jobs")
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(args.train_parquet)
    entries = {}
    for _, row in df.iterrows():
        e = _entry_of(row)
        entries[e["instance_id"]] = e

    episodes = load_latest_episodes(args.dump_dir)
    print(f"loaded {len(episodes)} instance episodes from {args.dump_dir}; "
          f"{len(entries)} train entries")

    # Build the job list: (instance, unsolved-F2P test) passing the heuristic.
    jobs: List[Tuple[dict, str, dict]] = []
    n_unsolved = n_insight = 0
    for iid, ep in episodes.items():
        entry = entries.get(iid)
        if not entry:
            continue
        f2p_map = ep.get("f2p_pass_map") or {}
        ok, stats = trajectory_is_insightful(ep, args.min_steps)
        for tid, passed in f2p_map.items():
            if passed:  # already solved → leave as-is
                continue
            n_unsolved += 1
            if not ok:
                continue
            n_insight += 1
            ep2 = dict(ep); ep2["_stats"] = stats
            jobs.append((entry, tid, ep2))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"unsolved F2P tests: {n_unsolved} | heuristic-insightful candidates: {n_insight} | "
          f"jobs: {len(jobs)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    client = _make_client()
    kept = skipped = errored = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex, open(args.out, "a") as fh:
        futs = {ex.submit(refine_one, client, args.model, e, tid, ep,
                          args.max_refined, not args.no_gate, args.timeout): (e["instance_id"], tid)
                for (e, tid, ep) in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            iid, tid = futs[fut]
            try:
                res = fut.result()
            except Exception as ex2:
                res = RefineResult(instance_id=iid, test_id=tid, error=f"worker: {ex2!r}")
            fh.write(json.dumps(res.to_record()) + "\n"); fh.flush()
            if res.refined:
                kept += 1
            elif res.action == "skip" or not res.insightful:
                skipped += 1
            else:
                errored += 1
            tag = (f"kept={len(res.refined)} act={res.action}" if res.refined
                   else (res.error or f"skip/{res.action}"))
            print(f"[{i}/{len(jobs)}] {iid}::{tid.split('::')[-1]}: {tag}")
    print(f"DONE: kept={kept} skipped={skipped} errored={errored} -> {args.out}")


if __name__ == "__main__":
    main()
