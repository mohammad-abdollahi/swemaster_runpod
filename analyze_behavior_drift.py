#!/usr/bin/env python3
"""Compare agent behavior across PPO steps from chat_completion jsonls.

Reads chat_completions/{step}.jsonl for each step, measures per-step
aggregate behavior signals (first-action type distribution, bash command
verbosity, response token usage, finish-action rate, ratio of `pytest`
invocations, references to active test IDs). Outputs a CSV-friendly
summary so we can see how the policy changed across PPO updates.

Usage:
  python analyze_behavior_drift.py [EXPERIMENT_DIR]

EXPERIMENT_DIR defaults to /home/temp/training_outputs/active_tests_swe_master/
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else
            "/home/temp/training_outputs/active_tests_swe_master")


def extract_first_action_command(asst_content: str) -> str:
    """Extract the first <function=NAME ...> name from an assistant message."""
    m = re.search(r"<function=([a-z_]+)\b", asst_content)
    return m.group(1) if m else "(no_function)"


def n_bash_chars(asst_content: str) -> int:
    """Approximate bash payload size from an assistant message."""
    m = re.search(r"<function=execute_bash>(.*?)</function>", asst_content, re.DOTALL)
    return len(m.group(1)) if m else 0


def main():
    # Find the most recent experiment dir
    candidates = sorted([p for p in ROOT.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
    if not candidates:
        print(f"[error] no experiment dirs under {ROOT}")
        return 1
    exp = candidates[-1]
    cc_dir = exp / "chat_completions"
    if not cc_dir.exists():
        print(f"[error] no chat_completions/ in {exp}")
        return 2

    print(f"# Agent behavior drift across PPO steps ({exp.name})")
    step_files = sorted(cc_dir.glob("*.jsonl"), key=lambda p: int(p.stem))
    if not step_files:
        print("[warning] no .jsonl files yet")
        return 0

    print(f"\n{'step':>4} {'n':>3} {'mean_turns':>10} {'mean_response_chars':>18} "
          f"{'finish_rate':>11} {'pytest_inv_rate':>15} {'first_actions':>30}")
    for sf in step_files:
        records = [json.loads(l) for l in sf.open() if l.strip()]
        if not records:
            continue
        n = len(records)
        turns = [sum(1 for m in r["chat_completion"] if m["role"] == "assistant") for r in records]
        # mean response chars per trajectory (assistant message bytes)
        resp = [sum(len(m.get("content","")) for m in r["chat_completion"] if m["role"]=="assistant")
                for r in records]
        finish_count = sum(1 for r in records
                            if r.get("termination_reason") == "ENV_DONE")
        pytest_count = sum(1 for r in records
                            if any("pytest" in m.get("content","")
                                   for m in r["chat_completion"] if m["role"]=="assistant"))
        # First-action distribution
        first_cmd = Counter()
        for r in records:
            asst = [m for m in r["chat_completion"] if m["role"]=="assistant"]
            if asst:
                first_cmd[extract_first_action_command(asst[0]["content"])] += 1
        top3 = ", ".join(f"{k}={v}" for k, v in first_cmd.most_common(3))
        print(
            f"{sf.stem:>4} {n:>3} {sum(turns)/n:>10.1f} {sum(resp)/n:>18.0f} "
            f"{100*finish_count/n:>10.0f}% {100*pytest_count/n:>14.0f}% {top3:>30}"
        )

    print("\nLegend:")
    print("  mean_turns         = mean assistant messages per trajectory")
    print("  mean_response_chars= mean total assistant-content bytes per trajectory")
    print("  finish_rate        = fraction of trajectories ending via ENV_DONE (submit/finish)")
    print("  pytest_inv_rate    = fraction that mention 'pytest' at any point")
    print("  first_actions      = top-3 first-action function names (count)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
