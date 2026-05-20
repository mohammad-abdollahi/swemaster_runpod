#!/usr/bin/env python3
"""Parse active-test-selection reward emissions from a training log.

Usage: python analyze_active_tests.py [LOGFILE]   (default: train_active_smoke.log)

Each rollout terminates with a print like:
  !!!!!!! ActiveTests terminal: frac_solved=0.500 r_solve=2.500 r_token=-0.050
          dense_sum=0.000 total=2.450 retired=1/2 ticks=0

This script extracts those lines and reports, per-PPO-step and per-task:
  - fraction of FAIL_TO_PASS solved at trajectory end
  - r_solve / r_token / dense_sum / total
  - how many tests the scheduler retired

Useful for confirming the new reward signal moves and for sanity-checking
token-budget penalty magnitudes against w_solve magnitudes.
"""
from __future__ import annotations

import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

LOG = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/temp/train_active_smoke.log")

STEP_MARK = re.compile(r"now in fit_agent, batch, epoch (\d+), step (\d+)")
# The default regex needs to be GREEDY past `-` (which is treated as a word
# character by some shells but Python's re uses ASCII word boundaries). The
# old non-greedy version matched at the first `-` (e.g., `acorg_1776_dark`
# instead of `acorg_1776_dark-matter-680`). Match a full instance-id by
# allowing alphanumerics, underscores, dots, and hyphens up to `:latest` or
# the next whitespace.
DOCKER_RE = re.compile(
    r"swerebench/sweb\.eval\.x86_64\.([A-Za-z0-9_.\-]+?)(?::latest|\s|$)"
)
ATS_RE = re.compile(
    r"ActiveTests terminal: "
    r"frac_solved=(?P<frac>[-0-9.eE]+) "
    r"r_solve=(?P<r_solve>[-0-9.eE]+) "
    r"r_token=(?P<r_token>[-0-9.eE]+) "
    r"dense_sum=(?P<dense>[-0-9.eE]+) "
    r"total=(?P<total>[-0-9.eE]+) "
    r"retired=(?P<ret_n>\d+)/(?P<ret_d>\d+) "
    r"ticks=(?P<ticks>\d+)"
)
SCHED_READY = re.compile(
    r"ActiveTests: scheduler ready, n_total=(\d+) m_init=(\d+) k_steps=(\d+)"
)


def main() -> int:
    if not LOG.exists():
        print(f"[error] log not found: {LOG}")
        return 1

    events = []
    n_total_seen = set()
    cur_step = (None, None)
    cur_task = None
    pending_task_for_step = None  # most recently seen Docker image before next ATS line

    with LOG.open(errors="ignore") as f:
        for ln in f:
            m = STEP_MARK.search(ln)
            if m:
                cur_step = (int(m.group(1)), int(m.group(2)))
                continue
            m = DOCKER_RE.search(ln)
            if m and "Docker image" in ln:
                cur_task = m.group(1)
                pending_task_for_step = cur_task
                continue
            m = SCHED_READY.search(ln)
            if m:
                n_total_seen.add(int(m.group(1)))
                continue
            m = ATS_RE.search(ln)
            if m:
                d = m.groupdict()
                events.append({
                    "epoch": cur_step[0],
                    "step": cur_step[1],
                    "task": pending_task_for_step,
                    "frac_solved": float(d["frac"]),
                    "r_solve": float(d["r_solve"]),
                    "r_token": float(d["r_token"]),
                    "dense_sum": float(d["dense"]),
                    "total": float(d["total"]),
                    "retired": int(d["ret_n"]),
                    "n_total": int(d["ret_d"]),
                    "ticks": int(d["ticks"]),
                })

    if not events:
        print("[warning] no ActiveTests terminal events parsed.")
        print(f"          searched {LOG} ({LOG.stat().st_size:,} bytes)")
        return 2

    print(f"# Active-Tests reward analysis ({LOG})")
    print(f"  events: {len(events)} | unique tasks: {len({e['task'] for e in events})}")
    if n_total_seen:
        print(f"  FAIL_TO_PASS sizes observed (n_total): {sorted(n_total_seen)}")

    by_step = defaultdict(list)
    for e in events:
        by_step[(e["epoch"], e["step"])].append(e)

    print("\n## Per-PPO-step summary")
    print(f"  {'epoch':>5} {'step':>4} {'n':>3} {'frac_solved_mean':>17} {'total_mean':>11} {'total_std':>10} {'all_zero?':>9}")
    for (ep, st_), rs in sorted(by_step.items(), key=lambda kv: (kv[0][0] or -1, kv[0][1] or -1)):
        fs = [r["frac_solved"] for r in rs]
        tot = [r["total"] for r in rs]
        all_zero = all(abs(t) < 1e-9 for t in tot)
        std = st.pstdev(tot) if len(tot) > 1 else 0.0
        print(
            f"  {str(ep):>5} {str(st_):>4} {len(rs):>3} "
            f"{st.mean(fs):>17.3f} {st.mean(tot):>11.3f} {std:>10.3f} {str(all_zero):>9}"
        )

    # Within-step variance is the GRPO/LOOP advantage signal.
    print("\n## Within-step reward variance (key signal for advantage / gradient)")
    variances = []
    for (ep, st_), rs in by_step.items():
        if len(rs) > 1:
            v = st.pvariance([r["total"] for r in rs])
            variances.append(v)
    if variances:
        print(f"  steps with >1 traj: {len(variances)}")
        print(f"  mean within-step variance: {st.mean(variances):.4f}")
        print(f"  median: {st.median(variances):.4f}")
        nonzero = sum(1 for v in variances if v > 1e-9)
        print(f"  steps with nonzero variance: {nonzero}/{len(variances)}  "
              f"({100*nonzero/len(variances):.1f}%)")

    # Note: per-task attribution is approximate. The log interleaves N
    # parallel rollouts' docker setup messages, so a single docker-image
    # line can be the "most recent" before any of those rollouts' terminal
    # ATS event. We log the aggregate and the *number of solves per step*
    # rather than attributing solves to specific tasks.
    print("\n## Solves per PPO step (frac_solved > 0)")
    print(f"  {'epoch':>5} {'step':>4} {'solves':>6} {'n':>3} {'best_frac':>9}")
    for (ep, st_), rs in sorted(by_step.items(), key=lambda kv: (kv[0][0] or -1, kv[0][1] or -1)):
        solves = [r["frac_solved"] for r in rs if r["frac_solved"] > 0]
        best = max((r["frac_solved"] for r in rs), default=0.0)
        print(f"  {str(ep):>5} {str(st_):>4} {len(solves):>6} {len(rs):>3} {best:>9.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
