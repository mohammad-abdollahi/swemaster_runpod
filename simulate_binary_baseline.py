#!/usr/bin/env python3
"""Simulate what the binary baseline would have produced on the same smoke.

Given the ActiveTests terminal log lines, we know per-trajectory `frac_solved`.
The binary baseline reward is `1 if frac_solved == 1 else 0`. We then ask
verl/rllm's group-variance filter: which groups would have been masked out?

Usage:
  python simulate_binary_baseline.py [LOGFILE]

Compares ATS vs binary for the same trajectory pool, showing how many PPO
steps would have produced zero gradient under the baseline.
"""
import re, sys
import statistics as st
from collections import defaultdict
from pathlib import Path

LOG = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/temp/train_active_smoke.log")

STEP_MARK = re.compile(r"now in fit_agent, batch, epoch (\d+), step (\d+)")
ATS_RE = re.compile(
    r"ActiveTests terminal: "
    r"frac_solved=(?P<frac>[-0-9.eE]+) "
    r"r_solve=(?P<r_solve>[-0-9.eE]+) "
    r"r_token=(?P<r_token>[-0-9.eE]+) "
    r"dense_sum=(?P<dense>[-0-9.eE]+) "
    r"total=(?P<total>[-0-9.eE]+) "
    r"retired=\d+/(?P<n>\d+) "
    r"ticks=\d+"
)


def main() -> int:
    if not LOG.exists():
        print(f"[error] log not found: {LOG}")
        return 1

    events = []
    cur_step = (None, None)
    with LOG.open(errors="ignore") as f:
        for ln in f:
            m = STEP_MARK.search(ln)
            if m:
                cur_step = (int(m.group(1)), int(m.group(2)))
                continue
            m = ATS_RE.search(ln)
            if m:
                d = m.groupdict()
                events.append({
                    "epoch": cur_step[0],
                    "step": cur_step[1],
                    "frac_solved": float(d["frac"]),
                    "r_ats": float(d["total"]),
                    "r_binary": 1.0 if float(d["frac"]) >= 0.999 else 0.0,
                })

    if not events:
        print("[error] no events parsed")
        return 1

    # Group by (epoch, step). Within each step, all events are 16 trajectories.
    # We do not have per-task uids in the log; for the simulation we treat each
    # consecutive group of 4 trajectories within a step as one uid (matches the
    # rollout.n=4 batching, even if the exact task pairing is approximate).
    print(f"# Binary-baseline simulation vs ATS (from {LOG})")
    print(f"  total trajectories observed: {len(events)}")
    by_step = defaultdict(list)
    for e in events:
        by_step[(e["epoch"], e["step"])].append(e)
    print()

    print("## Per-PPO-step comparison")
    print(
        f"  {'step':>4} {'n':>3} "
        f"{'ATS std':>10} {'ATS in [0,1]?':>14} "
        f"{'BIN std':>9} {'BIN groups all-0':>17} "
        f"{'BIN all-zero?':>14}"
    )
    for (ep, st_), rs in sorted(by_step.items(), key=lambda kv: (kv[0][0] or -1, kv[0][1] or -1)):
        ats_vals = [e["r_ats"] for e in rs]
        bin_vals = [e["r_binary"] for e in rs]
        ats_std = st.pstdev(ats_vals) if len(ats_vals) > 1 else 0.0
        bin_std = st.pstdev(bin_vals) if len(bin_vals) > 1 else 0.0
        # Pseudo-groups of 4 consecutive trajectories
        groups = [bin_vals[i:i+4] for i in range(0, len(bin_vals), 4)]
        all_zero_groups = sum(1 for g in groups if all(v == 0 for v in g))
        bin_all_zero = all(v == 0 for v in bin_vals)
        print(
            f"  {str(st_):>4} {len(rs):>3} "
            f"{ats_std:>10.4f} {'yes' if ats_std > 1e-9 else 'no':>14} "
            f"{bin_std:>9.4f} {all_zero_groups:>4}/{len(groups):<3} "
            f"{str(bin_all_zero):>14}"
        )

    # Key headline metric: across all observed steps, how many got a usable
    # gradient signal under each scheme?
    print("\n## Headline counts")
    ats_steps_with_var = sum(1 for rs in by_step.values()
                              if (st.pstdev([e["r_ats"] for e in rs]) if len(rs) > 1 else 0) > 1e-9)
    bin_steps_with_var = sum(1 for rs in by_step.values()
                              if (st.pstdev([e["r_binary"] for e in rs]) if len(rs) > 1 else 0) > 1e-9)
    total = len(by_step)
    print(f"  PPO steps with non-zero reward std under ATS:    {ats_steps_with_var}/{total}  "
          f"({100*ats_steps_with_var/total:.0f}%)")
    print(f"  PPO steps with non-zero reward std under binary: {bin_steps_with_var}/{total}  "
          f"({100*bin_steps_with_var/total:.0f}%)")
    print(f"  PPO steps wasted by binary (no gradient signal):  {total - bin_steps_with_var}/{total}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
