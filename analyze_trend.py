#!/usr/bin/env python3
"""Per-PPO-step learning-trend analyzer for an ATS training log.

Reports, per step and cumulatively:
  - mean reward (critic/rewards/mean)        -> noisy (task-mix + P2P penalty)
  - CUMULATIVE mean reward (running sum)     -> smooths step-to-step noise
  - moving-avg mean reward (last 3 steps)    -> trend without monotonic bias
  - grad_norm                                -> learning-signal magnitude
  - solves this step / CUMULATIVE solves     -> capability accrual
  - solve_partial / solve_none               -> per-step group composition
  - P2P regressions this step / cumulative   -> regression behavior over time
  - entropy_loss                             -> exploration health

Usage: python analyze_trend.py [LOGFILE]
"""
import re, sys
from pathlib import Path

LOG = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/temp/train_longrun20.log")

def field(line, key):
    m = re.search(re.escape(key) + r":([-0-9.eE]+)", line)
    return float(m.group(1)) if m else None

# Parse per-step metric lines
steps = []
for ln in LOG.open(errors="ignore"):
    if "traj/steps_mean" not in ln:
        continue
    m = re.search(r"step:(\d+) - traj/steps_mean", ln)
    if not m:
        continue
    steps.append({
        "step": int(m.group(1)),
        "reward_mean": field(ln, "critic/rewards/mean"),
        "reward_max": field(ln, "critic/rewards/max"),
        "grad_norm": field(ln, "actor/grad_norm"),
        "entropy": field(ln, "actor/entropy_loss"),
        "solve_partial": field(ln, "batch/solve_partial"),
        "solve_all": field(ln, "batch/solve_all"),
        "solve_none": field(ln, "batch/solve_none"),
    })

# Parse ATS terminal events to count solves + p2p regressions per step.
# (Terminal lines are interleaved; we attribute them to the step in progress.)
cur_step = 0
solves_by_step = {}
regr_by_step = {}
for ln in LOG.open(errors="ignore"):
    mk = re.search(r"epoch \d+, step (\d+)", ln)
    if mk:
        cur_step = int(mk.group(1))
        continue
    if "ATS terminal" in ln:
        fs = re.search(r"frac_solved=([0-9.]+)", ln)
        pb = re.search(r"frac_p2p_broken=([0-9.]+)", ln)
        if fs and float(fs.group(1)) > 0:
            solves_by_step[cur_step] = solves_by_step.get(cur_step, 0) + 1
        if pb and float(pb.group(1)) > 0:
            regr_by_step[cur_step] = regr_by_step.get(cur_step, 0) + 1

if not steps:
    print(f"[no PPO step metrics yet in {LOG}]")
    sys.exit(0)

print(f"# Learning-trend analysis ({LOG.name}) — {len(steps)} PPO steps\n")
hdr = f"{'step':>4} {'reward_mean':>11} {'cum_reward':>11} {'mov3_avg':>9} {'grad_norm':>9} {'solv':>4} {'cum_solv':>8} {'p2p_regr':>8} {'entropy':>8}"
print(hdr)
print("-" * len(hdr))
cum_reward = 0.0
cum_solves = 0
cum_regr = 0
rms = []
for s in steps:
    rm = s["reward_mean"] or 0.0
    cum_reward += rm
    rms.append(rm)
    mov3 = sum(rms[-3:]) / len(rms[-3:])
    st = s["step"]
    solv = solves_by_step.get(st, 0)
    regr = regr_by_step.get(st, 0)
    cum_solves += solv
    cum_regr += regr
    print(f"{st:>4} {rm:>11.3f} {cum_reward:>11.3f} {mov3:>9.3f} "
          f"{(s['grad_norm'] or 0):>9.3f} {solv:>4} {cum_solves:>8} {regr:>8} {(s['entropy'] or 0):>8.3f}")

# Simple trend verdict
print("\n## Trend signals")
if len(steps) >= 3:
    first3 = sum(rms[:3]) / 3
    last3 = sum(rms[-3:]) / 3
    print(f"  mean reward: first-3-avg={first3:.3f} -> last-3-avg={last3:.3f}  (delta {last3-first3:+.3f})")
print(f"  cumulative reward: {cum_reward:+.3f} over {len(steps)} steps")
print(f"  cumulative solves: {cum_solves} | cumulative P2P regressions: {cum_regr}")
gns = [s["grad_norm"] for s in steps if s["grad_norm"] is not None]
if gns:
    print(f"  grad_norm: mean={sum(gns)/len(gns):.3f} min={min(gns):.3f} max={max(gns):.3f} (all>0 ⇒ every step learns)")
