#!/usr/bin/env python3
"""Parse an ATS training log into a per-PPO-step CSV for later plotting.

Captures the metrics most useful for the paper's figures:
  step, reward_mean/max/min, grad_norm, pg_loss, entropy_loss,
  advantages_max/min, solve_all/partial/none, solves, cum_solves,
  p2p_regr, cum_p2p_regr, cum_reward, response_clip_ratio, step_time_s

Usage:
  python dump_metrics_csv.py [LOGFILE] [OUT_CSV]
Defaults: /home/temp/train_paper150.log -> /home/temp/train_paper150_metrics.csv
"""
import csv, re, sys
from pathlib import Path

LOG = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/temp/train_paper150.log")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/temp/train_paper150_metrics.csv")

def f(line, key):
    m = re.search(re.escape(key) + r":([-0-9.eE]+)", line)
    return float(m.group(1)) if m else ""

# Count solves + p2p regressions per PPO step from interleaved ATS terminal lines.
cur = 0
solves_by, regr_by = {}, {}
for ln in LOG.open(errors="ignore"):
    mk = re.search(r"epoch \d+, step (\d+)", ln)
    if mk:
        cur = int(mk.group(1)); continue
    if "ATS terminal" in ln:
        fs = re.search(r"frac_solved=([0-9.]+)", ln)
        pb = re.search(r"frac_p2p_broken=([0-9.]+)", ln)
        if fs and float(fs.group(1)) > 0:
            solves_by[cur] = solves_by.get(cur, 0) + 1
        if pb and float(pb.group(1)) > 0:
            regr_by[cur] = regr_by.get(cur, 0) + 1

rows = []
cum_r = 0.0; cum_s = 0; cum_g = 0
for ln in LOG.open(errors="ignore"):
    if "traj/steps_mean" not in ln:
        continue
    m = re.search(r"step:(\d+) - traj/steps_mean", ln)
    if not m:
        continue
    st = int(m.group(1))
    rm = f(ln, "critic/rewards/mean") or 0.0
    cum_r += rm if isinstance(rm, float) else 0.0
    solv = solves_by.get(st, 0); cum_s += solv
    regr = regr_by.get(st, 0); cum_g += regr
    rows.append({
        "step": st,
        "reward_mean": rm,
        "reward_max": f(ln, "critic/rewards/max"),
        "reward_min": f(ln, "critic/rewards/min"),
        "cum_reward": round(cum_r, 4),
        "grad_norm": f(ln, "actor/grad_norm"),
        "pg_loss": f(ln, "actor/pg_loss"),
        "entropy_loss": f(ln, "actor/entropy_loss"),
        "adv_max": f(ln, "critic/advantages/max"),
        "adv_min": f(ln, "critic/advantages/min"),
        "solve_all": f(ln, "batch/solve_all"),
        "solve_partial": f(ln, "batch/solve_partial"),
        "solve_none": f(ln, "batch/solve_none"),
        "solves": solv,
        "cum_solves": cum_s,
        "p2p_regr": regr,
        "cum_p2p_regr": cum_g,
        "response_clip_ratio": f(ln, "response_length/clip_ratio"),
        "step_time_s": f(ln, "timing_s/step"),
    })

if not rows:
    print(f"[no PPO step metrics yet in {LOG}]"); sys.exit(0)

with OUT.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"wrote {len(rows)} rows -> {OUT}")
print("columns:", ", ".join(rows[0].keys()))
