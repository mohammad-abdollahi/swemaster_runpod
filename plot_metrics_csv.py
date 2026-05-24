#!/usr/bin/env python3
"""Plot the per-PPO-step metrics CSV produced by dump_metrics_csv.py.

Produces a 4-panel figure for the paper:
  (1) reward: per-step mean (with min/max band) + cumulative reward
  (2) grad_norm per step (learning-signal magnitude)
  (3) solves: per-step solves + cumulative solves, and solve_all/partial/none stack
  (4) P2P regressions per step + cumulative (regression behavior over training)

Usage:
  python plot_metrics_csv.py [CSV] [OUT_PNG]
Defaults: /home/temp/train_paper150_metrics.csv -> /home/temp/train_paper150_curves.png
"""
import csv, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/temp/train_paper150_metrics.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/temp/train_paper150_curves.png"

def col(rows, k, cast=float):
    out = []
    for r in rows:
        try:
            out.append(cast(r[k]))
        except (ValueError, KeyError):
            out.append(0.0)
    return out

rows = list(csv.DictReader(CSV.open()))
if not rows:
    print(f"[empty CSV {CSV}]"); sys.exit(0)

step = col(rows, "step", int)
rmean = col(rows, "reward_mean"); rmax = col(rows, "reward_max"); rmin = col(rows, "reward_min")
cumr = col(rows, "cum_reward"); gn = col(rows, "grad_norm")
solves = col(rows, "solves", lambda x: int(float(x))); cums = col(rows, "cum_solves", lambda x: int(float(x)))
sa = col(rows, "solve_all"); sp = col(rows, "solve_partial"); sn = col(rows, "solve_none")
regr = col(rows, "p2p_regr", lambda x: int(float(x))); cumg = col(rows, "cum_p2p_regr", lambda x: int(float(x)))

fig, ax = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("ATS RL on SWE-Master-4B-SFT (max_steps=150, batch=16, SWE-rebench MIN_FTP=5)", fontsize=13, fontweight="bold")

# (1) reward
a = ax[0, 0]
a.fill_between(step, rmin, rmax, alpha=0.15, color="tab:blue", label="reward min-max")
a.plot(step, rmean, "o-", color="tab:blue", label="reward mean")
a.axhline(0, color="gray", ls="--", lw=0.5)
a2 = a.twinx(); a2.plot(step, cumr, "s--", color="tab:green", label="cumulative reward")
a2.set_ylabel("cumulative reward", color="tab:green")
a.set_title("Reward per step + cumulative"); a.set_xlabel("PPO step"); a.set_ylabel("reward"); a.legend(loc="upper left"); a.grid(alpha=0.3)

# (2) grad_norm
a = ax[0, 1]
a.plot(step, gn, "o-", color="tab:red")
a.set_title("grad_norm per step (learning-signal magnitude)"); a.set_xlabel("PPO step"); a.set_ylabel("grad_norm"); a.grid(alpha=0.3)

# (3) solves
a = ax[1, 0]
a.bar([s-0.2 for s in step], sn, 0.2, label="solve_none", color="tab:red")
a.bar(step, sp, 0.2, label="solve_partial", color="tab:orange")
a.bar([s+0.2 for s in step], sa, 0.2, label="solve_all", color="tab:green")
a3 = a.twinx(); a3.plot(step, cums, "s--", color="black", label="cumulative solves")
a3.set_ylabel("cumulative solves")
a.set_title("Solve composition + cumulative solves"); a.set_xlabel("PPO step"); a.set_ylabel("# task groups"); a.legend(loc="upper left"); a.grid(alpha=0.3)

# (4) P2P regressions
a = ax[1, 1]
a.bar(step, regr, 0.4, color="tab:purple", label="P2P regressions / step")
a4 = a.twinx(); a4.plot(step, cumg, "s--", color="black", label="cumulative regressions")
a4.set_ylabel("cumulative regressions")
a.set_title("PASS_TO_PASS regressions caught (v1.1 penalty)"); a.set_xlabel("PPO step"); a.set_ylabel("# trajectories"); a.legend(loc="upper left"); a.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT, dpi=120)
print(f"saved {OUT} ({len(rows)} steps)")
