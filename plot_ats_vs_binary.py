#!/usr/bin/env python3
"""Plot ATS reward distribution vs simulated-binary across PPO steps.

For each PPO step, plot the per-trajectory ATS reward (continuous) and the
simulated binary baseline (0/1) on the same axis to visualize the
"always-on gradient" property: ATS has non-trivial within-step variance
even when the binary baseline maps every trajectory to 0.
"""
import re
import sys
import statistics as st
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/temp/train_active_scaled.log")
OUT = sys.argv[2] if len(sys.argv) > 2 else "ats_vs_binary.png"

STEP_MARK = re.compile(r"now in fit_agent, batch, epoch (\d+), step (\d+)")
ATS_RE = re.compile(
    r"ActiveTests terminal: "
    r"frac_solved=(?P<frac>[-0-9.eE]+) .* "
    r"total=(?P<total>[-0-9.eE]+)"
)


def main():
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
                events.append({
                    "step": cur_step[1],
                    "frac_solved": float(m.group("frac")),
                    "ats": float(m.group("total")),
                    "bin": 1.0 if float(m.group("frac")) >= 0.999 else 0.0,
                })
    if not events:
        print(f"[error] no events parsed from {LOG}")
        return 1

    by_step = defaultdict(list)
    for e in events:
        by_step[e["step"]].append(e)

    steps = sorted(by_step.keys())
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: per-trajectory ATS reward (scatter) + per-step mean (line)
    ax = axes[0]
    ats_means = []
    bin_means = []
    for s in steps:
        rs = by_step[s]
        ats_vals = [r["ats"] for r in rs]
        bin_vals = [r["bin"] for r in rs]
        ax.scatter([s + (i - len(rs)/2) * 0.02 for i in range(len(rs))],
                   ats_vals, alpha=0.4, color="tab:blue", s=20)
        ats_means.append(sum(ats_vals) / len(ats_vals))
        bin_means.append(sum(bin_vals) / len(bin_vals))
    ax.plot(steps, ats_means, "o-", color="tab:blue", linewidth=2,
            label="ATS mean reward")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_ylabel("ATS reward per trajectory")
    ax.set_title("ATS reward distribution per PPO step (single H200, SWE-Master-4B-SFT, SWE-rebench MIN_FTP=5)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Bottom: within-step standard deviation, ATS vs binary
    ax = axes[1]
    ats_stds = []
    bin_stds = []
    for s in steps:
        rs = by_step[s]
        ats_stds.append(st.pstdev([r["ats"] for r in rs]) if len(rs) > 1 else 0.0)
        bin_stds.append(st.pstdev([r["bin"] for r in rs]) if len(rs) > 1 else 0.0)
    x = np.array(steps, dtype=float)
    w = 0.35
    ax.bar(x - w/2, ats_stds, w, color="tab:blue", label="ATS reward std")
    ax.bar(x + w/2, bin_stds, w, color="tab:orange", label="binary-baseline reward std (simulated)")
    ax.set_ylabel("Within-step reward std")
    ax.set_xlabel("PPO step")
    ax.set_title("Within-step reward variance: ATS vs simulated binary baseline (higher = more gradient signal)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    for i, (a, b) in enumerate(zip(ats_stds, bin_stds)):
        ax.text(x[i] - w/2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(x[i] + w/2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT, dpi=120)
    print(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
