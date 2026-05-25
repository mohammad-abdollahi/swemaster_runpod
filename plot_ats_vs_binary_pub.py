#!/usr/bin/env python3
"""Publication figure: ATS vs (simulated) binary-reward gradient signal.

For each PPO step we read the per-trajectory ATS terminal rewards from the
training log, and compute the within-step reward standard deviation under:
  - ATS (continuous): the actual total reward (winsorized for the documented
    step-8 token artifact, consistent with the learning-curve figure)
  - Binary baseline (simulated): r = 1 if the trajectory fully solved its
    FAIL_TO_PASS else 0  (the original SWE-Master reward)
Within-step reward std is what drives the GRPO/LOOP advantage; std≈0 ⇒ no
gradient. The figure shows ATS sustains signal across steps.

NOTE: the binary series is a *counterfactual* computed on ATS-policy
trajectories — a lower bound on the binary baseline's weakness, not a real
baseline run. Stated in paper/FIGURES_NOTES.md.
"""
import re, sys, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/temp/train_paper150.log")
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/temp/swemaster_runpod/paper/fig_ats_vs_binary.png"

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
    "axes.labelsize": 11, "axes.grid": True, "grid.alpha": 0.25,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "legend.frameon": False,
})

cur = 0
ats_by, bin_by = {}, {}
for ln in LOG.open(errors="ignore"):
    mk = re.search(r"epoch \d+, step (\d+)", ln)
    if mk:
        cur = int(mk.group(1)); continue
    if "terminal:" in ln and "frac_solved=" in ln:   # matches "ATS terminal" + legacy "ActiveTests terminal"
        fs = re.search(r"frac_solved=([0-9.]+)", ln)
        tot = re.search(r"total=(-?[0-9.]+)", ln)
        if fs and tot:
            ats_by.setdefault(cur, []).append(float(tot.group(1)))
            bin_by.setdefault(cur, []).append(1.0 if float(fs.group(1)) >= 0.999 else 0.0)

steps = sorted(s for s in ats_by if s >= 1)
ats_std, bin_std, ats_signal, bin_signal = [], [], 0, 0
for s in steps:
    a = ats_by[s]; b = bin_by[s]
    sa = st.pstdev(a) if len(a) > 1 else 0.0
    sb = st.pstdev(b) if len(b) > 1 else 0.0
    # winsorize the documented step-8 ATS std (token-bug outlier) for readability
    ats_std.append(min(sa, 5.0)); bin_std.append(sb)
    if sa > 1e-6: ats_signal += 1
    if sb > 1e-6: bin_signal += 1

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

a = ax[0]
x = range(len(steps)); w = 0.4
a.bar([i - w/2 for i in x], ats_std, w, color="#1f4e79", label="ATS (continuous)")
a.bar([i + w/2 for i in x], bin_std, w, color="#c0792e", label="binary baseline (simulated)")
a.set_xticks(list(x)); a.set_xticklabels(steps)
a.set_title("(a) Within-step reward std (drives GRPO advantage)")
a.set_xlabel("PPO step"); a.set_ylabel("reward std-dev")
a.legend(loc="upper right", fontsize=9)

a = ax[1]
labels = ["ATS\n(continuous)", "binary\n(simulated)"]
vals = [100*ats_signal/len(steps), 100*bin_signal/len(steps)]
a.bar(labels, vals, color=["#1f4e79", "#c0792e"], width=0.55)
for i, v in enumerate(vals):
    a.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=11, fontweight="bold")
a.set_title("(b) PPO steps with non-zero gradient signal")
a.set_ylabel("% of PPO steps"); a.set_ylim(0, 109)

fig.suptitle("Continuous test-based reward sustains the learning signal where binary stalls",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT)
print(f"saved {OUT} | ATS signal {ats_signal}/{len(steps)} steps, binary {bin_signal}/{len(steps)}")
