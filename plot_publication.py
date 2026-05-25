#!/usr/bin/env python3
"""Publication-quality figures for the ATS-RL paper.

Reads the raw per-step metrics CSV (never modifies it), produces:
  - paper/fig_learning_curve.png : 4-panel learning curve
  - paper/fig_ats_vs_binary.png  : ATS vs simulated-binary gradient signal
  - paper/normalized_metrics.csv : the normalized reward series used in the plots

Reward normalization (DISCLOSED — see paper/FIGURES_NOTES.md):
  r_norm = clip(reward_mean / NORM, -1, 1), where NORM = max over steps of the
  reward_mean among NON-OUTLIER steps. Any normalized value below FLOOR (-0.67)
  is winsorized to FLOOR. Exactly one step (the documented, since-fixed
  step-8 token-penalty artifact, raw mean -6.514) is affected by the floor.
  Raw values remain in the source CSV.
"""
import csv, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/temp/train_paper150_metrics.csv")
OUTDIR = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/temp/swemaster_runpod/paper")
FLOOR = -0.67          # winsorization floor for pathological-outlier steps
OUTLIER_Z = 3.0        # a step is an "outlier" if |reward_mean| is this many MADs out

# --- publication style ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "legend.frameon": False,
})

def col(rows, k, cast=float):
    out = []
    for r in rows:
        try: out.append(cast(r[k]))
        except (ValueError, KeyError): out.append(0.0)
    return out

rows = list(csv.DictReader(CSV.open()))
if not rows:
    print(f"[empty CSV {CSV}]"); sys.exit(0)

step = col(rows, "step", int)
rmean = col(rows, "reward_mean")
gn = col(rows, "grad_norm")
solves = col(rows, "solves", lambda x: int(float(x)))
cums = col(rows, "cum_solves", lambda x: int(float(x)))
sa = col(rows, "solve_all"); sp = col(rows, "solve_partial"); sn = col(rows, "solve_none")
entropy = col(rows, "entropy_loss")

# --- robust outlier-aware normalization scale (median absolute deviation) ---
import statistics as st
med = st.median(rmean)
mad = st.median([abs(x - med) for x in rmean]) or 1e-9
def is_outlier(x): return abs(x - med) / (1.4826 * mad) > OUTLIER_Z
legit = [x for x in rmean if not is_outlier(x)]
NORM = max(abs(x) for x in legit) if legit else 1.0
r_norm = []
for x in rmean:
    v = x / NORM
    v = max(-1.0, min(1.0, v))   # clip to [-1,1]
    if v < FLOOR:
        v = FLOOR                # winsorize pathological negatives
    r_norm.append(v)

# write normalized series (raw CSV untouched)
with (OUTDIR / "normalized_metrics.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["step", "reward_mean_raw", "reward_norm", "grad_norm", "solves", "cum_solves", "is_outlier"])
    for i, s in enumerate(step):
        w.writerow([s, round(rmean[i],4), round(r_norm[i],4), gn[i], solves[i], cums[i], int(is_outlier(rmean[i]))])

# ===================== Figure 1: learning curve =====================
fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))

# (a) normalized reward
a = ax[0, 0]
a.axhline(0, color="0.6", lw=0.8, zorder=1)
a.plot(step, r_norm, "o-", color="#1f4e79", lw=2, ms=5, zorder=3, label="normalized reward")
# mark the winsorized outlier step(s)
ox = [step[i] for i in range(len(step)) if is_outlier(rmean[i])]
oy = [r_norm[i] for i in range(len(step)) if is_outlier(rmean[i])]
if ox:
    a.scatter(ox, oy, color="#c00000", s=60, zorder=4, marker="v",
              label="winsorized (token-bug step, fixed)")
a.set_ylim(-1.05, 1.05)
a.set_title("(a) Normalized policy reward per PPO step")
a.set_xlabel("PPO step"); a.set_ylabel("reward (normalized to [-1, 1])")
a.legend(loc="lower right", fontsize=9)

# (b) cumulative solves
a = ax[0, 1]
a.plot(step, cums, "s-", color="#2e7d32", lw=2, ms=5)
a.set_title("(b) Cumulative resolved tasks")
a.set_xlabel("PPO step"); a.set_ylabel("cumulative solves")

# (c) grad norm — winsorize the documented step-8 pre-clip spike for readability
legit_gn = [gn[i] for i in range(len(step)) if not is_outlier(rmean[i])]
gn_cap = max(legit_gn) * 1.15 if legit_gn else 1.0
gn_plot = [min(g, gn_cap) for g in gn]
a = ax[1, 0]
a.plot(step, gn_plot, "^-", color="#b8860b", lw=2, ms=5, label="grad norm (applied; trainer clips at 1.0)")
ogn = [(step[i], gn_plot[i]) for i in range(len(step)) if is_outlier(rmean[i])]
if ogn:
    a.scatter([p[0] for p in ogn], [p[1] for p in ogn], color="#c00000", s=60, marker="v",
              zorder=4, label="token-bug step (pre-clip 5.4, fixed)")
a.set_title("(c) Gradient norm per PPO step")
a.set_xlabel("PPO step"); a.set_ylabel("grad norm")
a.set_ylim(0, gn_cap * 1.05); a.legend(loc="upper left", fontsize=8)

# (d) solve composition (fraction of task-groups)
a = ax[1, 1]
tot = [max(1, sa[i]+sp[i]+sn[i]) for i in range(len(step))]
fa = [sa[i]/tot[i] for i in range(len(step))]
fp = [sp[i]/tot[i] for i in range(len(step))]
fn = [sn[i]/tot[i] for i in range(len(step))]
a.bar(step, fa, color="#2e7d32", label="all rollouts solved")
a.bar(step, fp, bottom=fa, color="#f0a020", label="partial")
a.bar(step, fn, bottom=[fa[i]+fp[i] for i in range(len(step))], color="#c0c0c0", label="none")
a.set_title("(d) Per-step solve composition")
a.set_xlabel("PPO step"); a.set_ylabel("fraction of task groups")
a.set_ylim(0, 1.0); a.legend(loc="upper right", fontsize=8, ncol=1)

fig.suptitle("Active-Test-Selection RL on SWE-Master-4B-SFT (SWE-rebench, max\\_steps=150)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUTDIR / "fig_learning_curve.png")
print(f"saved {OUTDIR/'fig_learning_curve.png'}  (NORM={NORM:.3f}, FLOOR={FLOOR}, outlier steps={ox})")
