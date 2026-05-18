#!/usr/bin/env python3
"""Plot SWE-Master RL learning curves from a training log.

Usage: python plot_learning.py [LOGFILE] [OUT.png]
Defaults: /home/temp/train_h200_5epoch.log -> learning_curves.png

Panels:
  1. Reward learning curve — mean reward per EPOCH (the shuffle-safe aggregate)
  2. Per-task reward vs epoch (the real RL signal; one line per task)
  3. Per-step grad_norm (policy-update magnitude; 0 = all-equal-reward step)
  4. Solve-rate per epoch (none / partial / all)
"""
import re, sys, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = sys.argv[1] if len(sys.argv) > 1 else "/home/temp/train_h200_5epoch.log"
OUT = sys.argv[2] if len(sys.argv) > 2 else "learning_curves.png"

def num(line, key):
    m = re.search(re.escape(key) + r":([-0-9.eE]+)", line)
    return float(m.group(1)) if m else None

rows, step2task, step2epoch = [], {}, {}
cur = None
TASK = re.compile(r"swerebench/sweb\.eval\.x86_64\.([A-Za-z0-9_.-]+?)(?::latest)?\b")
MARK = re.compile(r"now in fit_agent, batch, epoch (\d+), step (\d+)")
with open(LOG, errors="ignore") as f:
    for ln in f:
        mk = MARK.search(ln)
        if mk:
            cur = int(mk.group(2)); step2epoch[cur] = int(mk.group(1))
        elif "Docker image:" in ln and cur is not None and cur not in step2task:
            mt = TASK.search(ln)
            if mt: step2task[cur] = mt.group(1)
        m = re.search(r"step:(\d+) - traj/steps_mean", ln)
        if not m: continue
        s = int(m.group(1))
        rows.append(dict(step=s,
                          reward=num(ln, "critic/rewards/mean"),
                          grad=num(ln, "actor/grad_norm"),
                          none=num(ln, "batch/solve_none"),
                          part=num(ln, "batch/solve_partial"),
                          alls=num(ln, "batch/solve_all")))
seen = {}
for r in rows: seen[r["step"]] = r
rows = [seen[s] for s in sorted(seen)]
if not rows:
    print(f"No completed steps in {LOG}"); sys.exit(0)

# epoch aggregates
ep = {}
for r in rows:
    e = step2epoch.get(r["step"])
    if e is None: continue
    ep.setdefault(e, []).append(r)
all_epochs = sorted(ep)
# drop trailing incomplete epoch(s): keep epochs with >= 50% of the modal
# (full) epoch step-count so a partially-run final epoch can't fake a "drop".
sizes = [len(ep[e]) for e in all_epochs]
full = max(sizes) if sizes else 0
epochs = [e for e in all_epochs if len(ep[e]) >= 0.5 * full]
dropped = [e for e in all_epochs if e not in epochs]
ep_reward = [st.mean([x["reward"] for x in ep[e] if x["reward"] is not None]) for e in epochs]
ep_none = [st.mean([x["none"] for x in ep[e] if x["none"] is not None]) for e in epochs]
ep_part = [st.mean([x["part"] for x in ep[e] if x["part"] is not None]) for e in epochs]
ep_all  = [st.mean([x["alls"] for x in ep[e] if x["alls"] is not None]) for e in epochs]

# per-task reward by epoch
by_task = {}
for r in rows:
    t, e = step2task.get(r["step"]), step2epoch.get(r["step"])
    if t is None or e is None or r["reward"] is None or e not in epochs: continue
    by_task.setdefault(t, {})[e] = r["reward"]

fig, ax = plt.subplots(2, 2, figsize=(16, 10))
_note = (f"  (excl. incomplete epoch(s) {dropped})" if dropped else "")
fig.suptitle(f"SWE-Master 4B full-param RL (H200, max_response=32768) — "
             f"{LOG.split('/')[-1]}{_note}",
             fontsize=13, fontweight="bold")

# 1. reward per epoch
ax[0, 0].plot(epochs, ep_reward, "o-", lw=2.5, color="#1f77b4")
ax[0, 0].set_title("Reward learning curve (mean test reward per epoch)")
ax[0, 0].set_xlabel("epoch"); ax[0, 0].set_ylabel("mean reward")
ax[0, 0].set_xticks(epochs); ax[0, 0].grid(alpha=.3)
for x, y in zip(epochs, ep_reward): ax[0, 0].annotate(f"{y:.3f}", (x, y),
    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

# 2. per-task reward vs epoch
for t, ev in sorted(by_task.items()):
    xs = sorted(ev); ax[0, 1].plot(xs, [ev[e] for e in xs], "o-", alpha=.5,
                                   lw=1.2, label=t[:18])
ax[0, 1].plot(epochs, ep_reward, "k--", lw=3, label="MEAN", zorder=10)
ax[0, 1].set_title("Per-task reward across epochs (real RL signal)")
ax[0, 1].set_xlabel("epoch"); ax[0, 1].set_ylabel("reward")
ax[0, 1].set_xticks(epochs); ax[0, 1].grid(alpha=.3)
ax[0, 1].legend(fontsize=6, ncol=2, loc="center left", bbox_to_anchor=(1.0, .5))

# 3. grad_norm per step
steps = [r["step"] for r in rows]
grads = [r["grad"] if r["grad"] is not None else 0 for r in rows]
ax[1, 0].bar(steps, grads, color=["#2ca02c" if g > 1e-9 else "#d62728" for g in grads])
ax[1, 0].set_title("Policy update magnitude per step (grad_norm; red = zero-gradient)")
ax[1, 0].set_xlabel("PPO step"); ax[1, 0].set_ylabel("grad_norm")
ax[1, 0].grid(alpha=.3, axis="y")
nz = sum(1 for g in grads if g > 1e-9)
ax[1, 0].text(.98, .95, f"{nz}/{len(grads)} steps updated",
              transform=ax[1, 0].transAxes, ha="right", va="top", fontsize=10,
              bbox=dict(boxstyle="round", fc="w", alpha=.7))

# 4. solve-rate per epoch (stacked)
ax[1, 1].bar(epochs, ep_none, label="solve_none", color="#d62728")
ax[1, 1].bar(epochs, ep_part, bottom=ep_none, label="solve_partial", color="#ff7f0e")
bot = [n + p for n, p in zip(ep_none, ep_part)]
ax[1, 1].bar(epochs, ep_all, bottom=bot, label="solve_all", color="#2ca02c")
ax[1, 1].set_title("Solve-rate composition per epoch")
ax[1, 1].set_xlabel("epoch"); ax[1, 1].set_ylabel("fraction of rollouts")
ax[1, 1].set_xticks(epochs); ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=.3, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"saved {OUT}  ({len(rows)} steps, epochs {epochs}, "
      f"epoch reward {['%.3f'%r for r in ep_reward]})")
