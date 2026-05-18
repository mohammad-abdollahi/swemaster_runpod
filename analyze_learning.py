#!/usr/bin/env python3
"""Parse SWE-Master RL training logs into a learning time-series + verdict.

Usage: python analyze_learning.py [LOGFILE]   (default: train_h200_5epoch.log)

Reads the per-step metric lines emitted by rllm/verl
("step:N - traj/... - critic/rewards/mean:.. - actor/grad_norm:.. ...") and
reports, per step and in aggregate, whether the model is actually LEARNING:
within-group reward variance -> non-zero advantage -> non-zero gradient.
"""
import re, sys, statistics as st

LOG = sys.argv[1] if len(sys.argv) > 1 else "/home/temp/train_h200_5epoch.log"

FIELDS = [
    "critic/rewards/mean", "critic/rewards/min", "critic/rewards/max",
    "critic/advantages/mean", "critic/advantages/max",
    "actor/grad_norm", "actor/pg_loss", "actor/entropy_loss",
    "batch/solve_none", "batch/solve_partial", "batch/solve_all",
    "response_length/clip_ratio", "traj/steps_mean",
    "timing_s/step",
]

def num(line, key):
    m = re.search(re.escape(key) + r":([-0-9.eE]+)", line)
    return float(m.group(1)) if m else None

rows = []
step2task = {}          # PPO step -> task id (from the rollout docker image)
step2epoch = {}         # PPO step -> epoch
cur_step = None
TASK_RE = re.compile(r"swerebench/sweb\.eval\.x86_64\.([A-Za-z0-9_.-]+?)(?::latest)?\b")
MARK_RE = re.compile(r"now in fit_agent, batch, epoch (\d+), step (\d+)")
with open(LOG, errors="ignore") as f:
    for ln in f:
        mk = MARK_RE.search(ln)
        if mk:
            cur_step = int(mk.group(2))
            step2epoch[cur_step] = int(mk.group(1))
        elif "Docker image:" in ln and cur_step is not None and cur_step not in step2task:
            mt = TASK_RE.search(ln)
            if mt:
                step2task[cur_step] = mt.group(1)
        m = re.search(r"step:(\d+) - traj/steps_mean", ln)
        if not m:
            continue
        r = {"step": int(m.group(1))}
        for k in FIELDS:
            r[k] = num(ln, k)
        rows.append(r)

# de-dup by step (keep last occurrence)
seen = {}
for r in rows:
    seen[r["step"]] = r
rows = [seen[s] for s in sorted(seen)]

if not rows:
    print(f"No completed PPO steps found in {LOG} yet.")
    sys.exit(0)

H = ("step  reward  rmin  rmax   adv_mean  grad_norm   pg_loss   entropy  "
     "none part  all  rclip  asteps  t(s)")
print(H); print("-" * len(H))
for r in rows:
    g = lambda k: ("%.3f" % r[k]) if r.get(k) is not None else "  -  "
    print(f"{r['step']:>4}  {g('critic/rewards/mean'):>6} {g('critic/rewards/min'):>5} "
          f"{g('critic/rewards/max'):>5}  {g('critic/advantages/mean'):>8}  "
          f"{g('actor/grad_norm'):>8}  {g('actor/pg_loss'):>8}  {g('actor/entropy_loss'):>7}  "
          f"{g('batch/solve_none'):>4}{g('batch/solve_partial'):>5}{g('batch/solve_all'):>5} "
          f"{g('response_length/clip_ratio'):>5}  {g('traj/steps_mean'):>5}  "
          f"{g('timing_s/step'):>5}")

n = len(rows)
gn = [r["actor/grad_norm"] for r in rows if r.get("actor/grad_norm") is not None]
rw = [r["critic/rewards/mean"] for r in rows if r.get("critic/rewards/mean") is not None]
nz_grad = [x for x in gn if x and x > 1e-9]
var_steps = sum(1 for r in rows
                if r.get("critic/rewards/max") is not None
                and r.get("critic/rewards/min") is not None
                and (r["critic/rewards/max"] - r["critic/rewards/min"]) > 1e-9)

print("\n=== VERDICT ===")
print(f"completed PPO steps        : {n}")
print(f"steps w/ non-zero grad_norm: {len(nz_grad)}/{n}  "
      f"(>0 => the policy actually updated)")
print(f"steps w/ reward variance   : {var_steps}/{n}  "
      f"(within-group spread => non-zero LOOP advantage => learning signal)")
if rw:
    half = max(1, n // 2)
    early, late = rw[:half], rw[-half:]
    print(f"reward mean  first {len(early)}: {st.mean(early):.3f}   "
          f"last {len(late)}: {st.mean(late):.3f}   "
          f"delta: {st.mean(late) - st.mean(early):+.3f}")
    print(f"reward range over run      : min {min(rw):.3f}  max {max(rw):.3f}")
if nz_grad:
    print(f"grad_norm (non-zero) mean  : {st.mean(nz_grad):.4f}  "
          f"max {max(nz_grad):.4f}")
# ---- task-keyed across-epoch learning (the real signal; dataloader shuffles)
by_task = {}   # task -> {epoch: reward}
for r in rows:
    s = r["step"]; t = step2task.get(s); e = step2epoch.get(s)
    rw_ = r.get("critic/rewards/mean")
    if t is None or e is None or rw_ is None:
        continue
    by_task.setdefault(t, {})[e] = rw_   # last occurrence in an epoch wins

multi = {t: ev for t, ev in by_task.items() if len(ev) >= 2}
print("\n=== PER-TASK LEARNING (same task across epochs; dataloader shuffles, "
      "so this — not step order — is the learning signal) ===")
if not multi:
    last_e = max(step2epoch.values()) if step2epoch else 0
    print(f"No task has been seen in 2+ epochs yet (currently epoch {last_e}). "
          f"Per-task deltas appear once epoch 1 completes (~step 40).")
else:
    print(f"{'task':<46} {'epochs(reward)':<34} first->last  delta")
    print("-" * 100)
    deltas = []
    for t in sorted(multi):
        ev = multi[t]
        es = sorted(ev)
        chain = " ".join(f"e{e}:{ev[e]:.2f}" for e in es)
        d = ev[es[-1]] - ev[es[0]]
        deltas.append(d)
        print(f"{t:<46} {chain:<34} {ev[es[0]]:.2f}->{ev[es[-1]]:.2f}  {d:+.3f}")
    up = sum(1 for d in deltas if d > 1e-9)
    dn = sum(1 for d in deltas if d < -1e-9)
    fl = sum(1 for d in deltas if abs(d) <= 1e-9)
    print("-" * 100)
    print(f"tasks revisited: {len(deltas)}   improved: {up}   regressed: {dn}"
          f"   flat: {fl}   mean delta: {st.mean(deltas):+.3f}")
    print("  (mean delta > 0 across revisited tasks = the policy is improving "
          "on tasks it has seen — the actual RL learning curve.)")

print("\nNote: all-equal rewards in an n=4 group => LOOP baseline cancels => "
      "grad_norm 0 by construction (intrinsic, not a bug). Learning shows up "
      "on the subset of steps with within-group reward variance.")
print(f"\nTensorBoard: tensorboard --logdir "
      f"/home/temp/training_outputs/swe_master_comprehensive --port 6006")
