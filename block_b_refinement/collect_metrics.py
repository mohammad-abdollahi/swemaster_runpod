#!/usr/bin/env python3
"""Collect comparable training metrics for one ablation arm into the Block-B metrics dir.

Pulls scalars from the run's newest TensorBoard event file (val solve-rate, reward-mean,
grad-norm) and, if a run log is given, tokens-per-solved from the `ATS terminal` lines.
Writes/updates a one-row CSV per arm and appends to metrics/RESULTS.md for side-by-side
comparison of control vs B1/B2/B3.

Usage:
  /home/temp/venv/bin/python3 block_b_refinement/collect_metrics.py \
     --arm B2_r22 --tb-dir <run_dir or tensorboard_log dir> [--log /home/temp/arm_B2.log] \
     [--note "k=6 gate; kept 41/64 tests"]
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re

HERE = os.path.dirname(__file__)
METRICS_DIR = os.path.join(HERE, "metrics")


def _newest_event_file(tb_dir: str) -> str | None:
    cands = []
    for root, _dirs, files in os.walk(tb_dir):
        for fn in files:
            if "tfevents" in fn:
                p = os.path.join(root, fn)
                cands.append((os.path.getmtime(p), p))
    if not cands:
        return None
    return sorted(cands)[-1][1]


def _read_scalars(event_file: str) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(event_file, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    out = {}
    for tag in tags:
        vals = [s.value for s in ea.Scalars(tag)]
        if vals:
            out[tag] = vals
    return out


def _tokens_per_solved(log_path: str) -> float | None:
    if not log_path or not os.path.exists(log_path):
        return None
    solved, tokens = 0, 0.0
    pat = re.compile(r"ATS terminal:.*frac_solved=([0-9.]+).*tokens=([0-9.]+)")
    with open(log_path, errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                fs, tk = float(m.group(1)), float(m.group(2))
                tokens += tk
                if fs >= 1.0:
                    solved += 1
    return round(tokens / solved, 1) if solved else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="e.g. control / B1_r12 / B2_r22 / B3_r3")
    ap.add_argument("--tb-dir", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    os.makedirs(METRICS_DIR, exist_ok=True)
    ev = _newest_event_file(a.tb_dir)
    row = {"arm": a.arm, "note": a.note, "tb_event": ev or "(none)"}

    if ev:
        sc = _read_scalars(ev)
        def last(*names):
            for n in names:
                if n in sc and sc[n]:
                    return round(sc[n][-1], 5)
            return None
        def peak(*names):
            for n in names:
                if n in sc and sc[n]:
                    return round(max(sc[n]), 5)
            return None
        row["val_solve_rate"] = last("val/test_score/swe", "val/test_score", "val/swe")
        row["final_reward_mean"] = last("critic/rewards/mean", "actor/reward_mean",
                                        "reward/mean", "training/reward_mean")
        row["peak_reward_mean"] = peak("critic/rewards/mean", "actor/reward_mean",
                                       "reward/mean", "training/reward_mean")
        row["final_grad_norm"] = last("actor/grad_norm", "training/grad_norm", "grad_norm")
        row["n_steps"] = max((len(v) for v in sc.values()), default=0)
    row["tokens_per_solved"] = _tokens_per_solved(a.log)

    # per-arm CSV
    csv_path = os.path.join(METRICS_DIR, f"{a.arm}.csv")
    cols = ["arm", "val_solve_rate", "final_reward_mean", "peak_reward_mean",
            "final_grad_norm", "tokens_per_solved", "n_steps", "note", "tb_event"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow({c: row.get(c, "") for c in cols})

    print(f"[metrics] {a.arm}")
    for c in cols:
        print(f"  {c:18s}: {row.get(c, '')}")
    print(f"  wrote -> {csv_path}")
    print(f"  remember to add a row to {os.path.join(METRICS_DIR, 'RESULTS.md')}")


if __name__ == "__main__":
    main()
