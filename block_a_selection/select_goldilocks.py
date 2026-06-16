#!/usr/bin/env python3
"""Block-A / Arm A2 — S1: Goldilocks difficulty banding.

ALGORITHM (IMPLEMENTATION_PLAN §4.A2). Train where the signal is richest: keep
instances with `0 < frac_solved_mean < 1`, drop the degenerate extremes
{0, 1}, and weight toward a target difficulty `p* ≈ 0.2` (SSR's cited optimum)
with a Gaussian band:

  w(i) = exp( -(frac_solved_mean(i) - p*)^2 / (2 sigma^2) )

We turn the continuous weight into a training set two ways (selectable):
  --mode filter  : keep band 0<frac<1, drop the rest (uniform within band)
  --mode weight  : duplicate each kept row round(w(i)*rep) times (∝ weight) so
                   a uniform sampler sees the difficulty distribution we want.

This *replaces* the coarse MIN_FTP heuristic with a MEASURED difficulty estimate
on the partial-credit pass-fraction (finer than SSR's binary solve-rate), over
real SWE-rebench instances — the novelty contrast.

Paper: Open-Reasoner-Zero arXiv:2503.24290 (annealing on a mined hard set —
code anchor); p*≈0.2 from SSR arXiv:2512.18552 (concept). Bengio CL ICML 2009.

Usage:
  python select_goldilocks.py --parquet <train.parquet> --probe <probe.csv> \
      --out <out.parquet> [--p-star 0.2] [--sigma 0.15] [--mode filter|weight] \
      [--rep 3]
"""
from __future__ import annotations
import argparse, math
import pandas as pd
from _common import load_probe, parquet_instance_id, write_filtered, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--p-star", type=float, default=0.2)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--mode", choices=["filter", "weight"], default="filter")
    ap.add_argument("--rep", type=int, default=3, help="max row reps in weight mode")
    a = ap.parse_args()

    df = pd.read_parquet(a.parquet)
    probe = load_probe(a.probe)
    src_ids = [parquet_instance_id(r) for _, r in df.iterrows()]

    scored, kept = {}, []
    dropped_easy, dropped_hard = [], []
    for iid in src_ids:
        p = probe.get(iid)
        if not p:
            continue
        f = p["frac_solved_mean"]
        w = math.exp(-((f - a.p_star) ** 2) / (2 * a.sigma ** 2))
        scored[iid] = {"weight": w, "frac_solved_mean": f,
                       "reward_var": p["reward_var"], "tokens_mean": p["tokens_mean"]}
        if f <= 0.0:
            dropped_hard.append(iid)
        elif f >= 1.0:
            dropped_easy.append(iid)
        else:
            kept.append(iid)

    print(f"[S1] {len(scored)} probed | band(0<f<1) keep {len(kept)} | "
          f"drop {len(dropped_easy)} all-solved + {len(dropped_hard)} all-failed | "
          f"p*={a.p_star} sigma={a.sigma}")
    report({i: scored[i] for i in kept}, "weight")

    audit = {"method": "S1_goldilocks", "p_star": a.p_star, "sigma": a.sigma,
             "mode": a.mode, "kept": kept, "dropped_all_solved": dropped_easy,
             "dropped_all_failed": dropped_hard,
             "weights": {i: round(scored[i]["weight"], 4) for i in kept}}

    if a.mode == "filter":
        write_filtered(df, kept, a.out, audit)
    else:
        # weight mode: duplicate rows ∝ weight (simple sampler)
        by_id = {parquet_instance_id(r): r for _, r in df.iterrows()}
        rows = []
        reps_used = {}
        for iid in kept:
            reps = max(1, round(scored[iid]["weight"] * a.rep))
            reps_used[iid] = reps
            rows.extend([by_id[iid]] * reps)
        out_df = pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)
        out_df.to_parquet(a.out)
        audit["reps"] = reps_used
        import json
        from pathlib import Path
        Path(str(a.out) + ".audit.json").write_text(json.dumps(audit, indent=2))
        print(f"[select] weight-mode wrote {len(out_df)} rows "
              f"({len(kept)} unique) -> {a.out}")


if __name__ == "__main__":
    main()
