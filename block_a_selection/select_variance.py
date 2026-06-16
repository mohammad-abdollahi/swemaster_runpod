#!/usr/bin/env python3
"""Block-A / Arm A1 — S2: Reward-variance instance curation.

ALGORITHM (IMPLEMENTATION_PLAN §4.A1). GRPO's group advantage is
`(r_i - mean(r)) / std(r)`; if `std(r)=0` the whole group contributes ZERO
gradient. So prefer instances whose `n` probe rollouts have HIGH reward
variance. With our continuous partial-credit reward this is a *continuous*
generalization of DAPO dynamic sampling (DAPO = the binary special case: drop
groups that are all-solved or all-failed).

  score v(i) = Var(reward_per_rollout)        # or Var(frac_solved) with --on-frac
  keep:  v(i) > eps  AND  top-K by v(i)        # default K = all surviving
  (optional) sample ∝ v(i)^alpha is left to the trainer; here we hard-filter.

This is a continuous DAPO: an all-correct or all-wrong instance has v(i)=0 and
is dropped exactly as DAPO would, but partially-solved instances are RANKED by
how much gradient signal they carry.

Paper: DAPO arXiv:2503.14476; GRPO arXiv:2402.03300. Mirror math in
volcengine/verl recipe/dapo/dapo_ray_trainer.py + trainer/ppo/core_algos.py.

Usage:
  python select_variance.py --parquet <train.parquet> --probe <probe.csv> \
      --out <out.parquet> [--top-k K] [--eps 1e-6] [--on-frac]
"""
from __future__ import annotations
import argparse, statistics as st
import pandas as pd
from _common import load_probe, parquet_instance_id, write_filtered, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=0, help="0 = keep all v>eps")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--on-frac", action="store_true",
                    help="score on Var(frac_solved) instead of Var(reward)")
    a = ap.parse_args()

    df = pd.read_parquet(a.parquet)
    probe = load_probe(a.probe)
    src_ids = [parquet_instance_id(r) for _, r in df.iterrows()]

    scored = {}
    for iid in src_ids:
        p = probe.get(iid)
        if not p:
            continue
        if a.on_frac:
            xs = p["frac_solved_list"]
            v = st.pvariance(xs) if len(xs) > 1 else 0.0
        else:
            v = p["reward_var"]
        scored[iid] = {"variance": v, "frac_solved_mean": p["frac_solved_mean"],
                       "reward_var": p["reward_var"], "tokens_mean": p["tokens_mean"]}

    survivors = {i: s for i, s in scored.items() if s["variance"] > a.eps}
    ranked = sorted(survivors, key=lambda i: -survivors[i]["variance"])
    if a.top_k > 0:
        ranked = ranked[:a.top_k]

    print(f"[S2] {len(scored)} probed -> {len(survivors)} with var>eps -> keep {len(ranked)}")
    dropped_flat = [i for i in scored if i not in survivors]
    print(f"[S2] dropped {len(dropped_flat)} zero-variance (all-solved/all-failed) instances")
    report({i: survivors[i] for i in ranked}, "variance")

    audit = {"method": "S2_reward_variance", "on_frac": a.on_frac, "eps": a.eps,
             "top_k": a.top_k, "kept": ranked, "dropped_zero_var": dropped_flat,
             "n_source": len(src_ids)}
    write_filtered(df, ranked, a.out, audit)


if __name__ == "__main__":
    main()
