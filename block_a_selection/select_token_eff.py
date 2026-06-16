#!/usr/bin/env python3
"""Block-A / Arm A3 — S11: Token-efficiency-weighted selection.

ALGORITHM (IMPLEMENTATION_PLAN §4.A3). Prefer instances that yield learning
signal CHEAPLY:

  score(i) = signal(i) / cost(i)
           = Var(reward)  /  (prompt_tokens_mean + completion_tokens_mean)

Keep top-K by score. This biases training toward informative-but-cheap
instances -> fewer tokens per solved instance (amplifies headline #2). It is S2's
numerator divided by rollout token cost — so it only ever keeps instances that
ALSO carry gradient signal (var>0), then ranks them by cost-efficiency.

Reuses the SWE-Effi "effectiveness under token budget" (EuTB) framing for
reporting. Paper: SWE-Effi arXiv:2509.09853 (metric); scorer is ours.

Usage:
  python select_token_eff.py --parquet <train.parquet> --probe <probe.csv> \
      --out <out.parquet> [--top-k K] [--eps 1e-6]
"""
from __future__ import annotations
import argparse
import pandas as pd
from _common import load_probe, parquet_instance_id, write_filtered, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=0, help="0 = keep all score>0")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--floor-tokens", type=float, default=1000.0,
                    help="cost floor to avoid div-by-zero / token blowups")
    a = ap.parse_args()

    df = pd.read_parquet(a.parquet)
    probe = load_probe(a.probe)
    src_ids = [parquet_instance_id(r) for _, r in df.iterrows()]

    scored = {}
    for iid in src_ids:
        p = probe.get(iid)
        if not p:
            continue
        signal = p["reward_var"]
        cost = max(a.floor_tokens, p["tokens_mean"])
        score = signal / cost
        scored[iid] = {"score": score, "frac_solved_mean": p["frac_solved_mean"],
                       "reward_var": p["reward_var"], "tokens_mean": p["tokens_mean"]}

    survivors = {i: s for i, s in scored.items() if s["score"] > a.eps}
    ranked = sorted(survivors, key=lambda i: -survivors[i]["score"])
    if a.top_k > 0:
        ranked = ranked[:a.top_k]

    print(f"[S11] {len(scored)} probed -> {len(survivors)} with signal>0 -> keep {len(ranked)}")
    report({i: survivors[i] for i in ranked}, "score")

    audit = {"method": "S11_token_efficiency", "eps": a.eps,
             "floor_tokens": a.floor_tokens, "top_k": a.top_k, "kept": ranked,
             "scores": {i: round(survivors[i]["score"], 8) for i in ranked}}
    write_filtered(df, ranked, a.out, audit)


if __name__ == "__main__":
    main()
