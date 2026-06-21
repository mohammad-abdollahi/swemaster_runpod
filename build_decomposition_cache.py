#!/usr/bin/env python3
"""Offline cache-build for B1/R12 online backtranslation.

For each instance in the train parquet, decompose every FAIL_TO_PASS test
(gpt-5.4-mini) and gate the sub-tests in the instance's Docker image (keep those
that FAIL pre-patch AND PASS gold). Survivors are persisted to a JSONL cache the
training env reads at rollout time (pure lookup — no LLM/Docker during training).

Usage:
  python build_decomposition_cache.py \
      --parquet data_examples/rl_data/swerebench_verlive/train_00000_verl.parquet \
      --out block_b_refinement/cache/decompositions.jsonl \
      --workers 6 --max-subtests 4 [--subset N] [--timeout 300]
"""
import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "R2E-Gym/src"))
logging.disable(logging.CRITICAL)

from r2egym.agenthub.runtime.test_refiner import (  # noqa: E402
    TestDecomposer, DecompositionCache, build_instance_cache,
)

DEF_PARQUET = "data_examples/rl_data/swerebench_verlive/train_00000_verl.parquet"
DEF_OUT = "block_b_refinement/cache/decompositions.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=DEF_PARQUET)
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-subtests", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--subset", type=int, default=0, help="only first N instances (0=all)")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    entries = [json.loads(df.iloc[i]["extra_info"]) for i in range(len(df))]
    if args.subset:
        entries = entries[: args.subset]
    print(f"cache-build: {len(entries)} instances | workers={args.workers} "
          f"| out={args.out}", flush=True)

    cache = DecompositionCache(args.out)           # thread-safe puts
    decomposer = TestDecomposer(max_subtests=args.max_subtests)  # shared OpenAI client

    done = 0
    tot_kept = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(build_instance_cache, e, decomposer, cache, args.timeout): e
                for e in entries}
        for fut in as_completed(futs):
            inst = futs[fut].get("instance_id", "?")
            try:
                s = fut.result()
            except Exception as exc:
                s = {"instance": inst, "error": repr(exc)}
            done += 1
            tot_kept += s.get("kept", 0)
            print(f"[{done}/{len(entries)}] {s.get('instance', inst)}: "
                  f"decomposed={s.get('decomposed', 0)} kept={s.get('kept', 0)} "
                  f"tests_with_survivors={s.get('tests_with_survivors', 0)}"
                  f"{(' ERR=' + s['error']) if s.get('error') else ''}", flush=True)

    print(f"\nDONE: {done} instances, {tot_kept} sub-tests kept, "
          f"cache size={len(cache)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
