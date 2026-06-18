#!/usr/bin/env python3
"""
Block-B test-refinement driver (offline parquet transform).

Reads a verl parquet, refines each instance's FAIL_TO_PASS set, and writes a new parquet
PLUS an audit trail. The ATS reward machinery is unchanged — only the set of tests an
instance carries changes (IMPLEMENTATION_PLAN.md §5).

Authoritative F2P source: the runtime builds its TestSpec from `extra_info["make_test_spec"]`
(docker.py L206/L230 `TestSpec(**json.loads(ds["make_test_spec"]))`), and surfaces results
keyed to `test_spec.FAIL_TO_PASS` (L1731). So we rewrite **make_test_spec["FAIL_TO_PASS"]**
(and mirror extra_info["FAIL_TO_PASS"] for bookkeeping).

Currently implemented: --method r22 (mutation-specificity gate, NO LLM).
Deferred (need teacher LLM via .env): --method r12, --method r3.

Usage:
  /home/temp/venv/bin/python3 refine_dataset.py --method r22 \
     --in  data_examples/rl_data/swerebench_live64/train_00000_verl.parquet \
     --out data_examples/rl_data/ats_B2_r22/train_00000_verl.parquet \
     --mutants 6 --workers 12 [--subset N] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

# make r2egym importable (installed in the training venv; also add src path as fallback)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "R2E-Gym", "src"))


def _refine_one(args_tuple):
    """Worker: refine a single row. Returns (idx, new_extra_info_str, audit_dict).
    Runs in its own process; Docker work happens here."""
    idx, extra_info_str, method, k, m_frac, timeout, verbose = args_tuple
    from r2egym.agenthub.runtime.test_refiner import specificity_gate

    ei = json.loads(extra_info_str) if isinstance(extra_info_str, str) else dict(extra_info_str)
    mts = ei.get("make_test_spec")
    mts = json.loads(mts) if isinstance(mts, str) else (mts or {})

    # authoritative universe = make_test_spec FAIL_TO_PASS; sync top-level for the gate
    authoritative_f2p = list(mts.get("FAIL_TO_PASS", ei.get("FAIL_TO_PASS", [])) or [])
    ei["FAIL_TO_PASS"] = authoritative_f2p

    if method == "r22":
        res = specificity_gate(ei, k=k, m_frac=m_frac, timeout=timeout, verbose=verbose)
    else:
        raise ValueError(f"method {method} not implemented offline (needs teacher LLM)")

    kept = res.kept_f2p or authoritative_f2p  # never empty
    # rewrite authoritative + mirror
    mts["FAIL_TO_PASS"] = kept
    ei["make_test_spec"] = json.dumps(mts)
    ei["FAIL_TO_PASS"] = kept
    ei["_refined"] = res.refiner
    return idx, json.dumps(ei), res.to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--method", default="r22", choices=["r22", "r12", "r3"])
    ap.add_argument("--mutants", type=int, default=6, help="k operator mutants (+1 pre-patch)")
    ap.add_argument("--m-frac", type=float, default=0.5, help="keep test if kill-rate >= m-frac")
    ap.add_argument("--subset", type=int, default=0, help="0 = all rows; else first N")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--audit-dir", default="block_b_refinement/audit")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="serial, verbose, print gold/base/kept per instance for the subset")
    a = ap.parse_args()

    if a.method in ("r12", "r3"):
        sys.exit(f"[refine] --method {a.method} needs a teacher LLM. Add an API key to .env "
                 f"first; see BLOCK_B_TEST_REFINEMENT_DESIGN.md §6.")

    df = pd.read_parquet(a.inp)
    n_total = len(df)
    idxs = list(range(n_total))
    if a.subset and a.subset < n_total:
        idxs = idxs[: a.subset]
    print(f"[refine] {a.inp}: {n_total} rows; refining {len(idxs)} via {a.method} "
          f"(k={a.mutants}, m_frac={a.m_frac}, workers={'1(smoke)' if a.smoke else a.workers})")

    tasks = [(i, df.iloc[i]["extra_info"], a.method, a.mutants, a.m_frac, a.timeout,
              a.verbose or a.smoke) for i in idxs]

    results = {}
    audits = []
    t0 = time.time()
    if a.smoke or a.workers <= 1:
        for tk in tasks:
            idx, ei_str, audit = _refine_one(tk)
            results[idx] = ei_str
            audits.append(audit)
            _log_one(audit)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(_refine_one, tk): tk[0] for tk in tasks}
            done = 0
            for fut in as_completed(futs):
                idx, ei_str, audit = fut.result()
                results[idx] = ei_str
                audits.append(audit)
                done += 1
                _log_one(audit, prefix=f"[{done}/{len(idxs)}] ")

    # write refined parquet (refined rows replaced; untouched rows pass through unchanged)
    new_ei = df["extra_info"].tolist()
    for idx, ei_str in results.items():
        new_ei[idx] = ei_str
    df["extra_info"] = new_ei
    if a.subset and a.subset < n_total:
        df = df.iloc[idxs].reset_index(drop=True)  # keep only the refined subset
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_parquet(a.out, index=False)

    # audit trail
    os.makedirs(a.audit_dir, exist_ok=True)
    ts = int(t0)
    audit_path = os.path.join(a.audit_dir, f"{a.method}_{ts}.jsonl")
    with open(audit_path, "w") as f:
        for au in audits:
            f.write(json.dumps(au) + "\n")

    # summary
    n_changed = sum(1 for au in audits if au["dropped_f2p"])
    n_err = sum(1 for au in audits if au["error"])
    n_fallback = sum(1 for au in audits if au["fallback_used"])
    tot_orig = sum(len(au["orig_f2p"]) for au in audits)
    tot_kept = sum(len(au["kept_f2p"]) for au in audits)
    tot_drop = sum(len(au["dropped_f2p"]) for au in audits)
    dt = time.time() - t0
    print("\n===== REFINE SUMMARY =====")
    print(f"method={a.method}  rows={len(audits)}  changed={n_changed}  "
          f"fallback={n_fallback}  errors={n_err}")
    print(f"F2P tests: orig={tot_orig}  kept={tot_kept}  dropped={tot_drop}  "
          f"({100*tot_drop/max(1,tot_orig):.1f}% dropped)")
    print(f"wrote parquet -> {a.out}")
    print(f"wrote audit   -> {audit_path}")
    print(f"elapsed {dt/60:.1f} min")


def _log_one(au, prefix=""):
    tag = "ERR " if au["error"] else ("FALLBACK " if au["fallback_used"] else "")
    krs = au["kill_rate_per_test"]
    avg_kr = (sum(krs.values()) / len(krs)) if krs else 0.0
    print(f"{prefix}{tag}{au['instance_id']}: f2p {len(au['orig_f2p'])}->{len(au['kept_f2p'])} "
          f"(dropped {len(au['dropped_f2p'])}, avg_kill_rate {avg_kr:.2f})"
          + (f"  [{au['error']}]" if au["error"] else ""))


if __name__ == "__main__":
    main()
