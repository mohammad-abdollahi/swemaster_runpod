#!/usr/bin/env python3
"""Build the phase-2 dataset by splicing OpenAI-refined tests into the original
ATS-RL dataset (swerebench_comprehensive). Branch `test-refinement-test-to-openai`.

For each instance that has ≥1 gate-surviving refined test (from
`test_to_openai_refiner.py`'s cache), we:
  1. add each refined test as a NEW file to the instance's `test_patch`
     (`r2e_tests/test_to_openai_<safe>_<k>.py`),
  2. update `FAIL_TO_PASS`: remove the original refined-away node id, add the new
     `<file>::<name>` node ids (other F2P + ALL P2P unchanged),
  3. regenerate `make_test_spec` (embedded test_spec the runtime reads directly) so
     its eval_script applies the augmented test_patch and runs the new F2P list.
Instances with no refinement are copied verbatim. Output:
`data_examples/rl_data/swerebench_comprehensive_refined/`.

Design: TEST_TO_OPENAI_DESIGN.md.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import dataclasses
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "R2E-Gym" / "src"))
from swebench_fork_swerebench.harness.test_spec.test_spec import (  # noqa: E402
    make_test_spec as mts_swe,
)


def _as_list(v) -> list:
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return []
    return []


def _entry_of(row) -> dict:
    ei = row["extra_info"]
    return json.loads(ei) if isinstance(ei, str) else dict(ei)


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def new_file_hunk(path: str, code: str) -> str:
    """A `git apply`-compatible new-file diff hunk creating `path` with `code`."""
    body = code if code.endswith("\n") else code + "\n"
    lines = body.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    n = len(lines)
    plus = "".join("+" + ln + "\n" for ln in lines)
    return (
        f"diff --git a/{path} b/{path}\n"
        f"new file mode 100644\n"
        f"index 0000000..1111111\n"
        f"--- /dev/null\n"
        f"+++ b/{path}\n"
        f"@@ -0,0 +1,{n} @@\n"
        f"{plus}"
    )


def load_refinements(cache_path: str) -> Dict[str, List[dict]]:
    """instance_id -> [refinement record] for records with gated survivors."""
    out: Dict[str, List[dict]] = defaultdict(list)
    if not Path(cache_path).exists():
        return out
    for line in Path(cache_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        refined = rec.get("refined") or []
        if refined and rec.get("instance_id") and rec.get("test_id"):
            out[rec["instance_id"]].append(rec)
    return out


def splice_instance(entry: dict, recs: List[dict]) -> Tuple[dict, dict]:
    """Return (updated_entry, stats). Mutates a copy of `entry`."""
    e = dict(entry)
    iid = e["instance_id"]
    f2p = list(_as_list(e.get("FAIL_TO_PASS")))
    test_patch = e.get("test_patch", "") or ""
    if test_patch and not test_patch.endswith("\n"):
        test_patch += "\n"
    added_nodes: List[str] = []
    removed_nodes: List[str] = []
    safe = _safe(iid)
    for k, rec in enumerate(recs):
        orig = rec["test_id"]
        path = f"r2e_tests/test_to_openai_{safe}_{k}.py"
        # one new file per refined original test; functions = gated survivors
        names = [n for n, _ in (tuple(x) for x in rec["refined"])]
        full_code = rec.get("full_code") or "\n\n".join(c for _, c in rec["refined"])
        test_patch += new_file_hunk(path, full_code)
        # FAIL_TO_PASS swap
        if orig in f2p:
            f2p.remove(orig)
            removed_nodes.append(orig)
        for nm in names:
            node = f"{path}::{nm}"
            f2p.append(node)
            added_nodes.append(node)
    e["FAIL_TO_PASS"] = f2p
    e["test_patch"] = test_patch
    # regenerate the embedded test_spec (eval_script applies new test_patch + F2P)
    ts = mts_swe(e)
    e["make_test_spec"] = json.dumps(
        dataclasses.asdict(ts) if dataclasses.is_dataclass(ts) else dict(ts.__dict__)
    )
    return e, {"instance": iid, "removed": removed_nodes, "added": added_nodes,
               "n_refined_tests": len(added_nodes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data_examples/rl_data/swerebench_comprehensive")
    ap.add_argument("--cache", default="block_b_refinement/cache/refinements_test_to_openai.jsonl")
    ap.add_argument("--out", default="data_examples/rl_data/swerebench_comprehensive_refined")
    ap.add_argument("--dry-run", action="store_true", help="report plan, don't write parquet")
    args = ap.parse_args()

    import pandas as pd
    refinements = load_refinements(args.cache)
    print(f"refinements: {sum(len(v) for v in refinements.values())} tests across "
          f"{len(refinements)} instances")

    os.makedirs(args.out, exist_ok=True)
    for split in ("train_00000_verl", "test_00000_verl"):
        src_p = os.path.join(args.src, f"{split}.parquet")
        if not os.path.exists(src_p):
            print(f"skip missing {src_p}")
            continue
        df = pd.read_parquet(src_p)
        rows = []
        n_spliced = 0
        for _, row in df.iterrows():
            e = _entry_of(row)
            iid = e["instance_id"]
            recs = refinements.get(iid) if split.startswith("train") else None
            if recs:
                e2, stats = splice_instance(e, recs)
                n_spliced += 1
                print(f"  spliced {iid}: -{len(stats['removed'])} F2P / "
                      f"+{stats['n_refined_tests']} refined")
                new_row = dict(row)
                new_row["extra_info"] = json.dumps(e2)
                rows.append(new_row)
            else:
                rows.append(dict(row))
        out_df = pd.DataFrame(rows)
        out_p = os.path.join(args.out, f"{split}.parquet")
        if args.dry_run:
            print(f"[dry-run] {split}: {len(out_df)} rows, {n_spliced} spliced (not written)")
        else:
            out_df.to_parquet(out_p)
            print(f"WROTE {out_p}: {len(out_df)} rows, {n_spliced} spliced")
    if not args.dry_run:
        # sanity: schema parity with source train
        src = pd.read_parquet(os.path.join(args.src, "train_00000_verl.parquet"))
        dst = pd.read_parquet(os.path.join(args.out, "train_00000_verl.parquet"))
        assert list(src.columns) == list(dst.columns), \
            f"column mismatch {list(dst.columns)} vs {list(src.columns)}"
        print(f"columns OK: {list(dst.columns)}")


if __name__ == "__main__":
    main()
