"""Shared helpers for the Block-A offline selection transforms.

Each select_*.py reads (1) a source verl parquet and (2) the probe CSV from
probe_instances.py, scores every instance, then writes a FILTERED parquet whose
rows are a subset/re-weighting of the source — the trainer is unchanged, so the
ONLY thing that differs between Block-A arms is the training set (cleanest
possible ablation, IMPLEMENTATION_PLAN §4).
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import pandas as pd


def load_probe(path):
    """probe CSV -> {instance_id: {frac_solved_mean, reward_var, tokens_mean, ...}}."""
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out[r["instance_id"]] = {
                "frac_solved_mean": float(r["frac_solved_mean"]),
                "frac_solved_list": json.loads(r["frac_solved_list"]),
                "reward_mean": float(r["reward_mean"]),
                "reward_var": float(r["reward_var"]),
                "reward_list": json.loads(r["reward_list"]),
                "tokens_mean": float(r.get("tokens_mean", 0) or 0),
            }
    return out


def parquet_instance_id(row):
    return json.loads(row["extra_info"])["instance_id"]


def write_filtered(src_df, keep_ids, out_path, audit=None):
    """Write the subset of src_df whose instance_id is in keep_ids (order
    preserved by keep_ids when it is a list; else source order)."""
    by_id = {}
    for _, row in src_df.iterrows():
        by_id.setdefault(parquet_instance_id(row), row)
    if isinstance(keep_ids, list):
        rows = [by_id[i] for i in keep_ids if i in by_id]
    else:
        rows = [r for i, r in by_id.items() if i in keep_ids]
    out_df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path)
    if audit is not None:
        Path(str(out_path) + ".audit.json").write_text(json.dumps(audit, indent=2))
    print(f"[select] wrote {len(out_df)}/{len(src_df)} instances -> {out_path}")
    return out_df


def report(scored, key, n=999):
    """Print a ranked table (instance_id, score, mean solve, var, tokens)."""
    print(f"{'instance_id':46s} {'score':>9} {'solve':>6} {'var':>8} {'tokens':>8}")
    for iid, s in sorted(scored.items(), key=lambda kv: -kv[1][key])[:n]:
        print(f"{iid:46s} {s[key]:9.4f} {s.get('frac_solved_mean',0):6.2f} "
              f"{s.get('reward_var',0):8.3f} {s.get('tokens_mean',0):8.0f}")
