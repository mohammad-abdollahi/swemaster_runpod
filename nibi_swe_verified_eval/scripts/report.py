#!/usr/bin/env python
"""Compute SWE-bench Verified resolve rate (pass@1) from an R2E-Gym trajectory jsonl.

Usage: python scripts/report.py <results_dir>/<exp>/<exp>.jsonl
"""
import json
import sys
from collections import Counter


def main():
    path = sys.argv[1]
    n = resolved = 0
    by_repo = Counter()
    repo_total = Counter()
    unresolved = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            n += 1
            ds = t.get("ds") or {}
            inst = ds.get("instance_id", t.get("instance_id", "?"))
            repo = ds.get("repo", "?")
            r = t.get("reward", 0) or 0
            repo_total[repo] += 1
            if r >= 1:
                resolved += 1
                by_repo[repo] += 1
            else:
                unresolved.append((inst, t.get("num_steps"), t.get("exit_reason")))
    print(f"instances graded : {n}")
    print(f"resolved         : {resolved}")
    print(f"resolve rate@1   : {resolved / n * 100:.2f}%" if n else "n/a")
    print("\nper-repo (resolved/total):")
    for repo in sorted(repo_total):
        print(f"  {repo:30s} {by_repo[repo]}/{repo_total[repo]}")
    if unresolved:
        print(f"\nunresolved ({len(unresolved)}): inst | steps | exit_reason")
        for inst, steps, ex in unresolved[:50]:
            print(f"  {inst}  {steps}  {ex}")


if __name__ == "__main__":
    main()
