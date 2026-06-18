#!/usr/bin/env python3
"""Offline self-check for OperatorMutator (no Docker).

Verifies: (1) mutants are produced from a realistic gold diff, (2) each mutant differs
from gold by exactly one line, (3) line COUNT is preserved (so the unified diff still
applies), (4) mutants are distinct. Run:
    /home/temp/venv/bin/python3 block_b_refinement/mutation_selfcheck.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "R2E-Gym", "src"))
from r2egym.agenthub.runtime.test_refiner import OperatorMutator  # noqa: E402

GOLD = """diff --git a/pkg/calc.py b/pkg/calc.py
index 1111111..2222222 100644
--- a/pkg/calc.py
+++ b/pkg/calc.py
@@ -1,5 +1,8 @@
 def hashes(algo):
-    return ["md5"]
+    supported = ["md5", "sha1", "sha256"]
+    if algo in supported and algo != "":
+        return algo
+    return supported[0]

 def is_even(n):
-    return n % 2 == 1
+    return n % 2 == 0
"""


def main():
    m = OperatorMutator(GOLD, seed=0)
    edits = m._candidate_edits()
    mutants = m.generate(8)
    print(f"candidate single-token edits: {len(edits)}")
    print(f"distinct mutants generated:   {len(mutants)}")

    gold_lines = GOLD.splitlines()
    ok = True
    for i, mut in enumerate(mutants):
        mut_lines = mut.splitlines()
        if len(mut_lines) != len(gold_lines):
            print(f"  ✗ mutant {i}: line COUNT changed ({len(mut_lines)} vs {len(gold_lines)})")
            ok = False
            continue
        diffs = [j for j in range(len(gold_lines)) if gold_lines[j] != mut_lines[j]]
        if len(diffs) != 1:
            print(f"  ✗ mutant {i}: {len(diffs)} lines changed (expected 1)")
            ok = False
            continue
        j = diffs[0]
        print(f"  ✓ mutant {i}: L{j}: '{gold_lines[j].strip()}' -> '{mut_lines[j].strip()}'")

    assert len(mutants) >= 3, "expected several mutants from this diff"
    assert len(set(mutants)) == len(mutants), "mutants must be distinct"
    assert ok, "some mutants malformed"
    print("\nALL CHECKS PASSED ✅")


if __name__ == "__main__":
    main()
