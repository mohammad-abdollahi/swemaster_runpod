#!/bin/bash
# Save R2E-Gym/SWE-Bench-Verified to disk (has docker_image, parsed_commit, run_tests columns
# the R2E-Gym runtime requires). Run on the LOGIN NODE.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

PY=/project/6087623/mohammadA/swebench_pro/.venv/bin/python
"$PY" - <<PY
from datasets import load_dataset
ds = load_dataset("R2E-Gym/SWE-Bench-Verified", split="test")
assert len(ds) == 500, len(ds)
for col in ("docker_image","parsed_commit","run_tests","instance_id","repo","version",
            "FAIL_TO_PASS","PASS_TO_PASS","problem_statement"):
    assert col in ds.column_names, f"missing column {col}"
ds.save_to_disk("$DATASET_DIR")
print("saved", len(ds), "instances to", "$DATASET_DIR")
PY
echo "Dataset staged at $DATASET_DIR"
