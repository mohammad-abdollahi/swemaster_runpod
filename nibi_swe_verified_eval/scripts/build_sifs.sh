#!/bin/bash
# Pre-build per-instance SIFs (run on the LOGIN NODE, which has internet).
# Args: START_IDX K   -> builds SIFs for dataset rows [START_IDX, START_IDX+K).
# Skips ones already present. Sequential (shares one apptainer cache safely).
set -uo pipefail
source "$(dirname "$0")/../env.sh"
module load $APPTAINER_MODULE

START_IDX=${1:-0}; K=${2:-5}
PY=/project/6087623/mohammadA/swebench_pro/.venv/bin/python

mapfile -t ROWS < <("$PY" - "$START_IDX" "$K" <<'PY'
import sys
from datasets import load_from_disk
import os
ds = load_from_disk(os.environ["DATASET_DIR"])
s, k = int(sys.argv[1]), int(sys.argv[2])
for i in range(s, min(s + k, len(ds))):
    print(ds[i]["instance_id"], ds[i]["docker_image"])
PY
)

for row in "${ROWS[@]}"; do
  inst=${row%% *}; img=${row#* }
  sif="$SIF_DIR/${inst}.sif"
  if [ -f "$sif" ]; then echo "skip (exists): $inst"; continue; fi
  echo "build: $inst <- $img"
  if apptainer build --force "$sif" "docker://${img}" > "$LOG_DIR/build_${inst}.log" 2>&1; then
    echo "  ok: $(ls -lah "$sif" | awk '{print $5}')"
  else
    echo "  FAILED: see $LOG_DIR/build_${inst}.log"
  fi
done
echo "build_sifs done for [$START_IDX, $((START_IDX+K)))"
