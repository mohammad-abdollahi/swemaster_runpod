#!/bin/bash
# Parallel per-instance SIF builder for the login node.
# Args: START_IDX K NWORKERS   (default 0 100 6)
# Splits [START,START+K) into NWORKERS contiguous chunks (contiguous => same-repo layer
# reuse within a worker's private cache). Each worker uses its own APPTAINER_CACHEDIR +
# APPTAINER_TMPDIR so concurrent builds never race on the same blob.
set -uo pipefail
source "$(dirname "$0")/../env.sh"
module load $APPTAINER_MODULE

START=${1:-0}; K=${2:-100}; NW=${3:-6}
PY=/project/6087623/mohammadA/swebench_pro/.venv/bin/python

# instance list
mapfile -t ROWS < <("$PY" - "$START" "$K" <<'PY'
import sys, os
from datasets import load_from_disk
ds = load_from_disk(os.environ["DATASET_DIR"])
s, k = int(sys.argv[1]), int(sys.argv[2])
for i in range(s, min(s + k, len(ds))):
    print(ds[i]["instance_id"], ds[i]["docker_image"])
PY
)
N=${#ROWS[@]}
# Hard cap at 2 workers: tmpfs unpack is fast but counts against the 16GB per-user memory
# cgroup; 2 concurrent rootfs (~8-9GB) is safe, 3+ OOM-kills mksquashfs. (NFS tmpdir avoids
# OOM but small-file unpack over NFS is pathologically slow, so tmpfs + cap is the sweet spot.)
[ "$NW" -gt 2 ] && NW=2
echo "building $N SIFs with $NW workers"

build_chunk() {
  local wid=$1; shift
  export APPTAINER_CACHEDIR="$WORK/apptainer_cache_w$wid"
  export APPTAINER_TMPDIR="/tmp/apptmp_w$wid.$USER"   # tmpfs: fast unpack (RAM-bounded by NW cap)
  rm -rf "$APPTAINER_TMPDIR"; mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
  for row in "$@"; do
    local inst=${row%% *}; local img=${row#* }
    local sif="$SIF_DIR/${inst}.sif"
    [ -f "$sif" ] && { echo "[w$wid] skip $inst"; continue; }
    if apptainer build --force "$sif" "docker://${img}" > "$LOG_DIR/build_${inst}.log" 2>&1; then
      echo "[w$wid] ok $inst"
    else
      echo "[w$wid] FAIL $inst (see $LOG_DIR/build_${inst}.log)"
    fi
  done
  rm -rf "$APPTAINER_TMPDIR" 2>/dev/null
}

# contiguous chunks (dataset is repo-sorted => same-repo layer reuse within a worker)
per=$(( (N + NW - 1) / NW ))
for ((w=0; w<NW; w++)); do
  start=$(( w * per )); [ $start -ge $N ] && break
  chunk=("${ROWS[@]:$start:$per}")
  build_chunk "$w" "${chunk[@]}" &
done
wait

# Sequential retry pass (1 build at a time => peak RAM well under the 16GB login cgroup;
# catches the larger images that OOM under parallel tmpfs unpack).
echo "=== sequential retry for missing SIFs ==="
export APPTAINER_CACHEDIR="$WORK/apptainer_cache_seq"
export APPTAINER_TMPDIR="/tmp/apptmp_seq.$USER"   # single build => tmpfs safe under 16GB
rm -rf "$APPTAINER_TMPDIR"; mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
for row in "${ROWS[@]}"; do
  inst=${row%% *}; img=${row#* }
  sif="$SIF_DIR/${inst}.sif"
  [ -f "$sif" ] && [ "$(stat -c%s "$sif" 2>/dev/null)" -gt 10000000 ] && continue
  rm -f "$sif"
  echo "[retry] $inst"
  apptainer build --force "$sif" "docker://${img}" > "$LOG_DIR/build_${inst}.log" 2>&1 \
    && echo "[retry] ok $inst" || echo "[retry] STILL FAILED $inst"
done
rm -rf "$APPTAINER_TMPDIR" 2>/dev/null
echo "ALL SIF BUILDS DONE for [$START,$START+$K). valid sifs: $(ls -la $SIF_DIR/*.sif | awk '$5>10000000' | wc -l)"
