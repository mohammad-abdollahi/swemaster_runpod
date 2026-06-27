#!/bin/bash
#SBATCH --job-name=diag_net
#SBATCH --account=def-hemmati-ac
#SBATCH --qos=normal
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/moham98/swemaster/logs/diag_%j.log
set -uo pipefail
echo "NODE=$(hostname)  SLURM_TMPDIR=$SLURM_TMPDIR"
module load apptainer/1.3.5
source /project/6087623/mohammadA/evaluation/env.sh

echo "=== 1) internet from compute node? ==="
timeout 15 curl -sI https://registry-1.docker.io/v2/ 2>&1 | head -2 || echo "NO HTTPS to dockerhub"
timeout 15 curl -sI https://pypi.org 2>&1 | head -1 || echo "NO HTTPS to pypi"

echo "=== 2) SLURM_TMPDIR fs + node mem ==="
df -hT "$SLURM_TMPDIR" 2>/dev/null | tail -1
free -g | awk '/Mem:/{print $7" GB avail / "$2" GB total"}'

echo "=== 3) build one SIF on the compute node (local SLURM_TMPDIR tmpdir) ==="
export APPTAINER_TMPDIR="$SLURM_TMPDIR/apptmp"; mkdir -p "$APPTAINER_TMPDIR"
export APPTAINER_CACHEDIR="$SLURM_TMPDIR/acache"; mkdir -p "$APPTAINER_CACHEDIR"
IMG="docker://slimshetty/swebench-verified:sweb.eval.x86_64.sympy__sympy-13895"
t0=$(date +%s)
if apptainer build --force "$SLURM_TMPDIR/test.sif" "$IMG" > "$LOG_DIR/diag_build.log" 2>&1; then
  echo "BUILD OK on compute node in $(( $(date +%s) - t0 ))s  size=$(du -h $SLURM_TMPDIR/test.sif|cut -f1)"
else
  echo "BUILD FAILED on compute node (see diag_build.log; likely offline)"; tail -3 "$LOG_DIR/diag_build.log"
fi
echo "DIAG DONE"
