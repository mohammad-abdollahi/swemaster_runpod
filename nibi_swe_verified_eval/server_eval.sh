#!/bin/bash
#SBATCH --job-name=swemaster_eval
#SBATCH --account=def-hemmati-ac
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=/scratch/moham98/swemaster/logs/job_%j.log

# SWE-Master-4B-SFT × SWE-bench Verified, Docker-free (Apptainer) eval on Nibi.
#   sbatch server_eval.sh                 # smoke: 5 instances, 2 workers
#   sbatch server_eval.sh 0 500 16 full   # full run
set -uo pipefail
echo "Node: $(hostname)  $(date)"

module load gcc python/3.11 arrow apptainer/1.3.5

# 6th positional arg selects the model variant (sft|rl). Set BEFORE sourcing env.sh so
# env.sh resolves MODEL_DIR/SERVED_MODEL_NAME/LLM_NAME accordingly. Positional args are
# stored by SLURM at submit time -> guaranteed correct even after disconnect.
export MODEL_VARIANT="${6:-${MODEL_VARIANT:-sft}}"

EVAL_ROOT=/project/6087623/mohammadA/evaluation
source "$EVAL_ROOT/env.sh"
echo "MODEL_VARIANT=$MODEL_VARIANT"

# node-local fast scratch for apptainer builds/overlays (not Lustre, not quota-counted)
export APPTAINER_TMPDIR="$SLURM_TMPDIR/apptainer_tmp"; mkdir -p "$APPTAINER_TMPDIR"
export HF_HUB_OFFLINE=1
# compute nodes are offline: use only pre-built SIFs; missing ones fail that instance
# gracefully (caught per-instance by edit.py) instead of hanging on a doomed build.
export APPTAINER_ALLOW_BUILD=0
# Unique vLLM port per job: SLURM can place 2 jobs on one multi-GPU node, and a fixed port
# (8000 + internal 8001) collides -> "EADDRINUSE ... Engine core initialization failed".
# Spacing of 8 keeps each job's api/engine/aux ports clear of neighbours.
export VLLM_PORT=$(( 20000 + (SLURM_JOB_ID % 5000) * 8 ))
echo "VLLM_PORT=$VLLM_PORT"

START_IDX=${1:-0}; K=${2:-5}; MAX_WORKERS=${3:-2}; EXP_NAME=${4:-smoke}
WAIT_SIFS=${5:-0}   # if >0, wait (up to ~45min) for at least this many of the K SIFs to exist
                    # before running. Lets the job be submitted while login-node builds finish.

# 1) launch vLLM server in the background (model load overlaps the SIF wait below)
echo "== starting vLLM =="
bash "$EVAL_ROOT/scripts/serve_vllm.sh" > "$LOG_DIR/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

# 2) wait until the server is healthy (model loaded)
echo "== waiting for vLLM on :$VLLM_PORT =="
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "vLLM ready after ${i}0s"; break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then echo "vLLM died; see vllm log"; exit 1; fi
  sleep 10
done

# 2b) optionally wait for login-node SIF builds to reach a threshold (shared filesystem).
if [ "$WAIT_SIFS" -gt 0 ]; then
  echo "== waiting for >=$WAIT_SIFS SIFs (built on login node) =="
  PY=/project/6087623/mohammadA/swebench_pro/.venv/bin/python
  mapfile -t INSTS < <("$PY" - "$START_IDX" "$K" <<'PYEOF'
import sys, os
from datasets import load_from_disk
ds = load_from_disk(os.environ["DATASET_DIR"])
s, k = int(sys.argv[1]), int(sys.argv[2])
for i in range(s, min(s+k, len(ds))): print(ds[i]["instance_id"])
PYEOF
)
  for t in $(seq 1 "${WAIT_SIF_TICKS:-240}"); do  # default ~2h; SIFs build on login node over hours
    have=0
    for inst in "${INSTS[@]}"; do
      f="$SIF_DIR/${inst}.sif"; [ -f "$f" ] && [ "$(stat -c%s "$f" 2>/dev/null)" -gt 10000000 ] && have=$((have+1))
    done
    echo "  SIFs ready: $have/$K"
    [ "$have" -ge "$WAIT_SIFS" ] && break
    sleep 30
  done
fi

# 3) run the agent eval
echo "== running eval: start=$START_IDX k=$K workers=$MAX_WORKERS exp=$EXP_NAME =="
bash "$EVAL_ROOT/scripts/run_eval.sh" "$START_IDX" "$K" "$MAX_WORKERS" "$EXP_NAME"

echo "Done $(date)"
