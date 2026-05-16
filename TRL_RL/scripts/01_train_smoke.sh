#!/usr/bin/env bash
# Run the TRL GRPO smoke training inside the apptainer container.
#
#   bash scripts/00_enter_container.sh bash TRL_RL/scripts/01_train_smoke.sh
set -euo pipefail

USER=${USER:-$(whoami)}
SCRATCH=${SCRATCH:-/scratch/$USER}
PROJECT_ROOT=${PROJECT_ROOT:-/project/def-hemmati-ac/$USER/SWE-Master}
VENV=${VENV:-$SCRATCH/swemaster/venv/rllm}

source "$VENV/bin/activate"
cd "$PROJECT_ROOT"

# Thread caps — still applies even without Ray, since polars/numpy/etc.
# default to one-thread-per-core and we have 192 cores.
export POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-4}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export WANDB_MODE=${WANDB_MODE:-offline}

# Pin to a single GPU; TRL on a 4B model fits comfortably on one H100.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Find the local model snapshot.
HF_HUB="${HUGGINGFACE_HUB_CACHE:-$SCRATCH/swemaster/cache/hf/hub}"
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-4B}
export HF_HUB MODEL_ID
MODEL_PATH=$(python - <<'PY'
import os, glob
hub = os.environ["HF_HUB"]
repo = os.environ.get("MODEL_ID", "Qwen/Qwen3-4B").replace("/", "--")
snaps = sorted(glob.glob(os.path.join(hub, f"models--{repo}", "snapshots", "*")))
print(snaps[-1] if snaps else os.environ["MODEL_ID"])
PY
)
echo "[smoke] MODEL_PATH=$MODEL_PATH"

TS=$(date +"%Y-%m-%d_%H-%M-%S")
PROJECT_NAME=${PROJECT_NAME:-"trl-smoke"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"yes-qwen3-4b-${TS}"}
RUN_ROOT=${RUN_ROOT:-$SCRATCH/swemaster/runs}
OUTPUT_DIR="${RUN_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}"
LOGGING_DIR="${OUTPUT_DIR}/tb"
mkdir -p "$OUTPUT_DIR" "$LOGGING_DIR"
echo "[smoke] OUTPUT_DIR=$OUTPUT_DIR"
echo "[smoke] LOGGING_DIR=$LOGGING_DIR"

# Smaller defaults than DeepSWE_RL — we just want metric lines.
MAX_STEPS=${MAX_STEPS:-5}
# generation_batch_size = batch_size * grad_accum * world. TRL requires it
# divisible by num_generations, so keep BATCH_SIZE >= NUM_GENERATIONS.
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_GENERATIONS=${NUM_GENERATIONS:-4}
DATASET=${DATASET:-yes}

python TRL_RL/src/train_grpo.py \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --logging-dir "$LOGGING_DIR" \
    --dataset "$DATASET" \
    --max-steps "$MAX_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --num-generations "$NUM_GENERATIONS" \
    --max-prompt-length 128 \
    --max-completion-length 64 \
    --learning-rate 1e-6 \
    --bf16
