#!/usr/bin/env bash
# Run TRL GRPO on the SWE-Master demo RL parquet (the "original" dataset
# format from the paper). Reward is a diff-similarity proxy against the
# gold patch — stand-in until the Apptainer SWE-bench backend is in place.
#
#   bash scripts/00_enter_container.sh bash TRL_RL/scripts/02_train_swe.sh
set -euo pipefail

USER=${USER:-$(whoami)}
SCRATCH=${SCRATCH:-/scratch/$USER}
PROJECT_ROOT=${PROJECT_ROOT:-/project/def-hemmati-ac/$USER/SWE-Master}
VENV=${VENV:-$SCRATCH/swemaster/venv/rllm}

source "$VENV/bin/activate"
cd "$PROJECT_ROOT"

# Thread caps for login-node policy (ulimit -u 1024).
export POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-4}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export WANDB_MODE=${WANDB_MODE:-offline}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Resolve local model snapshot. Default = SWE-Master-4B-SFT (the paper's 4B
# SFT-warmed checkpoint, qwen3 architecture). Override with MODEL_ID=... to
# fall back to the Qwen3-4B base or a different SFT.
HF_HUB="${HUGGINGFACE_HUB_CACHE:-$SCRATCH/swemaster/cache/hf/hub}"
MODEL_ID=${MODEL_ID:-RUC-AIBOX/SWE-Master-4B-SFT}
export HF_HUB MODEL_ID
MODEL_PATH=$(python - <<'PY'
import os, glob
hub = os.environ["HF_HUB"]
repo = os.environ.get("MODEL_ID", "Qwen/Qwen3-4B").replace("/", "--")
snaps = sorted(glob.glob(os.path.join(hub, f"models--{repo}", "snapshots", "*")))
print(snaps[-1] if snaps else os.environ["MODEL_ID"])
PY
)
echo "[swe] MODEL_PATH=$MODEL_PATH"

# Use the in-repo demo parquet unless caller overrides.
SWE_PARQUET=${SWE_PARQUET:-data_examples/rl_data/ood_data/equal_0.3999999_0.9999999_swe-gym_and_rebench_demo_for_RL/train_00000.parquet}
echo "[swe] dataset=$SWE_PARQUET"

# Run identifier
TS=$(date +"%Y-%m-%d_%H-%M-%S")
PROJECT_NAME=${PROJECT_NAME:-"trl-swe"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"qwen3-4b-${TS}"}
RUN_ROOT=${RUN_ROOT:-$SCRATCH/swemaster/runs}
OUTPUT_DIR="${RUN_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}"
LOGGING_DIR="${OUTPUT_DIR}/tb"
mkdir -p "$OUTPUT_DIR" "$LOGGING_DIR"
echo "[swe] OUTPUT_DIR=$OUTPUT_DIR"

# SWE problem statements are long; allow longer prompts/completions.
# generation_batch_size = BATCH_SIZE * grad_accum * world must be divisible by
# NUM_GENERATIONS. The demo parquet has only 3 rows, so we keep batch low.
MAX_STEPS=${MAX_STEPS:-5}
BATCH_SIZE=${BATCH_SIZE:-2}
NUM_GENERATIONS=${NUM_GENERATIONS:-2}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
# Bumped from 512 → 2048: the SFT model (RUC-AIBOX/SWE-Master-4B-SFT) writes
# agentic prose ("let me explore…") before emitting a final diff. With only
# 512 completion tokens it gets truncated before reaching the diff.
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-2048}

python TRL_RL/src/train_grpo.py \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --logging-dir "$LOGGING_DIR" \
    --dataset swe \
    --swe-parquet "$SWE_PARQUET" \
    --max-steps "$MAX_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --num-generations "$NUM_GENERATIONS" \
    --max-prompt-length "$MAX_PROMPT_LENGTH" \
    --max-completion-length "$MAX_COMPLETION_LENGTH" \
    --learning-rate 1e-6 \
    --bf16
