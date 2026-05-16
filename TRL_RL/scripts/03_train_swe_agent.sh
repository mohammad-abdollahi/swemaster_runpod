#!/usr/bin/env bash
# Multi-turn agent GRPO smoke training on the SWE-Master demo parquet.
# Uses SweSandbox + agent_rollout (tool calls against local repo clones)
# instead of TRL's single-shot GRPOTrainer.
#
#   bash scripts/00_enter_container.sh bash TRL_RL/scripts/03_train_swe_agent.sh
set -euo pipefail

USER=${USER:-$(whoami)}
SCRATCH=${SCRATCH:-/scratch/$USER}
PROJECT_ROOT=${PROJECT_ROOT:-/project/def-hemmati-ac/$USER/SWE-Master}
VENV=${VENV:-$SCRATCH/swemaster/venv/rllm}

source "$VENV/bin/activate"
cd "$PROJECT_ROOT"

# Thread caps for login-node ulimit -u 1024.
export POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-4}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export WANDB_MODE=${WANDB_MODE:-offline}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Resolve local model snapshot. Default = SWE-Master-4B-SFT.
HF_HUB="${HUGGINGFACE_HUB_CACHE:-$SCRATCH/swemaster/cache/hf/hub}"
MODEL_ID=${MODEL_ID:-RUC-AIBOX/SWE-Master-4B-SFT}
export HF_HUB MODEL_ID
MODEL_PATH=$(python - <<'PY'
import os, glob
hub = os.environ["HF_HUB"]
repo = os.environ.get("MODEL_ID").replace("/", "--")
snaps = sorted(glob.glob(os.path.join(hub, f"models--{repo}", "snapshots", "*")))
print(snaps[-1] if snaps else os.environ["MODEL_ID"])
PY
)
echo "[agent] MODEL_PATH=$MODEL_PATH"

SWE_PARQUET=${SWE_PARQUET:-data_examples/rl_data/ood_data/equal_0.3999999_0.9999999_swe-gym_and_rebench_demo_for_RL/train_00000.parquet}
echo "[agent] dataset=$SWE_PARQUET"

# Run id
TS=$(date +"%Y-%m-%d_%H-%M-%S")
PROJECT_NAME=${PROJECT_NAME:-"trl-swe-agent"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"swe-master-4b-sft-${TS}"}
RUN_ROOT=${RUN_ROOT:-$SCRATCH/swemaster/runs}
OUTPUT_DIR="${RUN_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}"
LOGGING_DIR="${OUTPUT_DIR}/tb"
mkdir -p "$OUTPUT_DIR" "$LOGGING_DIR"
echo "[agent] OUTPUT_DIR=$OUTPUT_DIR"

# Multi-turn = each rollout has up to max-turns assistant generations of
# max-gen-tokens each, so total assistant tokens per rollout can be larger
# than the single-shot setup.  Keep batch + num-gen small for the smoke.
MAX_STEPS=${MAX_STEPS:-3}
BATCH_SIZE=${BATCH_SIZE:-2}
NUM_GENERATIONS=${NUM_GENERATIONS:-2}
# Was 4 — the SFT model methodically reads 4-6 files before patching, so 4
# turns was barely past "explore" and never reached apply_patch. Bumped to
# give space for at least 8 exploration turns + 1 patch + 1 submit.
MAX_TURNS=${MAX_TURNS:-15}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-12288}
# Was 512 — an apply_patch with a real diff easily exceeds that, leaving
# the rollout stranded with a truncated, unclosed <tool> tag.
MAX_GEN_TOKENS=${MAX_GEN_TOKENS:-1024}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
# 'proxy' = diff-similarity (no env). 'test' = paper-faithful binary
# SWE-bench reward (run TRL_RL/scripts/04_setup_test_envs.sh first).
REWARD=${REWARD:-proxy}
TEST_TIMEOUT=${TEST_TIMEOUT:-600}

export SWE_TEST_ENVS_ROOT=${SWE_TEST_ENVS_ROOT:-$SCRATCH/swemaster/test_envs}
export SWE_REPOS_ROOT=${SWE_REPOS_ROOT:-$SCRATCH/swemaster/repos}

python TRL_RL/src/train_agent_grpo.py \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --logging-dir "$LOGGING_DIR" \
    --swe-parquet "$SWE_PARQUET" \
    --max-steps "$MAX_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --num-generations "$NUM_GENERATIONS" \
    --max-turns "$MAX_TURNS" \
    --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
    --max-gen-tokens "$MAX_GEN_TOKENS" \
    --learning-rate "$LEARNING_RATE" \
    --reward "$REWARD" \
    --test-timeout "$TEST_TIMEOUT" \
    --bf16
