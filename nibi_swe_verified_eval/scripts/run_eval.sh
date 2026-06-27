#!/bin/bash
# Run the R2E-Gym agent (SWE-Master) over SWE-bench Verified via the Apptainer backend.
# Assumes a vLLM server is already serving $SERVED_MODEL_NAME at $OPENAI_API_BASE.
# Args: START_IDX K MAX_WORKERS EXP_NAME   (all optional)
set -uo pipefail
source "$(dirname "$0")/../env.sh"

START_IDX=${1:-0}
K=${2:-5}
MAX_WORKERS=${3:-2}
EXP_NAME=${4:-smoke}

source "$VENV/bin/activate"

# litellm -> local vLLM (OpenAI-compatible)
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}
export OPENAI_API_BASE=${OPENAI_API_BASE:-http://127.0.0.1:${VLLM_PORT}/v1}

# Apptainer backend knobs (read by runtime/apptainer.py)
export SIF_DIR OVERLAY_DIR
export WHEEL_DIR
export APPTAINER_ALLOW_BUILD=${APPTAINER_ALLOW_BUILD:-1}
export APPTAINER_FAKEROOT=1
export APPTAINER_BIN=$(command -v apptainer)

cd "$R2EGYM_DIR"
python src/r2egym/agenthub/run/edit.py runagent_multiple \
  --traj_dir "$RESULTS_DIR/$EXP_NAME" \
  --max_workers "$MAX_WORKERS" \
  --start_idx "$START_IDX" \
  --k "$K" \
  --dataset "$DATASET_DIR" \
  --split "test" \
  --llm_name "${LLM_NAME:-hosted_vllm/swe-master-sft}" \
  --use_fn_calling False \
  --exp_name "$EXP_NAME" \
  --temperature 0.7 \
  --max_steps 200 \
  --max_steps_absolute 200 \
  --max_tokens 120000 \
  --max_reward_calc_time 1800 \
  --backend "apptainer" \
  --prepull_images False \
  --scaffold "openhands" \
  --used_yaml "./src/r2egym/agenthub/config/openhands/openhands_sp_non_fn_calling.yaml"

echo "=== eval finished. results under $RESULTS_DIR/$EXP_NAME ==="
