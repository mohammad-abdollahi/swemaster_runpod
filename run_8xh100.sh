#!/usr/bin/env bash
# =============================================================================
# One-command launcher for SWE-Master-4B RL on 8x H100.  Branch: train-8-gpus.
# =============================================================================
# Wraps train_active_tests_swe_master_4b_8xh100.sh and selects the reward arm:
#
#   REFINE_MODE=none     -> plain Active-Test-Selection RL (baseline; no refine)
#   REFINE_MODE=agentic  -> ATS-RL + Claude Code (`claude -p`) test refinement   [default]
#   REFINE_MODE=online   -> ATS-RL + OpenAI test refinement (needs OPENAI_API_KEY)
#
# Usage (from repo root):
#   BASE_MODEL=... TRAIN_PARQUET=... VAL_PARQUET=... EXPERIMENT_NAME=run1 \
#     bash run_8xh100.sh
# All train knobs (N_GPUS, TRAIN_BATCH_SIZE, MAX_RESPONSE, STEPS via
# TOTAL_TRAINING_STEPS, LR, ...) pass straight through to the launcher.
# See README_8XH100.md.
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=${REPO_DIR:-${_SCRIPT_DIR}}
cd "${REPO_DIR}"

TRAIN_SH=${TRAIN_SH:-DeepSWE_RL/rllm/examples/swe/train_active_tests_swe_master_4b_8xh100.sh}
REFINE_MODE=${REFINE_MODE:-agentic}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:?set EXPERIMENT_NAME (unique per run)}
export PROJECT_NAME=${PROJECT_NAME:-active_tests_swe_master_8xh100}

# ---- reward-arm wiring --------------------------------------------------
case "${REFINE_MODE}" in
  none)
    export RLLM_ACTIVE_TESTS_ENABLED=1
    export RLLM_ACTIVE_TESTS_REFINE=0
    export RLLM_ACTIVE_TESTS_AGENTIC_REFINE=0
    echo "[run] REFINE_MODE=none  (plain ATS-RL baseline)"
    ;;
  agentic)
    export RLLM_ACTIVE_TESTS_ENABLED=1
    export RLLM_ACTIVE_TESTS_REFINE=1
    export RLLM_ACTIVE_TESTS_AGENTIC_REFINE=1
    export RLLM_ACTIVE_TESTS_REFINE_AFTER=${RLLM_ACTIVE_TESTS_REFINE_AFTER:-6}
    export RLLM_ACTIVE_TESTS_REFINE_MAX=${RLLM_ACTIVE_TESTS_REFINE_MAX:-4}
    export RLLM_AGENTIC_REFINE_MODEL=${RLLM_AGENTIC_REFINE_MODEL:-claude-sonnet-4-6}
    export RLLM_AGENTIC_MAX_TRAJ=${RLLM_AGENTIC_MAX_TRAJ:-3}
    export RLLM_AGENTIC_TRAJ_DIVERSITY=${RLLM_AGENTIC_TRAJ_DIVERSITY:-0.30}
    export RLLM_AGENTIC_MAX_TURNS=${RLLM_AGENTIC_MAX_TURNS:-12}
    export RLLM_AGENTIC_REFINE_TIMEOUT=${RLLM_AGENTIC_REFINE_TIMEOUT:-900}
    export RLLM_AGENTIC_MAX_CONCURRENCY=${RLLM_AGENTIC_MAX_CONCURRENCY:-4}
    export RLLM_AGENTIC_REFINE_MAX=${RLLM_ACTIVE_TESTS_REFINE_MAX}
    export RLLM_AGENTIC_TRAJ_DIR=${RLLM_AGENTIC_TRAJ_DIR:-${REPO_DIR}/block_b_refinement/cache/agentic_traj}
    export RLLM_ACTIVE_TESTS_REFINE_CACHE=${REFINE_CACHE:-${REPO_DIR}/block_b_refinement/cache/agentic_refine_8xh100.jsonl}
    # locate the claude native binary (ships with the VS Code extension) unless overridden
    CLAUDE_BIN=${RLLM_AGENTIC_CLAUDE_BIN:-$(ls -1 "${HOME}"/.vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude 2>/dev/null | tail -1 || true)}
    CLAUDE_BIN=${CLAUDE_BIN:-$(command -v claude || true)}
    if [ -z "${CLAUDE_BIN}" ] || [ ! -x "${CLAUDE_BIN}" ]; then
      echo "FATAL: REFINE_MODE=agentic but no usable 'claude' binary found." >&2
      echo "       Install/login Claude Code on this box, or set RLLM_AGENTIC_CLAUDE_BIN=/path/to/claude," >&2
      echo "       or use REFINE_MODE=none for the baseline." >&2
      exit 1
    fi
    export RLLM_AGENTIC_CLAUDE_BIN="${CLAUDE_BIN}"
    echo "[run] REFINE_MODE=agentic  claude=${CLAUDE_BIN}  refine_after=${RLLM_ACTIVE_TESTS_REFINE_AFTER}"
    ;;
  online)
    export RLLM_ACTIVE_TESTS_ENABLED=1
    export RLLM_ACTIVE_TESTS_REFINE=1
    export RLLM_ACTIVE_TESTS_AGENTIC_REFINE=0
    export RLLM_ACTIVE_TESTS_REFINE_AFTER=${RLLM_ACTIVE_TESTS_REFINE_AFTER:-6}
    : "${OPENAI_API_KEY:?REFINE_MODE=online needs OPENAI_API_KEY}"
    echo "[run] REFINE_MODE=online  (OpenAI refinement)"
    ;;
  *)
    echo "FATAL: unknown REFINE_MODE='${REFINE_MODE}' (use none|agentic|online)" >&2; exit 1;;
esac

echo "[run] launching ${TRAIN_SH}"
exec bash "${TRAIN_SH}"
