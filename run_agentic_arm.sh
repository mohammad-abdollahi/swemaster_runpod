#!/bin/bash
# Agentic test-refinement launcher — ONE ATS-RL training run. Branch: test-refinement-agentic.
#
# Same ATS-RL pipeline as the OpenAI arm, but the stall->refine backend is the Claude Code
# agent run headless (`claude -p`) on the user's Max subscription (NO API charge). On stall the
# agent gets the issue + gold patch + original F2P test + DIVERSE prior trajectories, explores a
# clean base_commit checkout, writes refined test(s); a post-processor validates true-F2P in
# clean buggy+fixed containers (gold patch applied internally) before the refined test REPLACES
# the scored test.
#
# Caller exports: BASE_MODEL, TRAIN_PARQUET/VAL_PARQUET, TOTAL_TRAINING_STEPS, TOTAL_EPOCHS,
#                 EXPERIMENT_NAME, PROJECT_NAME, REFINE_CACHE (agentic JSONL).
set -euo pipefail
cd /home/temp/swemaster_runpod
REPO=/home/temp/swemaster_runpod

EXP=${EXPERIMENT_NAME:?set EXPERIMENT_NAME}
PROJ=${PROJECT_NAME:-agentic_refine}

# Locate the claude native binary (ships with the VS Code extension; auths via Max login at
# ~/.claude/.credentials.json — no ANTHROPIC_API_KEY, so no API billing).
CLAUDE_BIN=${RLLM_AGENTIC_CLAUDE_BIN:-$(ls -1 /home/temp/.vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude 2>/dev/null | tail -1)}
CLAUDE_BIN=${CLAUDE_BIN:-$(command -v claude || true)}
if [ -z "$CLAUDE_BIN" ] || [ ! -x "$CLAUDE_BIN" ]; then
  echo "FATAL: claude binary not found/executable ($CLAUDE_BIN). Run 'claude login' / check the extension." >&2
  exit 1
fi
echo "[agentic-arm] using claude: $CLAUDE_BIN"

env \
  HOME=${HOME:-/home/temp} \
  RLLM_AGENTIC_CLAUDE_BIN="$CLAUDE_BIN" \
  TRAIN_PARQUET=${TRAIN_PARQUET:-$REPO/data_examples/rl_data/swerebench_verified_live/train_00000_verl.parquet} \
  VAL_PARQUET=${VAL_PARQUET:-$REPO/data_examples/rl_data/swerebench_verified_live/test_00000_verl.parquet} \
  MODEL_PATH=${BASE_MODEL:-/home/temp/models/SWE-Master-4B-SFT} \
  TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16} PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16} \
  TOTAL_EPOCHS=${TOTAL_EPOCHS:-40} \
  TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-10} \
  AGENT_MAX_STEPS=${AGENT_MAX_STEPS:-150} TRAJECTORY_TIMEOUT=${TRAJECTORY_TIMEOUT:-5400} \
  MAX_RESPONSE=${MAX_RESPONSE:-24576} \
  GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.30} \
  RLLM_MAX_ROLLOUT_CONCURRENCY=${RLLM_MAX_ROLLOUT_CONCURRENCY:-8} \
  RLLM_ACTIVE_TESTS_ENABLED=1 \
  RLLM_ACTIVE_TESTS_REFINE=1 \
  RLLM_ACTIVE_TESTS_AGENTIC_REFINE=1 \
  RLLM_ACTIVE_TESTS_REFINE_AFTER=${RLLM_ACTIVE_TESTS_REFINE_AFTER:-2} \
  RLLM_ACTIVE_TESTS_REFINE_MAX=${RLLM_ACTIVE_TESTS_REFINE_MAX:-4} \
  RLLM_ACTIVE_TESTS_REFINE_CACHE=${REFINE_CACHE:-$REPO/block_b_refinement/cache/agentic_refine.jsonl} \
  RLLM_AGENTIC_REFINE_MODEL=${RLLM_AGENTIC_REFINE_MODEL:-claude-sonnet-4-6} \
  RLLM_AGENTIC_MAX_TRAJ=${RLLM_AGENTIC_MAX_TRAJ:-3} \
  RLLM_AGENTIC_TRAJ_DIVERSITY=${RLLM_AGENTIC_TRAJ_DIVERSITY:-0.30} \
  RLLM_AGENTIC_MAX_TURNS=${RLLM_AGENTIC_MAX_TURNS:-12} \
  RLLM_AGENTIC_REFINE_TIMEOUT=${RLLM_AGENTIC_REFINE_TIMEOUT:-900} \
  RLLM_AGENTIC_MAX_CONCURRENCY=${RLLM_AGENTIC_MAX_CONCURRENCY:-2} \
  RLLM_AGENTIC_REFINE_MAX=${RLLM_ACTIVE_TESTS_REFINE_MAX:-4} \
  RLLM_AGENTIC_TRAJ_DIR=${RLLM_AGENTIC_TRAJ_DIR:-$REPO/block_b_refinement/cache/agentic_traj} \
  SAVE_FREQ=${SAVE_FREQ:-1} TEST_FREQ=${TEST_FREQ:-999} \
  EXPERIMENT_NAME="$EXP" PROJECT_NAME="$PROJ" \
  bash DeepSWE_RL/rllm/examples/swe/train_active_tests_swe_master_4b_h200.sh
