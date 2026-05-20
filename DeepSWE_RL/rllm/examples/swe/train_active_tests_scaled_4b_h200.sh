#!/usr/bin/env bash
# Active-Test-Selection scaled training — overnight run on a single H200.
#
# Differences from train_active_tests_swe_master_4b_h200.sh smoke defaults:
#   - AGENT_MAX_STEPS=50  (vs 20 smoke; tasks need more turns to actually
#                          solve; the smoke saw 100% MAX_STEPS terminations)
#   - W_SOLVE=10.0        (vs 5.0; bigger jump when a test passes, so the
#                          within-group variance dominates token noise)
#   - LR=3e-6             (vs 1e-6; faster updates per step)
#   - TOTAL_TRAINING_STEPS=8 (vs 4 smoke; need more PPO updates to see
#                              trend across reward curve)
#   - MAX_RESPONSE=16384
#   - TRAJECTORY_TIMEOUT=2400 (40 min/traj cap)
#
# Expected wallclock: ~5 hours for 8 PPO steps on the H200 (1 step ~ 35-45 min
# = 4 batches × 16 traj/batch × ~10 min/traj × concurrency 4, with PPO update).
set -x

# Tune key knobs first (env-var override-able).
export AGENT_MAX_STEPS=${AGENT_MAX_STEPS:-50}
export RLLM_ACTIVE_TESTS_W_SOLVE=${RLLM_ACTIVE_TESTS_W_SOLVE:-10.0}
export MAX_RESPONSE=${MAX_RESPONSE:-16384}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-8}
export TRAJECTORY_TIMEOUT=${TRAJECTORY_TIMEOUT:-2400}
export SAVE_FREQ=${SAVE_FREQ:-4}
export TEST_FREQ=${TEST_FREQ:-999}
export PROJECT_NAME=${PROJECT_NAME:-active_tests_swe_master}
TS=$(date +"%Y-%m-%d_%H-%M-%S")
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-scaled_${TS}}

# Higher learning rate than the smoke (override the inner script's 1e-6
# via an env var; we patch the script value below).
export LR=${LR:-3e-6}

REPO_DIR=${REPO_DIR:-/home/temp/swemaster_runpod}
SCRIPT="${REPO_DIR}/DeepSWE_RL/rllm/examples/swe/train_active_tests_swe_master_4b_h200.sh"

# The inner script bakes in `actor_rollout_ref.actor.optim.lr=1e-6`. We sed
# it on-the-fly in a temp copy so we don't perma-edit the smoke script.
TMP_SCRIPT=$(mktemp /tmp/train_active_scaled.XXXXXX.sh)
sed "s|actor_rollout_ref.actor.optim.lr=1e-6|actor_rollout_ref.actor.optim.lr=${LR}|" "${SCRIPT}" > "${TMP_SCRIPT}"
chmod +x "${TMP_SCRIPT}"
echo "Launching scaled run via temp script: ${TMP_SCRIPT}"
bash "${TMP_SCRIPT}"
