#!/bin/bash
# Test-to-OpenAI launcher — ONE training phase of the original ATS-RL pipeline.
# Branch: test-refinement-test-to-openai.
#
# This is the ORIGINAL ATS-RL pipeline (REFINE=0, MID_EPISODE=0 — both train-script
# defaults), NOT the prompt-refinement arms. The only non-default things we set are
# the dataset, base model, step budget, and (phase-1 only) the trajectory dump dir.
#
# Caller exports:
#   BASE_MODEL              base weights (phase1: SFT; phase2: phase-1 export)
#   TRAIN_PARQUET/VAL_PARQUET
#   TOTAL_TRAINING_STEPS    (phase1: 70, phase2: 10)
#   TOTAL_EPOCHS            (must allow the step budget; 32 train/batch16 = 2 steps/epoch)
#   RLLM_DUMP_TRAJ_DIR      (phase1 only: where to dump trajectory+passmap per instance)
#   EXPERIMENT_NAME PROJECT_NAME
set -euo pipefail
cd /home/temp/swemaster_runpod
REPO=/home/temp/swemaster_runpod

EXP=${EXPERIMENT_NAME:?set EXPERIMENT_NAME}
PROJ=${PROJECT_NAME:-test_to_openai}

# NB: we deliberately DO NOT set RLLM_ACTIVE_TESTS_W_SOLVE / _K / _MID_EPISODE /
# _REFINE here, so the train script's ORIGINAL ATS-RL defaults stand
# (W_SOLVE=5.0, K=5, MID_EPISODE=0, REFINE=0). That keeps phase-1 = original pipeline.
env \
  ${RLLM_DUMP_TRAJ_DIR:+RLLM_DUMP_TRAJ_DIR="$RLLM_DUMP_TRAJ_DIR"} \
  RLLM_DUMP_TRAJ_OBS_CAP=${RLLM_DUMP_TRAJ_OBS_CAP:-2000} \
  TRAIN_PARQUET=${TRAIN_PARQUET:-$REPO/data_examples/rl_data/swerebench_comprehensive/train_00000_verl.parquet} \
  VAL_PARQUET=${VAL_PARQUET:-$REPO/data_examples/rl_data/swerebench_comprehensive/test_00000_verl.parquet} \
  MODEL_PATH=${BASE_MODEL:-/home/temp/models/SWE-Master-4B-SFT} \
  TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16} PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16} \
  TOTAL_EPOCHS=${TOTAL_EPOCHS:-40} \
  TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-70} \
  AGENT_MAX_STEPS=${AGENT_MAX_STEPS:-150} TRAJECTORY_TIMEOUT=${TRAJECTORY_TIMEOUT:-5400} \
  MAX_RESPONSE=${MAX_RESPONSE:-24576} \
  GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.30} \
  RLLM_MAX_ROLLOUT_CONCURRENCY=${RLLM_MAX_ROLLOUT_CONCURRENCY:-8} \
  RLLM_ACTIVE_TESTS_ENABLED=1 \
  RLLM_ACTIVE_TESTS_REFINE=${RLLM_ACTIVE_TESTS_REFINE:-0} \
  RLLM_ACTIVE_TESTS_ONLINE_REFINE=${RLLM_ACTIVE_TESTS_ONLINE_REFINE:-0} \
  RLLM_ACTIVE_TESTS_REFINE_AFTER=${RLLM_ACTIVE_TESTS_REFINE_AFTER:-2} \
  RLLM_ACTIVE_TESTS_REFINE_MAX=${RLLM_ACTIVE_TESTS_REFINE_MAX:-4} \
  RLLM_ACTIVE_TESTS_REFINE_CACHE=${REFINE_CACHE:-$REPO/block_b_refinement/cache/online_refine.jsonl} \
  RLLM_ONLINE_REFINE_MODEL=${RLLM_ONLINE_REFINE_MODEL:-gpt-5.4-mini} \
  RLLM_ONLINE_REFINE_MAX=${RLLM_ACTIVE_TESTS_REFINE_MAX:-4} \
  SAVE_FREQ=${SAVE_FREQ:-1} TEST_FREQ=${TEST_FREQ:-999} \
  EXPERIMENT_NAME="$EXP" PROJECT_NAME="$PROJ" \
  bash DeepSWE_RL/rllm/examples/swe/train_active_tests_swe_master_4b_h200.sh
