#!/bin/bash
# B1/R12 launcher — one arm of the matched refinement ablation on swerebench_verlive.
# ATS-RL v1 reward EXACTLY (RLLM_ACTIVE_TESTS_MID_EPISODE=0): the refinement tick runs
# only for stall-detection + prompt decomposition, never altering the reward.
#
# Arm is chosen by the REFINE flags the caller exports:
#   baseline (refine off): RLLM_ACTIVE_TESTS_REFINE=0 RLLM_ACTIVE_TESTS_REFINE_TICK=1
#   refine arm:            RLLM_ACTIVE_TESTS_REFINE=1   (REFINE_TICK auto-on)
# Both tick identically (stall detection) and share the v1 reward; they differ ONLY in
# whether the backtranslated decomposition is shown for a stalled test.
set -euo pipefail
cd /home/temp/swemaster_runpod
REPO=/home/temp/swemaster_runpod

EXP=${EXPERIMENT_NAME:?set EXPERIMENT_NAME}
PROJ=${PROJECT_NAME:-ablation_b1_r12}

TRAIN_PARQUET=${TRAIN_PARQUET:-$REPO/data_examples/rl_data/swerebench_verlive/train_00000_verl.parquet} \
VAL_PARQUET=${VAL_PARQUET:-$REPO/data_examples/rl_data/swerebench_verlive/test_00000_verl.parquet} \
MODEL_PATH=${BASE_MODEL:-/home/temp/models/SWE-Master-4B-ATS-B1-r12-smoke4step} \
TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=16 \
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2} \
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-6} \
AGENT_MAX_STEPS=150 TRAJECTORY_TIMEOUT=5400 MAX_RESPONSE=24576 \
GPU_MEM_UTIL=0.30 \
RLLM_MAX_ROLLOUT_CONCURRENCY=8 \
RLLM_ACTIVE_TESTS_ENABLED=1 RLLM_ACTIVE_TESTS_MID_EPISODE=0 \
RLLM_ACTIVE_TESTS_W_SOLVE=10.0 RLLM_ACTIVE_TESTS_K=20 \
RLLM_ACTIVE_TESTS_REFINE=${RLLM_ACTIVE_TESTS_REFINE:-0} \
RLLM_ACTIVE_TESTS_REFINE_TICK=${RLLM_ACTIVE_TESTS_REFINE_TICK:-0} \
RLLM_ACTIVE_TESTS_REFINE_AFTER=${RLLM_ACTIVE_TESTS_REFINE_AFTER:-1} \
RLLM_ACTIVE_TESTS_REFINE_MAX=${RLLM_ACTIVE_TESTS_REFINE_MAX:-4} \
RLLM_ACTIVE_TESTS_REFINE_CACHE=${REFINE_CACHE:-$REPO/block_b_refinement/cache/decompositions.jsonl} \
SAVE_FREQ=1 TEST_FREQ=999 \
EXPERIMENT_NAME="$EXP" PROJECT_NAME="$PROJ" \
bash DeepSWE_RL/rllm/examples/swe/train_active_tests_swe_master_4b_h200.sh
