#!/bin/bash
# ATS-RL CONTROL arm on the live64 dataset (branch ablation/control-live64).
# This is the frozen baseline every Block-A/B/C ablation arm is measured against.
# = the validated paper-matched ATS-RL pipeline (literal current reward; token
# penalty capped at 3.0 so the -989 outlier is impossible) on live64.
#
# Base model: RUCAIBox/SWE-Master-4B-SFT (/home/temp/models/SWE-Master-4B-SFT).
# All hard-won-lesson settings baked in: save_freq=1, max_steps=150, batch=16,
# trajectory_timeout=5400, token cap on.
set -euo pipefail
cd /home/temp/swemaster_runpod

EXP=${EXPERIMENT_NAME:-control_live64_$(date +%Y%m%d-%H%M%S)}
LOG=/home/temp/control_live64.log

TRAIN_PARQUET=/home/temp/swemaster_runpod/data_examples/rl_data/swerebench_live64/train_00000_verl.parquet \
VAL_PARQUET=/home/temp/swemaster_runpod/data_examples/rl_data/swerebench_live64/test_00000_verl.parquet \
MODEL_PATH=/home/temp/models/SWE-Master-4B-SFT \
TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=16 \
TOTAL_EPOCHS=5 \
AGENT_MAX_STEPS=150 TRAJECTORY_TIMEOUT=5400 MAX_RESPONSE=24576 \
GPU_MEM_UTIL=0.30 \
RLLM_MAX_ROLLOUT_CONCURRENCY=8 \
RLLM_ACTIVE_TESTS_ENABLED=1 RLLM_ACTIVE_TESTS_W_SOLVE=10.0 \
SAVE_FREQ=1 TEST_FREQ=999 \
EXPERIMENT_NAME="${EXP}" PROJECT_NAME=ablation_control \
bash DeepSWE_RL/rllm/examples/swe/train_active_tests_swe_master_4b_h200.sh
