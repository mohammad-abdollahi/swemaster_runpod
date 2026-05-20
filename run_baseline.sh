#!/usr/bin/env bash
# Binary-baseline comparison run — identical config to run_scaled.sh except
# RLLM_ACTIVE_TESTS_ENABLED=0 (legacy binary terminal reward).
#
# Used for the headline ATS-vs-binary comparison in paper §5.4. Same dataset,
# same model, same optimizer, same PPO step count, same n=4 LOOP rollouts.
# Only the reward function differs.
set -e

export TRAIN_PARQUET=/home/temp/swemaster_runpod/data_examples/rl_data/swerebench_min_ftp5/train_00000_verl.parquet
export VAL_PARQUET=/home/temp/swemaster_runpod/data_examples/rl_data/swerebench_min_ftp5/test_00000_verl.parquet
export TRAIN_BATCH_SIZE=4
export PPO_MINI_BATCH_SIZE=4
export TOTAL_TRAINING_STEPS=8
export TOTAL_EPOCHS=3
export AGENT_MAX_STEPS=50
export MAX_RESPONSE=16384
export RLLM_ACTIVE_TESTS_ENABLED=0    # <-- the only difference vs run_scaled.sh
export LR=3e-6
export TRAJECTORY_TIMEOUT=2400
export SAVE_FREQ=4
export TEST_FREQ=999
export EXPERIMENT_NAME="baseline_binary_$(date +%Y%m%d-%H%M%S)"
export PROJECT_NAME=active_tests_swe_master

cd /home/temp/swemaster_runpod
sg docker -c 'bash DeepSWE_RL/rllm/examples/swe/train_active_tests_scaled_4b_h200.sh'
