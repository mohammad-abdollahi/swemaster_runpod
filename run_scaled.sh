#!/usr/bin/env bash
# Launch the scaled ATS training run on the MIN_FTP=5 parquet.
#
# Triggered manually by Claude once the smoke run completes. The smoke
# verifies the pipeline + first-update gradient signal; the scaled run
# demonstrates trend across 8 PPO steps on a richer dataset (16 tasks,
# mean FAIL_TO_PASS = 11.4 vs the smoke's 3.8).
#
# Settings:
#   - parquet:        swerebench_min_ftp5 (16 train + 4 val)
#   - PPO steps:      8 (cap via TOTAL_TRAINING_STEPS)
#   - rollouts/step:  4 × 4 = 16 trajectories
#   - agent steps:    50 (vs 20 smoke; more chances to actually solve)
#   - max_response:   16384 (vs 12288 smoke)
#   - w_solve:        10.0 (vs 5.0 smoke; bigger reward jump per solve)
#   - lr:             3e-6 (vs 1e-6 smoke)
#   - trajectory_timeout: 2400s (40 min) per traj
#
# Expected wallclock: ~6-8 hours (n_total varies, agent_max_steps × concurrency
# = main wall-clock driver).
set -e

export TRAIN_PARQUET=/home/temp/swemaster_runpod/data_examples/rl_data/swerebench_min_ftp5/train_00000_verl.parquet
export VAL_PARQUET=/home/temp/swemaster_runpod/data_examples/rl_data/swerebench_min_ftp5/test_00000_verl.parquet
export TRAIN_BATCH_SIZE=4
export PPO_MINI_BATCH_SIZE=4
export TOTAL_TRAINING_STEPS=8
export TOTAL_EPOCHS=3
export AGENT_MAX_STEPS=50
export MAX_RESPONSE=16384
export RLLM_ACTIVE_TESTS_W_SOLVE=10.0
export LR=3e-6
export TRAJECTORY_TIMEOUT=2400
export SAVE_FREQ=4
export TEST_FREQ=999
export EXPERIMENT_NAME="scaled_minftp5_$(date +%Y%m%d-%H%M%S)"
export PROJECT_NAME=active_tests_swe_master

cd /home/temp/swemaster_runpod
sg docker -c 'bash DeepSWE_RL/rllm/examples/swe/train_active_tests_scaled_4b_h200.sh'
