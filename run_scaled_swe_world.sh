#!/usr/bin/env bash
# Launch the ATS scaled training run on SWE-World-4B-SFT (the base the user
# explicitly requested). Mirrors run_scaled.sh but flips the policy to
# RUC-AIBOX/SWE-World-4B-SFT (same Qwen3-4B architecture as SWE-Master-4B-SFT,
# so the training config carries over byte-for-byte).
#
# Settings (matched to the prior SWE-Master scaled config so results are
# directly comparable):
#   - parquet:        swerebench_min_ftp5 (16 train + 4 val, mean |F|=11.4)
#   - PPO steps:      8 (via TOTAL_TRAINING_STEPS)
#   - rollouts/step:  4 × 4 = 16 trajectories
#   - agent steps:    50
#   - max_response:   16384
#   - w_solve:        10.0
#   - lr:             3e-6
#   - trajectory_timeout: 2400s
#
# Expected wallclock: ~6 hours.
set -e

export MODEL_PATH=/home/temp/models/SWE-World-4B-SFT

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
export EXPERIMENT_NAME="scaled_sweworld_minftp5_$(date +%Y%m%d-%H%M%S)"
export PROJECT_NAME=active_tests_swe_world

cd /home/temp/swemaster_runpod
sg docker -c 'bash DeepSWE_RL/rllm/examples/swe/train_active_tests_scaled_4b_h200.sh'
