#!/usr/bin/env bash
# Binary baseline on SWE-World-4B-SFT — matched-compute control for the ATS
# scaled run that just completed (active_tests_swe_world/scaled_sweworld_minftp5).
#
# Identical to run_scaled_swe_world.sh except:
#   - RLLM_ACTIVE_TESTS_ENABLED=0   (legacy int(success) terminal reward)
#   - EXPERIMENT_NAME              (separate output dir for clean comparison)
#   - PROJECT_NAME=baseline_binary_swe_world
#
# Same dataset, same hyperparameters, same agent_max_steps, same train_batch_size,
# same number of PPO updates, same seed (verl's default is fixed across runs
# with identical config). This is the cleanest head-to-head we can produce on a
# single H200.
#
# Expected wallclock: ~80-90 minutes (matches the ATS run).
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
export LR=3e-6
export TRAJECTORY_TIMEOUT=2400
export SAVE_FREQ=4
export TEST_FREQ=999

# The headline ablation: ATS off. All scheduler env vars are ignored when this
# is 0, so binary terminal reward is used (int(all FAIL_TO_PASS resolved)),
# no <active_tests> prompt block is injected, and there is no per-step reward.
export RLLM_ACTIVE_TESTS_ENABLED=0

export EXPERIMENT_NAME="baseline_binary_sweworld_minftp5_$(date +%Y%m%d-%H%M%S)"
export PROJECT_NAME=baseline_binary_swe_world

cd /home/temp/swemaster_runpod
sg docker -c 'bash DeepSWE_RL/rllm/examples/swe/train_active_tests_scaled_4b_h200.sh'
