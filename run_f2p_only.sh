#!/bin/bash
# ============================================================================
# ATS-RL ABLATION — FAIL_TO_PASS-only reward (no PASS_TO_PASS signal)
# Branch: test-selection
# ============================================================================
# Identical to the verlive control / R22 recipe (SWE-Master-4B-SFT, 10 steps,
# batch 16, ATS reward, W_SOLVE=10.0, token cap on) EXCEPT the PASS_TO_PASS
# regression-guard reward is fully removed:
#     w_regress           = 0.0   (terminal regression penalty off)
#     delta_p2p_break     = 0.0   (dense per-tick break penalty off)
#     delta_p2p_recover   = 0.0   (dense per-tick recover bonus off)
# => the policy is rewarded ONLY for solving FAIL_TO_PASS tests; breaking
#    previously-passing (PASS_TO_PASS) tests costs nothing.
#
# This is the single-variable ablation of the regression-guard term. p2p tests
# are still EVALUATED each terminal step (get_f2p_and_p2p_status), so the
# p2p-break rate remains measurable for the comparison — it just no longer
# shapes the reward.
#
# Dataset: swerebench_verlive (UNREFINED — 64 train / 16 val), the same data the
# R22/B2 verlive arm trained on. Use this run as the f2p-only point against the
# pending B0-control-on-verlive baseline.
#
# Launch (detached, survives disconnect):
#   setsid nohup sg docker -c "bash run_f2p_only.sh" > /home/temp/f2p_only.log 2>&1 &
set -euo pipefail
cd /home/temp/swemaster_runpod

EXP=${EXPERIMENT_NAME:-f2p_only_verlive_$(date +%Y%m%d-%H%M%S)}

TRAIN_PARQUET=${TRAIN_PARQUET:-/home/temp/swemaster_runpod/data_examples/rl_data/swerebench_verlive/train_00000_verl.parquet} \
VAL_PARQUET=${VAL_PARQUET:-/home/temp/swemaster_runpod/data_examples/rl_data/swerebench_verlive/test_00000_verl.parquet} \
MODEL_PATH=/home/temp/models/SWE-Master-4B-SFT \
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-10} \
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5} \
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16} PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16} \
AGENT_MAX_STEPS=150 TRAJECTORY_TIMEOUT=5400 MAX_RESPONSE=24576 \
GPU_MEM_UTIL=0.30 \
RLLM_MAX_ROLLOUT_CONCURRENCY=8 \
RLLM_ACTIVE_TESTS_ENABLED=1 RLLM_ACTIVE_TESTS_W_SOLVE=10.0 \
RLLM_ACTIVE_TESTS_W_REGRESS=0.0 \
RLLM_ACTIVE_TESTS_DELTA_P2P_BREAK=0.0 \
RLLM_ACTIVE_TESTS_DELTA_P2P_RECOVER=0.0 \
SAVE_FREQ=1 TEST_FREQ=999 \
EXPERIMENT_NAME="${EXP}" PROJECT_NAME=ablation_test_selection \
bash DeepSWE_RL/rllm/examples/swe/train_active_tests_swe_master_4b_h200.sh
