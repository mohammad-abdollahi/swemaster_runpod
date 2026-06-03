#!/usr/bin/env bash
# Paper-faithful ATS-RL run on SWE-Master-4B-SFT — single H200.
#
# Differences from train_active_tests_swe_master_4b_h200.sh:
#   1. 320-task MIN_FTP=5 parquet × 5 epochs ≈ 100 PPO steps (paper-matched).
#   2. save_freq=1 → FSDP checkpoint after every PPO step (memory rule).
#   3. max_actor_ckpt_to_keep=2 → rolling FSDP cleanup; peak ~190 GB.
#   4. test_freq=-1 → no in-training validation; user runs val post-hoc.
#   5. Output goes to /home/temp/training_outputs/active_tests_50ep/<ts>.
#      Final HF export will live at /home/temp/models/SWE-Master-4B-ATS-RL-50ep.
#   6. Metrics: verl logs every step → 100 datapoints for plots (NOT 5).
#      A sidecar `live_metrics_csv.sh` watcher keeps a refreshed CSV on disk.
#   7. HF milestones: epoch-boundary saves (5 copies × 7.5 GB), via separate
#      auto_export_milestones.sh watcher.
#
# Paper-faithful agent: agent.max_steps=150, trajectory_timeout=5400,
# max_response=24576, train_batch_size=16, rollout.n=4.
set -x

# ---- vLLM / Ray base ---------------------------------------------------
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=${VLLM_ENGINE_ITERATION_TIMEOUT_S:-100000000000}
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RLLM_USE_PARENT_VENV=1
export RLLM_PROJECT_ROOT=/home/temp/swemaster_runpod
export RLLM_MAX_ROLLOUT_CONCURRENCY=${RLLM_MAX_ROLLOUT_CONCURRENCY:-16}  # raised 12→16 (CPU RAM ~110/196 GB; ~33% more parallel rollouts)

# ---- Active-Test-Selection scheduler -----------------------------------
export RLLM_ACTIVE_TESTS_ENABLED=${RLLM_ACTIVE_TESTS_ENABLED:-1}
export RLLM_ACTIVE_TESTS_MID_EPISODE=${RLLM_ACTIVE_TESTS_MID_EPISODE:-0}
export RLLM_ACTIVE_TESTS_K=${RLLM_ACTIVE_TESTS_K:-5}
export RLLM_ACTIVE_TESTS_COOLDOWN=${RLLM_ACTIVE_TESTS_COOLDOWN:-10}
export RLLM_ACTIVE_TESTS_M_INIT=${RLLM_ACTIVE_TESTS_M_INIT:-2}
export RLLM_ACTIVE_TESTS_M_MAX=${RLLM_ACTIVE_TESTS_M_MAX:-6}
export RLLM_ACTIVE_TESTS_EXPAND=${RLLM_ACTIVE_TESTS_EXPAND:-2}
export RLLM_ACTIVE_TESTS_ALPHA=${RLLM_ACTIVE_TESTS_ALPHA:-1.0}
export RLLM_ACTIVE_TESTS_BETA=${RLLM_ACTIVE_TESTS_BETA:-0.2}
export RLLM_ACTIVE_TESTS_DELTA=${RLLM_ACTIVE_TESTS_DELTA:-0.5}
export RLLM_ACTIVE_TESTS_GAMMA=${RLLM_ACTIVE_TESTS_GAMMA:-0.01}
export RLLM_ACTIVE_TESTS_ETA=${RLLM_ACTIVE_TESTS_ETA:-0.05}
export RLLM_ACTIVE_TESTS_W_SOLVE=${RLLM_ACTIVE_TESTS_W_SOLVE:-5.0}
export RLLM_ACTIVE_TESTS_LAM_TOKEN=${RLLM_ACTIVE_TESTS_LAM_TOKEN:-0.1}
export RLLM_ACTIVE_TESTS_TOKEN_REF=${RLLM_ACTIVE_TESTS_TOKEN_REF:-32000.0}
export RLLM_ACTIVE_TESTS_BLOCK_CHARS=${RLLM_ACTIVE_TESTS_BLOCK_CHARS:-4000}

export PYTHONPATH=./DeepSWE_RL/rllm:./R2E-Gym/src:${PYTHONPATH:-}

# ---- Ray cluster -------------------------------------------------------
export RAY_memory_monitor_refresh_ms=0
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_ADDRESS="local"
/home/temp/venv/bin/ray stop --force >/dev/null 2>&1 || true
/home/temp/venv/bin/ray start --head --num-cpus=16 \
    --object-store-memory=16000000000 \
    --memory=170000000000 \
    --disable-usage-stats >/dev/null 2>&1 || true
export RAY_ADDRESS="auto"

# ---- paths -------------------------------------------------------------
REPO_DIR=${REPO_DIR:-/home/temp/swemaster_runpod}
MODEL_PATH=${MODEL_PATH:-/home/temp/models/SWE-Master-4B-SFT}
TRAIN_PARQUET=${TRAIN_PARQUET:-${REPO_DIR}/data_examples/rl_data/swerebench_cached63/train_00000_verl.parquet}
VAL_PARQUET=${VAL_PARQUET:-${REPO_DIR}/data_examples/rl_data/swerebench_cached63/test_00000_verl.parquet}
PYTHON_BIN=${PYTHON_BIN:-/home/temp/venv/bin/python3}

# ---- paper-faithful tunables ------------------------------------------
MAX_RESPONSE=${MAX_RESPONSE:-24576}
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-$((8192 + MAX_RESPONSE))}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.45}      # raised 0.40→0.45: more vLLM KV cache; tight but actor is CPU-offloaded so this is safe
TOTAL_EPOCHS=${TOTAL_EPOCHS:-17}       # 63 tasks / batch 16 = 3 batches/epoch × 17 epochs = 51 steps; TOTAL_TRAINING_STEPS hard-caps at 50
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
AGENT_MAX_STEPS=${AGENT_MAX_STEPS:-150}
TRAJECTORY_TIMEOUT=${TRAJECTORY_TIMEOUT:-5400}
SAVE_FREQ=${SAVE_FREQ:-1}              # save at every PPO step (memory rule: avoid losing the final model). max_actor_ckpt_to_keep rolls disk usage.
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}
TEST_FREQ=${TEST_FREQ:--1}              # disabled: user will run val post-hoc
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-50}  # HARD-CAP at 50 — paper needs exactly 50 report points (one metric row per PPO step)

PROJECT_NAME=${PROJECT_NAME:-active_tests_paper320}
TS=$(date +"%Y-%m-%d_%H-%M-%S")
EXPERIMENT_NAME=${EXPERIMENT_NAME:-paper320_${TS}}

DEFAULT_LOCAL_DIR="/home/temp/training_outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}"
ROLLOUT_DATA_DIR="/home/temp/training_outputs_rollout/${PROJECT_NAME}/${EXPERIMENT_NAME}"
mkdir -p "${DEFAULT_LOCAL_DIR}" "${ROLLOUT_DATA_DIR}"

echo "Project: ${PROJECT_NAME}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Model: ${MODEL_PATH}"
echo "Train data: ${TRAIN_PARQUET}"
echo "Val data: ${VAL_PARQUET}  (test_freq=${TEST_FREQ} — val is disabled in-training)"
echo "Epochs: ${TOTAL_EPOCHS}, batch=${TRAIN_BATCH_SIZE}, mini=${PPO_MINI_BATCH_SIZE}, save_freq=${SAVE_FREQ}, max_keep=${MAX_ACTOR_CKPT_TO_KEEP}"
echo "ActiveTests: enabled=${RLLM_ACTIVE_TESTS_ENABLED} mid=${RLLM_ACTIVE_TESTS_MID_EPISODE} k=${RLLM_ACTIVE_TESTS_K} m_init=${RLLM_ACTIVE_TESTS_M_INIT}"

# ---- launch ------------------------------------------------------------
cd "${REPO_DIR}"
TOTAL_STEPS_ARG=()
if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
    TOTAL_STEPS_ARG=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
fi
exec "${PYTHON_BIN}" -m rllm.trainer.verl.train_agent_ppo \
    "${TOTAL_STEPS_ARG[@]}" \
    algorithm.adv_estimator=loop \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="${VAL_PARQUET}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=1 \
    data.max_prompt_length=8192 \
    data.max_response_length=${MAX_RESPONSE} \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=32 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.chat_scheduler=verl.schedulers.completions_scheduler.CompletionsScheduler \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.mask_truncated_samples=False \
    algorithm.clip_advantages=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${DEFAULT_LOCAL_DIR}" \
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP} \
    trainer.default_hdfs_dir=null \
    env.name=swe_arsenal \
    +env.env_args.delete_image=False \
    +env.env_args.scaffold="openhands" \
    +env.env_args.backend="docker" \
    agent.name=sweagent \
    agent.max_steps=${AGENT_MAX_STEPS} \
    agent.overlong_filter=True \
    agent.trajectory_timeout=${TRAJECTORY_TIMEOUT} \
    agent.async_engine=True \
    +agent.agent_args.scaffold="openhands" \
    trainer.total_epochs=${TOTAL_EPOCHS}
