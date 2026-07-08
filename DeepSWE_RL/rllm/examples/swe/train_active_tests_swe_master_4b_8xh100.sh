#!/usr/bin/env bash
# =============================================================================
# Active-Test-Selection + agentic-refine RL — SWE-Master-4B on 8x H100.
# =============================================================================
# Multi-GPU (8x H100 80GB) launcher, tuned close to the original SWE-Master
# paper recipe. This is the 8-GPU counterpart of
# train_active_tests_swe_master_4b_h200.sh (single H200).
#
# WHY 8 GPUs fixes the single-GPU crash:
#   On one GPU, vLLM (generation) and FSDP (training) time-share the same
#   device; at the rollout->update hand-off the FSDP gather + backward on a
#   long sequence collided with vLLM's memory -> "CUDA illegal memory access".
#   On 8 GPUs, FSDP FULL_SHARD splits the 4B model + grads + optimizer across
#   all 8 devices (tiny per-GPU), so there is ample headroom at the hand-off.
#   => full context, NO CPU offload, paper-scale batch all become feasible.
#
# Everything is env-overridable. Sensible 8xH100 defaults are set below.
# Point it at your machine by exporting REPO_DIR / PYTHON_BIN / MODEL_PATH /
# TRAIN_PARQUET / VAL_PARQUET (see README_8XH100.md), or rely on the defaults
# if you keep the repo at $REPO_DIR with a venv at $REPO_DIR/../venv.
set -x

# ---- portable paths (override per machine) -----------------------------
# REPO_DIR defaults to the repo this script lives in (…/DeepSWE_RL/rllm/examples/swe -> repo root).
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=${REPO_DIR:-$(cd "${_SCRIPT_DIR}/../../../.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_PATH=${MODEL_PATH:-${REPO_DIR}/models/SWE-Master-4B-SFT}
TRAIN_PARQUET=${TRAIN_PARQUET:-${REPO_DIR}/data_examples/rl_data/swerebench_verified_live/train_00000_verl.parquet}
VAL_PARQUET=${VAL_PARQUET:-${REPO_DIR}/data_examples/rl_data/swerebench_verified_live/test_00000_verl.parquet}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_DIR}/training_outputs}
ROLLOUT_ROOT=${ROLLOUT_ROOT:-${REPO_DIR}/training_outputs_rollout}

# ---- vLLM / env base ---------------------------------------------------
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=${VLLM_ENGINE_ITERATION_TIMEOUT_S:-100000000000}
export HYDRA_FULL_ERROR=1
export WANDB_MODE=${WANDB_MODE:-disabled}
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export RLLM_USE_PARENT_VENV=1
export RLLM_PROJECT_ROOT=${REPO_DIR}
# 8 GPUs -> much more rollout parallelism than the single-GPU recipe.
export RLLM_MAX_ROLLOUT_CONCURRENCY=${RLLM_MAX_ROLLOUT_CONCURRENCY:-32}

# ---- Active-Test-Selection scheduler (same knobs as single-GPU) --------
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
export RLLM_ACTIVE_TESTS_W_SOLVE=${RLLM_ACTIVE_TESTS_W_SOLVE:-10.0}
export RLLM_ACTIVE_TESTS_LAM_TOKEN=${RLLM_ACTIVE_TESTS_LAM_TOKEN:-0.1}
export RLLM_ACTIVE_TESTS_TOKEN_REF=${RLLM_ACTIVE_TESTS_TOKEN_REF:-32000.0}
export RLLM_ACTIVE_TESTS_TOKEN_PENALTY_CAP=${RLLM_ACTIVE_TESTS_TOKEN_PENALTY_CAP:-3.0}
export RLLM_ACTIVE_TESTS_BLOCK_CHARS=${RLLM_ACTIVE_TESTS_BLOCK_CHARS:-4000}
# Refinement (agentic / online). Off here; the agentic wrapper turns it on.
export RLLM_ACTIVE_TESTS_REFINE=${RLLM_ACTIVE_TESTS_REFINE:-0}
export RLLM_ACTIVE_TESTS_REFINE_TICK=${RLLM_ACTIVE_TESTS_REFINE_TICK:-0}
export RLLM_ACTIVE_TESTS_REFINE_AFTER=${RLLM_ACTIVE_TESTS_REFINE_AFTER:--1}
export RLLM_ACTIVE_TESTS_REFINE_MAX=${RLLM_ACTIVE_TESTS_REFINE_MAX:-4}
export RLLM_ACTIVE_TESTS_REFINE_CACHE=${RLLM_ACTIVE_TESTS_REFINE_CACHE:-${REPO_DIR}/block_b_refinement/cache/decompositions.jsonl}
[ -n "${RLLM_ACTIVE_TESTS_W_REGRESS+x}" ]         && export RLLM_ACTIVE_TESTS_W_REGRESS
[ -n "${RLLM_ACTIVE_TESTS_DELTA_P2P_BREAK+x}" ]   && export RLLM_ACTIVE_TESTS_DELTA_P2P_BREAK
[ -n "${RLLM_ACTIVE_TESTS_DELTA_P2P_RECOVER+x}" ] && export RLLM_ACTIVE_TESTS_DELTA_P2P_RECOVER

export PYTHONPATH=${REPO_DIR}/DeepSWE_RL/rllm:${REPO_DIR}/R2E-Gym/src:${PYTHONPATH:-}

# ---- Ray cluster (scale to the 8-GPU box) ------------------------------
export RAY_memory_monitor_refresh_ms=0
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_ADDRESS="local"
RAY_BIN=${RAY_BIN:-ray}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-$(nproc)}
# object store: default 64G (big boxes have plenty of RAM); override if needed.
RAY_OBJECT_STORE=${RAY_OBJECT_STORE:-64000000000}
"${RAY_BIN}" stop --force >/dev/null 2>&1 || true
RAY_START_ARGS=(--head --num-cpus="${RAY_NUM_CPUS}" --num-gpus="${N_GPUS:-8}" \
    --object-store-memory="${RAY_OBJECT_STORE}" --disable-usage-stats)
[ -n "${RAY_MEMORY:-}" ] && RAY_START_ARGS+=(--memory="${RAY_MEMORY}")
"${RAY_BIN}" start "${RAY_START_ARGS[@]}" >/dev/null 2>&1 || true
export RAY_ADDRESS="auto"

# ---- multi-GPU + paper-scale tunables ----------------------------------
N_GPUS=${N_GPUS:-8}                 # GPUs per node
NNODES=${NNODES:-1}
TP_SIZE=${TP_SIZE:-1}               # vLLM tensor-parallel (1 => 8 DP replicas; best throughput for a 4B model)
# On 8x H100 the FSDP shard is tiny, so offload OFF is safe and MUCH faster.
PARAM_OFFLOAD=${PARAM_OFFLOAD:-False}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-False}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.60}  # H100 80GB has room; colocated vLLM+FSDP
# Paper-faithful context + episode length (single-GPU used shorter for memory).
MAX_RESPONSE=${MAX_RESPONSE:-24576}
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-$((8192 + MAX_RESPONSE))}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
# Paper-scale batch: 8 GPUs comfortably handle a much larger global batch.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
MICRO_BATCH=${MICRO_BATCH:-1}
ROLLOUT_N=${ROLLOUT_N:-4}            # samples per prompt (group size for GRPO/loop)
AGENT_MAX_STEPS=${AGENT_MAX_STEPS:-150}
TRAJECTORY_TIMEOUT=${TRAJECTORY_TIMEOUT:-5400}
LR=${LR:-1e-6}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-40}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-}   # empty = epochs-bound; set for a fixed step count
SAVE_FREQ=${SAVE_FREQ:-1}
TEST_FREQ=${TEST_FREQ:-999}
CKPT_CONTENTS=${CKPT_CONTENTS:-[model,optimizer,extra]}
FILTER_WORKERS=${FILTER_WORKERS:-8}

PROJECT_NAME=${PROJECT_NAME:-active_tests_swe_master_8xh100}
TS=$(date +"%Y-%m-%d_%H-%M-%S")
EXPERIMENT_NAME=${EXPERIMENT_NAME:-swe_master_8xh100_${TS}}

DEFAULT_LOCAL_DIR="${OUTPUT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}"
ROLLOUT_DATA_DIR="${ROLLOUT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}"
mkdir -p "${DEFAULT_LOCAL_DIR}" "${ROLLOUT_DATA_DIR}"

echo "==================== 8xH100 SWE-Master RL ===================="
echo "Repo:        ${REPO_DIR}"
echo "Python:      ${PYTHON_BIN}"
echo "Model:       ${MODEL_PATH}"
echo "Train data:  ${TRAIN_PARQUET}"
echo "Val data:    ${VAL_PARQUET}"
echo "GPUs:        ${N_GPUS} (nnodes=${NNODES}, vLLM TP=${TP_SIZE})"
echo "Batch:       train=${TRAIN_BATCH_SIZE} mini=${PPO_MINI_BATCH_SIZE} rollout.n=${ROLLOUT_N}"
echo "Context:     prompt=${MAX_PROMPT_LENGTH} response=${MAX_RESPONSE} (ppo_max_token_len=${PPO_MAX_TOKEN_LEN})"
echo "Offload:     param=${PARAM_OFFLOAD} optim=${OPTIMIZER_OFFLOAD}  gpu_util=${GPU_MEM_UTIL}"
echo "Episode:     max_steps=${AGENT_MAX_STEPS} traj_timeout=${TRAJECTORY_TIMEOUT}s  lr=${LR}"
echo "Refine:      ats=${RLLM_ACTIVE_TESTS_ENABLED} refine=${RLLM_ACTIVE_TESTS_REFINE} agentic=${RLLM_ACTIVE_TESTS_AGENTIC_REFINE:-0}"
echo "Out:         ${DEFAULT_LOCAL_DIR}"
echo "=============================================================="

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
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE} \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=${FILTER_WORKERS} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.chat_scheduler=verl.schedulers.completions_scheduler.CompletionsScheduler \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
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
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    actor_rollout_ref.actor.checkpoint.contents=${CKPT_CONTENTS} \
    trainer.test_freq=${TEST_FREQ} \
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
