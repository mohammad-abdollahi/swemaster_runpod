#!/usr/bin/env bash
# Smoke test of SWE-Master RL pipeline on a single H200 (1 GPU).
# Adapted from swe_rl_debug_hls_try_arsenal_long_step_and_timeout3600_4b.sh
#
# Differences vs. the original 2x8-GPU script:
#   * 1 node, 1 GPU (was 2 nodes x 8 GPUs)
#   * No ulysses sequence-parallel, no tensor parallel
#   * Tiny batch sizes (train=1, mini=1, micro=1)
#   * Local demo parquet from data_examples/ used for train+val
#   * SWE-Master-4B-SFT model from local /workspace/models cache
#   * WANDB disabled (offline); only console + tensorboard
#   * Very short response length to fit in single-GPU memory
#   * One training epoch; save_freq pushed high so we don't write a checkpoint
set -x

# ----------------------- environment ------------------------------------
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Use parent venv directly: /workspace/venv already has every package the
# upstream runtime_env pip list pins. Cloning the venv into /tmp/ray and then
# pip-installing local wheels by relative path takes ~5min, then fails
# because Ray resolves './swebench_fork_*.whl' from a different cwd.
# See the docstring on run_ppo_agent in rllm.trainer.verl.train_agent_ppo.
export RLLM_USE_PARENT_VENV=1

# rllm package is not installed; expose it via PYTHONPATH
export PYTHONPATH=./DeepSWE_RL/rllm:./R2E-Gym/src:${PYTHONPATH:-}

# Choose how the per-task sandbox is realised. Default to "local" (subprocess
# on the host) because RunPod blocks unshare() so docker run cannot start a
# container. Override with RLLM_BACKEND=docker on a docker-capable host.
RLLM_BACKEND=${RLLM_BACKEND:-local}
if [ "${RLLM_BACKEND}" = "local" ]; then
    # Belt-and-suspenders: if anything inside r2egym tries to fall back to
    # docker, this env var forces the LocalRuntime path in docker.py.
    export R2EGYM_FORCE_LOCAL_BACKEND=1
fi
echo "Sandbox backend: ${RLLM_BACKEND}"

# ----------------------- paths ------------------------------------------
# Repo root defaults to /workspace/swemaster_runpod but auto-falls back to the
# script's containing repo so this works from /root/swemaster_runpod too.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT_DEFAULT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
REPO_ROOT=${REPO_ROOT:-${REPO_ROOT_DEFAULT}}
if [ -d /workspace/swemaster_runpod ]; then
    REPO_ROOT=/workspace/swemaster_runpod
fi

# Venv: use /workspace/venv if present, otherwise the user's currently-active
# python interpreter.
if [ -x /workspace/venv/bin/python3 ]; then
    PY=/workspace/venv/bin/python3
else
    PY=${PY:-$(command -v python3)}
fi

# Output directory: /workspace when available, otherwise under the repo.
OUT_ROOT=${OUT_ROOT:-/workspace}
if [ ! -d "${OUT_ROOT}" ]; then
    OUT_ROOT="${REPO_ROOT}/training_outputs"
fi

MODEL_PATH=${MODEL_PATH:-${OUT_ROOT}/models/SWE-Master-4B-SFT}
TRAIN_PARQUET=${TRAIN_PARQUET:-${REPO_ROOT}/data_examples/rl_data/ood_data/equal_0.3999999_0.9999999_swe-gym_and_rebench_demo_for_RL/train_00000_verl.parquet}
VAL_PARQUET=${VAL_PARQUET:-${TRAIN_PARQUET}}

PROJECT_NAME="swe_master_smoke_test"
TS=$(date +"%Y-%m-%d_%H-%M-%S")
EXPERIMENT_NAME="smoke_${TS}"

DEFAULT_LOCAL_DIR="${OUT_ROOT}/training_outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}"
ROLLOUT_DATA_DIR="${OUT_ROOT}/training_outputs_rollout/${PROJECT_NAME}/${EXPERIMENT_NAME}"
mkdir -p "${DEFAULT_LOCAL_DIR}" "${ROLLOUT_DATA_DIR}"

echo "Project: ${PROJECT_NAME}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Model: ${MODEL_PATH}"
echo "Train data: ${TRAIN_PARQUET}"
echo "Val data: ${VAL_PARQUET}"

# ----------------------- launch -----------------------------------------
cd "${REPO_ROOT}"
echo "Repo root: ${REPO_ROOT}"
echo "Python:    ${PY}"
exec "${PY}" -m rllm.trainer.verl.train_agent_ppo \
    algorithm.adv_estimator=loop \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="${VAL_PARQUET}" \
    data.train_batch_size=1 \
    data.val_batch_size=1 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=4 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8200 \
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
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
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
    trainer.save_freq=1000 \
    trainer.test_freq=-1 \
    trainer.default_hdfs_dir=null \
    env.name=swe_arsenal \
    +env.env_args.delete_image=False \
    +env.env_args.scaffold="openhands" \
    +env.env_args.backend="${RLLM_BACKEND}" \
    agent.name=sweagent \
    agent.max_steps=4 \
    agent.overlong_filter=True \
    agent.trajectory_timeout=300 \
    agent.async_engine=True \
    +agent.agent_args.scaffold="openhands" \
    trainer.total_epochs=1
