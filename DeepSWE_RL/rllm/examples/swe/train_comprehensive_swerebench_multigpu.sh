#!/usr/bin/env bash
# SWE-Master RL — FULL-PARAMETER (no LoRA), MULTI-GPU. Use this on a machine
# with >=2x 80GB GPUs. Full-param 4B PPO is IMPOSSIBLE on 1 GPU with this
# pipeline (verl keeps the ~48GB fp32 Adam+master state GPU-resident during
# optimizer.step(); vLLM floor ~24GB; 1x80GB can't hold both -- see
# SETUP_NOTES.md "H100 continuation" and memory/training-plan.md). With
# >=2 GPUs, FSDP FULL_SHARD splits params+grads+optimizer (~64GB) across the
# GPUs and vLLM runs tensor-parallel, which makes full-param feasible.
#
# This IS the original paper recipe (examples/swe/
# swe_rl_debug_hls_try_arsenal_long_step_and_timeout3600_4b.sh): full-param,
# adv_estimator=loop (the LOOP/RLOO sibling of GRPO the paper describes; set
# ADV_ESTIMATOR=grpo for paper-literal GRPO), n=4, max_steps=150,
# trajectory_timeout=3600, total_epochs=5, max_prompt=8192.
#
# ---- Sizing (per-GPU footprint, full-param 4B, FSDP FULL_SHARD) ----
#   static params+grads+AdamW(fp32) ~64GB total -> /NUM_GPUS per GPU
#   activations ~10-18GB @ 8-16K resp (shard only with ULYSSES>1)
#   vLLM ~24GB/TP per GPU
#   2x80GB: ~59GB/GPU @ 8K resp                 -> OK (recommended minimum)
#   4x80GB: ~34GB/GPU, raise MAX_RESP / ULYSSES -> comfortable
#   8-16x : needed for the exact paper (MAX_RESP=122880 via ULYSSES, BS=32)
# GPUs must be 80GB (2x48GB won't fit ~59GB; use 4x). NVLink preferred.
#
# ---- Tunables (env vars) ----
#   NUM_GPUS   (default 2)   trainer.n_gpus_per_node; FSDP DP/shard world size
#   TP         (default =NUM_GPUS) vLLM tensor_model_parallel_size (must
#                                  divide NUM_GPUS and the model head count)
#   ULYSSES    (default 1)   actor ulysses_sequence_parallel_size; raise
#                            (2/4/8, must divide NUM_GPUS) to shard long-seq
#                            activations and push MAX_RESP toward 122880
#   MAX_RESP   (default 16384) data.max_response_length (paper 122880 needs
#                              ULYSSES>1 + many GPUs)
#   TRAIN_BS   (default 1)   data.train_batch_size  (paper 32; raise with GPUs)
#   MINI_BS    (default 1)   ppo_mini_batch_size    (paper 16)
#   OFFLOAD    (default 1)   1=keep param/optimizer CPU offload (safe extra
#                            cushion); set 0 on >=4 GPUs for speed (FSDP shard
#                            already makes the per-GPU optimizer small)
#   GPU_MEM    (default 0.55) vLLM gpu_memory_utilization (sharded training
#                             memory leaves room; lower if OOM)
# All paper step counts (max_steps 150, timeout 3600, epochs 5, n=4) are kept.
set -x

NUM_GPUS=${NUM_GPUS:-2}
TP=${TP:-${NUM_GPUS}}
ULYSSES=${ULYSSES:-1}
MAX_RESP=${MAX_RESP:-16384}
MAX_PROMPT=${MAX_PROMPT:-8192}
TRAIN_BS=${TRAIN_BS:-1}
MINI_BS=${MINI_BS:-1}
OFFLOAD=${OFFLOAD:-1}
GPU_MEM=${GPU_MEM:-0.55}
ADV_ESTIMATOR=${ADV_ESTIMATOR:-loop}
# ppo_max_token_len_per_gpu must be >= max_prompt + max_response
PPO_MAXTOK=${PPO_MAXTOK:-$(( MAX_PROMPT + MAX_RESP ))}
if [ "${OFFLOAD}" = "1" ]; then OFF=True; else OFF=False; fi

# ----------------------- environment ------------------------------------
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RLLM_USE_PARENT_VENV=1
export RLLM_PROJECT_ROOT=${RLLM_PROJECT_ROOT:-/home/temp/swemaster_runpod}
# Concurrent SWE rollout containers. Scale with host RAM (each ~2-4GB). On a
# big multi-GPU box raise toward train_batch_size*n.
export RLLM_MAX_ROLLOUT_CONCURRENCY=${RLLM_MAX_ROLLOUT_CONCURRENCY:-4}
export PYTHONPATH=./DeepSWE_RL/rllm:./R2E-Gym/src:${PYTHONPATH:-}

# Let the trainer init Ray itself (do NOT pre-start a tiny single-GPU cluster
# -- that hack was only for the 31GB L40S). Just clear any stale cluster.
export RAY_DISABLE_DOCKER_CPU_WARNING=1
${PYTHON_BIN:-/home/temp/venv/bin/python3} -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true

# ----------------------- paths ------------------------------------------
REPO_DIR=${REPO_DIR:-/home/temp/swemaster_runpod}
MODEL_PATH=${MODEL_PATH:-/home/temp/models/SWE-Master-4B-SFT}
TRAIN_PARQUET=${TRAIN_PARQUET:-${REPO_DIR}/data_examples/rl_data/swerebench_comprehensive/train_00000_verl.parquet}
VAL_PARQUET=${VAL_PARQUET:-${REPO_DIR}/data_examples/rl_data/swerebench_comprehensive/test_00000_verl.parquet}
PYTHON_BIN=${PYTHON_BIN:-/home/temp/venv/bin/python3}

PROJECT_NAME="swe_master_comprehensive_fullft"
TS=$(date +"%Y-%m-%d_%H-%M-%S")
EXPERIMENT_NAME="fullft_${NUM_GPUS}gpu_${TS}"

DEFAULT_LOCAL_DIR="/home/temp/training_outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}"
ROLLOUT_DATA_DIR="/home/temp/training_outputs_rollout/${PROJECT_NAME}/${EXPERIMENT_NAME}"
mkdir -p "${DEFAULT_LOCAL_DIR}" "${ROLLOUT_DATA_DIR}"

echo "Project: ${PROJECT_NAME} | NUM_GPUS=${NUM_GPUS} TP=${TP} ULYSSES=${ULYSSES} MAX_RESP=${MAX_RESP} OFFLOAD=${OFF} GPU_MEM=${GPU_MEM} ADV=${ADV_ESTIMATOR}"
echo "Model: ${MODEL_PATH}"
echo "Train: ${TRAIN_PARQUET}"

# ----------------------- launch -----------------------------------------
cd "${REPO_DIR}"
exec "${PYTHON_BIN}" -m rllm.trainer.verl.train_agent_ppo \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="${VAL_PARQUET}" \
    data.train_batch_size="${TRAIN_BS}" \
    data.val_batch_size=1 \
    data.max_prompt_length="${MAX_PROMPT}" \
    data.max_response_length="${MAX_RESP}" \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=1 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BS}" \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAXTOK}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ULYSSES}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload="${OFF}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OFF}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.chat_scheduler=verl.schedulers.completions_scheduler.CompletionsScheduler \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM}" \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload="${OFF}" \
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
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    env.name=swe_arsenal \
    +env.env_args.delete_image=False \
    +env.env_args.scaffold="openhands" \
    +env.env_args.backend="docker" \
    agent.name=sweagent \
    agent.max_steps=150 \
    agent.overlong_filter=True \
    agent.trajectory_timeout=3600 \
    agent.async_engine=True \
    +agent.agent_args.scaffold="openhands" \
    trainer.total_epochs=5
