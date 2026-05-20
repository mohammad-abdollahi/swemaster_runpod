#!/usr/bin/env bash
# SWE-Master RL pipeline on a SINGLE NVIDIA H200 143 GB (196 GB host RAM),
# FULL-PARAMETER (no LoRA) — the paper recipe, now actually runnable on one GPU.
#
# Why this exists (see memory/training-plan.md + SETUP_NOTES.md "H100"):
#   On the 80 GB H100, full-param 4B PPO was DEFINITIVELY impossible: verl keeps
#   the ~64 GB params+grads+fp32-Adam state GPU-resident DURING optimizer.step()
#   (verl/workers/fsdp_workers.py:369 builds a plain on-GPU AdamW; :663
#   load_fsdp_optimizer(device_id=cuda)). vLLM init floor (~24 GB) + ~64 GB
#   optimizer/params + ~10-18 GB activations > 80 GB from both sides. 3 attempts
#   all OOMed (train_h100_fullft.log). LoRA was the only single-GPU fallback,
#   but LoRA step:2 dies on the vLLM wake_up resync KeyError.
#
#   The H200's 143 GB lifts that ceiling: ~64 GB optimizer + vLLM + activations
#   ~ 100-127 GB < 143 GB with headroom. Full-param ALSO removes the LoRA<->vLLM
#   resync bug that capped the H100/L40S runs at step 1. This is the box the
#   pipeline was waiting for.
#
# Paper recipe (examples/swe/swe_rl_debug_..._4b.sh) — adopted EXACTLY:
#   * FULL-PARAMETER, NO lora_rank
#   * actor param_offload + optimizer_offload (196 GB RAM holds the offload)
#   * ref.fsdp_config.param_offload=True
#   * data.max_prompt_length=8192
#   * agent.max_steps=150, agent.trajectory_timeout=3600
#   * actor_rollout_ref.rollout.n=4, trainer.total_epochs=5
#
# Paper-exact data-parallel knobs adopted (2026-05-19) — these are NOT
# memory-bound here: ppo_micro_batch_size_per_gpu=1 + use_dynamic_bsz=False
# means the actor backprops ONE sequence per backward regardless, just with
# more gradient-accumulation micro-steps before optimizer.step(). Peak stays
# the deterministic ~128.6 GB. Total rollout compute per epoch unchanged
# (num_tasks×n either way).
#   * train_batch_size=32             (paper-exact; needs >=32 train tasks)
#   * ppo_mini_batch_size=16          (paper-exact)
#   * log_prob_micro_batch_size_per_gpu=4 (paper-exact; unused w/ dynamic_bsz)
#   * filter_overlong_prompts_workers=32 (paper-exact)
#
# DEVIATED only on irreducibly multi-GPU sharding knobs (paper's 2x8=16 H800):
#   * max_response_length 122880 -> 32768  (paper's 120K needs ulysses
#       sequence_parallel_size=8; on 1 GPU the full-param backward of 122K
#       tokens needs ~150 GB > 143 GB. 32768 is the empirical 143 GB ceiling
#       — user-confirmed do NOT exceed. ulysses_sequence_parallel_size=1.)
#   * ppo_max_token_len_per_gpu = 8192 + max_response  (caps the per-microbatch
#       backward peak; paper's 16500 only fits because ulysses_sp=8 shards
#       122880/8 ≈ 15360 per GPU. No sharding here → must hold full sequence.)
#   * tensor_model_parallel_size 2 -> 1, ulysses_sp 8 -> 1  (need >=2/>=8 GPUs)
#   * gpu_memory_utilization 0.70 -> 0.30  (paper's 0.70 was per-GPU with the
#       optimizer FSDP-sharded ~3 GB/GPU across 16 GPUs; here the full ~64 GB
#       optimizer state lives on 1 GPU during step, so vLLM gets less budget.)
#   * RLLM_MAX_ROLLOUT_CONCURRENCY=4  (196 GB RAM holds n=4 SWE containers)
# Unchanged infra: 1 GPU, no TP/SP, in-repo SWE-rebench parquet, WANDB off
# (console+tensorboard), local SWE-Master-4B-SFT.
set -x

# ----------------------- environment ------------------------------------
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — vLLM
# 0.10.2's cumem allocator (device_allocator/cumem.py) asserts it is absent
# (incompatible with its memory pool, pytorch#147851). It was only an 80 GB
# H100 anti-fragmentation survival lever; the 143 GB H200 does not need it.
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
export RLLM_PROJECT_ROOT=/home/temp/swemaster_runpod
export RLLM_MAX_ROLLOUT_CONCURRENCY=4

export PYTHONPATH=./DeepSWE_RL/rllm:./R2E-Gym/src:${PYTHONPATH:-}

# ---- Ray cluster (196 GB RAM: generous object store, monitor off) -------
export RAY_memory_monitor_refresh_ms=0
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_ADDRESS="local"
/home/temp/venv/bin/ray stop --force >/dev/null 2>&1 || true
/home/temp/venv/bin/ray start --head --num-cpus=16 \
    --object-store-memory=16000000000 \
    --memory=170000000000 \
    --disable-usage-stats >/dev/null 2>&1 || true
export RAY_ADDRESS="auto"

# ----------------------- paths ------------------------------------------
REPO_DIR=${REPO_DIR:-/home/temp/swemaster_runpod}
MODEL_PATH=${MODEL_PATH:-/home/temp/models/SWE-Master-4B-SFT}
TRAIN_PARQUET=${TRAIN_PARQUET:-${REPO_DIR}/data_examples/rl_data/swerebench_comprehensive/train_00000_verl.parquet}
VAL_PARQUET=${VAL_PARQUET:-${REPO_DIR}/data_examples/rl_data/swerebench_comprehensive/test_00000_verl.parquet}
PYTHON_BIN=${PYTHON_BIN:-/home/temp/venv/bin/python3}

# Tunables (override via env). max_response is pushed as high as the 143 GB
# actor-backward peak allows; ppo_max_token_len_per_gpu must be >= the longest
# single sequence = max_prompt(8192) + max_response.
MAX_RESPONSE=${MAX_RESPONSE:-32768}
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-$((8192 + MAX_RESPONSE))}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.3}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
# Paper-exact 4B values (`...timeout3600_4b.sh`): 32 / 16. Memory-free here
# (micro_batch=1 → grad accumulation only). Requires >=TRAIN_BATCH_SIZE train
# tasks in the parquet (default parquet has 32; rebuild via
# build_comprehensive_swerebench.py with N_TRAIN to scale up).
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}

PROJECT_NAME="swe_master_comprehensive"
TS=$(date +"%Y-%m-%d_%H-%M-%S")
EXPERIMENT_NAME="h200_fullft_${TS}"

DEFAULT_LOCAL_DIR="/home/temp/training_outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}"
ROLLOUT_DATA_DIR="/home/temp/training_outputs_rollout/${PROJECT_NAME}/${EXPERIMENT_NAME}"
mkdir -p "${DEFAULT_LOCAL_DIR}" "${ROLLOUT_DATA_DIR}"

echo "Project: ${PROJECT_NAME}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Model: ${MODEL_PATH}"
echo "Train data: ${TRAIN_PARQUET}"
echo "Val data: ${VAL_PARQUET}"

# ----------------------- launch -----------------------------------------
cd "${REPO_DIR}"
exec "${PYTHON_BIN}" -m rllm.trainer.verl.train_agent_ppo \
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
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=4 \
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
    trainer.n_gpus_per_node=1 \
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
    trainer.total_epochs=${TOTAL_EPOCHS}
