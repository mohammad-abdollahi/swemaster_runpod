#!/usr/bin/env bash
# Smoke-test RL training: GRPO on FrozenLake with Qwen3-4B on a single node
# (4x H100 80GB on Trillium). Logs to console + tensorboard.
# Run inside the container — usually invoked by 04_train_frozenlake.sbatch.
set -euo pipefail

source "${VENV_BIN:-/scratch/$USER/swemaster/venv/rllm/bin}/activate"
cd "${PROJECT_ROOT:-/project/def-hemmati-ac/$USER/SWE-Master}/DeepSWE_RL/rllm"

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export WANDB_MODE=offline

# Login-node has ulimit -u 1024 — polars/rayon/openmp will busts that without
# explicit caps. Each cap below maps to a different thread pool.
export POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-4}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export RAY_num_cpus=${RAY_num_cpus:-32}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}

# How many GPUs to use for training. With ray.local_mode=True we MUST stay
# on a single GPU — multi-GPU FSDP requires real Ray actors. Trillium nodes
# are 4x H100 so for the real training (on a compute node where ulimit is OK)
# we'd flip RLLM_RAY_LOCAL_MODE=0 and bump N_GPUS=4.
N_GPUS=${N_GPUS:-1}
# 0 = use real Ray (recommended now that we fence raylet via LD_PRELOAD shim).
# 1 = local_mode (only works on ray<2.32, so unusable with verl>=0.7.1).
export RLLM_RAY_LOCAL_MODE=${RLLM_RAY_LOCAL_MODE:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# LD_PRELOAD shim: lie about online CPU count so raylet's boost::asio pool +
# polars + rayon + OMP each cap their default thread count at FAKECPUS_N
# instead of nproc (192 on this node). Without this we blow past ulimit -u
# (1024) during raylet boot on a login node.
export FAKECPUS_N=${FAKECPUS_N:-8}
FAKE_SO=/scratch/$USER/swemaster/fakecpus.so
if [[ -f "$FAKE_SO" ]]; then
  export LD_PRELOAD="$FAKE_SO${LD_PRELOAD:+:$LD_PRELOAD}"
  echo "[train] LD_PRELOAD=$LD_PRELOAD  FAKECPUS_N=$FAKECPUS_N"
else
  echo "[train] WARN: $FAKE_SO not built — raylet may exceed ulimit -u" >&2
fi

# Run identifier for tensorboard + checkpoint dirs.
TS=$(date +"%Y-%m-%d_%H-%M-%S")
PROJECT_NAME=${PROJECT_NAME:-"rllm-smoke"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"frozenlake-qwen3-4b-${TS}"}
RUN_ROOT=${RUN_ROOT:-/scratch/$USER/swemaster/runs}
LOCAL_DIR="${RUN_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}"
mkdir -p "${LOCAL_DIR}"
echo "[train] LOCAL_DIR=${LOCAL_DIR}"

# Model path: prefer locally-cached snapshot to avoid hitting HF inside Ray
# workers. Fall back to repo id (will redownload via HF_HOME cache).
export MODEL_ID=${MODEL_ID:-Qwen/Qwen3-4B}
export HF_HUB="${HUGGINGFACE_HUB_CACHE:-${HF_HOME_DIR:-/scratch/$USER/swemaster/cache/hf}/hub}"
MODEL_PATH=$(python - <<'PY'
import os, glob
hub = os.environ["HF_HUB"]
repo = os.environ.get("MODEL_ID","Qwen/Qwen3-4B").replace("/","--")
snapshots = glob.glob(os.path.join(hub, f"models--{repo}", "snapshots", "*"))
print(snapshots[0] if snapshots else os.environ["MODEL_ID"])
PY
)
echo "[train] MODEL_PATH=${MODEL_PATH}"

python -m examples.frozenlake.train_frozenlake_agent \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=16 \
    data.val_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.grad_norm_threshold=10 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.chat_scheduler=verl.schedulers.completions_scheduler.CompletionsScheduler \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.mask_truncated_samples=False \
    algorithm.clip_advantages=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${LOCAL_DIR}" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.default_hdfs_dir=null \
    trainer.rejection_sample=True \
    trainer.rejection_sample_multiplier=2 \
    +env.env_args.max_steps=4 \
    +env.env_args.is_slippery=False \
    agent.max_steps=4 \
    agent.async_engine=True \
    agent.use_stepwise_advantage=False \
    +agent.engine_args.disable_thinking=False \
    +agent.agent_args.max_steps=4 \
    +agent.agent_args.use_accumulate_history=True \
    trainer.total_epochs=1 \
    trainer.total_training_steps=5
