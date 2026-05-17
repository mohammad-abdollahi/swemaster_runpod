#!/usr/bin/env bash
# SWE-Master RL pipeline on a single H100 80 GB (196 GB host RAM), tracking
# the ORIGINAL SWE-Master 4B recipe (examples/swe/
# swe_rl_debug_hls_try_arsenal_long_step_and_timeout3600_4b.sh) as closely as
# one GPU allows. The paper recipe was authored for 2x8 H800 (16 GPUs).
#
# Adopted EXACTLY from the paper recipe (user asked for: full SFT not LoRA,
# longer input/output, and the paper's step counts):
#   * FULL-PARAMETER fine-tuning, NO LoRA               (paper has no lora_rank)
#   * actor param_offload=True, optimizer_offload=True  (paper-exact; the 4B
#       fp32 master+Adam ~48 GB offloads to the 196 GB host RAM -- this is
#       exactly why it was impossible on the 31 GB L40S and forced LoRA there)
#   * ref.fsdp_config.param_offload=True                (paper-exact)
#   * data.max_prompt_length=8192                       (paper-exact)
#   * agent.max_steps=150                               (paper-exact)
#   * agent.trajectory_timeout=3600                     (paper-exact)
#   * trainer.total_epochs=5                            (paper-exact)
#   * actor_rollout_ref.rollout.n=4                     (paper-exact)
#
# DEVIATED from the paper because a single 80 GB GPU cannot do otherwise
# (these are pure multi-GPU data-parallel / sharding knobs, not the method):
#   * data.train_batch_size 32 -> 1, ppo_mini_batch 16 -> 1
#       (paper's 32 is data-parallel across 16 GPUs; on 1 GPU it only makes a
#        PPO step 32x slower, no quality change at this smoke scale)
#   * data.max_response_length 122880 -> 8192
#       (paper's 120K episodic trajectory budget assumes 16-GPU ulysses
#        sequence-parallel sharding. On ONE 80 GB GPU the full-param
#        loss.backward() of a long trajectory OOMs: a 32768 attempt died
#        allocating +10.5 GB with only 8 GB free. 8192 is the empirical
#        single-GPU full-param ceiling here -- still 4x the L40S 2048, and
#        the length the LoRA H100 step:1 handled. Longer needs >1 GPU
#        (ulysses_sequence_parallel_size>1) -- not available on this box.)
#   * ppo_max_token_len_per_gpu=16384 (= prompt+response = 8192+8192)
#   * gpu_memory_utilization 0.70 -> 0.20 (full-param ~48 GB fp32 Adam+master
#        must be live on-GPU during optimizer.step() regardless of seq len;
#        offload only saves memory BETWEEN steps. vLLM shrunk to the floor
#        that still loads 4B weights so the step fits. paper's 0.70 was
#        per-GPU with FSDP-sharded optimizer across 16 GPUs.) + alloc conf
#        expandable_segments:True (anti-fragmentation). enforce_eager kept.
#   NOTE: this is the edge of single-GPU full-param 4B PPO. If optimizer.step
#   still OOMs, full-param needs >1 GPU (FSDP shards the ~48 GB optimizer);
#   LoRA is the proven single-GPU fallback (see memory/training-plan.md).
#   * RLLM_MAX_ROLLOUT_CONCURRENCY=4 (196 GB RAM holds n=4 SWE containers;
#        L40S's 31 GB forced 1)
# Unchanged infra: 1 GPU, no TP/SP, in-repo SWE-rebench parquet, WANDB off
# (console+tensorboard), local SWE-Master-4B-SFT. Full-param also removes the
# LoRA<->vLLM wake_up resync KeyError that killed step 2 of the LoRA run.
set -x

# ----------------------- environment ------------------------------------
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# expandable_segments:True lets the allocator grow/return segments instead of
# fragmenting into many fixed blocks. The full-param 4B Adam step OOMed asking
# for 386 MB while ~0.9 GB sat reserved-but-unallocated -- a fragmentation
# signature. True is the single-GPU full-param survival lever here.
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
# Use parent venv directly: /workspace/venv already has every package the
# upstream runtime_env pip list pins. Cloning the venv into /tmp/ray and then
# pip-installing local wheels by relative path takes ~5min, then fails
# because Ray resolves './swebench_fork_*.whl' from a different cwd.
# See the docstring on run_ppo_agent in rllm.trainer.verl.train_agent_ppo.
export RLLM_USE_PARENT_VENV=1
# train_agent_ppo's runtime_env rebuilds PYTHONPATH from this; default is the
# old /workspace path, so it MUST be set to this VM's repo.
export RLLM_PROJECT_ROOT=/home/temp/swemaster_runpod
# Run rollout trajectories one container at a time. 31GB host RAM cannot hold
# 2 concurrent astropy containers + vLLM + FSDP (kernel OOM-killed the run).
# n=2 trajectories still run for the GRPO group, just sequentially. Honored by
# the cap added in rllm/engine/agent_execution_engine.py.
export RLLM_MAX_ROLLOUT_CONCURRENCY=4

# rllm package is not installed; expose it via PYTHONPATH
export PYTHONPATH=./DeepSWE_RL/rllm:./R2E-Gym/src:${PYTHONPATH:-}

# ---- host-RAM containment (this VM has only 31 GB; H200 pod had ~2 TB) ----
# FSDP param/optimizer offload pushes the 4B model to CPU RAM. Ray's default
# auto-sized object store (~30% RAM ≈ 9 GB) plus its 0.95 OOM-killer leaves no
# headroom and kills the worker mid-init. Pre-start a Ray cluster with a tiny
# object store and the memory monitor disabled, then attach to it. The docker
# rollout boundary is reached in fit_agent BEFORE the optimizer step, so we
# only need to survive the init peak, not hold fp32 optimizer states.
export RAY_memory_monitor_refresh_ms=0
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_ADDRESS="local"
/home/temp/venv/bin/ray stop --force >/dev/null 2>&1 || true
/home/temp/venv/bin/ray start --head --num-cpus=16 \
    --object-store-memory=8000000000 \
    --memory=140000000000 \
    --disable-usage-stats >/dev/null 2>&1 || true
export RAY_ADDRESS="auto"

# ----------------------- paths ------------------------------------------
REPO_DIR=${REPO_DIR:-/home/temp/swemaster_runpod}
MODEL_PATH=${MODEL_PATH:-/home/temp/models/SWE-Master-4B-SFT}
TRAIN_PARQUET=${TRAIN_PARQUET:-${REPO_DIR}/data_examples/rl_data/swerebench_comprehensive/train_00000_verl.parquet}
VAL_PARQUET=${VAL_PARQUET:-${REPO_DIR}/data_examples/rl_data/swerebench_comprehensive/test_00000_verl.parquet}
PYTHON_BIN=${PYTHON_BIN:-/home/temp/venv/bin/python3}

PROJECT_NAME="swe_master_comprehensive"
TS=$(date +"%Y-%m-%d_%H-%M-%S")
EXPERIMENT_NAME="smoke_${TS}"

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
    data.train_batch_size=1 \
    data.val_batch_size=1 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=1 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.rollout.n=4 \
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
