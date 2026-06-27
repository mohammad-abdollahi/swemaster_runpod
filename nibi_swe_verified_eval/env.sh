#!/bin/bash
# Shared environment for SWE-Master-4B-SFT × SWE-bench Verified eval on Nibi (no Docker).
# Source this from every script and SLURM job:  source /project/6087623/mohammadA/evaluation/env.sh

# ---- modules (single source of truth) ----
# `arrow` provides pyarrow for datasets; the agent venv relies on it. apptainer is the runtime.
if command -v module >/dev/null 2>&1; then
  module load gcc python/3.11 arrow apptainer/1.3.5 >/dev/null 2>&1 || true
fi

# ---- paths ----
export EVAL_ROOT=/project/6087623/mohammadA/evaluation          # code (few files, on /project)
export WORK=/scratch/moham98/swemaster                          # heavy artifacts (on /scratch)
export R2EGYM_DIR=/project/6087623/mohammadA/swemaster_runpod/R2E-Gym

# ---- model / dataset ----
# MODEL_VARIANT selects which SWE-Master model to serve/eval: sft (default) or rl.
# Same scaffold/params/SIFs/dataset for both — only the model + served name change.
export MODEL_VARIANT=${MODEL_VARIANT:-sft}
if [ "$MODEL_VARIANT" = "rl" ]; then
  export MODEL_ID=RUC-AIBOX/SWE-Master-4B-RL
  export MODEL_DIR=$WORK/models/SWE-Master-4B-RL
  export SERVED_MODEL_NAME=swe-master-rl            # must NOT contain "qwen" (parser branch)
elif [ "$MODEL_VARIANT" = "ats" ]; then
  export MODEL_ID=mohammad-abdollahi/SWE-Master-4B-ATS-RL
  export MODEL_DIR=$WORK/models/SWE-Master-4B-ATS-RL
  export SERVED_MODEL_NAME=swe-master-ats
elif [ "$MODEL_VARIANT" = "sft32" ]; then
  export MODEL_ID=RUC-AIBOX/SWE-Master-32B-SFT
  export MODEL_DIR=$WORK/models/SWE-Master-32B-SFT
  export SERVED_MODEL_NAME=swe-master-32b-sft       # no "qwen" in served name
  export VLLM_TP=2                                   # 32B needs tensor-parallel across 2 H100s
else
  export MODEL_ID=RUC-AIBOX/SWE-Master-4B-SFT
  export MODEL_DIR=$WORK/models/SWE-Master-4B-SFT
  export SERVED_MODEL_NAME=swe-master-sft
fi
export VLLM_TP=${VLLM_TP:-1}                          # tensor-parallel size (1 for 4B, 2 for 32B)
export LLM_NAME=hosted_vllm/$SERVED_MODEL_NAME
export DATASET_DIR=$WORK/dataset/SWE-Bench-Verified_disk         # HF dataset saved_to_disk, with image_name column
export VLLM_PORT=${VLLM_PORT:-8000}   # per-job override (set in server_eval.sh) avoids port clashes when SLURM co-locates jobs

# ---- HuggingFace caches (off $HOME; compute nodes are offline so we pre-download) ----
export HF_HOME=$WORK/hf_cache
export HF_HUB_OFFLINE=0                                          # set to 1 inside compute-node jobs

# ---- Apptainer (the Docker replacement) ----
export APPTAINER_CACHEDIR=$WORK/apptainer_cache                  # layer/blob cache (persists; offline reuse)
# Build/unpack scratch dir: node-local tmpfs/NVMe is far faster than Lustre and is NOT
# counted against the /scratch inode quota. Inside a SLURM job use $SLURM_TMPDIR.
export APPTAINER_TMPDIR=${SLURM_TMPDIR:-/tmp}/apptainer_tmp.$USER
mkdir -p "$APPTAINER_TMPDIR" 2>/dev/null || true
export SIF_DIR=$WORK/sif                                         # per-instance read-only .sif (1 file each)
export OVERLAY_DIR=$WORK/overlays                                # per-instance writable overlay images
export APPTAINER_MODULE=apptainer/1.3.5

# ---- venv (fresh, on /scratch where inodes are available) ----
export VENV=$WORK/venv

# ---- results ----
export RESULTS_DIR=$WORK/results
export LOG_DIR=$WORK/logs

mkdir -p "$WORK"/{models,dataset,apptainer_cache,apptainer_tmp,sif,overlays,results,logs} 2>/dev/null || true
export WHEEL_DIR=$WORK/wheels
export APPTAINER_OVERLAY_MB=${APPTAINER_OVERLAY_MB:-4096}
