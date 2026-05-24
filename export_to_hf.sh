#!/usr/bin/env bash
# Convert the latest verl FSDP checkpoint of a training run into a
# HuggingFace-format model directory (safetensors + config + tokenizer),
# ready to upload to the Hub.
#
# Usage:
#   bash export_to_hf.sh [EXPERIMENT_DIR] [TARGET_DIR]
#
# Defaults to the paper150 run. Pick the highest global_step_* checkpoint.
#
# To upload afterwards (you do this yourself, with your HF token):
#   hf upload <your-username>/SWE-Master-4B-ATS-RL <TARGET_DIR> .
# or have the merger upload directly by adding:
#   --hf_upload_path <your-username>/SWE-Master-4B-ATS-RL [--private]
set -e

EXP_DIR=${1:-$(ls -d /home/temp/training_outputs/active_tests_paper150/*/ 2>/dev/null | tail -1)}
TARGET_DIR=${2:-/home/temp/models/SWE-Master-4B-ATS-RL}
MERGER=/home/temp/swemaster_runpod/DeepSWE_RL/rllm/verl/scripts/model_merger.py

if [ -z "$EXP_DIR" ] || [ ! -d "$EXP_DIR" ]; then
  echo "ERROR: experiment dir not found: $EXP_DIR"; exit 1
fi

# Find the highest-numbered global_step_* checkpoint that has an actor/ dir.
CKPT=$(ls -d "${EXP_DIR}"global_step_*/ 2>/dev/null | sort -t_ -k3 -n | tail -1)
if [ -z "$CKPT" ]; then
  echo "ERROR: no global_step_* checkpoint under $EXP_DIR"
  echo "Available:"; ls -la "$EXP_DIR"
  exit 1
fi
ACTOR_DIR="${CKPT}actor"
echo "Latest checkpoint: $CKPT"
echo "Actor dir:         $ACTOR_DIR"
echo "Target HF dir:     $TARGET_DIR"

mkdir -p "$TARGET_DIR"
/home/temp/venv/bin/python "$MERGER" merge \
  --backend fsdp \
  --local_dir "$ACTOR_DIR" \
  --target_dir "$TARGET_DIR"

echo ""
echo "=== HF model written to $TARGET_DIR ==="
ls -la "$TARGET_DIR"
echo ""
echo "To upload (with your HF token logged in via 'hf auth login'):"
echo "  hf upload <username>/SWE-Master-4B-ATS-RL $TARGET_DIR ."
