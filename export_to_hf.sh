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
# Repo-relative so this works regardless of where the repo is checked out
# (8xH100 box, etc.). Falls back to the venv python on the original machine.
_EXPORT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MERGER=${MERGER:-${_EXPORT_SCRIPT_DIR}/DeepSWE_RL/rllm/verl/scripts/model_merger.py}
PYTHON_BIN=${PYTHON_BIN:-$([ -x /home/temp/venv/bin/python ] && echo /home/temp/venv/bin/python || echo python3)}

if [ -z "$EXP_DIR" ] || [ ! -d "$EXP_DIR" ]; then
  echo "ERROR: experiment dir not found: $EXP_DIR"; exit 1
fi

# Find the highest-numbered global_step_* checkpoint that HAS a populated
# actor/ dir (verl's max_actor_ckpt_to_keep prunes older ones, so we must
# skip empty/stale step dirs). sort -V handles numeric components correctly
# inside path strings — the previous sort -t_ -k3 -n broke when EXP_DIR
# had multiple underscores (e.g. timestamped subdirs).
CKPT=""
for d in $(ls -d "${EXP_DIR}"global_step_*/ 2>/dev/null | sort -V -r); do
  # Require BOTH the FSDP shard AND config.json — config.json is written last,
  # so its presence is the green light that the checkpoint is complete.
  # Accept ANY FSDP world size: single-GPU writes model_world_size_1_rank_0.pt,
  # an 8-GPU run writes model_world_size_8_rank_0.pt. config.json is written
  # last, so its presence marks a complete checkpoint.
  if ls "${d}actor/"model_world_size_*_rank_0.pt >/dev/null 2>&1 && [ -f "${d}actor/config.json" ]; then
    CKPT="$d"; break
  fi
done
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
"${PYTHON_BIN}" "$MERGER" merge \
  --backend fsdp \
  --local_dir "$ACTOR_DIR" \
  --target_dir "$TARGET_DIR"

echo ""
echo "=== HF model written to $TARGET_DIR ==="
ls -la "$TARGET_DIR"
echo ""
echo "To upload (with your HF token logged in via 'hf auth login'):"
echo "  hf upload <username>/SWE-Master-4B-ATS-RL $TARGET_DIR ."
