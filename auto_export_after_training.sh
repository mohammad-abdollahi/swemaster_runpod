#!/usr/bin/env bash
# Wait for the paper150 training process to exit, then automatically convert
# the latest verl FSDP checkpoint to a HuggingFace-format model dir.
# Runs unattended so the model is saved even with no one watching.
set -e
EXP_GLOB="/home/temp/training_outputs/active_tests_paper150"
TARGET_DIR=/home/temp/models/SWE-Master-4B-ATS-RL
EXPORT_LOG=/home/temp/auto_export.log

echo "$(date -u) waiting for training (rllm.trainer.verl.train_agent_ppo) to exit..." | tee -a "$EXPORT_LOG"
while pgrep -f "rllm.trainer.verl.train_agent_ppo" >/dev/null 2>&1; do
  sleep 120
done
echo "$(date -u) training process gone; waiting 60s for final checkpoint flush" | tee -a "$EXPORT_LOG"
sleep 60

EXP_DIR=$(ls -d ${EXP_GLOB}/*/ 2>/dev/null | tail -1)
echo "$(date -u) experiment dir: $EXP_DIR" | tee -a "$EXPORT_LOG"
ls -la "$EXP_DIR" | tee -a "$EXPORT_LOG"

# Run the converter (writes HF model to TARGET_DIR)
if bash /home/temp/swemaster_runpod/export_to_hf.sh "$EXP_DIR" "$TARGET_DIR" >>"$EXPORT_LOG" 2>&1; then
  echo "$(date -u) EXPORT OK -> $TARGET_DIR" | tee -a "$EXPORT_LOG"
  # copy tokenizer from base model if merger didn't include it
  for f in tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json added_tokens.json generation_config.json; do
    [ -f "$TARGET_DIR/$f" ] || cp "/home/temp/models/SWE-Master-4B-SFT/$f" "$TARGET_DIR/" 2>/dev/null || true
  done
  echo "$(date -u) final HF dir contents:" | tee -a "$EXPORT_LOG"
  ls -la "$TARGET_DIR" | tee -a "$EXPORT_LOG"
else
  echo "$(date -u) EXPORT FAILED — see $EXPORT_LOG" | tee -a "$EXPORT_LOG"
fi
echo "AUTO_EXPORT_DONE $(date -u)" | tee -a "$EXPORT_LOG"
