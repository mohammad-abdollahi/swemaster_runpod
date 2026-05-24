#!/usr/bin/env bash
# Save the trained model to HF format as soon as the FINAL checkpoint is
# written — do not wait for the (unwanted) validation pass.
#
# Strategy: poll for the highest global_step_* checkpoint. When training is
# near done (step 10 expected), export from the latest checkpoint. Also exits
# the wait if the trainer process dies (fallback).
set -e
EXP_DIR=$(ls -d /home/temp/training_outputs/active_tests_paper150/*/ 2>/dev/null | tail -1)
TARGET_DIR=/home/temp/models/SWE-Master-4B-ATS-RL
LOG=/home/temp/auto_export.log
FINAL_STEP=10   # expected last checkpoint (5 epochs x ~2 steps, save_freq=5)

echo "$(date -u) checkpoint-export watcher started; exp=$EXP_DIR" | tee -a "$LOG"

# Wait until the final checkpoint appears OR the trainer exits.
while true; do
  LATEST=$(ls -d "${EXP_DIR}"global_step_*/ 2>/dev/null | sort -t_ -k3 -n | tail -1)
  LATEST_N=$(echo "$LATEST" | grep -oE 'global_step_[0-9]+' | grep -oE '[0-9]+' || echo 0)
  if [ -n "$LATEST" ] && [ -d "${LATEST}actor" ] && [ "${LATEST_N:-0}" -ge "$FINAL_STEP" ]; then
    echo "$(date -u) final checkpoint present: $LATEST" | tee -a "$LOG"
    sleep 60   # let the checkpoint finish flushing
    break
  fi
  if ! pgrep -f "rllm.trainer.verl.train_agent_ppo" >/dev/null 2>&1; then
    echo "$(date -u) trainer exited; exporting from latest checkpoint ($LATEST)" | tee -a "$LOG"
    sleep 30
    break
  fi
  sleep 60
done

# Export latest checkpoint -> HF format
EXP_DIR=$(ls -d /home/temp/training_outputs/active_tests_paper150/*/ 2>/dev/null | tail -1)
if bash /home/temp/swemaster_runpod/export_to_hf.sh "$EXP_DIR" "$TARGET_DIR" >>"$LOG" 2>&1; then
  for f in tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json added_tokens.json generation_config.json; do
    [ -f "$TARGET_DIR/$f" ] || cp "/home/temp/models/SWE-Master-4B-SFT/$f" "$TARGET_DIR/" 2>/dev/null || true
  done
  echo "$(date -u) EXPORT OK -> $TARGET_DIR" | tee -a "$LOG"
  ls -la "$TARGET_DIR" | tee -a "$LOG"
else
  echo "$(date -u) EXPORT FAILED — see $LOG" | tee -a "$LOG"
fi
echo "AUTO_EXPORT_DONE $(date -u)" | tee -a "$LOG"
