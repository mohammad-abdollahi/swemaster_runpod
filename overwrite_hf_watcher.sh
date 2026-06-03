#!/usr/bin/env bash
# Watch the FSDP checkpoint dir for new global_step_N folders and export each
# one to a SINGLE HF dir (overwriting the previous step's HF weights). This
# keeps the latest trained model always available at /home/temp/models/<NAME>/.
#
# Disk: ~7.5 GB at any time (one HF copy), regardless of step count.
#
# Usage: setsid nohup bash overwrite_hf_watcher.sh <EXP_DIR> <MODEL_NAME> <FINAL_STEP> \
#                       > /home/temp/overwrite_hf.log 2>&1 &
set -e
EXP_DIR=${1:?need EXP_DIR}
MODEL_NAME=${2:-SWE-Master-4B-ATS-RL-50steps}
FINAL_STEP=${3:-50}
TARGET=/home/temp/models/${MODEL_NAME}
LOG=/home/temp/overwrite_hf.log
mkdir -p "$TARGET"

echo "$(date -u) overwrite-HF watcher: exp=$EXP_DIR -> $TARGET final_step=$FINAL_STEP" | tee -a "$LOG"

LAST_EXPORTED=$(cat "$TARGET/.last_exported_step" 2>/dev/null || echo 0)
BEST_REWARD=$(cat "$TARGET/.best_reward" 2>/dev/null || echo "0")
REWARD_GUARD_RATIO=${REWARD_GUARD_RATIO:-0.5}   # skip export if step reward < ratio * best_so_far
METRICS_CSV=${METRICS_CSV:-/home/temp/train_paper50steps_metrics.csv}
echo "  resuming from last_exported_step=$LAST_EXPORTED  best_reward_so_far=$BEST_REWARD  guard_ratio=$REWARD_GUARD_RATIO" | tee -a "$LOG"

CONSEC_IDLE=0
while true; do
  # Find highest-N global_step_* checkpoint currently on disk
  LATEST_N=0
  for CKPT in $(ls -d "${EXP_DIR}"/global_step_*/ 2>/dev/null); do
    N=$(basename "$CKPT" | grep -oE '[0-9]+$')
    if [ -z "$N" ]; then continue; fi
    if [ ! -d "${CKPT}actor" ]; then continue; fi
    if [ "$N" -gt "$LATEST_N" ]; then LATEST_N=$N; fi
  done

  if [ "$LATEST_N" -gt "$LAST_EXPORTED" ]; then
    # Reward guard: check metrics CSV for this step's reward
    NEW_REWARD=$(awk -F, -v step="$LATEST_N" 'NR>1 && $1==step {print $2}' "$METRICS_CSV" 2>/dev/null | head -1)
    if [ -z "$NEW_REWARD" ]; then
      echo "$(date -u) step $LATEST_N: CSV row not yet written; waiting for metrics watcher" | tee -a "$LOG"
      sleep 30; continue
    fi
    # Update best-so-far
    if awk -v n="$NEW_REWARD" -v b="$BEST_REWARD" 'BEGIN{exit !(n>b)}'; then
      BEST_REWARD=$NEW_REWARD
      echo "$BEST_REWARD" > "$TARGET/.best_reward"
      echo "$(date -u) step $LATEST_N reward $NEW_REWARD is NEW BEST" | tee -a "$LOG"
    fi
    # Reward guard: skip if significantly lower than best
    THRESHOLD=$(awk -v b="$BEST_REWARD" -v r="$REWARD_GUARD_RATIO" 'BEGIN{printf "%.4f", b*r}')
    if awk -v n="$NEW_REWARD" -v t="$THRESHOLD" 'BEGIN{exit !(n<t)}'; then
      echo "$(date -u) step $LATEST_N reward $NEW_REWARD < threshold $THRESHOLD ($REWARD_GUARD_RATIO × best $BEST_REWARD); SKIP export — model dir stays at step $LAST_EXPORTED" | tee -a "$LOG"
      # Bump LAST_EXPORTED so we don't keep re-evaluating this step
      LAST_EXPORTED=$LATEST_N
      echo "$LAST_EXPORTED" > "$TARGET/.last_exported_step"
      sleep 30; continue
    fi
    echo "$(date -u) new step $LATEST_N (reward $NEW_REWARD ≥ threshold $THRESHOLD); overwriting $TARGET" | tee -a "$LOG"
    sleep 20  # FSDP shard flush
    if bash /home/temp/swemaster_runpod/export_to_hf.sh "$EXP_DIR" "$TARGET" >>"$LOG" 2>&1; then
      for f in tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json added_tokens.json generation_config.json; do
        [ -f "$TARGET/$f" ] || cp "/home/temp/models/SWE-Master-4B-SFT/$f" "$TARGET/" 2>/dev/null || true
      done
      LAST_EXPORTED=$LATEST_N
      echo "$LAST_EXPORTED" > "$TARGET/.last_exported_step"
      echo "$(date -u) EXPORT OK: step $LATEST_N reward $NEW_REWARD -> $TARGET ($(du -sh "$TARGET" | cut -f1))" | tee -a "$LOG"
    else
      echo "$(date -u) EXPORT FAILED for step $LATEST_N; will retry next poll" | tee -a "$LOG"
    fi
  fi

  # exit when we've exported the final step OR trainer is gone for 5 min
  if [ "$LAST_EXPORTED" -ge "$FINAL_STEP" ]; then
    echo "$(date -u) final step $FINAL_STEP exported; exiting" | tee -a "$LOG"
    break
  fi
  if ! pgrep -f "rllm.trainer.verl.train_agent_ppo" >/dev/null 2>&1; then
    CONSEC_IDLE=$((CONSEC_IDLE+1))
    if [ "$CONSEC_IDLE" -ge 10 ]; then
      echo "$(date -u) trainer gone for 5 min; final export attempt then exiting" | tee -a "$LOG"
      # Try one last export of the current latest
      LATEST_N=0
      for CKPT in $(ls -d "${EXP_DIR}"/global_step_*/ 2>/dev/null); do
        N=$(basename "$CKPT" | grep -oE '[0-9]+$')
        if [ -n "$N" ] && [ "$N" -gt "$LATEST_N" ]; then LATEST_N=$N; fi
      done
      if [ "$LATEST_N" -gt "$LAST_EXPORTED" ]; then
        bash /home/temp/swemaster_runpod/export_to_hf.sh "$EXP_DIR" "$TARGET" >>"$LOG" 2>&1 || true
        echo "$LATEST_N" > "$TARGET/.last_exported_step"
      fi
      break
    fi
  else
    CONSEC_IDLE=0
  fi
  sleep 30
done
echo "OVERWRITE_HF_WATCHER_DONE $(date -u)" | tee -a "$LOG"
