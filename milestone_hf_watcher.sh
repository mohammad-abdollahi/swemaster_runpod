#!/usr/bin/env bash
# Watch the FSDP checkpoint directory for new global_step_N folders. When N
# hits an epoch-boundary multiple, export it to HF format under a numbered
# subdir: /home/temp/models/SWE-Master-4B-ATS-RL-50ep/epoch_K/. The very
# last epoch's export is also mirrored to the bare model dir for easy upload.
#
# Defaults assume the paper320 config: 320 tasks / batch 16 = 20 steps/epoch
# × 5 epochs → epoch-boundary steps are {20, 40, 60, 80, 100}.
#
# Usage: setsid nohup bash milestone_hf_watcher.sh /path/to/exp_dir \
#                       SWE-Master-4B-ATS-RL-50ep 20 5 \
#                       >/tmp/milestone_hf.log 2>&1 &
set -e
EXP_DIR=${1:?need experiment dir (e.g. /home/temp/training_outputs/active_tests_50ep/<ts>)}
MODEL_NAME=${2:-SWE-Master-4B-ATS-RL-50ep}
STEPS_PER_EPOCH=${3:-20}
TOTAL_EPOCHS=${4:-5}
TARGET_ROOT=/home/temp/models/${MODEL_NAME}
LOG=/home/temp/milestone_hf.log
mkdir -p "$TARGET_ROOT"

echo "$(date -u) milestone-HF watcher: exp=$EXP_DIR steps_per_epoch=$STEPS_PER_EPOCH total_epochs=$TOTAL_EPOCHS" | tee -a "$LOG"
echo "  target: $TARGET_ROOT/epoch_{1..${TOTAL_EPOCHS}}/" | tee -a "$LOG"

# Track which epochs we've already exported (idempotent across restarts).
declare -A DONE
for K in $(ls -d "$TARGET_ROOT"/epoch_*/ 2>/dev/null); do
  N=$(basename "$K" | sed 's/epoch_//')
  DONE[$N]=1
  echo "  already exported: epoch_$N" | tee -a "$LOG"
done

while true; do
  for K in $(seq 1 "$TOTAL_EPOCHS"); do
    if [ -n "${DONE[$K]:-}" ]; then continue; fi
    STEP=$((K * STEPS_PER_EPOCH))
    CKPT="${EXP_DIR}/global_step_${STEP}"
    if [ -d "${CKPT}/actor" ]; then
      OUT="${TARGET_ROOT}/epoch_${K}"
      mkdir -p "$OUT"
      echo "$(date -u) epoch ${K} ready (step ${STEP}); exporting -> $OUT" | tee -a "$LOG"
      sleep 30  # let FSDP shard writes finish flushing
      if bash /home/temp/swemaster_runpod/export_to_hf.sh "$EXP_DIR" "$OUT" >>"$LOG" 2>&1; then
        for f in tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json added_tokens.json generation_config.json; do
          [ -f "$OUT/$f" ] || cp "/home/temp/models/SWE-Master-4B-SFT/$f" "$OUT/" 2>/dev/null || true
        done
        DONE[$K]=1
        echo "$(date -u) EXPORT OK: epoch_${K} -> $OUT ($(du -sh "$OUT" | cut -f1))" | tee -a "$LOG"
        # final epoch also goes to the bare model dir for easy upload
        if [ "$K" = "$TOTAL_EPOCHS" ]; then
          cp -ru "$OUT"/* "$TARGET_ROOT/" 2>/dev/null || true
          echo "$(date -u) mirrored final epoch_${K} to $TARGET_ROOT (for upload)" | tee -a "$LOG"
        fi
      else
        echo "$(date -u) EXPORT FAILED for epoch_${K}; will retry next poll" | tee -a "$LOG"
        rm -rf "$OUT" 2>/dev/null || true
      fi
    fi
  done

  # All epochs done?
  ALL_DONE=1
  for K in $(seq 1 "$TOTAL_EPOCHS"); do
    [ -z "${DONE[$K]:-}" ] && ALL_DONE=0
  done
  if [ "$ALL_DONE" = 1 ]; then
    echo "$(date -u) all $TOTAL_EPOCHS epochs exported; watcher exiting" | tee -a "$LOG"
    break
  fi
  # also exit if trainer is gone AND we've waited a bit
  if ! pgrep -f "rllm.trainer.verl.train_agent_ppo" >/dev/null 2>&1; then
    echo "$(date -u) trainer exited; one final pass then quitting" | tee -a "$LOG"
    sleep 120
    # one last pass through the loop
    for K in $(seq 1 "$TOTAL_EPOCHS"); do
      if [ -n "${DONE[$K]:-}" ]; then continue; fi
      STEP=$((K * STEPS_PER_EPOCH))
      CKPT="${EXP_DIR}/global_step_${STEP}"
      if [ -d "${CKPT}/actor" ]; then
        OUT="${TARGET_ROOT}/epoch_${K}"
        mkdir -p "$OUT"
        bash /home/temp/swemaster_runpod/export_to_hf.sh "$EXP_DIR" "$OUT" >>"$LOG" 2>&1 || true
      fi
    done
    break
  fi
  sleep 60
done
echo "MILESTONE_HF_WATCHER_DONE $(date -u)" | tee -a "$LOG"
