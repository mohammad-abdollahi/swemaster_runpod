#!/usr/bin/env bash
# Autonomous monitor — runs every 5 min, checks pipeline health, writes a
# compact human-readable status file the user can `cat` when waking up.
# Logs every observation to /home/temp/monitor_training.log for forensics.
#
# Usage: setsid nohup bash monitor_training.sh > /dev/null 2>&1 &

LOG=/home/temp/monitor_training.log
STATUS=/home/temp/training_status.txt
TRAIN_LOG=/home/temp/train_paper50steps.log
METRICS_CSV=/home/temp/train_paper50steps_metrics.csv
EXP_GLOB="/home/temp/training_outputs/active_tests_paper320/paper320_2026-06-01_05-10-29"
MODEL_DIR=/home/temp/models/SWE-Master-4B-ATS-RL-50steps
INTERVAL=300   # 5 min between checks

# Track stalls — if log mtime hasn't advanced for STALL_LIMIT we flag it
LAST_LOG_MTIME=0
STALL_COUNT=0
STALL_LIMIT=12  # 12 × 5 min = 1 hour of no log advance = "stalled"

write_status() {
  {
    echo "=== ATS-RL 50-step training status ==="
    echo "updated: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
    echo ""

    # 1) Trainer
    if pgrep -f "rllm.trainer.verl.train_agent_ppo" >/dev/null; then
      ETIME=$(ps -o etime= -p "$(pgrep -f 'rllm.trainer.verl.train_agent_ppo' | head -1)" | tr -d ' ')
      echo "trainer:        RUNNING (elapsed $ETIME)"
    else
      echo "trainer:        *** STOPPED ***"
    fi

    # 2) Watchers
    pgrep -f "live_metrics_csv" >/dev/null && echo "metrics watcher: running" || echo "metrics watcher: *** STOPPED ***"
    pgrep -f "overwrite_hf_watcher" >/dev/null && echo "HF watcher:      running" || echo "HF watcher:      *** STOPPED ***"

    # 3) Latest PPO step
    LAST_STEP=$(grep -aoE "epoch [0-9]+, step [0-9]+" "$TRAIN_LOG" 2>/dev/null | tail -1)
    echo "latest log step: ${LAST_STEP:-not yet started}"
    if [ -f "$METRICS_CSV" ]; then
      NROWS=$(($(wc -l < "$METRICS_CSV") - 1))
      echo "metrics rows:    $NROWS / 50"
    else
      echo "metrics rows:    csv not yet written"
    fi

    # 4) Latest reward + entropy (from CSV)
    if [ -f "$METRICS_CSV" ] && [ "$(wc -l < "$METRICS_CSV")" -gt 1 ]; then
      LATEST=$(tail -1 "$METRICS_CSV")
      REW=$(echo "$LATEST" | awk -F, '{print $2}')
      GRAD=$(echo "$LATEST" | awk -F, '{print $4}')
      echo "latest reward:   $REW"
      echo "latest grad:     $GRAD"
    fi

    # 5) HF model
    if [ -d "$MODEL_DIR" ]; then
      LAST_EXPORTED=$(cat "$MODEL_DIR/.last_exported_step" 2>/dev/null || echo "none")
      SZ=$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)
      echo "HF model:        step=$LAST_EXPORTED  size=$SZ"
    fi

    # 6) Disk
    df -h / | tail -1 | awk '{print "disk:           ", $3, "used,", $4, "free"}'

    # 7) GPU
    nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null \
      | head -1 | awk -F, '{print "GPU:            ", $1, "used,", $2, "free, util", $3}'

    # 8) Docker pulls (106 missing as of start)
    HAVE=$(sg docker -c "docker images --format '{{.Repository}}'" 2>/dev/null | grep -c swerebench)
    echo "swerebench imgs: $HAVE on disk (started 63, target 169 for full 123-task parquet)"
    if [ -f /home/temp/pull_retry.log ]; then
      LAST_PULL=$(tail -1 /home/temp/pull_retry.log 2>/dev/null)
      echo "pull-retry:      $LAST_PULL"
    fi

    # 9) Real errors only (skip benign Warnings/Tips)
    ERRORS=$(grep -aE "RuntimeError|Traceback|CUDA out of memory|OutOfMemoryError|trainer crashed|Killed" "$TRAIN_LOG" 2>/dev/null | grep -vE "FutureWarning|UserWarning|Tip:" | wc -l)
    if [ "${ERRORS:-0}" -gt 0 ]; then
      echo ""
      echo "*** REAL ERRORS DETECTED ($ERRORS); recent: ***"
      grep -aE "RuntimeError|Traceback|CUDA out of memory|OutOfMemoryError" "$TRAIN_LOG" | grep -vE "FutureWarning|UserWarning|Tip:" | tail -5
    fi

    # 10) Stall warning
    if [ "$STALL_COUNT" -ge "$STALL_LIMIT" ]; then
      echo ""
      echo "*** WARNING: training log hasn't advanced in $((STALL_COUNT * INTERVAL / 60)) min ***"
    fi
  } > "$STATUS"
}

cleanup_old_ckpts() {
  EXP_DIR=$(ls -dt /home/temp/training_outputs/active_tests_paper320/*/ 2>/dev/null | head -1)
  [ -z "$EXP_DIR" ] && return
  STEPS=$(ls -d "${EXP_DIR}"global_step_*/ 2>/dev/null | grep -oE 'global_step_[0-9]+' | grep -oE '[0-9]+' | sort -n)
  [ -z "$STEPS" ] && return
  COUNT=$(echo "$STEPS" | wc -l)
  if [ "$COUNT" -le 2 ]; then return; fi
  DEL=$((COUNT - 2))
  TO_DELETE=$(echo "$STEPS" | head -n "$DEL")
  NOW_S=$(date +%s)
  for N in $TO_DELETE; do
    CK="${EXP_DIR}global_step_$N"
    [ ! -d "$CK" ] && continue
    M=$(stat -c %Y "$CK" 2>/dev/null || echo 0)
    AGE=$((NOW_S - M))
    # Safety: only delete dirs older than 5 min (avoid racing trainer's active write)
    if [ "$AGE" -gt 300 ]; then
      SZ=$(du -sh "$CK" 2>/dev/null | awk '{print $1}')
      rm -rf "$CK" 2>/dev/null
      echo "$(date -u) cleanup: removed global_step_$N ($SZ, age ${AGE}s)" >> "$LOG"
    fi
  done
}

echo "$(date -u) monitor started" >> "$LOG"

while true; do
  write_status
  cleanup_old_ckpts

  # Stall detection: train log mtime
  if [ -f "$TRAIN_LOG" ]; then
    M=$(stat -c %Y "$TRAIN_LOG" 2>/dev/null || echo 0)
    if [ "$M" -gt "$LAST_LOG_MTIME" ]; then
      LAST_LOG_MTIME=$M
      STALL_COUNT=0
    else
      STALL_COUNT=$((STALL_COUNT + 1))
    fi
  fi

  # Restart HF watcher if it died but trainer is alive (rare, but rescuable)
  if pgrep -f "rllm.trainer.verl.train_agent_ppo" >/dev/null \
     && ! pgrep -f "overwrite_hf_watcher" >/dev/null; then
    EXP_DIR=$(ls -dt /home/temp/training_outputs/active_tests_paper320/*/ 2>/dev/null | head -1)
    if [ -n "$EXP_DIR" ]; then
      echo "$(date -u) HF watcher dead but trainer alive — restarting" >> "$LOG"
      setsid nohup bash /home/temp/swemaster_runpod/overwrite_hf_watcher.sh \
        "$EXP_DIR" SWE-Master-4B-ATS-RL-50steps 50 \
        >> /home/temp/overwrite_hf.log 2>&1 < /dev/null &
      disown
    fi
  fi
  if pgrep -f "rllm.trainer.verl.train_agent_ppo" >/dev/null \
     && ! pgrep -f "live_metrics_csv" >/dev/null; then
    echo "$(date -u) metrics watcher dead but trainer alive — restarting" >> "$LOG"
    setsid nohup bash /home/temp/swemaster_runpod/live_metrics_csv.sh \
      "$TRAIN_LOG" "$METRICS_CSV" \
      >> /home/temp/live_metrics.log 2>&1 < /dev/null &
    disown
  fi

  sleep "$INTERVAL"
done
