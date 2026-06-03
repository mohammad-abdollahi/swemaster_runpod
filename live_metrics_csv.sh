#!/usr/bin/env bash
# Sidecar: keep a per-PPO-step metrics CSV refreshed in real time during
# training. Re-parses the training log every 60s. Crash-safe — even if the
# trainer dies on step 73, the CSV on disk holds all 73 completed steps for
# plotting.
#
# Usage: setsid nohup bash live_metrics_csv.sh /home/temp/train_paper320.log \
#                       /home/temp/train_paper320_metrics.csv >/tmp/live_metrics.log 2>&1 &
set -e
LOG=${1:-/home/temp/train_paper320.log}
OUT=${2:-/home/temp/train_paper320_metrics.csv}
INTERVAL=${INTERVAL:-60}

echo "$(date -u) live-metrics watcher: $LOG -> $OUT (every ${INTERVAL}s)"
while true; do
  if [ -f "$LOG" ]; then
    /home/temp/venv/bin/python /home/temp/swemaster_runpod/dump_metrics_csv.py "$LOG" "$OUT" >/dev/null 2>&1 || true
    NROWS=$(($(wc -l < "$OUT" 2>/dev/null || echo 1) - 1))
    echo "$(date -u) refreshed $OUT: $NROWS rows"
  fi
  # Exit gracefully when trainer is gone AND log hasn't grown for 3 minutes
  if ! pgrep -f "rllm.trainer.verl.train_agent_ppo" >/dev/null 2>&1; then
    if [ -f "$LOG" ]; then
      LAST_MOD=$(stat -c %Y "$LOG")
      NOW=$(date +%s)
      if [ $((NOW - LAST_MOD)) -gt 180 ]; then
        /home/temp/venv/bin/python /home/temp/swemaster_runpod/dump_metrics_csv.py "$LOG" "$OUT" >/dev/null 2>&1 || true
        echo "$(date -u) trainer exited; final refresh done"
        break
      fi
    fi
  fi
  sleep "$INTERVAL"
done
