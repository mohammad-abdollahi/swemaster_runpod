#!/bin/bash
# Compact health report for the overnight control_live64 babysitting loop.
# Prints: phase, pull status, training proc/GPU, latest PPO step metrics, errors.
cd /home/temp/swemaster_runpod
PULL_LOG=data_examples/rl_data/swerebench_live64/pull.log
TRAIN_LOG=/home/temp/control_live64.log

echo "===== $(date -u '+%Y-%m-%d %H:%M:%S UTC') ====="

# --- pull ---
if [ -f "$PULL_LOG" ]; then
  ready=$(grep -cE 'OK|CACHED' "$PULL_LOG" 2>/dev/null || echo 0)
  fails=$(grep -c '^  FAIL' "$PULL_LOG" 2>/dev/null || echo 0)
  if pgrep -f pull_live64 >/dev/null; then pstat="PULLING"; else pstat="done"; fi
  echo "PULL: $ready/41 ready, $fails fail ($pstat)"
fi

# --- training process + GPU ---
if pgrep -f train_agent_ppo >/dev/null; then
  echo "TRAIN: RUNNING (pid $(pgrep -f train_agent_ppo | head -1))"
else
  echo "TRAIN: not running"
fi
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -1 | awk '{print "GPU: "$0}'
running=$(sg docker -c "docker ps -q" 2>/dev/null | wc -l)
echo "containers running: $running"

# --- latest step metrics from training log ---
if [ -f "$TRAIN_LOG" ]; then
  echo "--- training log (last step metrics) ---"
  # verl prints lines like: step:N ... critic/rewards/mean:X ... actor/grad_norm:Y
  laststep=$(grep -oE "step:[0-9]+" "$TRAIN_LOG" 2>/dev/null | tail -1)
  echo "last step line: ${laststep:-<none yet>}"
  grep -E "ATS terminal" "$TRAIN_LOG" 2>/dev/null | tail -1 | cut -c1-160
  # key verl metrics from the most recent metric dump
  grep -oE "(critic/rewards/mean|actor/grad_norm|val-core[^ ]*|response_length/mean):[0-9.eE+-]+" "$TRAIN_LOG" 2>/dev/null | tail -8
  echo "--- recent errors/warnings ---"
  grep -iE "NO_CONTAINER|CUDA out of memory|Traceback|RayActorError|Error executing|ray stop|Killed" "$TRAIN_LOG" 2>/dev/null | tail -4
  echo "log size: $(wc -l < "$TRAIN_LOG") lines, last write $(stat -c %y "$TRAIN_LOG" 2>/dev/null | cut -d. -f1)"
fi
echo "================================================"
