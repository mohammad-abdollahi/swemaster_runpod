#!/usr/bin/env bash
# Empirically find the largest data.max_response_length that completes a full
# PPO step:1 (rollout -> reward -> full-param actor update) on the 143 GB H200
# without OOM. Ascending ladder, 1 epoch (we kill right after step:1), stop at
# first failure; the largest success is written to /home/temp/best_max_response.
set -u
REPO=/home/temp/swemaster_runpod
SCRIPT="$REPO/DeepSWE_RL/rllm/examples/swe/train_comprehensive_swerebench_h200.sh"
RESULTS=/home/temp/max_response_search.results
: > "$RESULTS"
LADDER="24576 32768 40960 49152 57344 65536 81920 98304 122880"
PER_TEST_TIMEOUT=2100   # 35 min hard cap per setting

cleanup() {
  pkill -9 -f "venv/bin/python3 -m rllm.trainer" 2>/dev/null
  /home/temp/venv/bin/ray stop --force >/dev/null 2>&1
  sleep 3
  # remove ONLY swerebench/* rollout containers from this session, by id
  sg docker -c "docker ps -a --format '{{.ID}} {{.Image}}'" 2>/dev/null \
    | awk '/ swerebench\//{print $1}' \
    | while read -r c; do [ -n "$c" ] && sg docker -c "docker rm -f $c" >/dev/null 2>&1; done
  sleep 2
}

BEST=0
for MR in $LADDER; do
  cleanup
  LOG="/home/temp/search_mr_${MR}.log"
  echo "[$(date +%H:%M:%S)] === testing MAX_RESPONSE=$MR ===" | tee -a "$RESULTS"
  MAX_RESPONSE=$MR TOTAL_EPOCHS=1 \
    setsid bash -c "sg docker -c 'bash $SCRIPT'" > "$LOG" 2>&1 < /dev/null &
  start=$(date +%s)
  status="TIMEOUT"
  while :; do
    sleep 20
    now=$(date +%s); el=$((now-start))
    if grep -qE "step:1 - traj/steps_mean.*perf/max_memory_allocated_gb:" "$LOG" 2>/dev/null; then
      status="OK"; break
    fi
    if grep -qiE "OutOfMemoryError|CUDA out of memory|torch\.cuda\.OutOfMemory|HIP out of memory|Error executing job with overrides|Engine core initialization failed" "$LOG" 2>/dev/null; then
      status="FAIL"; break
    fi
    # process died without producing step:1
    if ! pgrep -f "venv/bin/python3 -m rllm.trainer" >/dev/null 2>&1; then
      if ! grep -qE "step:1 - traj/steps_mean" "$LOG" 2>/dev/null; then status="DIED"; break; fi
    fi
    [ "$el" -ge "$PER_TEST_TIMEOUT" ] && { status="TIMEOUT"; break; }
  done
  peak=$(grep -oE "perf/max_memory_allocated_gb:[0-9.]+" "$LOG" 2>/dev/null | tail -1)
  rew=$(grep -oE "critic/rewards/mean:[0-9.]+" "$LOG" 2>/dev/null | tail -1)
  echo "[$(date +%H:%M:%S)] MAX_RESPONSE=$MR -> $status  $peak  $rew  (elapsed ${el}s)" | tee -a "$RESULTS"
  if [ "$status" = "OK" ]; then
    BEST=$MR
    echo "$BEST" > /home/temp/best_max_response
  else
    echo "[$(date +%H:%M:%S)] stop ladder at first non-OK ($status); BEST=$BEST" | tee -a "$RESULTS"
    break
  fi
done
cleanup
echo "[$(date +%H:%M:%S)] SEARCH DONE. BEST MAX_RESPONSE=$BEST" | tee -a "$RESULTS"
