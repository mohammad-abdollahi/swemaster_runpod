#!/usr/bin/env bash
# Autonomous Docker pull-retry loop. Every 30 min, probe whether Docker Hub
# rate-limit has reset (single pull test). When it works, pull all 106
# missing images in parallel and exit. Logs everything.
#
# Usage: setsid nohup bash pull_retry_loop.sh > /home/temp/pull_retry.log 2>&1 &
LOG=/home/temp/pull_retry.log
NEED_LIST=/tmp/need_pull.txt
INTERVAL=1800  # 30 min between probes

# Re-derive list of needed images from the curated 123-task parquet
PARQ=/home/temp/swemaster_runpod/data_examples/rl_data/swerebench_paper320/train_00000_verl.parquet
if [ ! -f "$NEED_LIST" ] || [ ! -s "$NEED_LIST" ]; then
  /home/temp/venv/bin/python <<PY > /tmp/all_imgs.txt
import pandas as pd, json
df = pd.read_parquet("$PARQ")
for _, r in df.iterrows():
    ei = r['extra_info']
    if isinstance(ei, str): ei = json.loads(ei)
    img = ei.get('docker_image') or ''
    if ':' not in img.split('/')[-1]: img += ':latest'
    if img: print(img)
PY
  : > "$NEED_LIST"
  while IFS= read -r img; do
    if ! sg docker -c "docker image inspect '$img'" >/dev/null 2>&1; then
      echo "$img" >> "$NEED_LIST"
    fi
  done < /tmp/all_imgs.txt
fi
echo "$(date -u) pull-retry-loop started; need to pull $(wc -l < "$NEED_LIST") images" | tee -a "$LOG"

ATTEMPT=0
while true; do
  NEED=$(wc -l < "$NEED_LIST")
  if [ "$NEED" -le 0 ]; then
    echo "$(date -u) all images pulled — exiting" | tee -a "$LOG"
    break
  fi
  ATTEMPT=$((ATTEMPT+1))
  PROBE=$(head -1 "$NEED_LIST")
  echo "$(date -u) attempt $ATTEMPT  probe-pull $PROBE" | tee -a "$LOG"
  if sg docker -c "docker pull '$PROBE'" >/dev/null 2>&1; then
    echo "$(date -u) RATE LIMIT CLEARED — starting parallel pull of $NEED images" | tee -a "$LOG"
    # Pulls succeeded; do the rest in parallel
    cat "$NEED_LIST" | xargs -I{} -P 4 bash -c '
      IMG="$1"
      if sg docker -c "docker pull \"$IMG\"" >/dev/null 2>&1; then
        echo "  OK $IMG"
      else
        echo "  FAIL $IMG"
      fi
    ' _ {} 2>&1 | tee -a "$LOG"
    # Rebuild needed list from what's still missing
    TMP=$(mktemp)
    while IFS= read -r img; do
      if ! sg docker -c "docker image inspect '$img'" >/dev/null 2>&1; then
        echo "$img" >> "$TMP"
      fi
    done < "$NEED_LIST"
    mv "$TMP" "$NEED_LIST"
    REMAIN=$(wc -l < "$NEED_LIST")
    PULLED=$((NEED - REMAIN))
    echo "$(date -u) pulled $PULLED  remaining $REMAIN" | tee -a "$LOG"
    if [ "$REMAIN" -le 0 ]; then
      echo "$(date -u) all 106 images successfully pulled" | tee -a "$LOG"
      break
    fi
    # Some failed — likely temporary; wait short interval before retry
    echo "$(date -u) some failed — waiting 10 min before retry" | tee -a "$LOG"
    sleep 600
    continue
  fi
  ERR=$(sg docker -c "docker pull '$PROBE' 2>&1" | tail -1)
  echo "$(date -u) still rate-limited: $ERR" | tee -a "$LOG"
  echo "$(date -u) sleeping ${INTERVAL}s before next attempt" | tee -a "$LOG"
  sleep "$INTERVAL"
done

# Final summary
PRESENT=$(sg docker -c "docker images --format '{{.Repository}}'" 2>/dev/null | grep -c swerebench)
echo "$(date -u) DONE — swerebench images on disk: $PRESENT" | tee -a "$LOG"
