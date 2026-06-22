#!/bin/bash
# Pre-pull all swerebench_verlive (SWE-Live + SWE-Verified overlap) dataset images
# so future trainings have them cached (no mid-rollout re-pull stalls).
# Reads /tmp/pull_images.txt (one image ref per line). Sequential to avoid starving
# the live training's docker/network. Logs OK/FAIL per image.
LOG=/home/temp/pull_images.log
LIST=/tmp/pull_images.txt
total=$(grep -c . "$LIST")
n=0; ok=0; fail=0
echo "[$(date -u +%H:%M:%S)] === pulling $total verlive images ===" > "$LOG"
while read -r img; do
  [ -z "$img" ] && continue
  n=$((n+1))
  if sg docker -c "docker pull '$img'" >/dev/null 2>&1; then
    ok=$((ok+1)); echo "[$(date -u +%H:%M:%S)] OK ($n/$total) $img" >> "$LOG"
  else
    fail=$((fail+1)); echo "[$(date -u +%H:%M:%S)] FAIL ($n/$total) $img" >> "$LOG"
  fi
done < "$LIST"
echo "[$(date -u +%H:%M:%S)] === DONE: $ok ok, $fail failed of $total ===" >> "$LOG"
