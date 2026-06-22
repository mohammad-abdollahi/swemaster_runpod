#!/bin/bash
# Standing miner-killer watchdog (user request 2026-06-21). Kills any crypto-miner
# process the moment it appears. Targets ONLY known miner signatures so it never
# touches the training (ray/vLLM/python) or normal processes. Detached; logs kills.
LOG=/home/temp/miner_killer.log
echo "[$(date -u +%H:%M:%S)] miner-killer watchdog started (pid $$)" >> "$LOG"
while true; do
  # 1) exact process name 'cudaa'
  for p in $(ps -eo pid,comm | awk '$2=="cudaa"{print $1}'); do
    kill -9 "$p" 2>/dev/null && echo "[$(date -u +%H:%M:%S)] killed cudaa pid $p" >> "$LOG"
  done
  # 2) cmdline signatures of known miners (herominers/pearlhash/xmrig/stratum/.gpu)
  for p in $(ps -eo pid,cmd | grep -iE 'herominers|pearlhash|xmrig|stratum\+tcp|/\.gpu/' | grep -viE 'miner_killer|grep -iE' | awk '{print $1}'); do
    kill -9 "$p" 2>/dev/null && echo "[$(date -u +%H:%M:%S)] killed miner-sig pid $p" >> "$LOG"
  done
  # 3) remove the known miner staging dir if it reappears
  [ -d /var/tmp/.gpu ] && rm -rf /var/tmp/.gpu 2>/dev/null && echo "[$(date -u +%H:%M:%S)] removed /var/tmp/.gpu" >> "$LOG"
  sleep 8
done
