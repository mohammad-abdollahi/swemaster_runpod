#!/usr/bin/env bash
# Compact one-line health summary of an ATS training run. Designed for
# frequent polling so infra failures (NO_CONTAINER, OOM, rate limits) are
# caught within minutes instead of hours.
#
# Usage: bash training_status.sh /home/temp/train_active_paperbatch.log
LOG=${1:-/home/temp/train_active_paperbatch.log}

steps=$(grep -c "step:[0-9]* - traj/steps_mean" "$LOG" 2>/dev/null)
ats=$(grep -c "ATS terminal" "$LOG" 2>/dev/null)
solves=$(grep "ATS terminal" "$LOG" 2>/dev/null | grep -vc "frac_solved=0.000")
p2pbroken=$(grep -oE "p2p_broken=[0-9]+/[0-9]+" "$LOG" 2>/dev/null | tail -1)
nocontainer=$(grep -c "initial container failed" "$LOG" 2>/dev/null)
ratelimit=$(grep -c "toomanyrequests" "$LOG" 2>/dev/null)
traceback=$(grep -c "Traceback" "$LOG" 2>/dev/null)
oom=$(grep -ciE "out of memory|CUDA error|OOM" "$LOG" 2>/dev/null)
lasttraj=$(grep -oE "Number of Trajectories [0-9]+/[0-9]+" "$LOG" 2>/dev/null | tail -1)
curstep=$(grep -oE "now in fit_agent, batch, epoch [0-9]+, step [0-9]+" "$LOG" 2>/dev/null | tail -1)
gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tail -1)
ctr=$(sg docker -c "docker ps -q | wc -l" 2>/dev/null)
alive=$(pgrep -f "rllm.trainer.verl.train_agent_ppo" >/dev/null 2>&1 && echo yes || echo NO)

echo "$(date -u +%H:%M:%S) | alive=$alive | $curstep | ppo_steps_done=$steps | traj=$lasttraj | ATS_terminals=$ats solves=$solves | p2p=$p2pbroken | ERR: nocontainer=$nocontainer ratelimit=$ratelimit traceback=$traceback oom=$oom | gpu=${gpu} ctr=$ctr"
