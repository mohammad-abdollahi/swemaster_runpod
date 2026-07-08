#!/usr/bin/env bash
# =============================================================================
# Disconnect-proof orchestrator for the 8x H100 run.  Branch: train-8-gpus.
# =============================================================================
# Launches run_8xh100.sh, survives SSH disconnect, relaunches verl on death
# (verl auto-resumes from the latest checkpoint), prunes checkpoints to the
# latest-2, and exports the final model at the end. Clean Ray teardown on every
# (re)launch so a dead driver never leaves a half-cluster behind.
#
# Usage (from repo root):
#   EXPERIMENT_NAME=run1 BASE_MODEL=/path/to/base M_FINAL=/path/to/out \
#     TRAIN_PARQUET=... VAL_PARQUET=... STEPS=150 \
#     setsid nohup bash orchestrator_8xh100.sh > orch_8xh100.log 2>&1 < /dev/null &
# Everything else (N_GPUS, TRAIN_BATCH_SIZE, MAX_RESPONSE, REFINE_MODE, ...)
# passes through to run_8xh100.sh / the launcher.  See README_8XH100.md.
set +e

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=${REPO_DIR:-${_SCRIPT_DIR}}
cd "${REPO}"

LOG=${ORCH_LOG:-${REPO}/orch_8xh100.log}
ALOG=${ARM_LOG:-${REPO}/train_8xh100.log}
RAY_BIN=${RAY_BIN:-ray}
LOCK=/tmp/orchestrator_8xh100.lock

EXP=${EXPERIMENT_NAME:?set EXPERIMENT_NAME}
PROJ=${PROJECT_NAME:-active_tests_swe_master_8xh100}
OUTROOT=${OUTPUT_ROOT:-${REPO}/training_outputs}/${PROJ}
STEPS=${STEPS:-${TOTAL_TRAINING_STEPS:-150}}
M_FINAL=${M_FINAL:-${REPO}/models/SWE-Master-4B-8xh100-${EXP}}
MAXRELAUNCH=${MAXRELAUNCH:-8}
KILL_MINERS=${KILL_MINERS:-0}     # set 1 only on boxes with the RunPod miner malware

export EXPERIMENT_NAME="$EXP" PROJECT_NAME="$PROJ"
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-$STEPS}

if ! mkdir "$LOCK" 2>/dev/null; then echo "[orch] lock held ($LOCK); exit" | tee -a "$LOG"; exit 0; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
log(){ echo "[orch $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

killminers(){ [ "$KILL_MINERS" = "1" ] || return 0
  for p in $(ps -eo pid,comm 2>/dev/null | awk '$2=="cudaa"||$2=="herominers"{print $1}'); do kill -9 "$p" 2>/dev/null; done; }

alive(){ pgrep -f 'rllm.trainer.verl.train_agent_ppo' >/dev/null 2>&1; }
OD(){ ls -dt ${OUTROOT}/${1}*/ 2>/dev/null | head -1; }
prune(){ ls -d ${1}global_step_*/ 2>/dev/null | sort -V | head -n -2 | xargs -r rm -rf; }

# Full teardown so a dead/zombie driver never leaves a half-cluster (see the
# single-GPU zombie-driver lesson): kill driver + ray, wipe the ray tmp dir.
teardown(){
  pkill -9 -f 'rllm.trainer.verl.train_agent_ppo' 2>/dev/null
  pkill -9 -f 'run_8xh100.sh' 2>/dev/null
  for p in $(pgrep -f 'ray/core/src/ray|ray::' 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  "${RAY_BIN}" stop --force >/dev/null 2>&1
  sleep 3
  rm -rf /tmp/ray 2>/dev/null
}

launch(){ killminers
  # wait for GPUs to be free enough before (re)launching
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    [ -z "$used" ] && break
    [ "$used" -lt 40000 ] && break
    log "GPUs busy (max ${used}MiB used); waiting..."; sleep 30; killminers
  done
  setsid nohup bash "${REPO}/run_8xh100.sh" > "$ALOG" 2>&1 < /dev/null &
  log "launched ${EXP} (steps=${STEPS})"; sleep 90
}

log "=== 8xH100 orchestrator start: exp=${EXP} proj=${PROJ} steps=${STEPS} refine=${REFINE_MODE:-agentic} base=${BASE_MODEL:-<default>} -> ${M_FINAL} ==="
teardown
launch

RELAUNCH=0
while true; do
  sleep 60; killminers
  d=$(OD "$EXP"); [ -n "$d" ] && prune "$d"
  if grep -qaE 'training done|Final validation metrics' "$ALOG" 2>/dev/null; then log "$EXP done-marker"; break; fi
  if ! alive; then
    if grep -qaE 'training done|Final validation metrics' "$ALOG" 2>/dev/null; then break; fi
    if [ "$RELAUNCH" -lt "$MAXRELAUNCH" ]; then
      RELAUNCH=$((RELAUNCH+1)); log "$EXP died -> relaunch #$RELAUNCH (verl auto-resume)"
      teardown; launch
    else log "FATAL: exceeded $MAXRELAUNCH relaunches"; break; fi
  fi
done

d=$(OD "$EXP")
log "$EXP finished; exporting from ${d} -> ${M_FINAL}"
CUDA_VISIBLE_DEVICES='' bash "${REPO}/export_to_hf.sh" "$d" "$M_FINAL" >> "$LOG" 2>&1 \
  && log "$EXP exported -> ${M_FINAL}" || log "$EXP export FAILED (merge manually from ${d})"
"${RAY_BIN}" stop --force >> "$LOG" 2>&1
log "=== 8xH100 orchestrator complete (model: ${M_FINAL}) ==="
