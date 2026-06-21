#!/bin/bash
# B1/R12 matched-ablation orchestrator — runs TWO 10-step arms sequentially on one GPU,
# both ATS-RL v1 reward (MID_EPISODE=0) on swerebench_verlive, SFT base:
#   arm 1 = baseline (refine OFF, tick ON)  -> /home/temp/models/SWE-Master-4B-ATS-B0-refoff
#   arm 2 = refine   (refine ON)            -> /home/temp/models/SWE-Master-4B-ATS-B1-refon
# Both tick identically; they differ ONLY in whether the decomposition is shown.
# Survives disconnect; relaunches verl on death (auto-resume); prunes checkpoints to
# latest-2; exports each final model. Logs to /home/temp/b1_orchestrator.log.
set +e
cd /home/temp/swemaster_runpod
LOG=/home/temp/b1_orchestrator.log
RAY=/home/temp/venv/bin/ray
PROJ=ablation_b1_r12
LOCK=/tmp/b1_orchestrator.lock

if ! mkdir "$LOCK" 2>/dev/null; then echo "[orch] lock held; exit" >> $LOG; exit 0; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
log(){ echo "[orch $(date -u +%H:%M:%S)] $*" >> $LOG; }
log "=== b1 orchestrator started ==="

OD(){ ls -dt /home/temp/training_outputs/${PROJ}/${1}*/ 2>/dev/null | head -1; }
prune(){ ls -d ${1}global_step_*/ 2>/dev/null | sort -t_ -k3 -n | head -n -2 | xargs -r rm -rf; }
alive(){ pgrep -f 'rllm.trainer.verl.train_agent_ppo' >/dev/null 2>&1; }

run_arm(){
  local EXP=$1 ALOG=$2 RENV=$3 TGT=$4
  log "ARM $EXP starting (env: $RENV)"
  if alive; then
    log "trainer already running -> supervise existing"
  else
    while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do
      log "GPU busy, waiting..."; sleep 30
    done
    env $RENV EXPERIMENT_NAME=$EXP PROJECT_NAME=$PROJ setsid nohup bash run_refine_arm.sh > $ALOG 2>&1 &
    log "launched $EXP"; sleep 60
  fi
  local RELAUNCH=0 MAXRELAUNCH=6
  while true; do
    sleep 60
    local d; d=$(OD $EXP); [ -n "$d" ] && prune "$d"
    if grep -qE 'training done|Final validation metrics' "$ALOG" 2>/dev/null; then
      log "$EXP done-marker found"; break
    fi
    if ! alive; then
      if grep -qE 'training done|Final validation metrics' "$ALOG" 2>/dev/null; then break; fi
      if [ "$RELAUNCH" -lt "$MAXRELAUNCH" ]; then
        RELAUNCH=$((RELAUNCH+1)); log "$EXP trainer died -> relaunch #$RELAUNCH (verl auto-resume)"
        $RAY stop --force >> $LOG 2>&1; sleep 10
        env $RENV EXPERIMENT_NAME=$EXP PROJECT_NAME=$PROJ setsid nohup bash run_refine_arm.sh > $ALOG 2>&1 &
        sleep 60
      else
        log "FATAL: $EXP exceeded $MAXRELAUNCH relaunches; giving up"; break
      fi
    fi
  done
  local d; d=$(OD $EXP)
  log "$EXP finished; exporting from $d -> $TGT"
  CUDA_VISIBLE_DEVICES='' bash export_to_hf.sh "$d" "$TGT" >> $LOG 2>&1 \
    && log "$EXP exported -> $TGT" || log "$EXP export FAILED"
  $RAY stop --force >> $LOG 2>&1; sleep 15
}

run_arm B0_refoff_verlive /home/temp/arm_B0_refoff.log \
  "RLLM_ACTIVE_TESTS_REFINE=0 RLLM_ACTIVE_TESTS_REFINE_TICK=1" \
  /home/temp/models/SWE-Master-4B-ATS-B0-refoff

run_arm B1_refon_verlive /home/temp/arm_B1_refon.log \
  "RLLM_ACTIVE_TESTS_REFINE=1 RLLM_ACTIVE_TESTS_REFINE_TICK=1" \
  /home/temp/models/SWE-Master-4B-ATS-B1-refon

log "=== b1 orchestrator complete ==="
