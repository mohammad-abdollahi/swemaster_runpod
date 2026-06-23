#!/bin/bash
# B1/R12 NEW-DATA run — identical backtranslation refinement (REFINE=1, v1 reward,
# MID_EPISODE=0) trained on a FRESH 64/16 dataset of NEW swerebench (SWE-Verified+Live)
# instances (swerebench_newverlive), CONTINUING from the previous model
# SWE-Master-4B-ATS-B1-r12 (HF-weight load). Same method, new instances, same size.
# Survives disconnect; relaunches verl on death (auto-resume); prunes checkpoints to
# latest-2; exports the final model to /home/temp/models/SWE-Master-4B-ATS-B1-r12-newverlive.
# Logs to /home/temp/b1_newdata_orchestrator.log.
set +e
cd /home/temp/swemaster_runpod
LOG=/home/temp/b1_newdata_orchestrator.log
RAY=/home/temp/venv/bin/ray
PROJ=ablation_b1_newdata
LOCK=/tmp/b1_newdata_orchestrator.lock

if ! mkdir "$LOCK" 2>/dev/null; then echo "[orch] lock held; exit" >> $LOG; exit 0; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
log(){ echo "[orch $(date -u +%H:%M:%S)] $*" >> $LOG; }
log "=== b1 newdata orchestrator started ==="

OD(){ ls -dt /home/temp/training_outputs/${PROJ}/${1}*/ 2>/dev/null | head -1; }
prune(){ ls -d ${1}global_step_*/ 2>/dev/null | sort -t_ -k3 -n | head -n -2 | xargs -r rm -rf; }
alive(){ pgrep -f 'rllm.trainer.verl.train_agent_ppo' >/dev/null 2>&1; }

run_arm(){
  local EXP=$1 ALOG=$2 RENV=$3 TGT=$4
  log "ARM $EXP starting (env: $RENV)"
  if alive; then
    log "trainer already running -> supervise existing"
  else
    while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 40000 ]; do
      log "GPU busy (>40GB, real trainer), waiting..."; sleep 30
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

run_arm B1_newdata /home/temp/arm_B1_newdata.log \
  "RLLM_ACTIVE_TESTS_REFINE=1 RLLM_ACTIVE_TESTS_REFINE_TICK=1 \
BASE_MODEL=/home/temp/models/SWE-Master-4B-ATS-B1-r12 \
TRAIN_PARQUET=/home/temp/swemaster_runpod/data_examples/rl_data/swerebench_newverlive/train_00000_verl.parquet \
VAL_PARQUET=/home/temp/swemaster_runpod/data_examples/rl_data/swerebench_newverlive/test_00000_verl.parquet \
REFINE_CACHE=/home/temp/swemaster_runpod/block_b_refinement/cache/decompositions_newverlive.jsonl \
TOTAL_TRAINING_STEPS=10 TOTAL_EPOCHS=3" \
  /home/temp/models/SWE-Master-4B-ATS-B1-r12-newverlive

log "=== b1 newdata orchestrator complete ==="
