#!/bin/bash
# Online test-to-OpenAI refinement orchestrator — branch test-refinement-test-to-openai.
# SINGLE ATS-RL run (refinement INTEGRATED into the loop, not a separate phase): when a
# FAIL_TO_PASS test stalls, OpenAI gpt-5.4 rewrites/splits it live and the refined test
# REPLACES the scored test (drives the reward). Survives disconnect; relaunches verl on
# death; prunes checkpoints to latest-2; kills miners; exports the final model.
# Logs: /home/temp/online_refine_orchestrator.log (+ /home/temp/arm_online_refine.log).
set +e
cd /home/temp/swemaster_runpod
REPO=/home/temp/swemaster_runpod
LOG=/home/temp/online_refine_orchestrator.log
RAY=/home/temp/venv/bin/ray
PROJ=online_refine
LOCK=/tmp/online_refine_orchestrator.lock
OUTROOT=/home/temp/training_outputs/${PROJ}

DATASET=${DATASET:-swerebench_verified_live}
COMP=${REPO}/data_examples/rl_data/${DATASET}
SFT=/home/temp/models/SWE-Master-4B-SFT
M_FINAL=${M_FINAL:-/home/temp/models/SWE-Master-4B-TestToOpenAI-online}
STEPS=${STEPS:-10}
BATCH=${BATCH:-16}
REFINE_AFTER=${REFINE_AFTER:-2}
OPENAI_MODEL=${OPENAI_MODEL:-gpt-5.4-mini}
CACHE=${REPO}/block_b_refinement/cache/online_refine_${DATASET}.jsonl

if ! mkdir "$LOCK" 2>/dev/null; then echo "[orch] lock held; exit" >> $LOG; exit 0; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
log(){ echo "[orch $(date -u +%H:%M:%S)] $*" >> $LOG; }
killminers(){ for p in $(ps -eo pid,comm|awk '$2=="cudaa"{print $1}'); do kill -9 $p 2>/dev/null; done
  if ! ps -eo args|grep -q '[m]iner_killer.sh'; then setsid nohup bash miner_killer.sh >/dev/null 2>&1 & fi; }
log "=== online_refine orchestrator started (dataset=$DATASET steps=$STEPS batch=$BATCH refine_after=$REFINE_AFTER model=$OPENAI_MODEL) ==="

OD(){ ls -dt ${OUTROOT}/${1}*/ 2>/dev/null | head -1; }
prune(){ ls -d ${1}global_step_*/ 2>/dev/null | sort -V | head -n -2 | xargs -r rm -rf; }
alive(){ pgrep -f 'rllm.trainer.verl.train_agent_ppo' >/dev/null 2>&1; }

EXP=online
ALOG=/home/temp/arm_online_refine.log
RENV="BASE_MODEL=$SFT \
TRAIN_PARQUET=$COMP/train_00000_verl.parquet \
VAL_PARQUET=$COMP/test_00000_verl.parquet \
TOTAL_TRAINING_STEPS=$STEPS TOTAL_EPOCHS=80 \
TRAIN_BATCH_SIZE=$BATCH PPO_MINI_BATCH_SIZE=$BATCH \
RLLM_ACTIVE_TESTS_REFINE=1 RLLM_ACTIVE_TESTS_ONLINE_REFINE=1 \
RLLM_ACTIVE_TESTS_REFINE_AFTER=$REFINE_AFTER \
RLLM_ONLINE_REFINE_MODEL=$OPENAI_MODEL \
REFINE_CACHE=$CACHE"

killminers
if ! alive; then
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 40000 ]; do
    log "GPU busy (>40GB), waiting..."; sleep 30; killminers
  done
  env $RENV EXPERIMENT_NAME=$EXP PROJECT_NAME=$PROJ setsid nohup bash run_test_to_openai_arm.sh > $ALOG 2>&1 &
  log "launched $EXP"; sleep 60
fi
RELAUNCH=0; MAXRELAUNCH=8
while true; do
  sleep 60; killminers
  d=$(OD $EXP); [ -n "$d" ] && prune "$d"
  if grep -qE 'training done|Final validation metrics' "$ALOG" 2>/dev/null; then log "$EXP done-marker"; break; fi
  if ! alive; then
    if grep -qE 'training done|Final validation metrics' "$ALOG" 2>/dev/null; then break; fi
    if [ "$RELAUNCH" -lt "$MAXRELAUNCH" ]; then
      RELAUNCH=$((RELAUNCH+1)); log "$EXP died -> relaunch #$RELAUNCH (verl auto-resume)"
      $RAY stop --force >> $LOG 2>&1; sleep 10
      env $RENV EXPERIMENT_NAME=$EXP PROJECT_NAME=$PROJ setsid nohup bash run_test_to_openai_arm.sh > $ALOG 2>&1 &
      sleep 60
    else log "FATAL: exceeded $MAXRELAUNCH relaunches"; break; fi
  fi
done
d=$(OD $EXP)
log "$EXP finished; exporting from $d -> $M_FINAL"
CUDA_VISIBLE_DEVICES='' bash export_to_hf.sh "$d" "$M_FINAL" >> $LOG 2>&1 \
  && log "$EXP exported -> $M_FINAL" || log "$EXP export FAILED"
$RAY stop --force >> $LOG 2>&1
log "=== online_refine orchestrator complete (model: $M_FINAL) ==="
