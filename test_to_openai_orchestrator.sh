#!/bin/bash
# Test-to-OpenAI orchestrator — branch test-refinement-test-to-openai.
# Four phases, each gated on the previous, survives disconnect, relaunches the
# trainer on death (verl auto-resume), prunes checkpoints to latest-2, kills miners:
#
#   PHASE 1  train 70 steps, ORIGINAL ATS-RL (REFINE=0), swerebench_comprehensive,
#            SWE-Master-4B-SFT base, RLLM_DUMP_TRAJ_DIR on  -> export theta70
#   REFINE   test_to_openai_refiner.py: unsolved+insightful F2P -> gpt-5.4 rewrite/split
#            -> Docker gate -> refinements cache
#   BUILD    build_refined_dataset.py: splice refined tests -> comprehensive_refined
#   PHASE 2  continue +10 steps from theta70 on comprehensive_refined -> export final
#
# Logs: /home/temp/test_to_openai_orchestrator.log (+ per-phase arm logs).
set +e
cd /home/temp/swemaster_runpod
REPO=/home/temp/swemaster_runpod
LOG=/home/temp/test_to_openai_orchestrator.log
RAY=/home/temp/venv/bin/ray
PY=/home/temp/venv/bin/python
PROJ=test_to_openai
LOCK=/tmp/test_to_openai_orchestrator.lock

OUTROOT=/home/temp/training_outputs/${PROJ}
DUMP_DIR=${OUTROOT}/traj_dump
CACHE=${REPO}/block_b_refinement/cache/refinements_test_to_openai.jsonl
REFINED=${REPO}/data_examples/rl_data/swerebench_comprehensive_refined
COMP=${REPO}/data_examples/rl_data/swerebench_comprehensive
SFT=/home/temp/models/SWE-Master-4B-SFT
M_PHASE1=/home/temp/models/SWE-Master-4B-TestToOpenAI-phase1
M_FINAL=/home/temp/models/SWE-Master-4B-TestToOpenAI
P1_STEPS=${P1_STEPS:-70}
P2_STEPS=${P2_STEPS:-10}
OPENAI_MODEL=${OPENAI_MODEL:-gpt-5.4}

if ! mkdir "$LOCK" 2>/dev/null; then echo "[orch] lock held; exit" >> $LOG; exit 0; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
log(){ echo "[orch $(date -u +%H:%M:%S)] $*" >> $LOG; }
killminers(){ for p in $(ps -eo pid,comm|awk '$2=="cudaa"{print $1}'); do kill -9 $p 2>/dev/null; done
  if ! ps -eo args|grep -q '[m]iner_killer.sh'; then setsid nohup bash miner_killer.sh >/dev/null 2>&1 & fi; }
log "=== test_to_openai orchestrator started (P1=${P1_STEPS} P2=${P2_STEPS} model=${OPENAI_MODEL}) ==="

OD(){ ls -dt ${OUTROOT}/${1}*/ 2>/dev/null | head -1; }
prune(){ ls -d ${1}global_step_*/ 2>/dev/null | sort -V | head -n -2 | xargs -r rm -rf; }
alive(){ pgrep -f 'rllm.trainer.verl.train_agent_ppo' >/dev/null 2>&1; }

# ---- generic training phase with crash-relaunch + export -------------------
run_phase(){
  # $1 EXP  $2 ALOG  $3 ENV(extra)  $4 TGT_export
  local EXP=$1 ALOG=$2 RENV=$3 TGT=$4
  killminers
  if [ -d "$TGT" ] && [ -n "$(ls -A "$TGT" 2>/dev/null)" ] && grep -q "$EXP exported" $LOG 2>/dev/null; then
    log "$EXP already exported -> skip"; return 0
  fi
  log "PHASE $EXP starting (env: $RENV)"
  if ! alive; then
    while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 40000 ]; do
      log "GPU busy (>40GB), waiting..."; sleep 30; killminers
    done
    env $RENV EXPERIMENT_NAME=$EXP PROJECT_NAME=$PROJ setsid nohup bash run_test_to_openai_arm.sh > $ALOG 2>&1 &
    log "launched $EXP"; sleep 60
  fi
  local RELAUNCH=0 MAXRELAUNCH=8
  while true; do
    sleep 60; killminers
    local d; d=$(OD $EXP); [ -n "$d" ] && prune "$d"
    if grep -qE 'training done|Final validation metrics' "$ALOG" 2>/dev/null; then
      log "$EXP done-marker found"; break
    fi
    if ! alive; then
      if grep -qE 'training done|Final validation metrics' "$ALOG" 2>/dev/null; then break; fi
      if [ "$RELAUNCH" -lt "$MAXRELAUNCH" ]; then
        RELAUNCH=$((RELAUNCH+1)); log "$EXP died -> relaunch #$RELAUNCH (verl auto-resume)"
        $RAY stop --force >> $LOG 2>&1; sleep 10
        env $RENV EXPERIMENT_NAME=$EXP PROJECT_NAME=$PROJ setsid nohup bash run_test_to_openai_arm.sh > $ALOG 2>&1 &
        sleep 60
      else log "FATAL: $EXP exceeded $MAXRELAUNCH relaunches; giving up"; break; fi
    fi
  done
  local d; d=$(OD $EXP)
  log "$EXP finished; exporting from $d -> $TGT"
  CUDA_VISIBLE_DEVICES='' bash export_to_hf.sh "$d" "$TGT" >> $LOG 2>&1 \
    && log "$EXP exported -> $TGT" || log "$EXP export FAILED"
  $RAY stop --force >> $LOG 2>&1; sleep 15
}

# ======================= PHASE 1 — train 70 steps ==========================
mkdir -p "$DUMP_DIR"
run_phase phase1 /home/temp/arm_test_to_openai_p1.log \
  "BASE_MODEL=$SFT \
TRAIN_PARQUET=$COMP/train_00000_verl.parquet \
VAL_PARQUET=$COMP/test_00000_verl.parquet \
TOTAL_TRAINING_STEPS=$P1_STEPS TOTAL_EPOCHS=40 \
RLLM_DUMP_TRAJ_DIR=$DUMP_DIR" \
  "$M_PHASE1"

# ======================= REFINE — OpenAI gpt-5.4 ============================
killminers
if [ -s "$CACHE" ] && grep -q '"refined": \[\[' "$CACHE" 2>/dev/null; then
  log "refine cache already populated -> skip"
else
  log "REFINE: running test_to_openai_refiner.py over $DUMP_DIR (model=$OPENAI_MODEL)"
  sg docker -c "$PY $REPO/test_to_openai_refiner.py \
    --dump-dir $DUMP_DIR \
    --train-parquet $COMP/train_00000_verl.parquet \
    --out $CACHE --model $OPENAI_MODEL --min-steps 5 --max-refined 4 --workers 4" \
    >> $LOG 2>&1
  log "REFINE done; cache lines: $(wc -l < $CACHE 2>/dev/null)"
fi

# ======================= BUILD — refined dataset ===========================
killminers
if [ -s "$REFINED/train_00000_verl.parquet" ]; then
  log "refined dataset already built -> skip"
else
  log "BUILD: build_refined_dataset.py -> $REFINED"
  $PY $REPO/build_refined_dataset.py --src $COMP --cache $CACHE --out $REFINED >> $LOG 2>&1
  log "BUILD done"
fi

# ======================= PHASE 2 — continue +10 steps ======================
# If no tests were refined (empty/identical dataset), still continue on the
# original dataset so phase-2 is well-defined; log the situation.
P2_TRAIN=$REFINED/train_00000_verl.parquet
[ -s "$P2_TRAIN" ] || { log "WARN: refined train missing; falling back to comprehensive"; P2_TRAIN=$COMP/train_00000_verl.parquet; }
run_phase phase2 /home/temp/arm_test_to_openai_p2.log \
  "BASE_MODEL=$M_PHASE1 \
TRAIN_PARQUET=$P2_TRAIN \
VAL_PARQUET=$COMP/test_00000_verl.parquet \
TOTAL_TRAINING_STEPS=$P2_STEPS TOTAL_EPOCHS=40" \
  "$M_FINAL"

log "=== test_to_openai orchestrator complete (final model: $M_FINAL) ==="
