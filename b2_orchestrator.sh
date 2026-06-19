#!/bin/bash
# Detached Block-B B2 orchestrator — survives Claude session disconnect.
# Waits for the R22 refine to finish, commits the gated parquet, launches B2 training
# (10 steps), preserves a step-5 HF snapshot, relaunches on death (verl auto-resume),
# then exports the final model + metrics and commits. Logs to b2_orchestrator.log.
set +e
cd /home/temp/swemaster_runpod
LOG=/home/temp/b2_orchestrator.log
PY=/home/temp/venv/bin/python3
RAY=/home/temp/venv/bin/ray
ARMDIR=data_examples/rl_data/swerebench_verlive_r22
TRAIN=$ARMDIR/train_00000_verl.parquet
VAL=data_examples/rl_data/swerebench_verlive/test_00000_verl.parquet
LOCK=/tmp/b2_orchestrator.lock

# single-instance guard
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "[orch] another orchestrator holds the lock; exiting" >> $LOG; exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

log(){ echo "[orch $(date -u +%H:%M:%S)] $*" >> $LOG; }
log "=== orchestrator started ==="

# 1) wait until the refine OUTPUT is written + summary present (robust to lingering workers)
while [ ! -f "$TRAIN" ] || ! grep -q 'REFINE SUMMARY' /home/temp/r22_verlive.log 2>/dev/null; do
  sleep 15
done
log "refine output ready"
if [ ! -f "$TRAIN" ]; then log "FATAL: refine output missing ($TRAIN)"; exit 1; fi

# 2) ensure val present + commit the gated parquet (idempotent)
cp -f "$VAL" "$ARMDIR/" 2>/dev/null
git add "$ARMDIR" block_b_refinement/audit/ 2>/dev/null
git commit -q -m "block-b R22 (gentler gate): refined verlive parquet + audit (32.8% dropped)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" 2>/dev/null \
  && log "committed gated parquet" || log "nothing new to commit (parquet)"
git push -q origin test-refinement 2>/dev/null && log "pushed" || log "push skipped/failed"

# 3) launch B2 training (10 steps) — fixed EXPERIMENT_NAME so relaunch resumes
EXP=B2_r22_verlive_orch
launch_b2(){
  MODEL_PATH=/home/temp/models/SWE-Master-4B-SFT TOTAL_TRAINING_STEPS=10 \
  TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=16 \
  EXPERIMENT_NAME=$EXP PROJECT_NAME=ablation_block_b \
  TRAIN_PARQUET=$TRAIN VAL_PARQUET=$VAL MAX_CKPT_KEEP=2 \
  setsid nohup bash run_control_live64.sh > /home/temp/arm_B2.log 2>&1 &
  log "launched B2 (exp=$EXP, 10 steps)"
}
# If a B2 trainer is already running (e.g. orchestrator restarted to pick up a fix),
# DON'T launch a second one — just supervise the existing run.
if pgrep -f 'rllm.trainer.verl.train_agent_ppo' >/dev/null 2>&1; then
  log "B2 trainer already running -> supervise existing (skip launch)"
else
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do
    log "GPU busy, waiting..."; sleep 30
  done
  launch_b2
  sleep 60   # let it spin up
fi

# NOTE: the train script hardcodes default_local_dir under ablation_control/ (ignores
# PROJECT_NAME), so checkpoints land at ablation_control/${EXP}, NOT ablation_block_b/.
OD(){ ls -dt /home/temp/training_outputs/ablation_control/${EXP}*/ 2>/dev/null | head -1; }

# 4) supervise: step-5 snapshot, relaunch-on-death, completion
STEP5=0; RELAUNCH=0; MAXRELAUNCH=6
done_marker(){ grep -qE 'training done|Final validation metrics' /home/temp/arm_B2.log 2>/dev/null; }
while true; do
  sleep 60
  d=$(OD)
  if done_marker; then log "B2 completion marker found"; break; fi

  # step-5 snapshot (isolate global_step_5 since export_to_hf merges the HIGHEST step)
  if [ "$STEP5" -eq 0 ] && [ -n "$d" ] && [ -d "${d}global_step_5" ]; then
    log "preserving step-5 snapshot"
    rm -rf /tmp/b2s5; mkdir -p /tmp/b2s5
    cp -r "${d}global_step_5" /tmp/b2s5/ 2>/dev/null
    cp "${d}"*.txt /tmp/b2s5/ 2>/dev/null
    CUDA_VISIBLE_DEVICES='' bash export_to_hf.sh /tmp/b2s5/ /home/temp/models/SWE-Master-4B-ATS-B2-r22-step5 >> $LOG 2>&1 \
      && log "step-5 model exported -> SWE-Master-4B-ATS-B2-r22-step5" || log "step-5 export FAILED"
    STEP5=1
  fi

  # relaunch on death (no trainer proc AND no completion)
  if ! pgrep -f 'rllm.trainer.verl.train_agent_ppo' >/dev/null 2>&1; then
    if ! done_marker; then
      if [ "$RELAUNCH" -lt "$MAXRELAUNCH" ]; then
        RELAUNCH=$((RELAUNCH+1))
        log "B2 trainer died w/o completion -> relaunch #$RELAUNCH (verl auto-resume)"
        $RAY stop --force >> $LOG 2>&1
        sleep 10
        launch_b2
        sleep 60
      else
        log "FATAL: exceeded $MAXRELAUNCH relaunches; giving up"; break
      fi
    fi
  fi
done

# 5) final export + metrics + commit
d=$(OD)
log "B2 finished; exporting final model from $d"
CUDA_VISIBLE_DEVICES='' bash export_to_hf.sh "$d" /home/temp/models/SWE-Master-4B-ATS-B2-r22 >> $LOG 2>&1 \
  && log "final model exported -> SWE-Master-4B-ATS-B2-r22" || log "final export FAILED"
$PY block_b_refinement/collect_metrics.py --arm B2_r22_verlive --tb-dir "$d" --log /home/temp/arm_B2.log >> $LOG 2>&1 \
  && log "metrics collected" || log "metrics FAILED"
git add block_b_refinement/metrics/ 2>/dev/null
git commit -q -m "block-b B2 (R22 verlive): 10-step training metrics + model exports

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" 2>/dev/null \
  && log "committed metrics" || log "nothing to commit (metrics)"
git push -q origin test-refinement 2>/dev/null && log "pushed metrics" || log "push skipped/failed"
log "=== orchestrator complete ==="
