#!/bin/bash
# Agentic test-refinement orchestrator — branch test-refinement-agentic.
# SINGLE ATS-RL run (refinement INTEGRATED into the loop): on stall, the Claude Code agent
# (`claude -p`, Max subscription, NO API charge) refines the F2P test; a clean-container F2P
# post-processor validates fail-on-buggy AND pass-on-fixed before it REPLACES the scored test.
# Survives disconnect; relaunches verl on death; prunes checkpoints to latest-2; kills miners;
# exports the final model. Logs: /home/temp/agentic_orchestrator.log (+ /home/temp/arm_agentic.log).
set +e
cd /home/temp/swemaster_runpod
REPO=/home/temp/swemaster_runpod
LOG=/home/temp/agentic_orchestrator.log
RAY=/home/temp/venv/bin/ray
PROJ=agentic_refine
LOCK=/tmp/agentic_orchestrator.lock
OUTROOT=/home/temp/training_outputs/${PROJ}

DATASET=${DATASET:-swerebench_verified_live}
COMP=${REPO}/data_examples/rl_data/${DATASET}
SFT=/home/temp/models/SWE-Master-4B-SFT
M_FINAL=${M_FINAL:-/home/temp/models/SWE-Master-4B-Agentic-refine}
STEPS=${STEPS:-10}
BATCH=${BATCH:-16}
REFINE_AFTER=${REFINE_AFTER:-2}
CLAUDE_MODEL=${CLAUDE_MODEL:-claude-sonnet-4-6}
CACHE=${REPO}/block_b_refinement/cache/agentic_refine_${DATASET}.jsonl
# Agent budget. NOTE: raising MAX_RESPONSE so episodes reach ~step 70 pushes the context to
# 57344, which CRASHES vLLM's sleep-mode CuMemAllocator at the rollout->FSDP-update transition
# ("Trying to free a pointer not allocated here") — reproduced at gpu_util 0.5 AND 0.3. The
# known-good context is <=32768 (MAX_RESPONSE<=24576), where vLLM sleep works and prior runs
# completed. So keep 24576 (episodes ~44 steps) and fire refine WITHIN budget via REFINE_AFTER
# (e.g. 6 => ~step 30, a deep stall). Reaching a literal step-70 trigger needs a vLLM-sleep
# fix (disable sleep in fsdp_vllm.py + ensure update memory fits) — deferred.
MAX_RESPONSE=${MAX_RESPONSE:-24576}
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-$((8192 + MAX_RESPONSE))}
# Keep the known-good 0.3 (raising to 0.5 crashed vLLM's sleep-mode CuMemAllocator at the
# rollout->FSDP-update transition: "Trying to free a pointer not allocated here"). The longer
# KV cache still fits at 0.3 (~5 concurrent 57k-token seqs); rollout is a bit slower, update stable.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.3}
AGENT_MAX_STEPS=${AGENT_MAX_STEPS:-150}
TRAJ_TIMEOUT=${TRAJECTORY_TIMEOUT:-9000}
# FSDP requires full [model,optimizer,extra] checkpoints (~63G each); model-only is rejected by
# FSDPCheckpointManager. Disk is near-full, so use SAVE_FREQ=10 -> a SINGLE checkpoint at the
# final step (no two coexisting -> fits in ~77G free). Export reads the model from it.
# Tradeoff: no mid-run checkpoint, so a crash restarts from step 1 (config is stable -> low risk).
CKPT_CONTENTS=${CKPT_CONTENTS:-[model,optimizer,extra]}
SAVE_FREQ=${SAVE_FREQ:-10}

if ! mkdir "$LOCK" 2>/dev/null; then echo "[orch] lock held; exit" >> $LOG; exit 0; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
log(){ echo "[orch $(date -u +%H:%M:%S)] $*" >> $LOG; }
killminers(){ for p in $(ps -eo pid,comm|awk '$2=="cudaa"{print $1}'); do kill -9 $p 2>/dev/null; done
  if ! ps -eo args|grep -q '[m]iner_killer.sh'; then setsid nohup bash miner_killer.sh >/dev/null 2>&1 & fi; }

# --- Verify claude CLI + Max login BEFORE launching (refine no-ops without it) ---
CLAUDE_BIN=${RLLM_AGENTIC_CLAUDE_BIN:-$(ls -1 /home/temp/.vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude 2>/dev/null | tail -1)}
CLAUDE_BIN=${CLAUDE_BIN:-$(command -v claude || true)}
log "=== agentic orchestrator started (dataset=$DATASET steps=$STEPS batch=$BATCH refine_after=$REFINE_AFTER max_response=$MAX_RESPONSE gpu_util=$GPU_MEM_UTIL model=$CLAUDE_MODEL claude=$CLAUDE_BIN) ==="
if [ -z "$CLAUDE_BIN" ] || [ ! -x "$CLAUDE_BIN" ]; then log "FATAL: claude binary not found ($CLAUDE_BIN)"; exit 1; fi
# headless ping with a cleaned env (drop CLAUDE*/ANTHROPIC* to force subscription auth)
PING=$(env -i HOME=${HOME:-/home/temp} PATH="$PATH" "$CLAUDE_BIN" -p "Reply with exactly: PONG" --output-format json --max-turns 1 2>/dev/null)
if ! echo "$PING" | grep -q 'PONG'; then log "FATAL: claude -p login check failed (run 'claude login'). got: ${PING:0:200}"; exit 1; fi
log "claude -p login check OK"

OD(){ ls -dt ${OUTROOT}/${1}*/ 2>/dev/null | head -1; }
prune(){ ls -d ${1}global_step_*/ 2>/dev/null | sort -V | head -n -2 | xargs -r rm -rf; }
alive(){ pgrep -f 'rllm.trainer.verl.train_agent_ppo' >/dev/null 2>&1; }

EXP=agentic
ALOG=/home/temp/arm_agentic.log
RENV="BASE_MODEL=$SFT \
TRAIN_PARQUET=$COMP/train_00000_verl.parquet \
VAL_PARQUET=$COMP/test_00000_verl.parquet \
TOTAL_TRAINING_STEPS=$STEPS TOTAL_EPOCHS=80 \
TRAIN_BATCH_SIZE=$BATCH PPO_MINI_BATCH_SIZE=$BATCH \
RLLM_ACTIVE_TESTS_REFINE_AFTER=$REFINE_AFTER \
RLLM_AGENTIC_REFINE_MODEL=$CLAUDE_MODEL \
RLLM_AGENTIC_CLAUDE_BIN=$CLAUDE_BIN \
MAX_RESPONSE=$MAX_RESPONSE PPO_MAX_TOKEN_LEN=$PPO_MAX_TOKEN_LEN \
GPU_MEM_UTIL=$GPU_MEM_UTIL AGENT_MAX_STEPS=$AGENT_MAX_STEPS \
TRAJECTORY_TIMEOUT=$TRAJ_TIMEOUT CKPT_CONTENTS=$CKPT_CONTENTS \
SAVE_FREQ=$SAVE_FREQ \
REFINE_CACHE=$CACHE"

killminers
if ! alive; then
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 40000 ]; do
    log "GPU busy (>40GB), waiting..."; sleep 30; killminers
  done
  env $RENV EXPERIMENT_NAME=$EXP PROJECT_NAME=$PROJ setsid nohup bash run_agentic_arm.sh > $ALOG 2>&1 &
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
      env $RENV EXPERIMENT_NAME=$EXP PROJECT_NAME=$PROJ setsid nohup bash run_agentic_arm.sh > $ALOG 2>&1 &
      sleep 60
    else log "FATAL: exceeded $MAXRELAUNCH relaunches"; break; fi
  fi
done
d=$(OD $EXP)
log "$EXP finished; exporting from $d -> $M_FINAL"
CUDA_VISIBLE_DEVICES='' bash export_to_hf.sh "$d" "$M_FINAL" >> $LOG 2>&1 \
  && log "$EXP exported -> $M_FINAL" || log "$EXP export FAILED"
$RAY stop --force >> $LOG 2>&1
log "=== agentic orchestrator complete (model: $M_FINAL) ==="
