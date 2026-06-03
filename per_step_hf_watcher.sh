#!/usr/bin/env bash
# Watch the FSDP checkpoint dir for new global_step_N folders and export each
# one to HF format under /home/temp/models/<MODEL_NAME>/step_N/.
#
# With save_freq=1 and max_actor_ckpt_to_keep=2, FSDP keeps a rolling 2 on
# disk — the watcher must grab each step BEFORE the next save sweeps it.
# Per-step time ≈ 55-75 min, HF export ≈ 5 min, so plenty of slack.
#
# Usage: setsid nohup bash per_step_hf_watcher.sh <EXP_DIR> <MODEL_NAME> <FINAL_STEP> \
#                       > /home/temp/per_step_hf.log 2>&1 &
set -e
EXP_DIR=${1:?need EXP_DIR}
MODEL_NAME=${2:-SWE-Master-4B-ATS-RL-50steps}
FINAL_STEP=${3:-50}
TARGET_ROOT=/home/temp/models/${MODEL_NAME}
LOG=/home/temp/per_step_hf.log
mkdir -p "$TARGET_ROOT"

echo "$(date -u) per-step HF watcher: exp=$EXP_DIR target=$TARGET_ROOT final_step=$FINAL_STEP" | tee -a "$LOG"

declare -A DONE
for K in $(ls -d "$TARGET_ROOT"/step_*/ 2>/dev/null); do
  N=$(basename "$K" | sed 's/step_//')
  DONE[$N]=1
done
echo "  resuming with ${#DONE[@]} already-exported steps" | tee -a "$LOG"

CONSEC_IDLE=0
while true; do
  # Look at currently-present FSDP checkpoints (max 2 due to rolling cleanup)
  for CKPT in $(ls -d "${EXP_DIR}"/global_step_*/ 2>/dev/null); do
    N=$(basename "$CKPT" | grep -oE '[0-9]+$')
    if [ -z "$N" ]; then continue; fi
    if [ -n "${DONE[$N]:-}" ]; then continue; fi
    if [ ! -d "${CKPT}actor" ]; then continue; fi

    OUT="${TARGET_ROOT}/step_${N}"
    echo "$(date -u) detected new step_${N}; exporting -> $OUT" | tee -a "$LOG"
    mkdir -p "$OUT"
    sleep 20  # FSDP shard write flush
    if bash /home/temp/swemaster_runpod/export_to_hf.sh "$EXP_DIR" "$OUT" >>"$LOG" 2>&1; then
      for f in tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json added_tokens.json generation_config.json; do
        [ -f "$OUT/$f" ] || cp "/home/temp/models/SWE-Master-4B-SFT/$f" "$OUT/" 2>/dev/null || true
      done
      DONE[$N]=1
      SZ=$(du -sh "$OUT" 2>/dev/null | cut -f1)
      echo "$(date -u) EXPORT OK: step_${N} ($SZ)" | tee -a "$LOG"
      # Also mirror the final step to the bare model dir for easy upload
      if [ "$N" -ge "$FINAL_STEP" ]; then
        cp -ru "$OUT"/* "$TARGET_ROOT/" 2>/dev/null || true
        echo "$(date -u) final step_${N} mirrored to $TARGET_ROOT" | tee -a "$LOG"
      fi
    else
      echo "$(date -u) EXPORT FAILED for step_${N}; will retry next poll" | tee -a "$LOG"
      rm -rf "$OUT"
    fi
  done

  # exit when all steps done OR trainer is gone and no new ckpts for 5 min
  if [ "${#DONE[@]}" -ge "$FINAL_STEP" ]; then
    echo "$(date -u) all $FINAL_STEP steps exported; exiting" | tee -a "$LOG"
    break
  fi
  if ! pgrep -f "rllm.trainer.verl.train_agent_ppo" >/dev/null 2>&1; then
    CONSEC_IDLE=$((CONSEC_IDLE+1))
    if [ "$CONSEC_IDLE" -ge 10 ]; then  # 10 × 30s = 5 min
      echo "$(date -u) trainer gone for 5 min; exiting with ${#DONE[@]} exports" | tee -a "$LOG"
      break
    fi
  else
    CONSEC_IDLE=0
  fi
  sleep 30
done
echo "PER_STEP_HF_WATCHER_DONE $(date -u)" | tee -a "$LOG"
