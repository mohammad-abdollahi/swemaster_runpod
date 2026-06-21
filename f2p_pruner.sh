#!/bin/bash
# Detached disk-guard for the f2p-only run: while training is alive, keep only
# the latest 2 COMPLETE checkpoints (each ~63G) so save_freq=1 over 10 steps
# can't fill the disk. Only ever deletes checkpoints older than the newest 2,
# which are complete — never the one being written. Logs to f2p_pruner.log.
set +e
EXP=$(grep -oE 'f2p_only_verlive_[0-9-]+' /home/temp/f2p_only.expname | head -1)
RUNDIR=/home/temp/training_outputs/ablation_test_selection/${EXP}
LOG=/home/temp/f2p_pruner.log
KEEP=2
echo "[prune $(date -u +%H:%M:%S)] started RUNDIR=$RUNDIR keep=$KEEP" >> $LOG
while pgrep -f 'rllm.trainer.verl.train_agent_ppo' >/dev/null 2>&1; do
  mapfile -t cks < <(ls -d "${RUNDIR}"/global_step_*/ 2>/dev/null | sort -V)
  n=${#cks[@]}
  if [ "$n" -gt "$KEEP" ]; then
    for ((i=0; i<n-KEEP; i++)); do
      rm -rf "${cks[$i]}" && echo "[prune $(date -u +%H:%M:%S)] removed $(basename ${cks[$i]}) (free: $(df -h / | awk 'NR==2{print $4}'))" >> $LOG
    done
  fi
  sleep 120
done
echo "[prune $(date -u +%H:%M:%S)] trainer gone; pruner exit" >> $LOG
