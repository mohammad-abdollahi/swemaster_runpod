#!/bin/bash
# Detached supervisor for the f2p-only ablation run (branch test-selection).
# Waits for the 10-step training to finish (relaunches on death, verl auto-resume),
# exports the final model to HF format, extracts the reward trend from tensorboard,
# prunes intermediate checkpoints to save disk, and commits metrics. Survives
# Claude-session disconnect. Logs to /home/temp/f2p_supervisor.log.
set +e
cd /home/temp/swemaster_runpod
LOG=/home/temp/f2p_supervisor.log
RAY=/home/temp/venv/bin/ray
PY=/home/temp/venv/bin/python3
EXP=$(grep -oE 'f2p_only_verlive_[0-9-]+' /home/temp/f2p_only.expname | head -1)
RUNDIR=/home/temp/training_outputs/ablation_test_selection/${EXP}
OUTMODEL=/home/temp/models/SWE-Master-4B-ATS-f2p-only
TRAINLOG=/home/temp/f2p_only.log
LOCK=/tmp/f2p_supervisor.lock

if ! mkdir "$LOCK" 2>/dev/null; then echo "[sup] another supervisor holds lock; exit" >> $LOG; exit 0; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
log(){ echo "[sup $(date -u +%H:%M:%S)] $*" >> $LOG; }
log "=== supervisor started: EXP=$EXP RUNDIR=$RUNDIR ==="

done_marker(){ grep -qiE 'training done|Final validation metrics|fit_agent .*finished|10 steps' "$TRAINLOG" 2>/dev/null; }
final_ckpt_ready(){ [ -f "${RUNDIR}/global_step_10/actor/config.json" ]; }
trainer_alive(){ pgrep -f 'rllm.trainer.verl.train_agent_ppo' >/dev/null 2>&1; }

RELAUNCH=0; MAXRELAUNCH=4
# 1) supervise until final checkpoint exists
while true; do
  sleep 60
  if final_ckpt_ready; then log "final checkpoint global_step_10 ready"; break; fi
  if ! trainer_alive; then
    if final_ckpt_ready; then break; fi
    if [ "$RELAUNCH" -lt "$MAXRELAUNCH" ]; then
      RELAUNCH=$((RELAUNCH+1))
      log "trainer died without final ckpt -> relaunch #$RELAUNCH (verl auto-resume)"
      $RAY stop --force >> $LOG 2>&1; sleep 10
      EXPERIMENT_NAME=$EXP setsid nohup sg docker -c "EXPERIMENT_NAME=$EXP bash run_f2p_only.sh" >> $TRAINLOG 2>&1 &
      sleep 90
    else
      log "FATAL: exceeded $MAXRELAUNCH relaunches; proceeding with best available ckpt"; break
    fi
  fi
done

# let the trainer flush/exit cleanly
sleep 20

# 2) export final model
log "exporting final model -> $OUTMODEL"
CUDA_VISIBLE_DEVICES='' bash export_to_hf.sh "${RUNDIR}/" "$OUTMODEL" >> $LOG 2>&1 \
  && log "EXPORT OK -> $OUTMODEL" || log "EXPORT FAILED"

# 3) extract reward trend from tensorboard
log "extracting metrics"
$PY - "$RUNDIR" >> $LOG 2>&1 <<'PYEOF'
import sys, glob, os, csv
rundir = sys.argv[1]
evs = glob.glob(os.path.join(rundir, "**", "events.out.tfevents*"), recursive=True)
out = os.path.join(os.getcwd(), "ablation_test_selection_metrics", "f2p_only_verlive.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
rows = {}
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    for ev in evs:
        ea = EventAccumulator(ev); ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            if any(k in tag for k in ("reward", "grad_norm", "val", "solve")):
                for s in ea.Scalars(tag):
                    rows.setdefault(s.step, {})[tag] = s.value
except Exception as e:
    print("metrics extract error:", e)
tags = sorted({t for v in rows.values() for t in v})
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["step"] + tags)
    for st in sorted(rows): w.writerow([st] + [rows[st].get(t, "") for t in tags])
print("wrote", out, "steps:", sorted(rows), "tags:", tags)
PYEOF
log "metrics done"

# 4) prune intermediate checkpoints — keep only the final one (model already exported)
log "pruning intermediate checkpoints (keep global_step_10 only)"
for d in "${RUNDIR}"/global_step_*/; do
  case "$d" in *global_step_10/) : ;; *) rm -rf "$d" && log "pruned $(basename $d)";; esac
done

# 5) commit metrics
git add ablation_test_selection_metrics/ 2>/dev/null
git commit -q -m "test-selection f2p-only: 10-step training metrics (reward trend)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" 2>/dev/null \
  && log "committed metrics" || log "nothing to commit (metrics)"
log "=== supervisor complete ==="
echo "F2P_SUPERVISOR_DONE" >> $TRAINLOG
