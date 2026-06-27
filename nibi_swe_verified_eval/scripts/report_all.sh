#!/bin/bash
# Merge every chunk for each model and print resolve rate + metrics over all instances.
# SFT exps: runA, runB, sft_100..sft_450 ; RL exps: runA_rl, runB_rl, rl_100..rl_450
source "$(dirname "$0")/../env.sh"
source "$VENV/bin/activate"
cd "$(dirname "$0")/.."

SFT=$(ls $RESULTS_DIR/runA/runA.jsonl $RESULTS_DIR/runB/runB.jsonl $RESULTS_DIR/sft_*/sft_*.jsonl 2>/dev/null)
RL=$(ls $RESULTS_DIR/runA_rl/runA_rl.jsonl $RESULTS_DIR/runB_rl/runB_rl.jsonl $RESULTS_DIR/rl_*/rl_*.jsonl 2>/dev/null)
ATS=$(ls $RESULTS_DIR/ats_*/ats_*.jsonl 2>/dev/null)
SFT32=$(ls $RESULTS_DIR/sft32_*/sft32_*.jsonl 2>/dev/null)

echo "############################## SFT ##############################"
[ -n "$SFT" ] && python scripts/metrics.py $SFT --csv $RESULTS_DIR/metrics_sft_500.csv || echo "no SFT results yet"
echo
echo "############################## RL ##############################"
[ -n "$RL" ] && python scripts/metrics.py $RL --csv $RESULTS_DIR/metrics_rl_500.csv || echo "no RL results yet"
echo
echo "############################## ATS-RL ##############################"
[ -n "$ATS" ] && python scripts/metrics.py $ATS --csv $RESULTS_DIR/metrics_ats_500.csv || echo "no ATS-RL results yet"
echo
echo "############################## 32B-SFT ##############################"
[ -n "$SFT32" ] && python scripts/metrics.py $SFT32 --csv $RESULTS_DIR/metrics_sft32_500.csv || echo "no 32B-SFT results yet"
