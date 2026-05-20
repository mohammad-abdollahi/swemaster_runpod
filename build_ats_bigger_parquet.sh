#!/usr/bin/env bash
# Build a bigger ATS training parquet for the post-scaled-run experiment.
#
# After the SWE-World scaled run finishes (~16 train tasks, mean |F|=11.4,
# 8 PPO steps), if learning shows up we want to scale the dataset itself.
# This script builds a 64-task parquet (4x the current size), still filtered
# to MIN_FTP=5 so the continuous reward stays meaningfully partial-credit.
#
# Downstream: train_active_tests_scaled_4b_h200.sh picks this up via
# TRAIN_PARQUET=... override (the launcher script run_scaled_swe_world_big.sh).
#
# Expected wallclock: 5-10 minutes for the manifest probes.
set -e

cd /home/temp/swemaster_runpod
N_TRAIN=64 \
N_VAL=8 \
MIN_FTP=5 \
MAX_PROBE=500 \
OUT_DIR=/home/temp/swemaster_runpod/data_examples/rl_data/swerebench_min_ftp5_64 \
/home/temp/venv/bin/python build_comprehensive_swerebench.py

echo "Built parquet at /home/temp/swemaster_runpod/data_examples/rl_data/swerebench_min_ftp5_64/"
/home/temp/venv/bin/python -c "
import pandas as pd, json
for f in ['train_00000_verl.parquet','test_00000_verl.parquet']:
    df = pd.read_parquet(f'/home/temp/swemaster_runpod/data_examples/rl_data/swerebench_min_ftp5_64/{f}')
    ftps = []
    for _, row in df.iterrows():
        x = row['extra_info']
        if isinstance(x, str):
            x = json.loads(x)
        ftps.append(len(x.get('FAIL_TO_PASS', [])))
    print(f'{f}: {len(df)} rows, |F| range=[{min(ftps)},{max(ftps)}] mean={sum(ftps)/len(ftps):.1f}')
"
