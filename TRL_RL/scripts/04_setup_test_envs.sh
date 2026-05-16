#!/usr/bin/env bash
# Pre-build one conda env per SWE-bench demo instance so the paper-faithful
# test reward can run. One-time (idempotent via .env_ready sentinels).
#
#   bash scripts/00_enter_container.sh bash TRL_RL/scripts/04_setup_test_envs.sh
set -euo pipefail

USER=${USER:-$(whoami)}
SCRATCH=${SCRATCH:-/scratch/$USER}
PROJECT_ROOT=${PROJECT_ROOT:-/project/def-hemmati-ac/$USER/SWE-Master}
VENV=${VENV:-$SCRATCH/swemaster/venv/rllm}

source "$VENV/bin/activate"
cd "$PROJECT_ROOT"

export CONDA_EXE=${CONDA_EXE:-/opt/conda/bin/conda}
export SWE_TEST_ENVS_ROOT=${SWE_TEST_ENVS_ROOT:-$SCRATCH/swemaster/test_envs}
export SWE_REPOS_ROOT=${SWE_REPOS_ROOT:-$SCRATCH/swemaster/repos}
# Keep conda's package cache on /scratch (home is full).
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-$SCRATCH/swemaster/cache/conda_pkgs}
mkdir -p "$SWE_TEST_ENVS_ROOT" "$CONDA_PKGS_DIRS"

echo "[setup-test-envs] envs root: $SWE_TEST_ENVS_ROOT"
python TRL_RL/src/build_test_envs.py
