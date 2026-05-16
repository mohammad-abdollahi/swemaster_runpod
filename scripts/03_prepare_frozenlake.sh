#!/usr/bin/env bash
# Register the FrozenLake dataset via rllm's DatasetRegistry. Run inside the
# container:
#   bash scripts/00_enter_container.sh bash scripts/03_prepare_frozenlake.sh
set -euo pipefail

source "${VENV_BIN:-/scratch/$USER/swemaster/venv/rllm/bin}/activate"
cd "${PROJECT_ROOT:-/project/def-hemmati-ac/$USER/SWE-Master}/DeepSWE_RL/rllm"

# Small dataset is enough for a smoke test.
python -m examples.frozenlake.prepare_frozenlake_data
