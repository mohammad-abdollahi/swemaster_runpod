#!/usr/bin/env bash
# Wrap any command in the project's Apptainer container with all the right
# binds and cache redirections. Examples:
#   bash scripts/00_enter_container.sh bash scripts/01_build_env.sh
#   bash scripts/00_enter_container.sh python -c 'import torch; print(torch.cuda.is_available())'
#   bash scripts/00_enter_container.sh                # drop into shell
set -euo pipefail

USER=${USER:-$(whoami)}
SCRATCH=${SCRATCH:-/scratch/$USER}
PROJECT_ROOT=${PROJECT_ROOT:-/project/def-hemmati-ac/$USER/SWE-Master}
SIF=${SIF:-$SCRATCH/swemaster/sif/pytorch_27_cu128}     # sandbox dir OR .sif file
VENV=${VENV:-$SCRATCH/swemaster/venv/rllm}
HF_HOME_DIR=${HF_HOME_DIR:-$SCRATCH/swemaster/cache/hf}

if [[ ! -e "$SIF" ]]; then
  echo "[enter-container] container image not found at $SIF; build it first" >&2
  exit 1
fi

# Forbid cluster modules from leaking into the container.
module --force purge 2>/dev/null || true

mkdir -p "$HF_HOME_DIR" "$SCRATCH/swemaster/tmp"

# Container env vars: keep HF, pip, ray, torch caches on /scratch
export APPTAINERENV_HF_HOME="$HF_HOME_DIR"
export APPTAINERENV_HUGGINGFACE_HUB_CACHE="$HF_HOME_DIR/hub"
export APPTAINERENV_TRANSFORMERS_CACHE="$HF_HOME_DIR/transformers"
export APPTAINERENV_TORCH_HOME="$SCRATCH/swemaster/cache/torch"
export APPTAINERENV_PIP_CACHE_DIR="$SCRATCH/swemaster/cache/pip"
export APPTAINERENV_TMPDIR=/tmp
export APPTAINERENV_PROJECT_ROOT="$PROJECT_ROOT"
export APPTAINERENV_SCRATCH="$SCRATCH"
export APPTAINERENV_VENV="$VENV"
export APPTAINERENV_HF_HOME_DIR="$HF_HOME_DIR"
# Propagate LD_PRELOAD + FAKECPUS_N into the container (apptainer --cleanenv
# strips host env). Need raylet (C++) to see the shim too.
if [[ -n "${LD_PRELOAD:-}" ]]; then export APPTAINERENV_LD_PRELOAD="$LD_PRELOAD"; fi
if [[ -n "${FAKECPUS_N:-}" ]]; then export APPTAINERENV_FAKECPUS_N="$FAKECPUS_N"; fi

# Forward common training knobs so `REWARD=test bash 00_enter_container.sh ...`
# works without the caller needing the APPTAINERENV_ prefix (--cleanenv would
# otherwise silently drop them and the launcher would fall back to defaults).
for _v in REWARD MAX_STEPS TEST_TIMEOUT BATCH_SIZE NUM_GENERATIONS MAX_TURNS \
          MAX_PROMPT_TOKENS MAX_GEN_TOKENS LEARNING_RATE MODEL_ID DATASET \
          SWE_PARQUET PROJECT_NAME EXPERIMENT_NAME; do
  if [[ -n "${!_v:-}" ]]; then export "APPTAINERENV_${_v}=${!_v}"; fi
done

# Source the venv inside the container if it exists
APPTAINERENV_VENV_BIN=""
if [[ -d "$VENV/bin" ]]; then
  APPTAINERENV_VENV_BIN="$VENV/bin"
fi
export APPTAINERENV_VENV_BIN

# Build the command. Default is an interactive bash.
if [[ $# -eq 0 ]]; then
  CMD=(bash)
else
  CMD=("$@")
fi

# Binds: project (ro is too restrictive — many caches write into the repo);
# scratch for everything mutable.
exec apptainer exec --nv --cleanenv \
  --env "PATH=${VENV}/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "USER=${USER}" \
  --env "HOME=${HOME}" \
  -B "$PROJECT_ROOT" \
  -B "$SCRATCH" \
  "$SIF" "${CMD[@]}"
