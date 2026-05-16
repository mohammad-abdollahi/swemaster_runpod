#!/usr/bin/env bash
# Runs INSIDE the Apptainer container. Creates a Python venv on /scratch and
# installs rllm + verl + their RL deps from PyPI source — no cluster modules.
#
# Invoke from the project root, NOT from inside the container, like:
#   bash scripts/00_enter_container.sh scripts/01_build_env.sh
set -euo pipefail

SCRATCH=${SCRATCH:-/scratch/$USER}
PROJECT_ROOT=${PROJECT_ROOT:-/project/def-hemmati-ac/$USER/SWE-Master}
VENV=${VENV:-$SCRATCH/swemaster/venv/rllm}
HF_HOME_DIR=${HF_HOME_DIR:-$SCRATCH/swemaster/cache/hf}
PIP_CACHE=${PIP_CACHE:-$SCRATCH/swemaster/cache/pip}

mkdir -p "$VENV" "$HF_HOME_DIR" "$PIP_CACHE"

# Use the container's python (3.11 inside pytorch:2.7.1-cu128 image)
PYTHON_BIN=${PYTHON_BIN:-$(command -v python3)}
echo "[build-env] python: $PYTHON_BIN -> $($PYTHON_BIN --version)"
echo "[build-env] venv:   $VENV"
echo "[build-env] HF:     $HF_HOME_DIR"

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi

# shellcheck disable=SC1090
source "$VENV/bin/activate"

export PIP_CACHE_DIR="$PIP_CACHE"
export PIP_DISABLE_PIP_VERSION_CHECK=1

python -m pip install --upgrade pip wheel setuptools

# (1) Install vllm FIRST so it gets to pick the torch version it needs.
# vllm 0.10.x pulls torch 2.8.0+cu128 as a hard requirement. Installing it
# AFTER torch 2.7 would have the bad surprise of silently upgrading torch
# and breaking any C-extensions already built against 2.7.
pip install "vllm>=0.10,<0.11"

# (2) flash-attn from prebuilt wheel matched to torch 2.8 + cu12 + cp311 +
# cxx11abi=True. PyPI's flash-attn never has a wheel; only github releases do.
# --force-reinstall is required because pip strips the local-version tag
# (+cu12torch2.8cxx11abiTRUE) when comparing installed vs requested, so a
# stale wheel built against a different torch would be silently kept.
FA_VER="2.8.3"
FA_WHL="flash_attn-${FA_VER}+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FA_VER}/${FA_WHL}"
pip install --no-build-isolation --force-reinstall --no-deps "$FA_URL"

# (3) Editable installs of this repo's rllm and verl (already cloned on host).
pip install --no-deps -e "$PROJECT_ROOT/DeepSWE_RL/rllm"

VERL_DIR="$PROJECT_ROOT/DeepSWE_RL/rllm/verl"
if [[ ! -d "$VERL_DIR" ]]; then
  echo "[build-env] verl not found at $VERL_DIR — clone it on the host first:" >&2
  echo "  git clone --depth 1 --branch release/v0.7.1 https://github.com/volcengine/verl.git $VERL_DIR" >&2
  exit 1
fi
pip install --no-deps -e "$VERL_DIR"

# (4) Runtime deps. Pin numpy<2.0 LAST so it sticks (vllm pulls 2.x).
pip install \
  "transformers>=4.45" \
  "accelerate>=0.34" \
  "datasets>=2.20" \
  "hydra-core>=1.3" \
  "omegaconf>=2.3" \
  "ray[default]>=2.30" \
  "tensorboard" \
  "tensorboardX" \
  "wandb" \
  "codetiming" \
  "polars" \
  "dill" \
  "pyarrow" \
  "pandas" \
  "scipy" \
  "tqdm" \
  "regex" \
  "pyyaml" \
  "psutil" \
  "peft" \
  "deepspeed" \
  "dm-tree" \
  "gym>=0.26.2" \
  "gymnasium[toy-text]>=1.0" \
  "fire" \
  "sentence-transformers" \
  "torchmetrics" \
  "einops" \
  "msgpack" \
  "huggingface_hub" \
  "tensordict>=0.8,<=0.10" \
  "torchdata" \
  "pybind11" \
  "pylatexenc"

# Re-pin numpy<2.0 last (vllm + several deps default to numpy 2.x which
# breaks verl + flash-attn).
pip install "numpy<2.0"

# --- (5) sanity check ----------------------------------------------------
python - <<'PY'
import importlib, sys
for mod in ("torch", "flash_attn", "rllm", "verl", "vllm", "transformers",
            "ray", "hydra", "omegaconf", "tensorboard", "datasets"):
    try:
        m = importlib.import_module(mod)
        print(f"[ok] {mod:<16} {getattr(m,'__version__','?')}  {getattr(m,'__file__','?')[:80]}")
    except Exception as e:
        print(f"[MISS] {mod}: {type(e).__name__}: {e}")
PY
echo "[build-env] DONE"
