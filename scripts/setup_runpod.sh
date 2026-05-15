#!/usr/bin/env bash
# Bootstraps the SWE-Master RL pipeline on a single-H200 RunPod pod (or any
# linux host with NVIDIA CUDA 12.8+, ~150GB free disk, and uv installed).
#
# It will:
#   1. Create a Python 3.11 venv under ${VENV_DIR}
#   2. Install torch 2.7.1+cu128 first, then the project's editable installs
#   3. Pin the package versions that were verified working on 2026-05-15
#   4. Install vllm 0.10.2 and flash-attn 2.8.3 (the long step)
#
# After this script, run:
#   bash DeepSWE_RL/rllm/examples/swe/smoke_test_4b_single_h200.sh
#
# Variables (override on the command line if needed):
#   REPO_ROOT  - path to this checkout    (default: dir containing this script)
#   VENV_DIR   - venv location            (default: /workspace/venv if /workspace
#                                          exists, else ${REPO_ROOT}/.venv)
#   UV_CACHE_DIR / PIP_CACHE_DIR - cache dirs (default: under ${HOME}/.cache)
set -euxo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}

if [ -d /workspace ] && [ -w /workspace ]; then
    DEFAULT_VENV=/workspace/venv
    DEFAULT_CACHE=/workspace
else
    DEFAULT_VENV="${REPO_ROOT}/.venv"
    DEFAULT_CACHE="${HOME}/.cache"
fi
VENV_DIR=${VENV_DIR:-${DEFAULT_VENV}}
export UV_CACHE_DIR=${UV_CACHE_DIR:-${DEFAULT_CACHE}/uv-cache}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-${DEFAULT_CACHE}/pip-cache}
mkdir -p "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}"

command -v uv >/dev/null || { echo "uv not found - install with: pip install uv"; exit 1; }

# 0) venv -------------------------------------------------------------------
if [ ! -x "${VENV_DIR}/bin/python" ]; then
    uv venv --python 3.11 "${VENV_DIR}"
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

# 1) torch CUDA 12.8 first --------------------------------------------------
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.1 torchvision torchaudio

# 2) clone agentica's verl fork if missing (NOT volcengine - it lacks _timer).
VERL_DIR="${REPO_ROOT}/DeepSWE_RL/rllm/verl"
if [ ! -d "${VERL_DIR}" ]; then
    git clone --depth 1 https://github.com/agentica-project/verl.git "${VERL_DIR}"
fi

# 3) editable installs ------------------------------------------------------
uv pip install -e "${VERL_DIR}"
uv pip install -e "${REPO_ROOT}/R2E-Gym"
uv pip install \
    "${REPO_ROOT}/swebench_fork_swegym-2.0.13-py3-none-any.whl" \
    "${REPO_ROOT}/swebench_fork_swerebench-4.0.3-py3-none-any.whl" \
    "${REPO_ROOT}/swesmith-0.0.7-py3-none-any.whl"
uv pip install -e "${REPO_ROOT}/DeepSWE_RL/rllm" --no-deps

# 4) pinned package versions verified on 2026-05-15 -------------------------
# These pins make the trainer reach fit_agent on a single H200.
#   - pkg_resources removed in setuptools 81+ (verl imports it)
#   - AutoModelForVision2Seq dropped in transformers 5.x (verl FSDP worker imports it)
#   - HfFolder removed in huggingface_hub 1.0 (r2egym imports it)
uv pip install "setuptools<81" "transformers>=4.45,<5" "huggingface_hub<1" tensorboard

# 5) vllm + flash-attn (the long step, ~20 min build for flash-attn) --------
uv pip install vllm==0.10.2
uv pip install --no-build-isolation flash-attn==2.8.3

# 6) sanity check -----------------------------------------------------------
python - <<'PY'
import rllm, verl, r2egym, torch, vllm, flash_attn  # noqa: F401
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("vllm:", vllm.__version__)
print("flash-attn:", flash_attn.__version__)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo "Done. Smoke test entrypoint:"
echo "    bash DeepSWE_RL/rllm/examples/swe/smoke_test_4b_single_h200.sh"
