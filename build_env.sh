#!/usr/bin/env bash
# Rebuild the SWE-Master RL environment on this L40S VM (/home/temp).
# Mirrors the verified recipe in SETUP_NOTES.md, with /workspace -> /home/temp
# and the import-side pins/forks that the H200 work established.
set -Eeuo pipefail

REPO=/home/temp/swemaster_runpod
VENV=/home/temp/venv
export UV_CACHE_DIR=/home/temp/uv-cache
export PIP_CACHE_DIR=/home/temp/pip-cache
export HF_HOME=/home/temp/hf-cache
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME"

log() { echo -e "\n========== $* ==========" ; }

log "1. install uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

log "2. create py3.11 venv at $VENV"
uv venv --python 3.11 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python --version

log "3. torch cu128 (pinned first so vllm builds against it)"
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.1 torchvision torchaudio

log "4. clone agentica-project/verl fork (public volcengine fork lacks _timer)"
cd "$REPO/DeepSWE_RL/rllm"
if [ ! -d verl/.git ]; then
  rm -rf verl
  git clone --depth 1 https://github.com/agentica-project/verl.git verl
fi
git -C verl log --oneline -1

log "5. install verl (editable)"
uv pip install -e ./verl

# verl is gitignored (cloned fresh), so its patches live here, not in git.
# Fix the multi-step LoRA<->vLLM resync bug: `'cpu' in torch.device` raises
# TypeError (device not iterable) and crashes every LoRA run at step 2.
# Idempotent.
sed -i "s#orig_dev = 'cpu' if 'cpu' in next(model.parameters()).device else 'cuda'#orig_dev = 'cpu' if next(model.parameters()).device.type == 'cpu' else 'cuda'#g" \
    verl/verl/workers/sharding_manager/fsdp_vllm.py
echo "applied verl fsdp_vllm device-check patch"

log "6. install R2E-Gym (editable)"
uv pip install -e "$REPO/R2E-Gym"

log "7. install swebench forks + swesmith wheels"
uv pip install \
    "$REPO/swebench_fork_swegym-2.0.13-py3-none-any.whl" \
    "$REPO/swebench_fork_swerebench-4.0.3-py3-none-any.whl" \
    "$REPO/swesmith-0.0.7-py3-none-any.whl"

log "8. install rllm (editable, no deps)"
uv pip install -e "$REPO/DeepSWE_RL/rllm" --no-deps

log "9. vllm 0.10.2 (bumps torch -> 2.8 cu128)"
uv pip install vllm==0.10.2

log "10. flash-attn 2.8.3 (no build isolation)"
# --no-build-isolation means flash-attn builds against THIS venv, so its
# build deps (wheel, packaging, setuptools, torch) must already be present.
uv pip install wheel packaging
uv pip install --no-build-isolation flash-attn==2.8.3

log "11. import-side pins established on H200"
uv pip install 'setuptools<81' 'transformers>=4.45,<5' 'huggingface_hub<1' tensorboard

log "12. smoke import verify"
python - <<'PY'
import rllm, verl, r2egym, torch, vllm, flash_attn
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm", vllm.__version__, "flash_attn", flash_attn.__version__)
print("device", torch.cuda.get_device_name(0))
import transformers, huggingface_hub, setuptools
print("transformers", transformers.__version__, "hub", huggingface_hub.__version__, "setuptools", setuptools.__version__)
print("ALL IMPORTS OK")
PY
log "DONE"
