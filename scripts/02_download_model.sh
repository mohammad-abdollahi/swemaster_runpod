#!/usr/bin/env bash
# Download Qwen3-4B (the model used by the rllm frozenlake example) into the
# HF cache on /scratch. Run inside the container:
#   bash scripts/00_enter_container.sh bash scripts/02_download_model.sh
set -euo pipefail

source "${VENV_BIN:-/scratch/$USER/swemaster/venv/rllm/bin}/activate"

: "${HF_HOME_DIR:?HF_HOME_DIR must be set (00_enter_container.sh exports it)}"
export HF_HOME="$HF_HOME_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_HOME_DIR/hub"

python - <<'PY'
import os
from huggingface_hub import snapshot_download
model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3-4B")
path = snapshot_download(
    repo_id=model_id,
    cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
    allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt", "*.model"],
)
print("[download-model] cached at:", path)
PY
