#!/bin/bash
# Download SWE-Master-4B-SFT to a local snapshot so compute nodes (offline) can serve it.
# Run on the LOGIN NODE.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

# Use whichever python has huggingface_hub; the serve venv (swebench_pro/.venv) does.
PY=/project/6087623/mohammadA/swebench_pro/.venv/bin/python

mkdir -p "$MODEL_DIR"
"$PY" - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="$MODEL_ID",
    local_dir="$MODEL_DIR",
    local_dir_use_symlinks=False,           # real files: portable + offline-safe
    ignore_patterns=["*.pt", "*.pth", "original/*"],  # keep safetensors only
)
print("downloaded to", p)
PY
echo "=== contents ==="
ls -lah "$MODEL_DIR" | head
echo "Model staged at $MODEL_DIR"
