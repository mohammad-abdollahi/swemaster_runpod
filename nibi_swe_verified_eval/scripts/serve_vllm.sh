#!/bin/bash
# Launch the vLLM OpenAI-compatible server for SWE-Master-4B-SFT.
# Uses the existing swebench_pro/.venv (vllm 0.19.1). Runs in foreground; backgrounded by the job.
set -uo pipefail
source "$(dirname "$0")/../env.sh"

SERVE_PY=/project/6087623/mohammadA/swebench_pro/.venv/bin/python
export HF_HUB_OFFLINE=1   # compute nodes are offline; model is a local snapshot

exec "$SERVE_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 127.0.0.1 --port "$VLLM_PORT" \
  --tensor-parallel-size ${VLLM_TP:-1} \
  --gpu-memory-utilization 0.90 \
  --dtype auto \
  --max-model-len "${VLLM_MAXLEN:-131072}" \
  --enable-prefix-caching
