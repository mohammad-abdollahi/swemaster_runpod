#!/bin/bash
# Build the R2E-Gym *agent* venv on /scratch (where inodes are available).
# This venv runs the agent client (talks to the vLLM server over HTTP). It does NOT need vllm.
# vLLM serving reuses the existing /project/.../swebench_pro/.venv (vllm 0.19.1).
# Run on the LOGIN NODE (needs internet for PyPI).
set -euo pipefail
source "$(dirname "$0")/../env.sh"

module load gcc python/3.11 arrow

if [ ! -x "$VENV/bin/python" ]; then
  python -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install -U pip wheel setuptools

# Curated deps first (skips google-cloud-aiplatform / anthropic[vertex] -> avoids grpc backtracking),
# then install R2E-Gym itself WITHOUT re-resolving its (heavy) declared deps.
pip install -r "$EVAL_ROOT/requirements_agent.txt"
pip install -e "$R2EGYM_DIR" --no-deps

# SWE dataset forks used by docker.py top-level imports (swesmith + the two swebench forks).
RP=/project/6087623/mohammadA/swemaster_runpod
# --no-deps: swebench_fork_swerebench declares ytsaurus-yson which isn't on PyPI; we only
# need these packages importable for docker.py's top-level imports, not their full deps.
pip install --no-deps "$RP/swesmith-0.0.7-py3-none-any.whl" \
            "$RP/swebench_fork_swegym-2.0.13-py3-none-any.whl" \
            "$RP/swebench_fork_swerebench-4.0.3-py3-none-any.whl"

echo "=== sanity: import the pieces we rely on ==="
python - <<'PY'
import swebench, litellm, datasets, r2egym
from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS, FAIL_TO_PASS, PASS_TO_PASS
from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
print("swebench", swebench.__version__, "| datasets", datasets.__version__, "| litellm OK | r2egym OK")
PY
echo "Agent venv ready at $VENV"
