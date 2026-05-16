#!/usr/bin/env bash
# Add TRL into the existing apptainer venv. Reuses the same venv we built for
# the verl/rllm path — we just layer ``trl`` on top.
#
#   bash scripts/00_enter_container.sh bash TRL_RL/scripts/00_install_trl.sh
set -euo pipefail

VENV=${VENV:-/scratch/$USER/swemaster/venv/rllm}
source "$VENV/bin/activate"

# TRL 0.13+ has GRPOTrainer. Pin to a known-good range and let pip pick a
# version compatible with our transformers (5.x) and torch (2.8).
pip install --no-build-isolation \
    "trl>=0.13,<0.30" \
    "accelerate>=0.34"

python - <<'PY'
import importlib
for m in ("trl", "transformers", "accelerate", "torch", "datasets"):
    mod = importlib.import_module(m)
    print(f"[ok] {m:<14} {getattr(mod, '__version__', '?')}")
from trl import GRPOConfig, GRPOTrainer  # noqa: F401
print("[ok] GRPOTrainer importable")
PY
echo "[install-trl] DONE"
