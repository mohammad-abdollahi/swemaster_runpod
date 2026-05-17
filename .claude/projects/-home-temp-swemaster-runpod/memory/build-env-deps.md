---
name: build-env-deps
description: Two reproducibility fixes for build_env.sh / rllm runtime when rebuilding the swemaster env on a fresh machine
metadata:
  node_type: memory
  type: feedback
---

When rebuilding the SWE-Master env on a fresh machine via `build_env.sh`, two gaps surface every time (both now handled, but verify on the next move):

1. **flash-attn `--no-build-isolation` needs build deps present first.** On a clean venv the flash-attn 2.8.3 build dies with `ModuleNotFoundError: No module named 'wheel'`. Fixed in `build_env.sh` step 10 by `uv pip install wheel packaging` before the flash-attn install. **Why:** `--no-build-isolation` builds against the target venv, so wheel/packaging/setuptools/torch must already be installed there.

2. **rllm is installed `-e --no-deps`, so its runtime deps are NOT pulled.** `build_env.sh`'s import check only imports top-level packages and passes, but the actual trainer pulls a deeper graph. Missing seen so far: `polars dm-tree datasets accelerate peft pylatexenc tensordict torchdata pyarrow` AND `sentence_transformers` (via `rllm.rewards.math_reward` → `rllm.utils`). **How to apply:** after `build_env.sh`, before launching training, run an import sweep of the real runtime modules (`rllm.trainer.verl.train_agent_ppo`, `rllm.rewards.math_reward`, `rllm.utils`, `rllm.agents.sweagent`, `rllm.engine.agent_execution_engine`, `r2egym.agenthub.runtime.docker`) in a loop that pip-installs each `No module named X` until clean — much faster than discovering them one full training relaunch at a time. Consider folding `sentence_transformers` + the polars set into `build_env.sh` step 11.

See [[vm-resources-h100]], [[training-plan]].
