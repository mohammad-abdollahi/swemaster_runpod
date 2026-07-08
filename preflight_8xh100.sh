#!/usr/bin/env bash
# =============================================================================
# Preflight for the 8x H100 run — validates the box BEFORE you launch, so you
# don't discover a missing piece 40 minutes into init.  Non-destructive.
# =============================================================================
# Usage (from repo root), same env vars you'll launch with:
#   MODEL_PATH=... TRAIN_PARQUET=... VAL_PARQUET=... REFINE_MODE=agentic \
#     bash preflight_8xh100.sh
set -u
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=${REPO_DIR:-${_SCRIPT_DIR}}
PYTHON_BIN=${PYTHON_BIN:-python3}
N_GPUS=${N_GPUS:-8}
MODEL_PATH=${MODEL_PATH:-${REPO_DIR}/models/SWE-Master-4B-SFT}
TRAIN_PARQUET=${TRAIN_PARQUET:-${REPO_DIR}/data_examples/rl_data/swerebench_verified_live/train_00000_verl.parquet}
VAL_PARQUET=${VAL_PARQUET:-${REPO_DIR}/data_examples/rl_data/swerebench_verified_live/test_00000_verl.parquet}
REFINE_MODE=${REFINE_MODE:-agentic}
FAIL=0; WARN=0
pass(){ echo "  [PASS] $*"; }
warn(){ echo "  [WARN] $*"; WARN=$((WARN+1)); }
fail(){ echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }

echo "== 1. GPUs =="
if command -v nvidia-smi >/dev/null 2>&1; then
  n=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
  mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "  detected ${n} GPU(s), ~${mem}MiB each: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  [ "$n" -ge "$N_GPUS" ] && pass "${n} GPUs >= N_GPUS=${N_GPUS}" || fail "only ${n} GPUs but N_GPUS=${N_GPUS}"
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
  [ "${used:-0}" -lt 5000 ] && pass "GPUs idle (max ${used}MiB used)" || warn "a GPU already has ${used}MiB used — free it first"
else fail "nvidia-smi not found"; fi

echo "== 2. Python / torch / vllm =="
if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  pass "python: $($PYTHON_BIN --version 2>&1)"
  "$PYTHON_BIN" - <<'PY' 2>/dev/null && pass "torch+vllm import OK" || fail "torch/vllm import failed under $PYTHON_BIN (activate the venv or set PYTHON_BIN)"
import torch, vllm
assert torch.cuda.is_available(), "cuda not available"
print("    torch", torch.__version__, "| cuda dev:", torch.cuda.device_count(), "| vllm", vllm.__version__)
PY
  "$PYTHON_BIN" -c "import torch;print('    torch sees', torch.cuda.device_count(),'CUDA devices')" 2>/dev/null
else fail "PYTHON_BIN not found: $PYTHON_BIN"; fi

echo "== 3. rLLM / R2E-Gym importable =="
PYTHONPATH="${REPO_DIR}/DeepSWE_RL/rllm:${REPO_DIR}/R2E-Gym/src:${PYTHONPATH:-}" \
  "$PYTHON_BIN" -c "import rllm, r2egym; print('    ok')" 2>/dev/null \
  && pass "rllm + r2egym import OK" || fail "rllm/r2egym not importable (check submodules present under DeepSWE_RL/ and R2E-Gym/)"

echo "== 4. Ray =="
command -v "${RAY_BIN:-ray}" >/dev/null 2>&1 && pass "ray: $(${RAY_BIN:-ray} --version 2>&1 | head -1)" || warn "ray not on PATH (set RAY_BIN or activate venv)"

echo "== 5. Base model =="
if [ -f "${MODEL_PATH}/config.json" ] && ls "${MODEL_PATH}"/*.safetensors >/dev/null 2>&1; then
  pass "model at ${MODEL_PATH} ($(du -sh "${MODEL_PATH}" 2>/dev/null | cut -f1))"
else fail "base model not found/incomplete at ${MODEL_PATH} (need config.json + *.safetensors)"; fi

echo "== 6. Dataset =="
[ -f "$TRAIN_PARQUET" ] && pass "train parquet: ${TRAIN_PARQUET}" || fail "train parquet missing: ${TRAIN_PARQUET}"
[ -f "$VAL_PARQUET" ]   && pass "val parquet:   ${VAL_PARQUET}"   || fail "val parquet missing: ${VAL_PARQUET}"

echo "== 7. Docker + images =="
if command -v docker >/dev/null 2>&1 && docker ps >/dev/null 2>&1; then
  imgs=$(docker images --format '{{.Repository}}' 2>/dev/null | grep -ciE 'sweb|swerebench|r2e' || true)
  tot=$(docker images -q 2>/dev/null | wc -l)
  pass "docker OK — ${tot} images total, ${imgs} look SWE-bench-ish"
  [ "${imgs:-0}" -ge 1 ] || warn "no obvious SWE-bench images matched (check your image naming; training pulls on miss)"
else fail "docker not usable (need daemon up + permission; try: docker ps)"; fi

echo "== 8. Disk =="
freeg=$(df --output=avail -BG "${REPO_DIR}" 2>/dev/null | tail -1 | tr -dc 0-9)
echo "  free at ${REPO_DIR}: ${freeg}G"
[ "${freeg:-0}" -ge 200 ] && pass "disk >=200G free" || warn "only ${freeg}G free — 4B FSDP checkpoints are ~30-60G each"

echo "== 9. Refinement backend (REFINE_MODE=${REFINE_MODE}) =="
case "$REFINE_MODE" in
  none) pass "baseline ATS-RL, no refine backend needed";;
  agentic)
    cb=${RLLM_AGENTIC_CLAUDE_BIN:-$(ls -1 "${HOME}"/.vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude 2>/dev/null | tail -1 || true)}
    cb=${cb:-$(command -v claude || true)}
    if [ -n "$cb" ] && [ -x "$cb" ]; then
      ping=$(env -i HOME="${HOME}" PATH="$PATH" "$cb" -p "Reply with exactly: PONG" --output-format json --max-turns 1 2>/dev/null)
      echo "$ping" | grep -q PONG && pass "claude -p login OK ($cb)" || fail "claude found but login failed — run 'claude login' (or use REFINE_MODE=none)"
    else fail "REFINE_MODE=agentic but no claude binary (set RLLM_AGENTIC_CLAUDE_BIN, or use REFINE_MODE=none)"; fi;;
  online) [ -n "${OPENAI_API_KEY:-}" ] && pass "OPENAI_API_KEY set" || fail "REFINE_MODE=online needs OPENAI_API_KEY";;
esac

echo ""
echo "==================================================="
if [ "$FAIL" -eq 0 ]; then echo "PREFLIGHT OK  (${WARN} warning(s)).  You can launch."
else echo "PREFLIGHT: ${FAIL} FAIL, ${WARN} warn — fix the FAILs before launching."; fi
echo "==================================================="
exit $FAIL
