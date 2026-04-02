#!/usr/bin/env bash
set -euo pipefail

# Setup script for GRPO + vLLM on A100 (Ampere sm_80, CUDA 12.4).
#
# Usage:
#   bash scripts/setup/grpo_vllm_venv_a100.sh .venv_grpo_a100
#
# Optional env vars:
#   PYTHON_BIN=python3.12
#   CUDA_TAG=cu124
#   TORCH_VERSION=2.5.1
#   VLLM_VERSION=0.17.1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${1:-.venv_grpo_a100}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
CUDA_TAG="${CUDA_TAG:-cu124}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
VLLM_VERSION="${VLLM_VERSION:-0.17.1}"

cd "${ROOT_DIR}"

echo "=== Creating venv: ${VENV_DIR} ==="
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel uv
python -m pip install --upgrade "setuptools==80.10.2"

# --- PyTorch (cu124 stable for A100) ---
echo "=== Installing PyTorch ${TORCH_VERSION}+${CUDA_TAG} ==="
uv pip install \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
  --extra-index-url https://pypi.org/simple \
  "torch==${TORCH_VERSION}+${CUDA_TAG}" \
  "torchvision" \
  "torchaudio"

# --- Core packages (vLLM from PyPI — pre-built A100 wheels available) ---
echo "=== Installing GRPO + vLLM packages ==="
uv pip install \
  --extra-index-url https://pypi.org/simple \
  -r requirements/grpo_vllm_cuda124.txt

# --- Flash Attention 2 (A100 sm_80 fully supported) ---
# Requires torch headers already installed above.
echo "=== Installing flash-attn ==="
uv pip install flash-attn --no-build-isolation

python -m pip check

echo ""
echo "=== GRPO + vLLM (A100) environment ready at ${VENV_DIR} ==="
echo "Activate: source ${VENV_DIR}/bin/activate"
echo ""
echo "Smoke test:"
echo "  python -c \"import torch, transformers, trl, vllm, flash_attn; \\"
echo "    print(torch.__version__, transformers.__version__, trl.__version__, vllm.__version__, flash_attn.__version__); \\"
echo "    print('CUDA:', torch.cuda.get_device_name(0), '| bf16:', torch.cuda.is_bf16_supported())\""
