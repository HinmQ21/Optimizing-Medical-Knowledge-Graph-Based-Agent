#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/setup/grpo_vllm_venv_dgx_spark.sh .venv_grpo_vllm
# Optional env vars:
#   PYTHON_BIN=python3.12
#   CUDA_TAG=cu130
#   TORCH_VERSION=2.10.0
#   TORCHVISION_VERSION=0.25.0
#   TORCHAUDIO_VERSION=2.10.0
#   VLLM_VERSION=0.17.1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${1:-.venv_grpo_vllm}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
CUDA_TAG="${CUDA_TAG:-cu130}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.10.0}"
VLLM_VERSION="${VLLM_VERSION:-0.17.1}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel uv
python -m pip install --upgrade "setuptools==80.10.2"

uv pip install \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
  --extra-index-url https://pypi.org/simple \
  "torch==${TORCH_VERSION}+${CUDA_TAG}" \
  "torchvision==${TORCHVISION_VERSION}+${CUDA_TAG}" \
  "torchaudio==${TORCHAUDIO_VERSION}+${CUDA_TAG}"

uv pip install \
  --index-url "https://wheels.vllm.ai/${VLLM_VERSION}/${CUDA_TAG}" \
  --extra-index-url https://pypi.org/simple \
  -r requirements/grpo_vllm_cuda13.txt

python -m pip check

echo "GRPO + vLLM environment ready at ${VENV_DIR}"
echo "Activate with: source ${VENV_DIR}/bin/activate"
echo "Smoke test:"
echo "  python -c \"import torch, transformers, trl, vllm; print(torch.__version__, transformers.__version__, trl.__version__, vllm.__version__)\""
