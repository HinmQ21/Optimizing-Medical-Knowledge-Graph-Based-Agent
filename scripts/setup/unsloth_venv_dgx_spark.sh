#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/setup/unsloth_venv_dgx_spark.sh .venv_unsloth
# Optional env vars:
#   PYTHON_BIN=python3.12
#   BUILD_XFORMERS_FROM_SOURCE=1
#   TORCH_CUDA_ARCH_LIST=12.1

VENV_DIR="${1:-.venv_unsloth}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel uv
uv pip install unsloth trl wandb

if [[ "${BUILD_XFORMERS_FROM_SOURCE:-0}" == "1" ]]; then
  python -m pip uninstall -y xformers || true
  python -m pip install ninja
  export TORCH_CUDA_ARCH_LIST
  if [[ ! -d xformers ]]; then
    git clone --depth=1 https://github.com/facebookresearch/xformers --recursive
  fi
  (
    cd xformers
    python setup.py install
  )
fi

echo "Unsloth environment ready at ${VENV_DIR}"
echo "Activate with: source ${VENV_DIR}/bin/activate"
